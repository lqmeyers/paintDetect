import argparse
import logging
import os
import random
import datetime
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
#import torchvision.transforms as transforms
#import torchvision.transforms.functional as TF
from pathlib import Path
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import yaml 

import wandb
from evaluate import evaluate, pixel_wise_confusion_matrix
from unet.unet_model import UNet
from utils.data_loading import *
from utils.dice_score import dice_loss, calculate_iou
from utils.dice_score import multiclass_dice_coeff, dice_coeff


def predict_img(net,
                full_img,
                device,
                scale_factor=1,
                out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(None, full_img, scale_factor, is_mask=False))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
    output = F.interpolate(output, (full_img.size[1], full_img.size[0]), mode='bilinear')
    if net.n_classes > 1:
        mask = output.argmax(dim=1)
    else:
        mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()



def mask_to_image(mask: np.ndarray, mask_values):
    if isinstance(mask_values[0], list):
        out = np.zeros((mask.shape[-2], mask.shape[-1], len(mask_values[0])), dtype=np.uint8)
    elif mask_values == [0, 1]:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=bool)
    else:
        out = np.zeros((mask.shape[-2], mask.shape[-1]), dtype=np.uint8)

    if mask.ndim == 3:
        mask = np.argmax(mask, axis=0)

    for i, v in enumerate(mask_values):
        out[mask == i] = v

    return Image.fromarray(out)

def threshold_predictions(predictions, threshold=0.5):
    # Set elements above the threshold to 1, and below or equal to the threshold to 0
    binary_array = (predictions > threshold).astype(int)
    return binary_array


def train_and_eval(config_file):
    
    #Load config file yml 
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f)
        model_config = config['model_settings'] # settings for model building
        train_config = config['train_settings'] # settings for model training
        data_config = config['data_settings'] # settings for data loading
        eval_config = config['eval_settings'] # settings for evaluation
        torch_seed = config['torch_seed']
        verbose = config['verbose']
    except Exception as e:
        print('ERROR - unable to open experiment config file. Terminating.')
        print('Exception msg:',e)
        return -1
    
    #set vars from configs
    epochs = train_config['epochs']
    batch_size = train_config['batch_size']
    learning_rate = float(train_config['learning_rate'])
    val_percent = train_config['val_percent']
    save_checkpoint = train_config['save_checkpoint']
    img_scale = data_config['img_scale']
    amp = train_config['amp']
    weight_decay = float(train_config['weight_decay'])
    momentum = train_config['momentum']
    gradient_clipping = train_config['gradient_clipping']

    #set datapaths
    dir_img = data_config['data_paths']['dir_train_img']
    dir_mask = data_config['data_paths']['dir_mask']
    dir_test_mask = data_config['data_paths']['dir_test_mask']
    dir_xml = data_config['data_paths']['dir_xml']
    dir_checkpoint = train_config['dir_checkpoint']
    
    # 0 Build model abd define device
    os.environ["CUDA_VISIBLE_DEVICES"]=str(train_config['gpu'])
    print('Using GPU',train_config['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # config num_classes counts foreground classes only.
    # Binary (1 foreground class) -> a single sigmoid channel; background is implicit (sigmoid < 0.5).
    # Multiclass (>1) -> one channel per foreground class plus an explicit background channel.
    num_fg_classes = model_config['num_classes']
    n_total_classes = 1 if num_fg_classes == 1 else num_fg_classes + 1
    model = UNet(n_channels=model_config['num_channels'], n_classes=n_total_classes, bilinear=model_config['bilinear'])
    model = model.to(memory_format=torch.channels_last)

    logging.info(f'Network:\n'
                    f'\t{model.n_channels} input channels\n'
                    f'\t{model.n_classes} output channels (classes)\n'
                    f'\t{"Bilinear" if model.bilinear else "Transposed conv"} upscaling')
    model.to(device=device)
    
    # 1. Create dataset
    if dir_xml is not None:
        dataset = LabelMeDataset(dir_img,dir_xml,dir_mask,img_scale)
    else:
        dataset = BasicDataset(dir_img, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    print("number of batches of validation rounds is "+str(len(val_loader)))

    # (Initialize logging)
    experiment = wandb.init(dir=train_config['wandb_dir_path'], entity=train_config['wandb_entity_name'], project=train_config['wandb_project_name'])
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
            val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {learning_rate}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_checkpoint}
        Device:          {device.type}
        Images scaling:  {img_scale}
        Mixed Precision: {amp}
    ''')
    
    # 4. Set up the optimizer, the loss, the learning rate scheduler and the loss scaling for AMP
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    #optimizer = optim.RMSprop(model.parameters(),
                          #    lr=learning_rate, weight_decay=weight_decay, momentum=momentum, foreach=True)
    # evaluate() returns validation loss (lower is better), so use mode='min'
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5,factor=.5,min_lr=1e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    # single-channel sigmoid + BCE for binary; softmax + cross-entropy for multiclass
    criterion = nn.BCEWithLogitsLoss() if model.n_classes == 1 else nn.CrossEntropyLoss()
    
    global_step = 0
    #"""
    # 5. Begin training
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch}/{epochs}', unit='img') as pbar:
            for batch in train_loader:

                images, true_masks = batch['image'], batch['mask']

                assert images.shape[1] == model.n_channels, \
                    f'Network has been defined with {model.n_channels} input channels, ' \
                    f'but loaded images have {images.shape[1]} channels. Please check that ' \
                    'the images are loaded correctly.'

                images = images.to(device=device, dtype=torch.float32, memory_format=torch.channels_last)
                true_masks = true_masks.to(device=device, dtype=torch.long)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
                    if model.n_classes == 1:
                        # BCE learns the foreground only; squeeze the channel dim to match (B,H,W) float targets
                        loss = criterion(masks_pred.squeeze(1), true_masks.float())
                    else:
                        loss = criterion(masks_pred, true_masks)

                optimizer.zero_grad(set_to_none=True)
                grad_scaler.scale(loss).backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
                grad_scaler.step(optimizer)
                grad_scaler.update()

                pbar.update(images.shape[0])
                global_step += 1
                epoch_loss += loss.item()
                experiment.log({
                    'train loss': loss.item(),
                    'step': global_step,
                    'epoch': epoch
                })
                pbar.set_postfix(**{'loss (batch)': loss.item()})

                # Evaluation round
                division_step = (n_train // (5 * batch_size))
                if division_step > 0:
                    if global_step % division_step == 0:
                        histograms = {}
                        for tag, value in model.named_parameters():
                            tag = tag.replace('/', '.')
                            if not (torch.isinf(value) | torch.isnan(value)).any():
                                histograms['Weights/' + tag] = wandb.Histogram(value.data.cpu())
                            if not (torch.isinf(value.grad) | torch.isnan(value.grad)).any():
                                histograms['Gradients/' + tag] = wandb.Histogram(value.grad.data.cpu())

                        val_score = evaluate(model, val_loader, device, amp)
                        print("score of validation round is "+str(val_score))
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            # build a 2D (H,W) predicted mask for visualization
                            if model.n_classes == 1:
                                pred_vis = (torch.sigmoid(masks_pred[0, 0]) > 0.5).float()
                            else:
                                pred_vis = masks_pred.argmax(dim=1)[0].float()
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    # wandb.Image permutes torch tensors as 3D CHW, so pass 2D masks as numpy
                                    'true': wandb.Image(true_masks[0].float().cpu().numpy()),
                                    'pred': wandb.Image(pred_vis.cpu().numpy()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception as e:
                            print(e)
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            #state_dict['mask_values'] = dataset.mask_values
            torch.save(state_dict, os.path.join(dir_checkpoint,'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')
    #"""

    #save_model as .pth
    model_name = model_config['model_out_path']+str(model_config['num_classes'])+'_classes_'+str(datetime.datetime.now())+'.pth'
    torch.save(model,model_name)

    ##--------------- Run prediction on test set and eval model------------------
 
    # 0. Parse eval configs 
    dir_test_img = data_config['data_paths']['dir_test_img']
    save_dir = eval_config['save_dir']

    #---------Ok I am going to do this img-wise for now with option to parrelelize later
    # 1. Create test dataset and dataloader 
    # test_set = LabelMeDataset(dir_test_img,dir_xml,dir_mask,img_scale)
    # test_loader = DataLoader(test_set, shuffle=True, **loader_args)

    # 2. Initialize dirs to save predicted masks
    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    directory_name = os.path.join(save_dir,'predict'+current_date)
    
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)
    
    in_files = []
    out_files = []

    dir_list = os.walk(dir_test_img)
    for root, dirs, files in dir_list:
        for f in files: 
            #print(root+f)
            in_files.append(root+f)
            out_files.append(os.path.join(directory_name,f[:-5]+eval_config['prediction_suffix']+'.conf_matr.npy'))

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    model.eval()

    # number of foreground masks scored per image: 1 for binary, one per foreground class otherwise
    num_fg_classes = 1 if model.n_classes == 1 else model.n_classes - 1
    results_array = np.zeros((len(in_files), num_fg_classes))
    for idx, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        if dir_xml is not None:
            xml_file_true = dir_xml+os.path.basename(filename)[:-4]+'.xml'
            mask_true = assemble_mask_from_xml(xml_file_true,dir_mask)
        else:
           mask_true = np.array(Image.open(glob(dir_test_mask+os.path.basename(filename)[:-4]+'.*.png')[0]))
           # masks are stored as 0/255; binarize to 0/1 for IoU and confusion matrix
           mask_true = (mask_true > 0).astype(np.int8)

        img = torch.from_numpy(BasicDataset.preprocess(None, img, img_scale, is_mask=False))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(img)  # (1, n_classes, H, W)

        # build one binary predicted mask per foreground class
        if model.n_classes == 1:
            prob = torch.sigmoid(output)[0, 0].detach().cpu().numpy()
            pred_masks = [(prob > 0.5).astype(np.int8)]
        else:
            # per-class argmax over softmax; skip background class 0.
            # NOTE: per-class IoU is only meaningful with per-class ground truth (dir_xml path);
            # against a single binary GT mask only the binary case is fully meaningful.
            labels = output.softmax(dim=1).argmax(dim=1)[0].detach().cpu().numpy()
            pred_masks = [(labels == c).astype(np.int8) for c in range(1, model.n_classes)]

        cf_array = np.zeros((num_fg_classes,) + mask_true.shape, dtype=np.int8)
        iou_for_sample = np.zeros(num_fg_classes, dtype=np.float64)
        for i, mask in enumerate(pred_masks):
            iou_for_sample[i] = calculate_iou(mask_true, mask)
            cf_array[i] = pixel_wise_confusion_matrix(mask, mask_true)

        out_filename = out_files[idx]
        logging.info(f'Mask saved to {out_filename}')

        #SAVE ARRAY TO .NPY FILE
        print('saving file',out_filename[:-4]+'.npy')
        np.save(out_filename[:-4]+'.npy',cf_array)

        #Aggregate eval_scores and display
        results_array[idx] = iou_for_sample
    print('Total IoUs calculated in shape: ',results_array.shape)
    print(idx)
 
    # Calculate macro-average
    macro_average = np.mean(results_array)
    # f'Macro-average of IoU for all classes across {idx} files: {macro_average}'

    # Calculate class-wise averages
    class_wise_averages = np.mean(results_array, axis=0)
    # for ic, c in enumerate(class_wise_averages):
    #     print(f'Class{ic} average IoU: {c}')

    # Calculate histogram
    histogram, bin_edges = np.histogram(class_wise_averages, bins=10)

    # Display results
    print("Macro Average:", macro_average)
    print("Class-wise Averages:", class_wise_averages)

    # Display histogram
    print("\nHistogram:")
    for i in range(len(bin_edges) - 1):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {'#' * histogram[i]}")
    
    # Save iou file 
    results_path = os.path.join(directory_name,'results_ious.npy')
    np.save(results_path,results_array)


print("beginning execution")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_file", help="yaml file with experiment settings", type=str)
    args = parser.parse_args()
    
    try:
        train_and_eval(args.config_file)
    except MemoryError:
        logging.error('Detected OutOfMemoryError! '
                      'Enabling checkpointing to reduce memory usage, but this slows down training. '
                      'Consider enabling AMP (--amp) for fast and memory efficient training')
        torch.cuda.empty_cache()
        #model.use_checkpointing()
        train_and_eval(args.config_file)
