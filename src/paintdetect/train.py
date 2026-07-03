"""Training entry point.

Single canonical trainer, consolidated from the near-duplicate ``train.py`` and
``train_and_eval.py``. The training loop, optimizer, scheduler, loss, and data
split are preserved exactly (identical numerics); the only changes are:

* the final model is saved via :func:`paintdetect.serialization.save_model`
  (HF ``save_pretrained`` dir) instead of a whole-object pickle with spaces and
  colons in the filename;
* the wandb project name is read from the config instead of hardcoded.

Call :func:`train_model` to train; pass ``run_eval=True`` (or use
``paintdetect train --eval``) to also run the test-set scoring pass afterwards.
"""

import datetime
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import wandb

from .config import load_config
from .data import LabelMeDataset
from .evaluate import evaluate
from .inference import run_test_eval
from .model import UNet
from .serialization import save_model


def train_model(config_file, run_eval=False):
    config = load_config(config_file)
    model_config = config['model_settings']
    train_config = config['train_settings']
    data_config = config['data_settings']

    # set vars from configs
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

    # set datapaths
    dir_img = data_config['data_paths']['dir_train_img']
    dir_mask = data_config['data_paths']['dir_mask']
    dir_xml = data_config['data_paths']['dir_xml']
    dir_checkpoint = train_config['dir_checkpoint']

    # 0. Build model and define device. CUDA_VISIBLE_DEVICES must be set before
    # any CUDA context is created, so keep this at the top of the function.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(train_config['gpu'])
    print('Using GPU', train_config['gpu'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = UNet(n_channels=model_config['num_channels'],
                 n_classes=model_config['num_classes'],
                 bilinear=model_config['bilinear'])
    model = model.to(memory_format=torch.channels_last)
    model.to(device=device)

    # 1. Create dataset
    dataset = LabelMeDataset(dir_img, dir_xml, dir_mask, img_scale)

    # 2. Split into train / validation partitions
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train_set, val_set = random_split(dataset, [n_train, n_val], generator=torch.Generator().manual_seed(0))

    # 3. Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)
    print("number of batches of validation rounds is " + str(len(val_loader)))

    # (Initialize logging)
    experiment = wandb.init(project=train_config.get('wandb_project_name', 'U-Net'),
                            resume='allow')
    experiment.config.update(
        dict(epochs=epochs, batch_size=batch_size, learning_rate=learning_rate,
             val_percent=val_percent, save_checkpoint=save_checkpoint, img_scale=img_scale, amp=amp)
    )

    # 4. Set up optimizer, loss, LR scheduler, and AMP loss scaling
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5, factor=.5, min_lr=1e-6)
    grad_scaler = torch.cuda.amp.GradScaler(enabled=amp)
    criterion = nn.CrossEntropyLoss()

    global_step = 0
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
                true_masks = true_masks.to(device=device, dtype=torch.float32)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    masks_pred = model(images)
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
                        scheduler.step(val_score)

                        logging.info('Validation Dice score: {}'.format(val_score))
                        try:
                            experiment.log({
                                'learning rate': optimizer.param_groups[0]['lr'],
                                'validation loss': val_score,
                                'images': wandb.Image(images[0].cpu()),
                                'masks': {
                                    'true': wandb.Image(true_masks[0].float().cpu()),
                                    'pred': wandb.Image(masks_pred.argmax(dim=1)[0].float().cpu()),
                                },
                                'step': global_step,
                                'epoch': epoch,
                                **histograms
                            })
                        except Exception:
                            pass

        if save_checkpoint:
            Path(dir_checkpoint).mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, os.path.join(dir_checkpoint, 'checkpoint_epoch{}.pth'.format(epoch)))
            logging.info(f'Checkpoint {epoch} saved!')

    # Save final model in the canonical HF format (config.json + safetensors).
    # Filesystem-safe timestamp (no spaces/colons, unlike the old datetime.now()).
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d-%H%M%S")
    out_dir = os.path.join(model_config['model_out_path'],
                           f"{model_config['num_classes']}_classes_{timestamp}")
    save_model(model, out_dir)
    print('Saved final model to', out_dir)

    if run_eval:
        run_test_eval(model, config, device)

    return out_dir
