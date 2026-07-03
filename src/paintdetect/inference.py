"""Inference and standalone test-set evaluation.

``predict_img`` / ``mask_to_image`` are the single-image inference helpers
(previously duplicated across ``predict.py`` and ``train_and_eval.py``).
``run_test_eval`` is the test-set scoring pass that used to be bolted onto the
end of ``train_and_eval.py`` — it walks a test image dir, compares predictions
to the LabelMe ground truth, and writes per-image confusion matrices + IoUs.
"""

import datetime
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from .data import assemble_mask_from_xml, preprocess_image
from .evaluate import pixel_wise_confusion_matrix
from .metrics import calculate_iou


def predict_img(net, full_img, device, scale_factor=1, out_threshold=0.5):
    """Predict a single-image mask (argmax over classes), upscaled to full size."""
    net.eval()
    img = torch.from_numpy(preprocess_image(full_img, scale_factor))
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
    """Convert a class-index mask into a PIL image via a per-class value LUT."""
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
    """Binarize raw network outputs at ``threshold``."""
    return (predictions > threshold).astype(int)


def run_test_eval(model, config, device):
    """Predict over the test set and score per-class IoU + confusion matrices.

    Reproduces the evaluation section of the original ``train_and_eval.py``:
    writes one ``.npy`` confusion-matrix array per image plus an aggregate
    ``results_ious.npy`` under a timestamped output directory, and prints the
    macro / class-wise IoU summary.
    """
    data_config = config['data_settings']
    eval_config = config['eval_settings']
    model_config = config['model_settings']

    dir_xml = data_config['data_paths']['dir_xml']
    dir_mask = data_config['data_paths']['dir_mask']
    dir_test_img = data_config['data_paths']['dir_test_img']
    img_scale = data_config['img_scale']
    save_dir = eval_config['save_dir']
    num_classes = model_config['num_classes']

    current_date = datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    directory_name = os.path.join(save_dir, 'predict' + current_date)
    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    in_files = []
    out_files = []
    for root, dirs, files in os.walk(dir_test_img):
        for f in files:
            in_files.append(os.path.join(root, f))
            out_files.append(os.path.join(
                directory_name, f[:-5] + eval_config['prediction_suffix'] + '.conf_matr.npy'))

    model.eval()

    results_array = np.zeros((len(in_files), num_classes))
    for idx, filename in enumerate(in_files):
        logging.info(f'Predicting image {filename} ...')
        img = Image.open(filename)
        xml_file_true = dir_xml + os.path.basename(filename)[:-4] + '.xml'
        mask_true = assemble_mask_from_xml(xml_file_true, dir_mask)

        img = torch.from_numpy(preprocess_image(img, img_scale))
        img = img.unsqueeze(0)
        img = img.to(device=device, dtype=torch.float32)

        with torch.no_grad():
            output = model(img)

        output = output.squeeze().detach().cpu().numpy()
        output = threshold_predictions(output)

        cf_array = np.zeros_like(mask_true)
        iou_for_sample = np.array([], dtype=np.float64)
        for i, mask in enumerate(output):
            iou = calculate_iou(mask_true[i], mask)
            iou_for_sample = np.append(iou_for_sample, iou)
            cf = pixel_wise_confusion_matrix(mask, mask_true[i])
            cf_array[i] = cf

        out_filename = out_files[idx]
        np.save(out_filename[:-4] + '.npy', cf_array)
        logging.info(f'Confusion matrix saved to {out_filename}')

        iou_for_sample = iou_for_sample.reshape(1, -1)
        results_array[idx] = iou_for_sample

    # Aggregate and display
    macro_average = np.mean(results_array)
    class_wise_averages = np.mean(results_array, axis=0)
    histogram, bin_edges = np.histogram(class_wise_averages, bins=10)

    print("Macro Average:", macro_average)
    print("Class-wise Averages:", class_wise_averages)
    print("\nHistogram:")
    for i in range(len(bin_edges) - 1):
        print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {'#' * histogram[i]}")

    results_path = os.path.join(directory_name, 'results_ious.npy')
    np.save(results_path, results_array)
    return results_array
