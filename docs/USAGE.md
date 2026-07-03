# paintdetect usage

The project is now an installable package (`src/paintdetect/`). The legacy
`Pytorch-UNet/` scripts still work but are superseded by the package below.

## Install

```bash
# into an existing env that already has a CUDA-matched torch:
pip install -r requirements.txt      # adds hf/safetensors/wandb/opencv etc.
pip install -e .                     # editable install of paintdetect

# optional analysis tools (bee pose estimation, needs OpenCV):
pip install -e .[analysis]
```

## Command line

```bash
# Train (and optionally evaluate) from a config
paintdetect train --config configs/unet_segmentation.yml
paintdetect train --config configs/unet_segmentation.yml --eval

# Inference on images (model = save_pretrained dir, legacy .pth, or HF repo id)
paintdetect predict -m <model> -i bee.jpg -o out.png

# Score a model on the config's test set
paintdetect evaluate --config configs/unet_segmentation.yml -m <model>

# Convert a legacy checkpoint to the HuggingFace format (config.json + safetensors)
paintdetect export -m old_checkpoint.pth -o my-model-dir
```

GPU selection is via the `gpu` field in the YAML (sets `CUDA_VISIBLE_DEVICES`);
set it to `""` to force CPU.

## Python API

```python
from paintdetect import (
    UNet, LabelMeDataset, load_model, save_model,
    predict_img, mask_to_image, train_model,
)

# Load any checkpoint format and run inference
import torch
from PIL import Image
net = load_model("my-model-dir")            # or legacy .pth, or HF repo id
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)
mask = predict_img(net=net, full_img=Image.open("bee.jpg"), device=device, scale_factor=1.0)

# Train from a config
train_model("configs/unet_segmentation.yml", run_eval=True)
```

## Model save/load and HuggingFace

`UNet` is a `huggingface_hub.PyTorchModelHubMixin`, so:

```python
save_model(net, "my-model-dir")             # config.json + model.safetensors
UNet.from_pretrained("my-model-dir")
net.push_to_hub("<username>/paintdetect-<part>")   # upload (requires HF login)
```

`load_model` transparently reads three formats: the new `save_pretrained` dir,
legacy whole-object pickles (`torch.save(model, ...)`), and legacy state_dict
checkpoints (constructor args inferred from tensor shapes). See `MODEL_CARD.md`
for a model-card template + demo snippet.

## What changed vs. the old code

- Single canonical trainer (`paintdetect.train.train_model`) — the near-duplicate
  `train.py` / `train_and_eval.py` were merged; training numerics are unchanged.
- Final models save in the HuggingFace format instead of an unloadable pickle.
- One image-preprocessing function (`preprocess_image`) replaces the two divergent
  `BasicDataset.preprocess` / `LabelMeDataset.preprocess` implementations.
- Analysis tooling (`segmentation`, `bee_angle`) moved under
  `paintdetect.analysis` and no longer runs batch jobs at import time.
- Dead one-off scripts and the old Keras code moved to `archive/`.

## Notebooks (not yet migrated)

`eval_masks.ipynb`, `mask_summary.ipynb`, `background_masking.ipynb` use the
root `utils.py` (kept in place). `batch_predict.ipynb` and `full_pytorch.ipynb`
still reference the legacy `Pytorch-UNet/` tree. These will be updated to the
`paintdetect` API when the legacy tree is removed.
```
