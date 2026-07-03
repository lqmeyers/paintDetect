---
license: mit
tags:
  - image-segmentation
  - unet
  - pytorch
  - biology
library_name: paintdetect
---

# paintdetect U-Net — honeybee segmentation

U-Net for multi-class semantic segmentation of honeybee images: paint marks
(used for individual bee ID) and/or body parts (head, thorax, abdomen).
Built on a fork of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet).

## Model details

- **Architecture:** U-Net (`paintdetect.model.UNet`), transposed-conv upsampling.
- **Input:** RGB image, `n_channels=3`.
- **Output:** `n_classes` logit channels (multi-label; classes are not mutually
  exclusive). Take `argmax` over channels for a single-label mask, or threshold
  per channel for multi-label masks.
- **Config:** `n_channels`, `n_classes`, `bilinear` are stored in `config.json`.

<!-- Fill in per trained model: -->
- **num_classes:** `<N>` (e.g. 4)
- **Classes:** `<list, e.g. head, thorax, abdomen, paint>`
- **Training data:** `<dataset description, e.g. 192 CVAT/LabelMe-annotated images>`
- **Metrics:** `<e.g. macro IoU / Dice on held-out test set>`

## Usage

```python
from paintdetect import load_model, predict_img
from PIL import Image
import torch

net = load_model("<hf-repo-id-or-local-dir>")   # or a legacy .pth path
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net.to(device)

img = Image.open("bee.jpg")
mask = predict_img(net=net, full_img=img, device=device, scale_factor=1.0)
# `mask` is an (H, W) array of class indices.
```

Or from the command line:

```bash
paintdetect predict -m <hf-repo-id-or-local-dir> -i bee.jpg -o out.png
```

## Loading / saving

```python
from paintdetect import UNet, save_model, load_model

# Save (HuggingFace format: config.json + model.safetensors)
save_model(net, "my-model-dir")
UNet.from_pretrained("my-model-dir")     # or load_model("my-model-dir")

# push_to_hub is available via the PyTorchModelHubMixin:
# net.push_to_hub("<username>/paintdetect-<part>")
```

`load_model` also reads legacy checkpoints from the original codebase (whole-object
pickles and raw state_dicts), inferring the architecture automatically. Convert
an old checkpoint to this format with:

```bash
paintdetect export -m old_checkpoint.pth -o my-model-dir
```

## Limitations

- Trained on a specific imaging setup; may not generalize to different
  backgrounds, lighting, or bee species.
- Multi-label design: overlapping classes (e.g. paint over thorax) are expected.
