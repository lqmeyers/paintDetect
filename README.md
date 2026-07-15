# paintDetect

**UNet-based semantic segmentation of bee images** — trained to detect paint spots (used to
identify individual bees), as well as related targets such as thorax, head, and background.
Built on top of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet) and adapted
into a configurable, YAML-driven training/evaluation pipeline.

<p align="left">Sample prediction:</p>

- *Image name: f17x2022_06_28.mp4.track000206.frame006589.jpg*
- *Dice coefficient: 0.9417218543046357*

<p align="center"><img src=https://github.com/lqmeyers/paintDetect/assets/107192889/ab8ebf3e-6554-4e12-8e5d-10017724facb /></p>
<p align="center">Green = True Mask, Blue = Predicted Mask, Red = Overlap</p>

<!--
<p align="left">Training visualized with Weights &amp; Biases:</p>
<p align="center"><img src=https://github.com/lqmeyers/paintDetect/assets/107192889/1e0c1838-2022-434a-a7fe-715eb1411c46 width="600" height="300" border="10"/></p>
-->

---

## Repository structure

```
paintDetect/
├── Pytorch-UNet/                       # all active code lives here
│   ├── train_and_eval.py               # ★ main entry point: train → save → predict → eval
│   ├── train.py                        # legacy training script (multiclass/LabelMe, out of sync)
│   ├── evaluate.py                     # validation loss + pixel-wise confusion matrix
│   ├── predict.py                      # original upstream CLI predictor (argparse-based)
│   ├── UNet_segmentation_train_config.yml   # example experiment config
│   ├── unet/                           # model definition
│   │   ├── unet_model.py               # UNet (also a HuggingFace PyTorchModelHubMixin)
│   │   └── unet_parts.py               # DoubleConv / Down / Up / OutConv building blocks
│   └── utils/
│       ├── data_loading.py             # BasicDataset & LabelMeDataset
│       ├── dice_score.py               # IoU, Dice coefficient/loss
│       └── utils.py                    # plotting helpers
├── notebooks/                          # analysis & batch inference (see below)
├── data/                               # images + masks (gitignored)
├── checkpoints/ · models/ · wandb/     # training artifacts (gitignored)
└── old/ · rerun_2026/                  # scratch / run outputs (gitignored)
```

---

## Data layout & conventions

Data lives under `data/` (gitignored). Each **segmentation target** gets its own mask folder,
and every folder is split into `training/` and `testing/`:

```
data/
├── images/            {training,testing}/   # RGB source images (.jpg)
├── paint_only_masks/  {training,testing}/   # paint-spot masks   ← default target
├── thorax_only_masks/ {training,testing}/
├── head_only_masks/   {training,testing}/
├── full_masks/        {training,testing}/
└── background_masks/   {training,testing}/
```

<!-- Add hf dataset link here -->
Conventions the data loaders rely on:

- **Filename matching.** A mask is matched to its image by the image's base name:
  `image = <name>.jpg`, `mask = <name>.<suffix>.png`. The loader globs `<name>.*` in the mask
  dir, so there must be exactly one match per image.
- **Mask values.** Ground-truth masks are stored as `0` / `255` PNGs and binarized to `0` / `1`
  during evaluation. `BasicDataset` scans the mask folder to derive its class values.
- **Prediction outputs** are written to timestamped `predict<datetime>/` folders (you'll see
  many of these already under each `*_masks/` dir).

Point the pipeline at a target by editing `dir_mask` / `dir_test_mask` in the config (e.g.
`paint_only_masks` vs. `thorax_only_masks`).


### Hardware

Full-resolution UNet training needs a real GPU (the config default is `img_scale: 1`, i.e. no
downsampling). On a machine with only a small GPU, run smoke tests on **CPU** by setting
`gpu: ""` in the config and disabling W&B with `WANDB_MODE=disabled`.

---

## Configuration

Everything is driven by a single YAML file — there are (almost) no command-line flags. The
example is `Pytorch-UNet/UNet_segmentation_train_config.yml`. Sections:

| Section | Key settings |
| --- | --- |
| top-level | `torch_seed`, `verbose` |
| `model_settings` | `num_channels` (3=RGB), `num_classes` (**foreground classes only**), `bilinear`, `model_out_path` |
| `train_settings` | `epochs`, `batch_size`, `learning_rate`, `val_percent`, checkpointing, `gpu`, `amp`, and W&B (`wandb_project_name`, `wandb_entity_name`, `wandb_dir_path`) |
| `data_settings` | `img_scale`, and `data_paths` (`dir_train_img`, `dir_test_img`, `dir_mask`, `dir_test_mask`, `dir_xml`) |
| `eval_settings` | `save_dir`, `prediction_suffix`, `out_threshold` |

**`num_classes` counts foreground classes only** 

- `num_classes: 1` (binary) → the network has a single sigmoid output channel; background is
  implicit (`sigmoid < 0.5`). Loss is `BCEWithLogitsLoss`.
- `num_classes: N > 1` (multiclass) → the network gets `N + 1` output channels (one explicit
  background channel + one per foreground class), softmax + `CrossEntropyLoss`.

---

## Training + evaluation

The main workflow is a single end-to-end script that trains, saves the model, then predicts
and scores the test set:

```bash
cd Pytorch-UNet
python train_and_eval.py --config_file UNet_segmentation_train_config.yml
```

What it does:

1. Builds the `UNet`, choosing output channels from `num_classes` (see above).
2. Loads data via `BasicDataset` (image dir + mask dir) — or `LabelMeDataset` if `dir_xml` is
   set — and splits off `val_percent` for validation (seeded).
3. Trains with Adam + `ReduceLROnPlateau`, logging to W&B.
4. Saves per-epoch `state_dict` checkpoints to `dir_checkpoint`, and the final whole model as
   `<model_out_path><num_classes>_classes_<timestamp>.pth`.
5. Runs inference over the test set and writes evaluation outputs (below).

## Evaluation outputs

Written under a timestamped `predict<datetime>/` folder inside `eval_settings.save_dir`:

- **Per-image confusion matrices** — `<name>.pred.conf_matr.npy`, a pixel-wise map where
  `0=FN, 1=FP, 2=TN, 3=TP` (`pixel_wise_confusion_matrix` in `evaluate.py`).
- **`results_ious.npy`** — per-image (and per-class) IoU array.
- Printed **macro-average** and **class-wise** IoU, plus an ASCII histogram.

---

## Prediction (standalone)

`predict.py` is the original upstream CLI:

```bash
cd Pytorch-UNet
python predict.py --model MODEL.pth --input img1.jpg img2.jpg --output out1.png out2.png \
                  --scale 1 --mask-threshold 0.5 --classes 2
```

<!-- mention loading the model from the hub -->

For batch prediction over a folder, the notebooks (below) are generally the path used in
practice.

---

## Notebooks

Under `notebooks/` — used for batch inference and post-hoc analysis:

- **`batch_predict.ipynb`** — run a trained model over a directory of images.
- **`eval_masks.ipynb`** — evaluate / compare predicted vs. ground-truth masks.
- **`mask_summary.ipynb`** — summary statistics over masks.
- **`background_masking.ipynb`** — background-mask experiments.
- **`full_pytorch.ipynb`** — end-to-end exploratory training/inference.

---

## Model persistence & HuggingFace

Models are saved two ways: per-epoch `state_dict` `.pth` checkpoints, and a final whole-object
pickle (`torch.save(model, ...)`). Because `UNet` subclasses
`huggingface_hub.PyTorchModelHubMixin`, `save_pretrained(...)` and `push_to_hub(...)` are also
available for sharing weights via the Hub.

---

## License

See [`LICENSE`](LICENSE).
