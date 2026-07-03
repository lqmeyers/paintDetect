# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Purpose

Trains and runs U-Net models for semantic segmentation of honeybee images — detecting paint marks (used for individual bee ID) and body parts (head, thorax, abdomen). Built on a customized fork of [milesial/Pytorch-UNet](https://github.com/milesial/Pytorch-UNet). Multi-class models predict background + N classes in a single network.

## Commands

The code is now an installable package (`src/paintdetect/`). Install and use the `paintdetect` CLI:

```bash
pip install -r requirements.txt && pip install -e .   # add pip install -e .[analysis] for the pose tools

# Train (add --eval to also score the test set); config-driven
paintdetect train --config configs/unet_segmentation.yml

# Inference (model = save_pretrained dir, legacy .pth, or HF repo id)
paintdetect predict -m <model> -i input.jpg -o out.png

# Score a model on the config test set; convert a legacy checkpoint to HF format
paintdetect evaluate --config configs/unet_segmentation.yml -m <model>
paintdetect export -m old_checkpoint.pth -o my-model-dir
```

See `docs/USAGE.md` for the Python API. GPU selection is via the `gpu` field in the YAML config (sets `CUDA_VISIBLE_DEVICES`, `""` forces CPU), **not** a command-line flag. There is no test suite or linter; smoke-test with a 1-epoch CPU run.

The legacy `Pytorch-UNet/` tree (`train_and_eval.py`, `predict.py`, run from inside that dir) still works but is superseded by the package. Dead one-off scripts live in `archive/`.

## Configuration

`configs/unet_segmentation.yml` is the single source of truth for a run (the legacy `Pytorch-UNet/UNet_segmentation_train_config.yml` is the pre-refactor copy). It is split into `model_settings`, `train_settings`, `data_settings`, and `eval_settings`. Training and data paths (image dirs, mask dir, XML dir) are all set here — copy and edit this file per experiment rather than passing flags. `num_classes` in `model_settings` must match the number of object classes in the LabelMe XML exports. `paintdetect.config.load_config` normalizes the historical `out_threshold=` typo key.

## Data pipeline (the key architectural concept)

Masks are **not** stored as single label images. They come from CVAT "LabelMe 3.0" exports:

- `dir_img` — the input `.jpg` images.
- `dir_mask` — a folder of individual **binary** PNG masks, one per object/class instance.
- `dir_xml` — one `.xml` per image; each XML lists the object class names and the relative path to that class's binary mask (`object/name` and `object/segm/mask`).

`LabelMeDataset` (in `Pytorch-UNet/utils/data_loading.py`) matches images to their XML by filename, then `assemble_mask_from_xml()` reads each referenced binary mask, thresholds it, and stacks them into a multi-class array. Class 0 is a synthesized background channel (bitwise-OR of all object masks, inverted). When adding/changing data, the XML→mask path linkage is what must stay consistent, not a naming convention on the masks themselves.

`data_loading.py` also contains the upstream `BasicDataset`/`CarvanaDataset` classes — these are unused legacy code from the original repo; the active path is `LabelMeDataset`.

## Model save/load conventions

The package standardizes on the HuggingFace format: `save_model()` writes a `save_pretrained` directory (`config.json` + `model.safetensors`), and `UNet` is a `PyTorchModelHubMixin` (so `from_pretrained`/`push_to_hub` work). Always load via `paintdetect.serialization.load_model(path)` — it transparently reads the new format **and** the two legacy formats still on disk: whole-object pickles (`torch.save(model, ...)`, the old `train_and_eval.py` final save) and raw `state_dict` checkpoints (constructor args inferred from tensor shapes). Convert legacy weights with `paintdetect export`.

The legacy `Pytorch-UNet/predict.py` still expects a bare `state_dict` and hardcodes `n_channels=3`; prefer the package. `current_models.py` (repo root) is a stale per-part registry with `/home/lmeyers/...` paths — treat paths as user-supplied.

## Downstream / analysis code (`paintdetect.analysis`)

- `paintdetect.analysis.segmentation` — `Segmentation` class chaining head/thorax/full part-models over one image (loads via `load_model`). WIP.
- `paintdetect.analysis.pose` — bee body orientation from head+thorax masks (`get_angle(...)`; `batch_process(...)` for bulk). Needs OpenCV (`[analysis]` extra). Import is side-effect free (the old `bee_angle.py` ran a batch job at import).
- `paintdetect.viz` — `overlay_masks` / `plot_img_and_mask`. `paintdetect.analysis.paths` — `getPath`/`getName`.
- Root `utils.py`, `segmentation.py`, `bee_angle.py`, `current_models.py` are the pre-refactor originals, kept only until the notebooks are migrated off them.

## Experiment tracking

Training logs to Weights & Biases (`wandb`). The package trainer reads `wandb_project_name` from `train_settings` (the old `train_and_eval.py` hardcoded `project='U-Net'`). Run artifacts accumulate under `wandb/` (gitignored); set `WANDB_MODE=disabled` for offline smoke tests.

## Gitignored (won't appear but are expected to exist locally)

`data/`, `models/`, `checkpoints/`, `wandb/`, and all `*.pth`, `*.jpg` files are gitignored. A checkout will not contain training data or model weights.
