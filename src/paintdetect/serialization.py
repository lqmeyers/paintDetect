"""Model persistence with backward compatibility.

The new canonical format is HuggingFace ``save_pretrained`` (a directory with
``config.json`` + ``model.safetensors``), which round-trips the architecture
(``n_channels``/``n_classes``/``bilinear``) so callers never have to guess it.

``load_model`` also transparently reads the two legacy formats that exist on
disk from the original code:

1. Whole pickled model objects — ``torch.save(model, ...)`` (the old *final*
   save that ``predict.py`` could never load).
2. Raw ``state_dict`` checkpoints — ``torch.save(model.state_dict(), ...)``
   (the old per-epoch ``checkpoint_epoch{N}.pth``), from which the constructor
   args are inferred by inspecting tensor shapes.
"""

import os

import torch

from .model import UNet


def save_model(model, out_dir):
    """Save ``model`` in the canonical HF format (config.json + safetensors).

    Weights are made contiguous first so ``channels_last`` models serialize
    cleanly to safetensors.
    """
    for p in model.parameters():
        if not p.is_contiguous():
            p.data = p.data.contiguous()
    model.save_pretrained(out_dir)
    return out_dir


def _is_pretrained_dir(path):
    return os.path.isdir(path) and os.path.isfile(os.path.join(path, "config.json"))


def _unet_from_state_dict(sd, map_location="cpu"):
    """Rebuild a UNet from a raw state_dict by inferring constructor args.

    - n_channels: input channels of the first conv (``inc.double_conv.0.weight``).
    - n_classes:  output channels of the final 1x1 conv (``outc.conv.weight``).
    - bilinear:   True when the up path has no ConvTranspose weights
      (``up1.up.*`` only exists when bilinear=False).
    """
    sd = dict(sd)
    sd.pop("mask_values", None)  # legacy checkpoints sometimes embedded this
    n_channels = sd["inc.double_conv.0.weight"].shape[1]
    n_classes = sd["outc.conv.weight"].shape[0]
    bilinear = not any(k.startswith("up1.up.") for k in sd)
    net = UNet(n_channels=n_channels, n_classes=n_classes, bilinear=bilinear)
    net.load_state_dict(sd)
    return net.to(map_location)


def load_model(path, map_location="cpu"):
    """Load a UNet from any supported format.

    Accepts: a ``save_pretrained`` directory, a HuggingFace repo id, a legacy
    whole-object ``.pth``, or a legacy state_dict ``.pth``.
    """
    # New HF format (local dir) or a hub repo id.
    if _is_pretrained_dir(path):
        return UNet.from_pretrained(path).to(map_location)
    if not os.path.exists(path):
        # Not a local path: assume a HuggingFace Hub repo id.
        return UNet.from_pretrained(path).to(map_location)

    # Legacy .pth: could be a pickled model object or a state_dict.
    # weights_only=False is required to unpickle whole-object saves; PyTorch 2.6+
    # defaults this to True, which would otherwise refuse to load them.
    obj = torch.load(path, map_location=map_location, weights_only=False)
    if isinstance(obj, UNet):
        return obj.to(map_location)
    if isinstance(obj, dict):
        return _unet_from_state_dict(obj, map_location=map_location)
    raise TypeError(
        f"Unrecognized checkpoint at {path!r}: expected a save_pretrained dir, "
        f"a UNet object, or a state_dict, got {type(obj)}."
    )
