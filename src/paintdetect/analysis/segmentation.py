"""Chain multiple part-models (head, thorax, full) over one image.

WIP tooling ported from the repo-root ``segmentation.py``. Cleaned up: no
``sys.path`` hacks, no import-time ``CUDA_VISIBLE_DEVICES`` mutation, model
loading routed through :func:`paintdetect.serialization.load_model` (so it
accepts save_pretrained dirs, legacy pickles, or state_dicts), and the abandoned
abstract-base-class experiment dropped.

``model_lib`` is a dict mapping part names to model paths. Paths are
user-supplied (the historical registry pointed at machine-specific, gitignored
locations); fill it in for your machine.
"""

import torch
import PIL.JpegImagePlugin
from PIL import Image

from ..serialization import load_model
from ..inference import predict_img, mask_to_image

# Example registry — replace paths with models available on your machine.
trained_models = {
    'full': '',
    'head': '',
    'paint': '',
    'thorax': '',
    'abdomen': '',
    'background': '',
}


class Segmentation:
    def __init__(self, model_lib, image):
        if not isinstance(image, PIL.JpegImagePlugin.JpegImageFile):
            image = Image.open(image)
        self.model_lib = model_lib
        self.image = image
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # Legacy checkpoints never stored mask_values, so this stays [0, 1].
        self.mask_values = [0, 1]
        self.head = self.mask_2_image(self.predict(self.image, self.load_model('head')))
        self.thorax = self.mask_2_image(self.predict(self.image, self.load_model('thorax')))
        self.full_segmentation = self.mask_2_image(self.predict(self.image, self.load_model('full')))

    def load_model(self, model_key):
        path = self.model_lib[model_key]
        net = load_model(path, map_location=self.device)
        net.to(device=self.device)
        return net

    def predict(self, image, model):
        """Return a class-index segmentation array."""
        self.current_model = model
        return predict_img(net=model, full_img=image, scale_factor=1,
                           out_threshold=.5, device=self.device)

    def mask_2_image(self, mat):
        """Return an RGB PIL.Image from a class map."""
        return mask_to_image(mat, self.mask_values)
