import torch
import torch.nn as nn

from transformers import Owlv2ImageProcessor, Owlv2ForObjectDetection, Owlv2Config


class Owlv2VisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = Owlv2Config.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    def load_model(self):
        self.image_processor = Owlv2ImageProcessor.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = Owlv2ForObjectDetection.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower.requires_grad_(False)

        self.is_loaded = True

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower.image_embedder(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = image_forward_out[0]
                image_features.append(image_feature)
        else:
            image_forward_out = self.image_tower.image_embedder(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = image_forward_out[0]

        return image_features

    @property
    def dtype(self):
        return self.image_tower.dtype

    @property
    def device(self):
        return self.image_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only
