import torch
import torch.nn as nn

from transformers import AutoImageProcessor, DepthAnythingConfig, DepthAnythingForDepthEstimation


class DepthAnythingVisionTower(nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        self.select_layer = args.mm_vision_select_layer
        self.cache_dir = cache_dir

        if not delay_load:
            self.load_model()
        else:
            self.cfg_only = DepthAnythingConfig.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)

    def load_model(self):
        self.image_processor = AutoImageProcessor.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower = DepthAnythingForDepthEstimation.from_pretrained(self.image_tower_name, cache_dir=self.cache_dir)
        self.image_tower.requires_grad_(False)

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        return image_features

    @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature[:, 1:])
        else:
            image_forward_outs = self.image_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)

        return image_features[:, 1:]

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

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

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2
