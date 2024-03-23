from dataclasses import dataclass, field
from typing import Optional
import torch
from .depth_anything_encoder import DepthAnythingVisionTower
from .siglip_encoder import SiglipVisionTower
from .owlv2_encoder import Owlv2VisionTower
from transformers.processing_utils import ProcessorMixin
import torch.nn.functional as F
import math

@dataclass
class VisionConfig:
    image_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=False)
    mm_vision_select_feature: Optional[str] = field(default="patch")

def parse_subargs(args):
    clip_args = VisionConfig()
    clip_args.image_tower = args.vision_expert_clip_image_tower
    clip_args.mm_vision_select_layer = args.vision_expert_clip_mm_vision_select_layer
    clip_args.mm_vision_select_feature = args.vision_expert_clip_mm_vision_select_feature
    da_args = VisionConfig()
    da_args.image_tower = args.vision_expert_depth_anything_image_tower
    da_args.mm_vision_select_layer = args.vision_expert_depth_anything_mm_vision_select_layer
    da_args.mm_vision_select_feature = args.vision_expert_depth_anything_mm_vision_select_feature
    owl_args = VisionConfig()
    owl_args.image_tower = args.vision_expert_owlv2_image_tower
    owl_args.mm_vision_select_layer = args.vision_expert_owlv2_mm_vision_select_layer
    owl_args.mm_vision_select_feature = args.vision_expert_owlv2_mm_vision_select_feature
    return clip_args, da_args, owl_args



class VisualExpertsGroup(torch.nn.Module):
    def __init__(self, image_tower, args, delay_load=False, cache_dir='./cache_dir'):
        super().__init__()

        self.is_loaded = False

        self.image_tower_names = image_tower
        self.experts = torch.nn.ModuleList()
        self.cache_dir = cache_dir
        clip_args, da_args, owl_args = parse_subargs(args)

        self.experts.append(SiglipVisionTower(clip_args.image_tower, clip_args, True, cache_dir))        
        self.experts.append(DepthAnythingVisionTower(da_args.image_tower, da_args, True, cache_dir))
        self.experts.append(Owlv2VisionTower(owl_args.image_tower, owl_args, True, cache_dir))

        if not delay_load:
            self.load_model()
        
    def load_model(self):
        for expert in self.experts:
            expert.load_model()
        self.image_processor = self.experts[1].image_processor
        self.is_loaded = True
    
    def prepare_square_images(self, images, image_processor):
        batch,channel,height,width = images.shape
        target_height = image_processor.size['height']
        target_width = image_processor.size['width']
        if width > height:
            square_images = torch.ones((batch, channel,width,width)).to(device=self.device, dtype=self.dtype)
            square_images[:,0,:,:] *= image_processor.image_mean[0]
            square_images[:,1,:,:] *= image_processor.image_mean[1]
            square_images[:,2,:,:] *= image_processor.image_mean[2]
            height_start = (width - height) // 2
            height_end = width - height_start
            square_images[:,:,height_start:height_end,:] = images
            tl = (0, height_start/width)
            br = (width/width, height_end/width)
        elif width < height:
            square_images = torch.ones((batch, channel,height,height)).to(device=self.device, dtype=self.dtype)
            square_images[:,0,:,:] *= image_processor.image_mean[0]
            square_images[:,1,:,:] *= image_processor.image_mean[1]
            square_images[:,2,:,:] *= image_processor.image_mean[2]
            width_start = (height - width) // 2
            width_end = height - width_start
            square_images[:,:,:,width_start:width_end] = images
            tl = (width_start/height, 0)
            br = (width_end/height, height/height)
        else:
            square_images = images
            tl = (0, 0)
            br = (width/height, height/height)
        square_images = F.interpolate(square_images, size=(target_height, target_width), mode="bilinear", align_corners=False)
        
        square_images[:,0,:,:] -= image_processor.image_mean[0]
        square_images[:,1,:,:] -= image_processor.image_mean[1]
        square_images[:,2,:,:] -= image_processor.image_mean[2]

        square_images[:,0,:,:] /= image_processor.image_std[0]
        square_images[:,1,:,:] /= image_processor.image_std[1]
        square_images[:,2,:,:] /= image_processor.image_std[2]

        return square_images, tl, br

    def stack_features(self, da_images, clip_features, da_features, owl_fetures, tl, br, patch_size):
        batch, owl_h, owl_w, owl_d = owl_fetures.shape
        _, _, da_d = da_features.shape
        clip_h = int(math.sqrt((clip_features.shape[1] + 1)))
        x0 = int(tl[0] * owl_h // 1)
        y0 = int(tl[1] * owl_h // 1)
        x1 = int(br[0] * owl_h // 1)
        y1 = int(br[1] * owl_h // 1)
        h = y1 - y0
        w = x1 - x0
        dummy = torch.zeros((batch,da_d,owl_h,owl_h)).to(device=self.device, dtype=self.dtype)
        dummy[:,:,y0:y1, x0:x1] = F.interpolate(da_features.reshape(batch,da_images.shape[2]//patch_size, da_images.shape[3]//patch_size, da_features.shape[2]).permute(0, 3, 1, 2), size=(h, w), mode="bilinear", align_corners=False)
        dummy = F.interpolate(torch.concat([owl_fetures.permute(0, 3, 1, 2), dummy], dim=1), size=(clip_h, clip_h), mode="bilinear", align_corners=False)
        d_d = dummy.shape[1]
        dummy = dummy.permute(0,2,3,1).reshape(batch,-1,d_d)[:,1:]
        return torch.concat([clip_features, dummy], dim=-1)

    def process(self, image):
        da_features = self.experts[1](image)
        
        image[:,0,:,:] *= self.image_processor.image_std[0]
        image[:,1,:,:] *= self.image_processor.image_std[1]
        image[:,2,:,:] *= self.image_processor.image_std[2]

        image[:,0,:,:] += self.image_processor.image_mean[0]
        image[:,1,:,:] += self.image_processor.image_mean[1]
        image[:,2,:,:] += self.image_processor.image_mean[2]

        clip_image, _, _ = self.prepare_square_images(image, self.experts[0].image_processor)
        owl_image, tl, br = self.prepare_square_images(image, self.experts[2].image_processor)
        
        clip_features = self.experts[0](clip_image)
        owl_fetures = self.experts[2](owl_image)
        return self.stack_features(image, clip_features, da_features, owl_fetures, tl, br, self.experts[0].config.patch_size)

    @torch.no_grad()
    def forward(self, images):

        if type(images) is list:
            image_features = []
            for image in images:
                image_features.append(self.process(image))
        else:
            image_features = self.process(images)    
        
        return image_features

    @property
    def dtype(self):
        return self.experts[1].dtype

    @property
    def device(self):
        return self.experts[1].device