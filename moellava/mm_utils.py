from PIL import Image
from io import BytesIO
import base64
from torchvision.transforms import functional as tvf
import torch

from transformers import StoppingCriteria
from moellava.constants import IMAGE_TOKEN_INDEX

def load_image_from_base64(image):
    return Image.open(BytesIO(base64.b64decode(image)))


# def expand2square(pil_img, background_color):
#     width, height = pil_img.size
#     if width == height:
#         result = Image.new("RGBA", (width, width), background_color)
#         result.paste(pil_img, (0, 0))
#         return result
#     elif width > height:
#         result = Image.new("RGBA", (width, width), background_color)
#         result.paste(pil_img, (0, (width - height) // 2))
#         result = np.array(result)
#         height_start = (width - height) // 2
#         height_end = width - (width - height) // 2
#         result[:,:,3] = 0
#         result[height_start:height_end,:,3] = 255
#         return Image.fromarray(result)
#     else:
#         result = Image.new("RGBA", (height, height), background_color)
#         result.paste(pil_img, ((height - width) // 2, 0))
#         result = np.array(result)
#         width_start = (height - width) // 2
#         width_end = height - (height - width) // 2
#         result[:,:,3] = 0
#         result[:,width_start:width_end,3] = 255
#         return Image.fromarray(result)

def expand2square(pil_img, background_color):
    width, height = pil_img.size
    if width == height:
        return pil_img
    elif width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    else:
        result = Image.new(pil_img.mode, (height, height), background_color)
        result.paste(pil_img, ((height - width) // 2, 0))
        return result
    
def longest_resize(pil_img, size, longest_side, pad_to, background_color):
    pil_img = tvf.resize(pil_img, size=size, max_size=longest_side)
    width, height = pil_img.size
    if max(width, height) == longest_side and min(width, height) < size:
        if width < height:
            pad_left = (pad_to - width) // 2
            pad_top = 0
            pad_right = pad_to - width - pad_left
            pad_bottom = 0
        else:
            pad_left = 0
            pad_top = (pad_to - height) // 2
            pad_right = 0
            pad_bottom = pad_to - height - pad_top
        pil_img = tvf.pad(pil_img, (pad_left, pad_top, pad_right, pad_bottom), background_color)
        
    return pil_img


def process_images(images, image_processor, model_cfg):
    image_aspect_ratio = getattr(model_cfg, "image_aspect_ratio", None)
    new_images = []
    if image_aspect_ratio == 'pad':
        for image in images:
            image = expand2square(image, tuple(int(x*255) for x in image_processor.image_mean))
            image = image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            new_images.append(image)
    else:
        return image_processor(images, return_tensors='pt')['pixel_values']
    if all(x.shape == new_images[0].shape for x in new_images):
        new_images = torch.stack(new_images, dim=0)
    return new_images
    


def tokenizer_image_token(prompt, tokenizer, image_token_index=IMAGE_TOKEN_INDEX, return_tensors=None):
    prompt_chunks = [tokenizer(chunk).input_ids for chunk in prompt.split('<image>')]

    def insert_separator(X, sep):
        return [ele for sublist in zip(X, [sep]*len(X)) for ele in sublist][:-1]

    input_ids = []
    offset = 0
    if len(prompt_chunks) > 0 and len(prompt_chunks[0]) > 0 and prompt_chunks[0][0] == tokenizer.bos_token_id:
        offset = 1
        input_ids.append(prompt_chunks[0][0])

    for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
        input_ids.extend(x[offset:])

    if return_tensors is not None:
        if return_tensors == 'pt':
            return torch.tensor(input_ids, dtype=torch.long)
        raise ValueError(f'Unsupported tensor type: {return_tensors}')
    return input_ids


def get_model_name_from_path(model_path):
    model_path = model_path.strip("/")
    model_paths = model_path.split("/")
    if model_paths[-1].startswith('checkpoint-'):
        return model_paths[-2] + "_" + model_paths[-1]
    else:
        return model_paths[-1]

class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.keyword_ids = []
        self.max_keyword_len = 0
        for keyword in keywords:
            cur_keyword_ids = tokenizer(keyword).input_ids
            if len(cur_keyword_ids) > 1 and cur_keyword_ids[0] == tokenizer.bos_token_id:
                cur_keyword_ids = cur_keyword_ids[1:]
            if len(cur_keyword_ids) > self.max_keyword_len:
                self.max_keyword_len = len(cur_keyword_ids)
            self.keyword_ids.append(torch.tensor(cur_keyword_ids))
        self.tokenizer = tokenizer
        self.start_len = input_ids.shape[1]
    
    def call_for_batch(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        offset = min(output_ids.shape[1] - self.start_len, self.max_keyword_len)
        self.keyword_ids = [keyword_id.to(output_ids.device) for keyword_id in self.keyword_ids]
        for keyword_id in self.keyword_ids:
            if (output_ids[0, -keyword_id.shape[0]:] == keyword_id).all():
                return True
        outputs = self.tokenizer.batch_decode(output_ids[:, -offset:], skip_special_tokens=True)[0]
        for keyword in self.keywords:
            if keyword in outputs:
                return True
        return False
    
    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        outputs = []
        for i in range(output_ids.shape[0]):
            outputs.append(self.call_for_batch(output_ids[i].unsqueeze(0), scores))
        return all(outputs)
