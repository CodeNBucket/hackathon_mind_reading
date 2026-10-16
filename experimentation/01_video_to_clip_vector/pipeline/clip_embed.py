"""Stage 2: frames -> CLIP embeddings (512 numbers per frame, and one per video).

This 512-d vector is what an EEG decoder would be trained to predict.
"""
import torch
from transformers import CLIPModel, CLIPProcessor

MODEL_NAME = "openai/clip-vit-base-patch32"


def load():
    return CLIPModel.from_pretrained(MODEL_NAME).eval(), CLIPProcessor.from_pretrained(MODEL_NAME)


@torch.no_grad()
def embed(model, proc, images, texts):
    """Returns (frame_vectors [n,512], video_vector [512], text_vectors [k,512], logit_scale)."""
    inp = proc(text=texts, images=images, return_tensors="pt", padding=True)
    out = model(**inp)
    frames = out.image_embeds / out.image_embeds.norm(dim=-1, keepdim=True)
    text = out.text_embeds / out.text_embeds.norm(dim=-1, keepdim=True)
    video = frames.mean(0)
    video = video / video.norm()
    return frames, video, text, model.logit_scale.exp()
