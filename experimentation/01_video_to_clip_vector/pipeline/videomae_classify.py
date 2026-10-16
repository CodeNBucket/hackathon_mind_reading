"""Optional stage: 16 frames -> VideoMAE action label (Kinetics-400) + 768-d clip feature."""
import numpy as np
import torch
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor

MODEL_NAME = "MCG-NJU/videomae-base-finetuned-kinetics"


def load():
    return (VideoMAEForVideoClassification.from_pretrained(MODEL_NAME).eval(),
            VideoMAEImageProcessor.from_pretrained(MODEL_NAME))


@torch.no_grad()
def classify(model, proc, images, top_k=5):
    """Returns (top-k [(label, prob)], pooled feature [768])."""
    inp = proc([np.array(im) for im in images], return_tensors="pt")
    out = model(**inp, output_hidden_states=True)
    probs = out.logits.softmax(-1)[0]
    top = probs.topk(top_k)
    pooled = out.hidden_states[-1].mean(1)[0]  # (1568 tokens, 768) -> (768,)
    labels = [(model.config.id2label[i], p) for p, i in zip(top.values.tolist(), top.indices.tolist())]
    return labels, pooled
