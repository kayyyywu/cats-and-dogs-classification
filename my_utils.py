import torch
import clip
import xml.etree.ElementTree as ET
from PIL import Image


# Check if CUDA (GPU) is available, otherwise use CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
# Load the CLIP model and preprocessing pipeline
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

target_class_ids = [15, 16, 17, 18, 19, 20, 21, 22]
CLASSES = [ 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 
            'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
            'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 
            'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
            'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
            'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 
            'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
            'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 
            'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
            'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
            'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 
            'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 
            'scissors', 'teddy bear', 'hair drier', 'toothbrush' ]

def make_square(im, min_size=256, fill_color=(0, 0, 0, 0)):
    x, y = im.size
    size = max(min_size, x, y)
    new_im = Image.new('RGBA', (size, size), fill_color)
    new_im.paste(im, (int((size - x) / 2), int((size - y) / 2)))
    return new_im
