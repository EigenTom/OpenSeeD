"""
IMAGES DIR: ./images
OUTPUT DIR: ./output
"""

import os
import sys
import pathlib
from panoseg import PanopticSegmenter

file_dir = str(pathlib.Path(__file__).parent.resolve())
sys.path.append(file_dir)
print(file_dir)

# Define paths and configurations
conf_files = [file_dir + "/configs/openseed/openseed_swint_lang.yaml"]
weight_path = file_dir + "/../../model_weights/openseed_weights/model_state_dict_swint_51.2ap.pt"  # Path to model weights

thing_classes = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
                 "fire hydrant", "street sign", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
                 "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "hat", "backpack", "umbrella", "shoe",
                 "eye glasses", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                 "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "plate",
                 "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                 "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed",
                 "mirror", "dining table", "window", "desk", "toilet", "door", "tv", "laptop", "mouse", "remote",
                 "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "blender", "book",
                 "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush", "hair brush"]
stuff_classes = ["banner", "blanket", "branch", "bridge", "building-other", "bush", "cabinet", "cage", "cardboard",
                 "carpet", "ceiling-other", "ceiling-tile", "cloth", "clothes", "clouds", "counter", "cupboard",
                 "curtain", "desk-stuff", "dirt", "door-stuff", "fence", "floor-marble", "floor-other", "floor-stone",
                 "floor-tile", "floor-wood", "flower", "fogiew", "food-other", "fruit", "furniture-other", "grass",
                 "gravel", "ground-other", "hill", "house", "leaves", "light", "matiew", "metal", "mirror-stuff",
                 "moss", "mountain", "mudiew", "napkin", "netiew", "paper", "pavement", "pillow", "plant-other",
                 "plastic", "platform", "playingfield", "railing", "railroad", "river", "road", "rock", "roof",
                 "rugiew", "salad", "sand", "seaiew", "shelf", "sky-other", "skyscraper", "snow", "solid-other",
                 "stairs", "stone", "straw", "structural-other", "table", "tent", "textile-other", "towel", "tree",
                 "vegetable", "wall-brick", "wall-concrete", "wall-other", "wall-panel", "wall-stone", "wall-tile",
                 "wall-wood", "water-other", "waterdrops", "window-blind", "window-other", "wood"]

# Instantiate the PanopticSegmenter
segmenter = PanopticSegmenter(
    conf_files=conf_files,
    weight_path=weight_path,
    thing_classes=thing_classes,
    stuff_classes=stuff_classes,
)

image_dir = file_dir + '/images/'
images = [f for f in os.listdir(image_dir) if os.path.isfile(image_dir + f)]

for image in images:
    image_path = image_dir + image  # Path to the input image
    output_path = file_dir + "/output/" + os.path.splitext(image)[0]  # Path to the output directory

    try:
        # Run inference
        segmenter.run_inference(image_path=image_path, output_path=output_path)

        print(f"Inference completed. Results are saved in {output_path}")

    except Exception as e:
        print(image_path, e)
