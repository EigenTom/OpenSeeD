# example_script.py
import sys
sys.path.append("/home/axton/axton-workspace/csc2125/models/openseed")

from panoseg import PanopticSegmenter



# Define paths and configurations
conf_files = ["/home/axton/axton-workspace/csc2125/models/openseed/configs/openseed/openseed_swint_lang.yaml"]
weight_path = "/home/axton/axton-workspace/csc2125/model_weights/openseed_weights/model_state_dict_swint_51.2ap.pt"  # Path to model weights
thing_classes = ["ostrich", "giraffe", "zebra"]
stuff_classes = ["building", "sky", "river", "ground", "car", "grass"]
# user_dir = "/optional/user/directory"  # Optional user directory
image_path = "/home/axton/axton-workspace/csc2125/models/openseed/images/animals.png"  # Path to the input image
output_path = "/home/axton/axton-workspace/csc2125/models/openseed/output_tmp"  # Path to the output directory

# Instantiate the PanopticSegmenter
segmenter = PanopticSegmenter(
    conf_files=conf_files,
    weight_path=weight_path,
    thing_classes=thing_classes,
    stuff_classes=stuff_classes,
    # user_dir=user_dir
)

# Run inference
segmenter.run_inference(image_path=image_path, output_path=output_path)

print(f"Inference completed. Results are saved in {output_path}")
