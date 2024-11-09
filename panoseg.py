# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Modified as a reusable class with post-processing for mask smoothing
# --------------------------------------------------------

import os
import sys
import logging
import argparse
import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

class PanopticSegmenter:
    def __init__(self, conf_files, weight_path, thing_classes, stuff_classes, user_dir=None):
        # Load options and setup paths
        # opt, _ = load_opt_command(['--conf_files'] + conf_files + ['--overrides', 'WEIGHT', weight_path])
        opt, _ = load_opt_command(['evaluate'] + ['--conf_files'] + conf_files + ['--overrides', 'WEIGHT', weight_path])    
        
        if user_dir:
            absolute_user_dir = os.path.abspath(user_dir)
            opt['user_dir'] = absolute_user_dir
        
        # Set up model
        self.model = BaseModel(opt, build_model(opt)).from_pretrained(weight_path).eval().cuda()
        
        # Set up classes and metadata
        self.thing_classes = thing_classes
        self.stuff_classes = stuff_classes
        self.metadata = self._create_metadata(thing_classes, stuff_classes)
        
        # Initialize image transformations
        self.transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])
    
    def _create_metadata(self, thing_classes, stuff_classes):
        # Generate colors for classes and map ids for segmentation
        thing_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(thing_classes))]
        stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(stuff_classes))]
        
        thing_dataset_id_to_contiguous_id = {x: x for x in range(len(thing_classes))}
        stuff_dataset_id_to_contiguous_id = {x + len(thing_classes): x for x in range(len(stuff_classes))}
        
        # Register metadata with Detectron2
        MetadataCatalog.get("demo").set(
            thing_colors=thing_colors,
            thing_classes=thing_classes,
            thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
            stuff_colors=stuff_colors,
            stuff_classes=stuff_classes,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
        )
        
        # Prepare model metadata
        metadata = MetadataCatalog.get('demo')
        self.model.model.metadata = metadata
        self.model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
        self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
        
        return metadata

    def _post_process_mask(self, mask):
        # Fill holes and smooth edges using OpenCV
        kernel = np.ones((5, 5), np.uint8)
        # Close small holes inside the mask
        closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # Smooth the edges
        smoothed = cv2.GaussianBlur(closed, (5, 5), 0)
        return smoothed

    def run_inference(self, image_path, output_path):
        os.makedirs(output_path, exist_ok=True)
        
        # Load and preprocess image
        image_ori = Image.open(image_path).convert("RGB")
        image = self.transform(image_ori)
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).cuda()
        
        # Create visualizer and prepare input for model inference
        visualizer = Visualizer(np.asarray(image_ori), metadata=self.metadata)
        batch_inputs = [{'image': image_tensor, 'height': image_ori.height, 'width': image_ori.width}]
        
        with torch.no_grad():
            outputs = self.model.forward(batch_inputs)
            pano_seg, pano_seg_info = outputs[-1]['panoptic_seg']
        
        # Update category IDs in `pano_seg_info`
        for obj in pano_seg_info:
            if obj['category_id'] in self.metadata.thing_dataset_id_to_contiguous_id:
                obj['category_id'] = self.metadata.thing_dataset_id_to_contiguous_id[obj['category_id']]
            else:
                obj['isthing'] = False
                obj['category_id'] = self.metadata.stuff_dataset_id_to_contiguous_id[obj['category_id']]
        
        # Generate and save the panoptic segmentation result
        pano_result = visualizer.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info).get_image()
        pano_output_path = os.path.join(output_path, 'panoptic_result.png')
        Image.fromarray(pano_result).save(pano_output_path)
        logger.info(f"Saved panoptic segmentation result to {pano_output_path}")
        
        # Save individual masks in category folders
        category_counts = {}
        for obj in pano_seg_info:
            # Determine category name and create folder if not exists
            category_id = obj['category_id']
            is_thing = obj['isthing']
            category_name = self.thing_classes[category_id] if is_thing else self.stuff_classes[category_id - len(self.thing_classes)]
            category_folder = os.path.join(output_path, category_name)
            os.makedirs(category_folder, exist_ok=True)
            
            # Generate mask, apply post-processing, and save it with sequential naming
            mask = (pano_seg.cpu().numpy() == obj['id']).astype(np.uint8) * 255
            processed_mask = self._post_process_mask(mask)
            object_count = category_counts.get(category_name, 0) + 1
            category_counts[category_name] = object_count
            mask_filename = f"{object_count:03d}.jpg"
            mask_output_path = os.path.join(category_folder, mask_filename)
            mask_image = Image.fromarray(processed_mask)
            mask_image.save(mask_output_path)
            logger.info(f"Saved mask for {category_name} to {mask_output_path}")

# Command-line interface to run inference with the class
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run panoptic segmentation on a single image.")
    parser.add_argument("--conf_files", nargs='+', required=True, help="Path(s) to the config file(s).")
    parser.add_argument("--overrides", nargs=2, help="Weight override in cmdline")
    parser.add_argument("--thing_classes", required=True, help="Space-separated list of object categories.")
    parser.add_argument("--stuff_classes", required=True, help="Space-separated list of background categories.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--user_dir", type=str, required=False, help="Path to user directory.")
    
    args = parser.parse_args()
    
    # Initialize the PanopticSegmenter
    segmenter = PanopticSegmenter(
        conf_files=args.conf_files,
        weight_path=args.overrides[1],
        thing_classes=args.thing_classes.split(),
        stuff_classes=args.stuff_classes.split(),
        user_dir=args.user_dir
    )
    
    # Run inference
    segmenter.run_inference(args.image_path, args.output_path)
