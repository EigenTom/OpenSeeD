# --------------------------------------------------------
# X-Decoder -- Generalized Decoding for Pixel, Image, and Language
# Modified for terminal execution and category-based output folders
# --------------------------------------------------------

import os
import sys

pth = '/'.join(sys.path[0].split('/')[:-1])
sys.path.insert(0, pth)
sys.path.append("/home/axton/axton-workspace/csc2125/models/OpenSeeD")

import logging
import argparse
from PIL import Image
import numpy as np
import torch
from torchvision import transforms
from utils.arguments import load_opt_command
from detectron2.data import MetadataCatalog
from detectron2.utils.colormap import random_color
from openseed.BaseModel import BaseModel
from openseed import build_model
from utils.visualizer import Visualizer

logger = logging.getLogger(__name__)

def main(args=None):
    # Load options and setup paths
    opt, cmdline_args = load_opt_command(args)    
    
    if cmdline_args.user_dir:
        absolute_user_dir = os.path.abspath(cmdline_args.user_dir)
        opt['user_dir'] = absolute_user_dir

    args = cmdline_args
    
    pretrained_pth = os.path.join(opt['WEIGHT'])
    os.makedirs(args.output_path, exist_ok=True)

    # Model setup
    model = BaseModel(opt, build_model(opt)).from_pretrained(pretrained_pth).eval().cuda()

    # Set up transformations
    transform = transforms.Compose([transforms.Resize(512, interpolation=Image.BICUBIC)])

    # Load classes from arguments and prepare metadata
    thing_classes = args.thing_classes.split()
    stuff_classes = args.stuff_classes.split()
    
    print(thing_classes)
    print(stuff_classes)
    
    thing_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(thing_classes))]
    stuff_colors = [random_color(rgb=True, maximum=255).astype(np.int_).tolist() for _ in range(len(stuff_classes))]
    thing_dataset_id_to_contiguous_id = {x:x for x in range(len(thing_classes))}
    stuff_dataset_id_to_contiguous_id = {x+len(thing_classes):x for x in range(len(stuff_classes))}

    
    MetadataCatalog.get("demo").set(
        thing_colors=thing_colors,
        thing_classes=thing_classes,
        thing_dataset_id_to_contiguous_id=thing_dataset_id_to_contiguous_id,
        stuff_colors=stuff_colors,
        stuff_classes=stuff_classes,
        stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
    )
    
    # Prepare model metadata
    model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(thing_classes + stuff_classes, is_eval=False)
    metadata = MetadataCatalog.get('demo')
    model.model.metadata = metadata
    model.model.sem_seg_head.num_classes = len(thing_classes + stuff_classes)
    
    with torch.no_grad():
        # Load and preprocess image
        image_ori = Image.open(args.image_path).convert("RGB")
        image = transform(image_ori)
        image_tensor = torch.from_numpy(np.asarray(image)).permute(2, 0, 1).cuda()

        # Create visualizer and draw overall panoptic segmentation
        visualizer = Visualizer(np.asarray(image_ori), metadata=metadata)
        
        # Run inference
        batch_inputs = [{'image': image_tensor, 'height': image_ori.height, 'width': image_ori.width}]
        outputs = model.forward(batch_inputs)
        pano_seg, pano_seg_info = outputs[-1]['panoptic_seg']
        
        for i in range(len(pano_seg_info)):
            if pano_seg_info[i]['category_id'] in metadata.thing_dataset_id_to_contiguous_id.keys():
                pano_seg_info[i]['category_id'] = metadata.thing_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
            else:
                pano_seg_info[i]['isthing'] = False
                pano_seg_info[i]['category_id'] = metadata.stuff_dataset_id_to_contiguous_id[pano_seg_info[i]['category_id']]
        
        pano_result = visualizer.draw_panoptic_seg(pano_seg.cpu(), pano_seg_info).get_image()
        
        # Save panoptic segmentation result
        pano_output_path = os.path.join(args.output_path, 'panoptic_result.png')
        Image.fromarray(pano_result).save(pano_output_path)
        logger.info(f"Saved panoptic segmentation result to {pano_output_path}")

        # Save individual masks in category folders
        category_counts = {}
        for obj in pano_seg_info:
            # Determine category name and create a folder for it if it doesn't exist
            is_thing = obj['isthing']
            category_id = obj['category_id']
            category_name = thing_classes[category_id] if is_thing else stuff_classes[category_id - len(thing_classes)]
            category_folder = os.path.join(args.output_path, category_name)
            os.makedirs(category_folder, exist_ok=True)

            # Generate mask and save it with sequential naming
            mask = (pano_seg.cpu().numpy() == obj['id']).astype(np.uint8) * 255
            object_count = category_counts.get(category_name, 0) + 1
            category_counts[category_name] = object_count
            mask_filename = f"{object_count:03d}.jpg"
            mask_output_path = os.path.join(category_folder, mask_filename)
            mask_image = Image.fromarray(mask)
            mask_image.save(mask_output_path)
            logger.info(f"Saved mask for {category_name} to {mask_output_path}")

if __name__ == "__main__":
    main()
    sys.exit(0)

    """
    python /home/axton/axton-workspace/csc2125/models/OpenSeeD/demo/demo_panoseg_cli.py evaluate \
    --conf_files "configs/openseed/openseed_swint_lang.yaml" \
    --thing_classes "ostrich giraffe zebra" \
    --stuff_classes "building sky river ground car grass" \
    --output_path "/home/axton/axton-workspace/csc2125/models/OpenSeeD/output_tmp" \
    --image_path "/home/axton/axton-workspace/csc2125/models/OpenSeeD/images/animals.png" \
    --overrides WEIGHT "/home/axton/axton-workspace/csc2125/models/OpenSeeD/ckpt/model_state_dict_swint_51.2ap.pt"
    
    
    """
    