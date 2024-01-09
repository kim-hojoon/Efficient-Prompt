import numpy as np 
import os 
import sys 
sys.path.append("/workspace")
from vgenie.utils.predefined_paths import PREDEFINED_PATHS 
from vgenie.dataset.preprocess.hmdb51 import get_video_names_and_paths, get_ids_from_txt
from tqdm import tqdm 

DATASET = 'hmdb51'
for SPLIT in ["test"]:
    part_feature_path = os.path.join(PREDEFINED_PATHS[DATASET][f'{SPLIT}_feature_dir'], "openai_clip-vit-base-patch16")

    ids = get_ids_from_txt(SPLIT)
    for vid_id in tqdm(ids):
        feature_path = os.path.join("HMDB", f"{vid_id}.npy")
        if os.path.exists(feature_path):
            continue
        feature = np.load(feature_path)
        for frame_idx in range(feature.shape[0]):
            save_feature_name = os.path.join(part_feature_path, f"{vid_id}_o_{frame_idx}.npz")
            np.savez_compressed(save_feature_name, embeddings=feature[frame_idx, :])

    # save to vidid_o_{framenum}.npz
    # >>> a["embeddings"].shape
    # (512,)