import argparse
import os
import sys
import yaml
import numpy as np

# Python 3.12 removed distutils, but tensorflow still requires it.
try:
    import distutils
except ImportError:
    import setuptools
    sys.modules['distutils'] = setuptools._distutils

import tensorflow_datasets as tfds
from PIL import Image

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    np.random.seed(seed)

def main():
    parser = argparse.ArgumentParser(description="Ingest LFW dataset deterministically.")
    parser.add_argument("--config", type=str, required=True, help="Path to the config file.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)

    print(f"Loading {config['data_source']}...")
    # Load dataset. LFW provides 'train' split default usually, but tfds lfw might not have pre-defined splits other than 'train'
    ds, ds_info = tfds.load("lfw", split="train", with_info=True, as_supervised=False)
    
    # We will build a dictionary of identity -> list of (image_array, original_filename)
    # LFW in TFDS has 'label' which is integer, and 'image' which is array.
    # We need the actual string labels to sort them properly if we want identity names,
    # or we can just use the integer label if we sort deterministically.
    # LFW feature 'label' contains the text identity
    dataset_dict = {}
    print("Iterating over dataset to group by identity...")
    for i, example in enumerate(tfds.as_numpy(ds)):
        identity = example['label'].decode('utf-8')
        image = example['image']
        
        if identity not in dataset_dict:
            dataset_dict[identity] = []
        dataset_dict[identity].append(image)
        
    identities = sorted(list(dataset_dict.keys()))
    print(f"Found {len(identities)} identities and {sum(len(v) for v in dataset_dict.values())} total images.")
    
    # Shuffle identities deterministically
    np.random.shuffle(identities)
    
    # Simple split of identities
    n_total = len(identities)
    n_train = int(n_total * config["train_ratio"])
    n_val = int(n_total * config["val_ratio"])
    
    train_ids = identities[:n_train]
    val_ids = identities[n_train:n_train + n_val]
    test_ids = identities[n_train + n_val:]
    
    splits = {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }
    
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    counts = {}
    
    print("Saving images to disk...")
    for split_name, ids in splits.items():
        split_dir = os.path.join(output_dir, split_name)
        os.makedirs(split_dir, exist_ok=True)
        
        num_images = 0
        for identity in ids:
            identity_dir = os.path.join(split_dir, identity)
            os.makedirs(identity_dir, exist_ok=True)
            
            images = dataset_dict[identity]
            # Since tfds doesn't guarantee order inside identity, we ensure deterministic order by... wait, tfds arrays are indistinguishable.
            # We'll just save them in the order they appeared, which is deterministic given TFDS yields deterministically.
            for i, img_arr in enumerate(images):
                img = Image.fromarray(img_arr)
                img_path = os.path.join(identity_dir, f"{identity}_{i:04d}.jpg")
                img.save(img_path)
                num_images += 1
                
        counts[split_name] = {
            "identities": len(ids),
            "images": num_images
        }
        print(f"{split_name}: {len(ids)} identities, {num_images} images.")
        
    manifest = {
        "seed": seed,
        "split_policy": config["split_policy"],
        "counts": counts,
        "data_source": config["data_source"],
        "cache_location": output_dir
    }
    
    manifest_path = config["manifest_path"]
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)
    with open(manifest_path, 'w') as f:
        yaml.dump(manifest, f, sort_keys=False)
        
    print(f"Manifest written to {manifest_path}")

if __name__ == "__main__":
    main()
