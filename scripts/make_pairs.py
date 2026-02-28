import argparse
import os
import yaml
import numpy as np
import csv

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def set_seed(seed):
    np.random.seed(seed)
    
def get_images_per_identity(split_dir):
    """Returns a dict {identity: [list of relative image paths]}"""
    images_dict = {}
    if not os.path.exists(split_dir):
        return images_dict
        
    identities = sorted(os.listdir(split_dir))
    for identity in identities:
        identity_dir = os.path.join(split_dir, identity)
        if os.path.isdir(identity_dir):
            images = sorted(os.listdir(identity_dir))
            # Store paths relative to project root for portability
            images_paths = [os.path.relpath(os.path.join(identity_dir, img)) for img in images if img.endswith('.jpg')]
            if images_paths:
                images_dict[identity] = images_paths
    return images_dict

def generate_pairs_for_split(images_dict, num_pairs, split_name):
    """Generate num_pairs/2 positive and num_pairs/2 negative pairs."""
    identities = sorted(list(images_dict.keys()))
    
    # Positive pairs: need identities with at least 2 images
    pos_identities = [id for id in identities if len(images_dict[id]) >= 2]
    
    pairs = []
    
    # Generate positive pairs
    n_pos = num_pairs // 2
    for _ in range(n_pos):
        if not pos_identities:
            break
        identity = np.random.choice(pos_identities)
        imgs = images_dict[identity]
        idx1, idx2 = np.random.choice(len(imgs), 2, replace=False)
        pairs.append({
            'left_path': imgs[idx1],
            'right_path': imgs[idx2],
            'label': 1,
            'split': split_name
        })
        
    # Generate negative pairs
    n_neg = num_pairs - len(pairs)
    for _ in range(n_neg):
        if len(identities) < 2:
            break
        id1, id2 = np.random.choice(identities, 2, replace=False)
        img1 = np.random.choice(images_dict[id1])
        img2 = np.random.choice(images_dict[id2])
        pairs.append({
            'left_path': img1,
            'right_path': img2,
            'label': 0,
            'split': split_name
        })
        
    return pairs

def main():
    parser = argparse.ArgumentParser(description="Deterministic pair generation.")
    parser.add_argument("--config", type=str, required=True, help="Path to pairs config.")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = config.get("seed", 42)
    set_seed(seed)
    
    data_dir = config["data_dir"]
    output_dir = config["output_dir"]
    os.makedirs(output_dir, exist_ok=True)
    
    split_configs = {
        'train': config.get('train_pairs', 2000),
        'val': config.get('val_pairs', 500),
        'test': config.get('test_pairs', 500)
    }
    
    for split_name, num_pairs in split_configs.items():
        split_dir = os.path.join(data_dir, split_name)
        images_dict = get_images_per_identity(split_dir)
        
        if not images_dict:
            print(f"Warning: No images found for split {split_name} in {split_dir}")
            continue
            
        pairs = generate_pairs_for_split(images_dict, num_pairs, split_name)
        
        output_file = os.path.join(output_dir, f"{split_name}.csv")
        with open(output_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['left_path', 'right_path', 'label', 'split'])
            writer.writeheader()
            writer.writerows(pairs)
            
        print(f"[{split_name}] Generated {len(pairs)} pairs -> {output_file}")
        
    print("Pair generation complete.")

if __name__ == "__main__":
    main()
