import json
import os
import matplotlib.pyplot as plt
from PIL import Image

def plot_pairs(pairs, title, save_path):
    if not pairs:
        return
    
    # We plot top 3 pairs
    n = min(len(pairs), 3)
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n))
    fig.suptitle(title, fontsize=14)
    
    # Handle single row case
    if n == 1:
        axes = [axes]
        
    for i in range(n):
        pair = pairs[i]
        try:
            img_left = Image.open(pair['left'])
            img_right = Image.open(pair['right'])
            
            axes[i][0].imshow(img_left)
            axes[i][0].axis('off')
            axes[i][0].set_title(f"Score: {pair['score']:.3f}", fontsize=10)
            
            axes[i][1].imshow(img_right)
            axes[i][1].axis('off')
        except Exception as e:
            print(f"Could not load images for pair {i}: {e}")
            
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    errors_file = "outputs/eval/errors_Run5-PostChange-Test-Eval.json"
    if not os.path.exists(errors_file):
        print(f"File not found: {errors_file}")
        return
        
    with open(errors_file, 'r') as f:
        data = json.load(f)
        
    fps = data.get("false_positives", [])
    fns = data.get("false_negatives", [])
    
    # Sort FP: we want the HIGHEST scores (model was very confident they were the same, but they are not)
    fps_sorted = sorted(fps, key=lambda x: x['score'], reverse=True)
    
    # Sort FN: we want the LOWEST scores (model was very confident they were different, but they are the same)
    fns_sorted = sorted(fns, key=lambda x: x['score'])
    
    os.makedirs("reports", exist_ok=True)
    
    print("Plotting top 3 False Positives...")
    plot_pairs(fps_sorted, "False Positives (Different people, high score)", "reports/false_positives_examples.png")
    
    print("Plotting top 3 False Negatives...")
    plot_pairs(fns_sorted, "False Negatives (Same person, low score)", "reports/false_negatives_examples.png")

if __name__ == "__main__":
    main()
