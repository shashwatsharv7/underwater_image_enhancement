import nbimporter
import os
import numpy as np
from tqdm import tqdm  # For progress bars
from traditional_pipeline import full_pipeline, preprocess # type: ignore
from evaluation import calculate_metrics # type: ignore

# Configuration
DATASET_ROOT = "dataset"
TRAIN_A_PATH = os.path.join(DATASET_ROOT, "trainA")
TRAIN_B_PATH = os.path.join(DATASET_ROOT, "trainB")
RESULTS_FILE = "pipeline_performance.txt"

def evaluate_pipeline(num_samples=10000):
    """Evaluate pipeline on all images in trainA with ground truth in trainB"""
    psnr_values = []
    ssim_values = []
    processed_count = 0
    
    # Get list of images with matching pairs
    valid_images = []
    for img_name in os.listdir(TRAIN_A_PATH):
        gt_path = os.path.join(TRAIN_B_PATH, img_name)
        if os.path.exists(gt_path):
            valid_images.append(img_name)
    
    # Limit samples if specified
    if num_samples is not None:
        valid_images = valid_images[:num_samples]
    
    print(f"Evaluating pipeline on {len(valid_images)} image pairs...")
    
    for img_name in tqdm(valid_images, desc="Processing images"):
        try:
            # Process image
            input_path = os.path.join(TRAIN_A_PATH, img_name)
            results = full_pipeline(input_path)
            
            # Get ground truth
            gt_path = os.path.join(TRAIN_B_PATH, img_name)
            gt_image = preprocess(gt_path)
            
            # Calculate metrics
            metrics = calculate_metrics(results['final'], gt_image)
            
            psnr_values.append(metrics['PSNR'])
            ssim_values.append(metrics['SSIM'])
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_name}: {str(e)}")
            continue
    
    # Calculate statistics
    stats = {
        'total_images': len(valid_images),
        'processed': processed_count,
        'mean_psnr': np.mean(psnr_values),
        'std_psnr': np.std(psnr_values),
        'mean_ssim': np.mean(ssim_values),
        'std_ssim': np.std(ssim_values),
    }
    
    # Save results
    with open(RESULTS_FILE, 'w') as f:
        f.write("Pipeline Performance Report\n")
        f.write("===========================\n")
        f.write(f"Total image pairs: {stats['total_images']}\n")
        f.write(f"Successfully processed: {stats['processed']}\n")
        f.write(f"Mean PSNR: {stats['mean_psnr']:.2f} ± {stats['std_psnr']:.2f} dB\n")
        f.write(f"Mean SSIM: {stats['mean_ssim']:.4f} ± {stats['std_ssim']:.4f}\n")
    
    print(f"\nEvaluation complete! Results saved to {RESULTS_FILE}")
    return stats

if __name__ == "__main__":
    # Example: Evaluate on first 50 images for quick test
    test_stats = evaluate_pipeline(num_samples=None)
    
    # For full evaluation (remove num_samples parameter)
    # full_stats = evaluate_pipeline()