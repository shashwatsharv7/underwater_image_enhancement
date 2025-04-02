import os
import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def calculate_metrics(dataset_path):
    """Calculate metrics for enhanced images compared to reference images"""
    # Define paths
    full_pipeline_dir = os.path.join(dataset_path, 'sequential_pipeline_results')
    reference_dir = os.path.join(dataset_path, 'trainB')
    original_dir = os.path.join(dataset_path, 'trainA')
    
    # Check if directories exist
    if not os.path.exists(full_pipeline_dir):
        raise ValueError(f"Full pipeline results directory not found: {full_pipeline_dir}")
    if not os.path.exists(reference_dir):
        raise ValueError(f"Reference directory not found: {reference_dir}")
    
    # Get list of files in results directory
    result_files = [f for f in os.listdir(full_pipeline_dir) 
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Initialize metrics dataframe
    metrics_df = pd.DataFrame(columns=[
        'filename', 'psnr', 'ssim', 'contrast_improvement', 'brightness_change'
    ])
    
    # Process each file
    for filename in tqdm(result_files, desc="Calculating Metrics"):
        try:
            # Setup paths
            result_path = os.path.join(full_pipeline_dir, filename)
            ref_path = os.path.join(reference_dir, filename)
            orig_path = os.path.join(original_dir, filename)
            
            # Check if reference exists
            if not os.path.exists(ref_path):
                continue
                
            # Load images
            result_img = cv2.imread(result_path)
            ref_img = cv2.imread(ref_path)
            orig_img = cv2.imread(orig_path) if os.path.exists(orig_path) else None
            
            if result_img is None or ref_img is None:
                print(f"Warning: Could not read images for {filename}")
                continue
                
            # Convert to float for calculations
            result_float = result_img.astype(np.float32) / 255.0
            ref_float = ref_img.astype(np.float32) / 255.0
            
            # Calculate PSNR and SSIM
            psnr_value = psnr(ref_float, result_float, data_range=1.0)
            ssim_value = ssim(ref_float, result_float, data_range=1.0, channel_axis=2)
            
            # Calculate contrast and brightness metrics if original is available
            contrast_improvement = None
            brightness_change = None
            
            if orig_img is not None:
                orig_float = orig_img.astype(np.float32) / 255.0
                
                # Calculate contrast improvement
                orig_contrast = np.std(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY))
                result_contrast = np.std(cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY))
                contrast_improvement = (result_contrast - orig_contrast) / orig_contrast
                
                # Calculate brightness change
                orig_brightness = np.mean(cv2.cvtColor(orig_img, cv2.COLOR_BGR2GRAY))
                result_brightness = np.mean(cv2.cvtColor(result_img, cv2.COLOR_BGR2GRAY))
                brightness_change = (result_brightness - orig_brightness) / 255.0
            
            # Add to metrics dataframe
            metrics_df = pd.concat([metrics_df, pd.DataFrame([{
                'filename': filename,
                'psnr': psnr_value,
                'ssim': ssim_value,
                'contrast_improvement': contrast_improvement,
                'brightness_change': brightness_change
            }])], ignore_index=True)
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
    
    # Calculate average metrics
    avg_metrics = metrics_df.mean(numeric_only=True)
    
    # Save metrics to CSV
    metrics_df.to_csv(os.path.join(dataset_path, 'sequential_pipeline_metrics.csv'), index=False)
    
    # Print summary
    print("\nFull Pipeline Enhancement Metrics:")
    print(f"Total Images Evaluated: {len(metrics_df)}")
    print(f"Average PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"Average SSIM: {avg_metrics['ssim']:.3f}")
    
    if 'contrast_improvement' in avg_metrics and not np.isnan(avg_metrics['contrast_improvement']):
        print(f"Average Contrast Improvement: {avg_metrics['contrast_improvement']:.3f}")
    
    if 'brightness_change' in avg_metrics and not np.isnan(avg_metrics['brightness_change']):
        print(f"Average Brightness Change: {avg_metrics['brightness_change']:.3f}")
    
    return metrics_df

def compare_all_methods(dataset_path):
    """Compare metrics from all enhancement methods"""
    # Define paths for different result types
    metrics_paths = {
        'Image Processing': os.path.join(dataset_path, 'enhancement_metrics.csv'),
        'GAN': os.path.join(dataset_path, 'gan_metrics.csv'),
        'Full Pipeline': os.path.join(dataset_path, 'sequential_pipeline_metrics.csv')
    }
    
    # Check which metrics files exist
    available_metrics = {}
    for method, path in metrics_paths.items():
        if os.path.exists(path):
            try:
                metrics = pd.read_csv(path)
                available_metrics[method] = metrics
                print(f"Loaded metrics for {method}: {len(metrics)} images")
            except Exception as e:
                print(f"Error loading metrics for {method}: {str(e)}")
    
    if len(available_metrics) < 2:
        print("Not enough metrics files available for comparison")
        return
    
    # Create comparison table
    comparison = pd.DataFrame(columns=['Method', 'PSNR', 'SSIM', 'Contrast Improvement', 'Brightness Change'])
    
    for method, metrics in available_metrics.items():
        avg_metrics = metrics.mean(numeric_only=True)
        comparison = pd.concat([comparison, pd.DataFrame([{
            'Method': method,
            'PSNR': avg_metrics.get('psnr', np.nan),
            'SSIM': avg_metrics.get('ssim', np.nan),
            'Contrast Improvement': avg_metrics.get('contrast_improvement', np.nan),
            'Brightness Change': avg_metrics.get('brightness_change', np.nan)
        }])], ignore_index=True)
    
    # Print comparison table
    print("\nComparison of Enhancement Methods:")
    print(comparison.to_string(index=False, float_format="%.3f"))
    
    # Save comparison to CSV
    comparison.to_csv(os.path.join(dataset_path, 'methods_comparison.csv'), index=False)
    
    return comparison

def main():
    # Update this path to your dataset directory
    dataset_path = "/Users/shashwatsharv/Dev/Project 2/code/dataset"
    
    # Calculate metrics for full pipeline results
    metrics = calculate_metrics(dataset_path)
    
    # Compare all available methods
    compare_all_methods(dataset_path)

if __name__ == "__main__":
    main()

#Sequential pipeline 1 
# Comparison of Enhancement Methods:
#           Method   PSNR  SSIM  Contrast Improvement  Brightness Change
# Image Processing 16.950 0.707                 0.226              0.031
#              GAN 22.250 0.721                 0.030              0.001
#    Full Pipeline 21.909 0.729                 0.059             -0.013