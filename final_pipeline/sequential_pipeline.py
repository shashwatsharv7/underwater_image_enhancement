# GAN Trained using enhanced image from image processing pipeline

import os
import cv2
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd

# Import your existing image processing pipeline
from image_processing_pipeline import RefinedUnderwaterEnhancer

class SequentialPipeline:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.original_dir = os.path.join(dataset_path, 'trainA')
        self.reference_dir = os.path.join(dataset_path, 'trainB')
        self.sequential_pipeline_dir = os.path.join(dataset_path, 'sequential_pipeline_results')
        
        # Create output directory
        os.makedirs(self.sequential_pipeline_dir, exist_ok=True)
        
        # Initialize the image processing enhancer
        self.image_enhancer = RefinedUnderwaterEnhancer(dataset_path)
        
        # Load the trained GAN model
        self.gan_model_path = os.path.join(dataset_path, 'checkpoints', 'best_generator.h5')
        self.generator = tf.keras.models.load_model(self.gan_model_path)
        print(f"Loaded GAN model from {self.gan_model_path}")
        
        # Image size for GAN model
        self.img_size = 128
    
    def gan_enhancement(self, img):
        """Apply GAN enhancement to an image"""
        # Resize for the network
        original_size = img.shape[:2]
        img_resized = cv2.resize(img, (self.img_size, self.img_size))
        
        # Normalize to [-1, 1]
        img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
        
        # Generate enhanced image
        input_tensor = tf.expand_dims(img_normalized, 0)
        generated = self.generator(input_tensor, training=False)
        
        # Convert back to [0, 255]
        generated = ((generated[0].numpy() + 1) * 127.5).astype(np.uint8)
        
        # Resize back to original size
        if original_size != (self.img_size, self.img_size):
            generated = cv2.resize(generated, (original_size[1], original_size[0]))
        
        return generated
    
    def process_dataset(self):
        """Process all original images through the sequential pipeline"""
        # Get list of original images
        original_files = [f for f in os.listdir(self.original_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        total_processed = 0
        metrics = []
        
        for filename in tqdm(original_files, desc="Processing Sequential Pipeline"):
            try:
                # Setup paths
                original_path = os.path.join(self.original_dir, filename)
                reference_path = os.path.join(self.reference_dir, filename)
                output_path = os.path.join(self.sequential_pipeline_dir, filename)
                
                # Load original image
                original = cv2.imread(original_path)
                if original is None:
                    print(f"Warning: Could not read image {original_path}")
                    continue
                
                # Step 1: Apply image processing pipeline
                ip_enhanced = self.image_enhancer.enhance_image(original_path)
                
                # Step 2: Apply GAN enhancement
                final_enhanced = self.gan_enhancement(ip_enhanced)
                
                # Save the result
                cv2.imwrite(output_path, final_enhanced)
                
                # Calculate metrics if reference exists
                if os.path.exists(reference_path):
                    reference = cv2.imread(reference_path)
                    if reference is not None:
                        # Calculate metrics
                        ref_float = reference.astype(np.float32)/255
                        final_float = final_enhanced.astype(np.float32)/255
                        orig_float = original.astype(np.float32)/255
                        
                        psnr_value = psnr(ref_float, final_float, data_range=1.0)
                        ssim_value = ssim(ref_float, final_float, data_range=1.0, channel_axis=2)
                        
                        # Calculate contrast improvement
                        orig_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
                        final_contrast = np.std(cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY))
                        contrast_improvement = (final_contrast - orig_contrast) / orig_contrast
                        
                        # Calculate brightness change
                        orig_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
                        final_brightness = np.mean(cv2.cvtColor(final_enhanced, cv2.COLOR_BGR2GRAY))
                        brightness_change = (final_brightness - orig_brightness) / 255.0
                        
                        metrics.append({
                            'filename': filename,
                            'psnr': psnr_value,
                            'ssim': ssim_value,
                            'contrast_improvement': contrast_improvement,
                            'brightness_change': brightness_change
                        })
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        # Print summary
        print(f"\nSequential pipeline processing complete:")
        print(f"Total Processed Images: {total_processed}")
        print(f"Enhanced images saved to: {self.sequential_pipeline_dir}")
        
        # Calculate and save metrics
        if metrics:
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(os.path.join(self.dataset_path, 'sequential_pipeline_metrics.csv'), index=False)
            
            avg_metrics = metrics_df.mean(numeric_only=True)
            print(f"\nAverage PSNR: {avg_metrics['psnr']:.2f} dB")
            print(f"Average SSIM: {avg_metrics['ssim']:.3f}")
            print(f"Average Contrast Improvement: {avg_metrics['contrast_improvement']:.3f}")
            print(f"Average Brightness Change: {avg_metrics['brightness_change']:.3f}")
        
        return metrics

def main():
    # Update this path to your dataset directory
    dataset_path = "/Users/shashwatsharv/Dev/Project 2/code/dataset"
    
    # Initialize and run the sequential pipeline
    pipeline = SequentialPipeline(dataset_path)
    pipeline.process_dataset()

if __name__ == "__main__":
    main()

# Average PSNR: 21.88 dB
# Average SSIM: 0.725
# Average Contrast Improvement: 0.059
# Average Brightness Change: -0.013