import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class UnderwaterEnhancer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.trainA_dir = os.path.join(dataset_path, 'trainA')
        self.trainB_dir = os.path.join(dataset_path, 'trainB')
        self.output_dir = os.path.join(dataset_path, 'enhanced_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics = pd.DataFrame(columns=[
            'filename', 'psnr', 'ssim', 'contrast_improvement',
            'brightness_change', 'color_balance'
        ])

    def auto_white_balance(self, img):
        """Improved white balance with dynamic range adjustment"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # CLAHE with adaptive parameters
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        
        # Merge channels and convert back to BGR
        balanced = cv2.merge([l, a, b])
        return cv2.cvtColor(balanced, cv2.COLOR_LAB2BGR)

    def adaptive_gamma(self, img):
        """Dynamic gamma correction based on image brightness"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(gray)
        gamma = np.log(0.5) / np.log((mean + 1)/255)  # +1 to avoid log(0)
        return np.power(img/255.0, gamma) * 255

    def color_correction(self, img):
        """Conservative color correction"""
        # Convert to float32 for precise calculations
        img = img.astype(np.float32)
        
        # Calculate color ratios
        avg_color = np.mean(img, axis=(0,1))
        max_channel = np.argmax(avg_color)
        
        # Apply channel-specific correction
        correction = [1.0, 1.0, 1.0]
        if max_channel == 0:   # Red dominant
            correction = [1.0, 1.1, 1.2]
        elif max_channel == 1: # Green dominant
            correction = [1.1, 1.0, 1.1]
        else:                 # Blue dominant
            correction = [1.2, 1.1, 1.0]
            
        corrected = img * correction
        return np.clip(corrected, 0, 255).astype(np.uint8)

    def enhance_image(self, img_path):
        """Main enhancement pipeline"""
        # Read and validate image
        img = cv2.imread(img_path)
        if img is None or img.size == 0:
            raise ValueError(f"Invalid image: {img_path}")
            
        # Processing pipeline
        enhanced = self.auto_white_balance(img)
        enhanced = self.adaptive_gamma(enhanced)
        enhanced = self.color_correction(enhanced)
        
        return enhanced

    def calculate_metrics(self, original, enhanced, reference=None):
        """Calculate quality metrics"""
        metrics = {}
        
        if reference is not None:
            # Convert to float for accurate calculations
            ref_float = reference.astype(np.float32)/255
            enh_float = enhanced.astype(np.float32)/255
            orig_float = original.astype(np.float32)/255
            
            metrics['psnr'] = psnr(ref_float, enh_float, data_range=1.0)
            metrics['ssim'] = ssim(ref_float, enh_float, 
                                data_range=1.0,  # Explicit data range
                                channel_axis=2)  # Updated parameter name
            
            # Calculate contrast improvement
            orig_contrast = orig_float.std()
            enh_contrast = enh_float.std()
            metrics['contrast_improvement'] = enh_contrast - orig_contrast
            
            # Calculate brightness change
            orig_brightness = orig_float.mean()
            enh_brightness = enh_float.mean()
            metrics['brightness_change'] = enh_brightness - orig_brightness
            
        return metrics

    def process_dataset(self):
        """Process all images in the dataset"""
        valid_exts = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(self.trainA_dir) 
                      if f.lower().endswith(valid_exts)]
        
        for filename in tqdm(image_files, desc="Processing Images"):
            try:
                # Path setup
                img_path = os.path.join(self.trainA_dir, filename)
                ref_path = os.path.join(self.trainB_dir, filename)
                output_path = os.path.join(self.output_dir, filename)
                
                # Load images
                original = cv2.imread(img_path)
                reference = cv2.imread(ref_path) if os.path.exists(ref_path) else None
                
                # Enhance image
                enhanced = self.enhance_image(img_path)
                cv2.imwrite(output_path, enhanced)
                
                # Calculate metrics
                metrics = self.calculate_metrics(original, enhanced, reference)
                metrics['filename'] = filename
                self.metrics = pd.concat([self.metrics, pd.DataFrame([metrics])], 
                                      ignore_index=True)
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
        
        # Save metrics
        self.metrics.to_csv(os.path.join(self.dataset_path, 'enhancement_metrics.csv'), 
                          index=False)
        return self.metrics

    def show_results(self, num_samples=3):
        """Display sample results"""
        sample_files = [f for f in os.listdir(self.output_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        samples = np.random.choice(sample_files, 
                                 min(num_samples, len(sample_files)), 
                                 replace=False)
        
        plt.figure(figsize=(15, 10))
        for idx, filename in enumerate(samples):
            # Load images
            orig = cv2.imread(os.path.join(self.trainA_dir, filename))
            enh = cv2.imread(os.path.join(self.output_dir, filename))
            ref = cv2.imread(os.path.join(self.trainB_dir, filename)) if os.path.exists(
                os.path.join(self.trainB_dir, filename)) else None
            
            # Create subplots
            plt.subplot(num_samples, 3, idx*3+1)
            plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
            plt.title(f"Original: {filename}")
            plt.axis('off')
            
            plt.subplot(num_samples, 3, idx*3+2)
            plt.imshow(cv2.cvtColor(enh, cv2.COLOR_BGR2RGB))
            plt.title("Enhanced")
            plt.axis('off')
            
            if ref is not None:
                plt.subplot(num_samples, 3, idx*3+3)
                plt.imshow(cv2.cvtColor(ref, cv2.COLOR_BGR2RGB))
                plt.title("Reference")
                plt.axis('off')
        
        plt.tight_layout()
        plt.show()

    def generate_report(self):
        """Generate summary report"""
        if self.metrics.empty:
            return "No data available"
            
        report = [
            "Underwater Image Enhancement Report",
            "="*50,
            f"Total Processed Images: {len(self.metrics)}",
            f"Average PSNR: {self.metrics['psnr'].mean():.2f} dB",
            f"Average SSIM: {self.metrics['ssim'].mean():.3f}",
            f"Average Contrast Improvement: {self.metrics['contrast_improvement'].mean():.3f}",
            f"Average Brightness Change: {self.metrics['brightness_change'].mean():.3f}",
            "\nPerformance by Image Type:"
        ]
        
        # Add per-category metrics if available
        if 'category' in self.metrics:
            for category, group in self.metrics.groupby('category'):
                report.append(
                    f"{category.title()} ({len(group)} images): "
                    f"PSNR {group['psnr'].mean():.2f} dB, "
                    f"SSIM {group['ssim'].mean():.3f}"
                )
        
        return "\n".join(report)

def main():
    # Configuration
    dataset_path = "/Users/shashwatsharv/Dev/Project 2/code/dataset"  # Update this path
    num_samples = 5  # Number of samples to visualize
    
    # Initialize enhancer
    enhancer = UnderwaterEnhancer(dataset_path)
    
    # Process dataset
    print("Starting image enhancement...")
    metrics = enhancer.process_dataset()
    
    # Generate report
    print("\nEnhancement Report:")
    print(enhancer.generate_report())
    
    # Show sample results
    enhancer.show_results(num_samples)
    print(f"\nVisualization complete. Enhanced images saved to: {enhancer.output_dir}")

if __name__ == "__main__":
    main()