import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

class RefinedUnderwaterEnhancer:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.trainA_dir = os.path.join(dataset_path, 'trainA')
        self.trainB_dir = os.path.join(dataset_path, 'trainB')
        self.output_dir = os.path.join(dataset_path, 'enhanced_results')
        os.makedirs(self.output_dir, exist_ok=True)
        
        self.metrics = pd.DataFrame(columns=[
            'filename', 'psnr', 'ssim', 'contrast_improvement', 'brightness_change'
        ])

    def denoise(self, img, strength=7):
        """Apply bilateral filtering for edge-preserving denoising"""
        return cv2.bilateralFilter(img, d=7, sigmaColor=strength, sigmaSpace=strength)
    
    def simple_dehaze(self, img, omega=0.5, t0=0.1):
        """Refined dehazing for underwater images"""
        img_norm = img.astype(np.float32) / 255.0
        
        # Find dark channel
        min_channel = np.min(img_norm, axis=2)
        dark_channel = cv2.erode(min_channel, np.ones((5,5)))
        
        # Estimate atmospheric light
        size = dark_channel.shape[0] * dark_channel.shape[1]
        numpx = int(max(size * 0.001, 1))
        dark_vec = dark_channel.reshape(size)
        img_vec = img_norm.reshape(size, 3)
        indices = dark_vec.argsort()[-numpx:]
        atm_light = np.mean(img_vec[indices], axis=0)
        
        # Estimate transmission
        transmission = 1 - omega * dark_channel
        
        # Use guided filter instead of gaussian blur to preserve edges
        gray = cv2.cvtColor((img_norm * 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        transmission = cv2.ximgproc.guidedFilter(guide=gray, src=transmission, radius=50, eps=1e-3)
        
        transmission = np.maximum(transmission, t0)
        
        # Recover scene radiance
        result = np.zeros_like(img_norm)
        for i in range(3):
            result[:,:,i] = (img_norm[:,:,i] - atm_light[i]) / transmission + atm_light[i]
        
        return np.clip(result * 255, 0, 255).astype(np.uint8)
    
    def balanced_white_balance(self, img):
        """Apply white balance with sand detection for underwater images"""
        # Convert to LAB color space
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate average a* and b* values
        avg_a = np.mean(a)
        avg_b = np.mean(b)
        
        # Calculate distance from neutral gray (128, 128)
        a_offset = avg_a - 128
        b_offset = avg_b - 128
        
        # Detect potential sand/seafloor (bright areas with greenish/bluish cast)
        # This helps with images like the stingray one
        is_bright = l > 150
        is_greenish = (a < 120) & (b > 130)
        potential_sand = is_bright & is_greenish
        
        # Apply stronger correction to potential sand areas
        a_new = np.copy(a)
        b_new = np.copy(b)
        
        # General correction
        a_new = np.clip(a - a_offset * 1.2, 0, 255).astype(np.uint8)
        b_new = np.clip(b - b_offset * 1.2, 0, 255).astype(np.uint8)
        
        # Extra correction for sand
        a_new[potential_sand] = np.clip(a[potential_sand] - a_offset * 1.8, 0, 255).astype(np.uint8)
        b_new[potential_sand] = np.clip(b[potential_sand] - b_offset * 1.8, 0, 255).astype(np.uint8)
        
        # Merge channels and convert back to BGR
        balanced_lab = cv2.merge([l, a_new, b_new])
        balanced = cv2.cvtColor(balanced_lab, cv2.COLOR_LAB2BGR)
        
        return balanced
    
    def enhance_contrast(self, img, clip_limit=1.8):
        """Enhance contrast using CLAHE with refined parameters"""
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE with moderate clip limit
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        lab = cv2.merge((l, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        return enhanced
    
    def refined_color_correction(self, img):
        """Apply color correction with artifact prevention"""
        # Convert to HSV to analyze color distribution
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        
        # Analyze hue distribution to determine dominant color cast
        hist = cv2.calcHist([h], [0], None, [180], [0, 180])
        hist_smooth = cv2.GaussianBlur(hist, (5, 5), 0)
        dominant_hue = np.argmax(hist_smooth)
        
        # Convert to float for processing
        img_float = img.astype(np.float32) / 255.0
        
        # Apply correction based on dominant hue
        if 70 <= dominant_hue <= 130:  # Green range
            # Reduce green, boost red
            img_float[:,:,1] = np.clip(img_float[:,:,1] * 0.85, 0, 1)    # Green
            img_float[:,:,2] = np.clip(img_float[:,:,2] * 1.5, 0, 1)     # Red
        elif 130 <= dominant_hue <= 150:  # Blue-green range
            # Reduce blue and green, boost red
            img_float[:,:,0] = np.clip(img_float[:,:,0] * 0.85, 0, 1)    # Blue
            img_float[:,:,1] = np.clip(img_float[:,:,1] * 0.9, 0, 1)     # Green
            img_float[:,:,2] = np.clip(img_float[:,:,2] * 1.4, 0, 1)     # Red
        elif 150 <= dominant_hue <= 180 or 0 <= dominant_hue < 20:  # Purple-red range
            # Reduce blue, maintain red - prevents purple artifacts
            img_float[:,:,0] = np.clip(img_float[:,:,0] * 0.8, 0, 1)     # Blue
        else:  # Other colors
            # General correction - moderate red boost
            img_float[:,:,2] = np.clip(img_float[:,:,2] * 1.3, 0, 1)     # Red
        
        # Apply smoothing to prevent blocky artifacts
        result = (img_float * 255).astype(np.uint8)
        result = cv2.GaussianBlur(result, (3, 3), 0)
        
        return result
    
    def pyramid_fusion(self, images, weights):
        """Implement Laplacian pyramid fusion for smoother results"""
        # Number of levels in the pyramid
        levels = 5
        
        # Convert images to float
        images_float = [img.astype(np.float32) for img in images]
        
        # Generate Gaussian pyramids for each image
        gaussian_pyramids = []
        for img in images_float:
            gaussian = [img]
            for i in range(levels-1):
                gaussian.append(cv2.pyrDown(gaussian[i]))
            gaussian_pyramids.append(gaussian)
        
        # Generate Laplacian pyramids
        laplacian_pyramids = []
        for i, gaussian in enumerate(gaussian_pyramids):
            laplacian = []
            for j in range(levels-1):
                size = (gaussian[j].shape[1], gaussian[j].shape[0])
                laplacian.append(gaussian[j] - cv2.resize(cv2.pyrUp(gaussian[j+1]), size))
            laplacian.append(gaussian[-1])
            laplacian_pyramids.append(laplacian)
        
        # Blend pyramids
        blended_pyramid = []
        for level in range(levels):
            blended_level = np.zeros_like(laplacian_pyramids[0][level])
            for i in range(len(images)):
                # Expand weight map to match image dimensions
                if level == 0:
                    weight = weights[i]
                else:
                    weight = cv2.pyrDown(weights[i])
                    for j in range(1, level):
                        weight = cv2.pyrDown(weight)
                
                # Resize weight to match current level
                weight = cv2.resize(weight, (laplacian_pyramids[i][level].shape[1], 
                                           laplacian_pyramids[i][level].shape[0]))
                
                # Expand dimensions for broadcasting
                weight = np.expand_dims(weight, axis=2)
                if laplacian_pyramids[i][level].ndim == 3:  # Color image
                    weight = np.repeat(weight, 3, axis=2)
                
                blended_level += weight * laplacian_pyramids[i][level]
            
            blended_pyramid.append(blended_level)
        
        # Reconstruct image
        result = blended_pyramid[-1]
        for i in range(levels-2, -1, -1):
            size = (blended_pyramid[i].shape[1], blended_pyramid[i].shape[0])
            result = cv2.resize(cv2.pyrUp(result), size) + blended_pyramid[i]
        
        return np.clip(result, 0, 255).astype(np.uint8)
    
    def generate_weight_maps(self, img):
        """Generate weight maps for fusion"""
        img_norm = img.astype(np.float32) / 255.0
        
        # Local contrast weight
        gray = cv2.cvtColor(img_norm, cv2.COLOR_BGR2GRAY)
        laplacian = np.abs(cv2.Laplacian(gray, cv2.CV_32F))
        contrast_weight = laplacian / (np.max(laplacian) + 1e-6)
        
        # Saturation weight
        hsv = cv2.cvtColor(img_norm, cv2.COLOR_BGR2HSV)
        saturation_weight = hsv[:,:,1]
        
        # Exposedness weight
        sigma = 0.2
        exposedness = np.exp(-((img_norm - 0.5) ** 2) / (2 * sigma ** 2))
        exposedness_weight = np.prod(exposedness, axis=2)
        
        # Combine weights
        combined_weight = contrast_weight * saturation_weight * exposedness_weight
        
        return combined_weight
    
    def advanced_fusion(self, img):
        """Create and fuse multiple enhanced versions using pyramid fusion"""
        # Create multiple enhanced versions
        dehazed = self.simple_dehaze(img)
        white_balanced = self.balanced_white_balance(img)
        color_corrected = self.refined_color_correction(img)
        
        # Generate weight maps
        weight1 = self.generate_weight_maps(dehazed)
        weight2 = self.generate_weight_maps(white_balanced)
        weight3 = self.generate_weight_maps(color_corrected)
        
        # Normalize weights
        sum_weights = weight1 + weight2 + weight3 + 1e-6
        norm_weight1 = weight1 / sum_weights
        norm_weight2 = weight2 / sum_weights
        norm_weight3 = weight3 / sum_weights
        
        # Apply pyramid fusion
        fused = self.pyramid_fusion(
            [dehazed, white_balanced, color_corrected],
            [norm_weight1, norm_weight2, norm_weight3]
        )
        
        return fused
    
    def enhance_image(self, img_path):
        """Main enhancement pipeline with refined approach"""
        # Read image
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"Could not read image: {img_path}")
        
        # Step 1: Denoise the image
        denoised = self.denoise(img)
        
        # Step 2: Apply advanced fusion-based enhancement
        fused = self.advanced_fusion(denoised)
        
        # Step 3: Enhance contrast
        contrast_enhanced = self.enhance_contrast(fused)
        
        # Step 4: Final gentle denoising
        final = self.denoise(contrast_enhanced, strength=5)
        
        return final
    
    def calculate_metrics(self, original, enhanced, reference=None):
        """Calculate quality metrics"""
        metrics = {}
        
        if reference is not None:
            # Convert to float for accurate calculations
            ref_float = reference.astype(np.float32)/255
            enh_float = enhanced.astype(np.float32)/255
            orig_float = original.astype(np.float32)/255
            
            # Calculate PSNR
            metrics['psnr'] = psnr(ref_float, enh_float, data_range=1.0)
            
            # Calculate SSIM
            metrics['ssim'] = ssim(ref_float, enh_float, 
                                data_range=1.0,
                                channel_axis=2)
            
            # Calculate contrast improvement (using standard deviation as a simple measure)
            orig_contrast = np.std(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            enh_contrast = np.std(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
            metrics['contrast_improvement'] = (enh_contrast - orig_contrast) / orig_contrast
            
            # Calculate brightness change
            orig_brightness = np.mean(cv2.cvtColor(original, cv2.COLOR_BGR2GRAY))
            enh_brightness = np.mean(cv2.cvtColor(enhanced, cv2.COLOR_BGR2GRAY))
            metrics['brightness_change'] = (enh_brightness - orig_brightness) / 255.0
            
        return metrics
    
    def process_dataset(self):
        """Process all images in the dataset"""
        valid_exts = ('.jpg', '.jpeg', '.png')
        image_files = [f for f in os.listdir(self.trainA_dir) 
                      if f.lower().endswith(valid_exts)]
        
        total_processed = 0
        errors = 0
        
        for filename in tqdm(image_files, desc="Processing Images"):
            try:
                # Path setup
                img_path = os.path.join(self.trainA_dir, filename)
                ref_path = os.path.join(self.trainB_dir, filename)
                output_path = os.path.join(self.output_dir, filename)
                
                # Load images
                original = cv2.imread(img_path)
                reference = cv2.imread(ref_path) if os.path.exists(ref_path) else None
                
                if original is None:
                    print(f"Warning: Could not read image {img_path}")
                    continue
                    
                if reference is None and os.path.exists(ref_path):
                    print(f"Warning: Could not read reference image {ref_path}")
                
                # Enhance image
                enhanced = self.enhance_image(img_path)
                cv2.imwrite(output_path, enhanced)
                
                # Calculate metrics
                if reference is not None:
                    metrics = self.calculate_metrics(original, enhanced, reference)
                    metrics['filename'] = filename
                    self.metrics = pd.concat([self.metrics, pd.DataFrame([metrics])], 
                                          ignore_index=True)
                
                total_processed += 1
                
            except Exception as e:
                print(f"Error processing {filename}: {str(e)}")
                errors += 1
        
        # Save metrics
        metrics_path = os.path.join(self.dataset_path, 'enhancement_metrics.csv')
        self.metrics.to_csv(metrics_path, index=False)
        
        # Print summary
        print(f"\nProcessing complete:")
        print(f"Total Processed Images: {total_processed}")
        print(f"Errors: {errors}")
        
        if not self.metrics.empty:
            # Calculate and display average metrics
            avg_metrics = self.metrics.mean(numeric_only=True)
            print(f"\nAverage PSNR: {avg_metrics['psnr']:.2f} dB")
            print(f"Average SSIM: {avg_metrics['ssim']:.3f}")
            print(f"Average Contrast Improvement: {avg_metrics['contrast_improvement']:.3f}")
            print(f"Average Brightness Change: {avg_metrics['brightness_change']:.3f}")
        
        return self.metrics
    
    def show_results(self, num_samples=5):
        """Display sample results"""
        sample_files = [f for f in os.listdir(self.output_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not sample_files:
            print("No enhanced images found to display.")
            return
            
        # Randomly select samples
        if len(sample_files) > num_samples:
            samples = np.random.choice(sample_files, num_samples, replace=False)
        else:
            samples = sample_files
        
        # Create figure
        fig, axes = plt.subplots(len(samples), 3, figsize=(15, 5*len(samples)))
        
        # Handle case with only one sample
        if len(samples) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, filename in enumerate(samples):
            # Load images
            orig_path = os.path.join(self.trainA_dir, filename)
            enh_path = os.path.join(self.output_dir, filename)
            ref_path = os.path.join(self.trainB_dir, filename)
            
            orig = cv2.imread(orig_path)
            enh = cv2.imread(enh_path)
            ref = cv2.imread(ref_path) if os.path.exists(ref_path) else None
            
            if orig is None or enh is None:
                continue
                
            # Convert BGR to RGB for display
            orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
            enh = cv2.cvtColor(enh, cv2.COLOR_BGR2RGB)
            ref = cv2.cvtColor(ref, cv2.COLOR_BGR2RGB) if ref is not None else None
            
            # Display images
            axes[idx, 0].imshow(orig)
            axes[idx, 0].set_title(f"Original: {filename}")
            axes[idx, 0].axis('off')
            
            axes[idx, 1].imshow(enh)
            axes[idx, 1].set_title("Enhanced")
            axes[idx, 1].axis('off')
            
            if ref is not None:
                axes[idx, 2].imshow(ref)
                axes[idx, 2].set_title("Reference")
                axes[idx, 2].axis('off')
            else:
                axes[idx, 2].axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.dataset_path, 'sample_results.png'))
        
        try:
            plt.show()
        except Exception as e:
            print(f"Warning: Could not display plot: {str(e)}")
            print(f"Results saved to {os.path.join(self.dataset_path, 'sample_results.png')}")


def main():
    # Configuration
    dataset_path = "/Users/shashwatsharv/Dev/Project 2/code/dataset"  # Update this path
    num_samples = 5  # Number of samples to visualize
    
    # Initialize enhancer
    enhancer = RefinedUnderwaterEnhancer(dataset_path)
    
    # Process dataset
    print("Starting image enhancement...")
    metrics = enhancer.process_dataset()
    
    # Show sample results
    print("\nGenerating result visualizations...")
    enhancer.show_results(num_samples)
    
    print(f"\nEnhancement complete. Enhanced images saved to: {enhancer.output_dir}")

if __name__ == "__main__":
    main()

# Processing complete:
# Total Processed Images: 5885
# Errors: 0

# Average PSNR: 16.95 dB
# Average SSIM: 0.707
# Average Contrast Improvement: 0.226
# Average Brightness Change: 0.031