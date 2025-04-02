import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from tqdm import tqdm
import matplotlib.pyplot as plt

class UnderwaterGAN:
    def __init__(self, dataset_path, img_size=128, batch_size=2):
        self.dataset_path = dataset_path
        self.enhanced_dir = os.path.join(dataset_path, 'enhanced_results')
        self.reference_dir = os.path.join(dataset_path, 'trainB')
        self.gan_output_dir = os.path.join(dataset_path, 'gan_results')
        self.checkpoint_dir = os.path.join(dataset_path, 'checkpoints')
        self.samples_dir = os.path.join(dataset_path, 'samples')
        
        os.makedirs(self.gan_output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        self.img_size = img_size  # Changed from 256 to 128
        self.batch_size = batch_size
        
        # Build and compile the models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
        # Compile models
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
            metrics=['accuracy']
        )
        
        self.discriminator.trainable = False
        self.gan.compile(
            loss=['binary_crossentropy', 'mae'],
            loss_weights=[1, 100],  # L1 loss weight is higher
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5)
        )
        
        self.discriminator.trainable = True
    
    def build_generator(self):
        """U-Net generator for image-to-image translation"""
        inputs = Input(shape=[self.img_size, self.img_size, 3])
        
        # Encoder
        e1 = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(inputs)
        e1 = layers.LeakyReLU(0.2)(e1)
        
        e2 = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(e1)
        e2 = layers.BatchNormalization()(e2)
        e2 = layers.LeakyReLU(0.2)(e2)
        
        e3 = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(e2)
        e3 = layers.BatchNormalization()(e3)
        e3 = layers.LeakyReLU(0.2)(e3)
        
        e4 = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(e3)
        e4 = layers.BatchNormalization()(e4)
        e4 = layers.LeakyReLU(0.2)(e4)
        
        # Bottleneck
        b = layers.Conv2D(512, kernel_size=4, strides=2, padding='same')(e4)
        b = layers.ReLU()(b)
        
        # Decoder with skip connections
        d1 = layers.Conv2DTranspose(512, kernel_size=4, strides=2, padding='same')(b)
        d1 = layers.BatchNormalization()(d1)
        d1 = layers.Dropout(0.5)(d1)
        d1 = layers.ReLU()(d1)
        d1 = layers.Concatenate()([d1, e4])
        
        d2 = layers.Conv2DTranspose(256, kernel_size=4, strides=2, padding='same')(d1)
        d2 = layers.BatchNormalization()(d2)
        d2 = layers.ReLU()(d2)
        d2 = layers.Concatenate()([d2, e3])
        
        d3 = layers.Conv2DTranspose(128, kernel_size=4, strides=2, padding='same')(d2)
        d3 = layers.BatchNormalization()(d3)
        d3 = layers.ReLU()(d3)
        d3 = layers.Concatenate()([d3, e2])
        
        d4 = layers.Conv2DTranspose(64, kernel_size=4, strides=2, padding='same')(d3)
        d4 = layers.BatchNormalization()(d4)
        d4 = layers.ReLU()(d4)
        d4 = layers.Concatenate()([d4, e1])
        
        # Output layer
        outputs = layers.Conv2DTranspose(3, kernel_size=4, strides=2, padding='same', activation='tanh')(d4)
        
        return Model(inputs=inputs, outputs=outputs, name='generator')
    
    def build_discriminator(self):
        """PatchGAN discriminator"""
        inputs = Input(shape=[self.img_size, self.img_size, 3])
        target = Input(shape=[self.img_size, self.img_size, 3])
        
        x = layers.Concatenate()([inputs, target])
        
        x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same')(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        x = layers.Conv2D(512, kernel_size=4, strides=1, padding='same')(x)
        x = layers.BatchNormalization()(x)
        x = layers.LeakyReLU(0.2)(x)
        
        outputs = layers.Conv2D(1, kernel_size=4, strides=1, padding='same')(x)
        
        return Model(inputs=[inputs, target], outputs=outputs, name='discriminator')
    
    def build_gan(self):
        """Combined generator and discriminator model"""
        self.discriminator.trainable = False
        
        img_input = Input(shape=[self.img_size, self.img_size, 3])
        gen_output = self.generator(img_input)
        
        disc_output = self.discriminator([img_input, gen_output])
        
        return Model(inputs=img_input, outputs=[disc_output, gen_output], name='gan')
    
    def preprocess_image(self, image_path):
        """Load and preprocess image for the network"""
        img = tf.io.read_file(image_path)
        img = tf.image.decode_jpeg(img)
        img = tf.cast(img, tf.float32)
        img = tf.image.resize(img, [self.img_size, self.img_size])
        img = (img / 127.5) - 1  # Normalize to [-1, 1]
        return img
    
    def create_dataset(self):
        """Create training dataset from enhanced and reference images"""
        # Load all file paths
        enhanced_paths = [os.path.join(self.enhanced_dir, f) for f in os.listdir(self.enhanced_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        reference_paths = [os.path.join(self.reference_dir, f) for f in os.listdir(self.reference_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Find matching pairs
        enhanced_filenames = [os.path.basename(f) for f in enhanced_paths]
        reference_filenames = [os.path.basename(f) for f in reference_paths]
        common_files = set(enhanced_filenames).intersection(set(reference_filenames))
        
        # Create matched arrays
        matched_enhanced_paths = []
        matched_reference_paths = []
        for f in common_files:
            enhanced_path = os.path.join(self.enhanced_dir, f)
            reference_path = os.path.join(self.reference_dir, f)
            matched_enhanced_paths.append(enhanced_path)
            matched_reference_paths.append(reference_path)
        
        print(f"Found {len(matched_enhanced_paths)} matching image pairs")
        
        # Load and preprocess all images at once
        enhanced_images = []
        reference_images = []
        for e_path, r_path in zip(matched_enhanced_paths, matched_reference_paths):
            e_img = cv2.imread(e_path)
            r_img = cv2.imread(r_path)
            if e_img is not None and r_img is not None:
                e_img = cv2.resize(e_img, (self.img_size, self.img_size))
                r_img = cv2.resize(r_img, (self.img_size, self.img_size))
                enhanced_images.append((e_img.astype(np.float32) / 127.5) - 1)
                reference_images.append((r_img.astype(np.float32) / 127.5) - 1)
        
        enhanced_array = np.array(enhanced_images)
        reference_array = np.array(reference_images)
        
        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((enhanced_array, reference_array))
        
        # Split into training and validation
        val_size = int(len(enhanced_array) * 0.1)  # 10% for validation
        train_ds = dataset.skip(val_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = dataset.take(val_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def train(self, epochs=50, save_interval=5):
        """Train the GAN model with early stopping and model checkpointing"""
        # Create dataset
        train_ds, val_ds = self.create_dataset()
        
        # Create sample images for visualization
        sample_enhanced, sample_reference = next(iter(val_ds))
        
        # Setup callbacks
        checkpoint_callback = ModelCheckpoint(
            filepath=os.path.join(self.checkpoint_dir, 'best_generator.h5'),
            monitor='val_loss',
            save_best_only=True,
            save_weights_only=False,
            mode='min',
            verbose=1
        )
        
        early_stopping = EarlyStopping(
            monitor='val_loss',
            patience=10,
            restore_best_weights=True,
            verbose=1
        )
        
        # Training loop
        history = {'gen_loss': [], 'disc_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            
            # Initialize metrics
            d_losses = []
            g_losses = []
            
            # Train on batches
            for batch_idx, (enhanced_images, reference_images) in enumerate(tqdm(train_ds, desc="Training")):
                # Train discriminator
                generated_images = self.generator(enhanced_images, training=True)
                
                # Real images
                with tf.GradientTape() as tape:
                    real_output = self.discriminator([enhanced_images, reference_images], training=True)
                    d_real_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                        tf.ones_like(real_output), real_output)
                
                    # Fake images
                    fake_output = self.discriminator([enhanced_images, generated_images], training=True)
                    d_fake_loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                        tf.zeros_like(fake_output), fake_output)
                    
                    # Total discriminator loss
                    d_loss = d_real_loss + d_fake_loss
                
                # Apply discriminator gradients
                d_gradients = tape.gradient(d_loss, self.discriminator.trainable_variables)
                tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5).apply_gradients(
                    zip(d_gradients, self.discriminator.trainable_variables))
                
                d_losses.append(d_loss)
                
                # Train generator
                with tf.GradientTape() as tape:
                    # Generate images
                    generated_images = self.generator(enhanced_images, training=True)
                    
                    # Discriminator output on generated images
                    fake_output = self.discriminator([enhanced_images, generated_images], training=True)
                    
                    # Generator losses
                    g_loss_gan = tf.keras.losses.BinaryCrossentropy(from_logits=True)(
                        tf.ones_like(fake_output), fake_output)
                    g_loss_l1 = tf.reduce_mean(tf.abs(reference_images - generated_images)) * 100
                    g_loss = g_loss_gan + g_loss_l1
                
                # Apply generator gradients
                g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
                tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5).apply_gradients(
                    zip(g_gradients, self.generator.trainable_variables))
                
                g_losses.append(g_loss)
            
            # Evaluate on validation set
            val_losses = []
            for val_enhanced, val_reference in val_ds:
                val_generated = self.generator(val_enhanced, training=False)
                val_l1_loss = tf.reduce_mean(tf.abs(val_reference - val_generated)) * 100
                val_losses.append(val_l1_loss)
            
            val_loss = tf.reduce_mean(val_losses)
            
            # Record metrics
            avg_d_loss = tf.reduce_mean(d_losses)
            avg_g_loss = tf.reduce_mean(g_losses)
            history['disc_loss'].append(avg_d_loss.numpy())
            history['gen_loss'].append(avg_g_loss.numpy())
            history['val_loss'].append(val_loss.numpy())
            
            # Print epoch results
            print(f"Discriminator Loss: {avg_d_loss:.4f}, Generator Loss: {avg_g_loss:.4f}, Validation Loss: {val_loss:.4f}")
            
            # Generate and save sample images
            if (epoch + 1) % save_interval == 0 or epoch == 0:
                self.generate_samples(sample_enhanced, sample_reference, epoch)
            
            # Save model if it's the best so far
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.generator.save(os.path.join(self.checkpoint_dir, 'best_generator.h5'))
                self.discriminator.save(os.path.join(self.checkpoint_dir, 'best_discriminator.h5'))
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")
            else:
                patience_counter += 1
                
            # Check for early stopping
            if patience_counter >= 10:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
            
            # Save model periodically regardless of performance
            if (epoch + 1) % save_interval == 0:
                self.generator.save(os.path.join(self.checkpoint_dir, f'generator_epoch_{epoch+1}.h5'))
        
        # Save final model
        self.generator.save(os.path.join(self.checkpoint_dir, 'final_generator.h5'))
        
        # Plot training history
        self.plot_training_history(history)
        
        return history
    
    def plot_training_history(self, history):
        """Plot and save training history"""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(history['gen_loss'], label='Generator Loss')
        plt.plot(history['disc_loss'], label='Discriminator Loss')
        plt.title('Training Losses')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(history['val_loss'], label='Validation Loss')
        plt.title('Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.checkpoint_dir, 'training_history.png'))
        plt.close()
    
    def generate_samples(self, enhanced_images, reference_images, epoch):
        """Generate and save sample images"""
        generated_images = self.generator(enhanced_images, training=False)
        
        # Convert from [-1, 1] to [0, 1]
        enhanced_images = (enhanced_images + 1) / 2
        reference_images = (reference_images + 1) / 2
        generated_images = (generated_images + 1) / 2
        
        plt.figure(figsize=(15, 5 * min(3, len(enhanced_images))))
        
        for i in range(min(3, len(enhanced_images))):
            plt.subplot(3, 3, i*3 + 1)
            plt.imshow(enhanced_images[i])
            plt.title("Enhanced")
            plt.axis('off')
            
            plt.subplot(3, 3, i*3 + 2)
            plt.imshow(generated_images[i])
            plt.title("GAN Output")
            plt.axis('off')
            
            plt.subplot(3, 3, i*3 + 3)
            plt.imshow(reference_images[i])
            plt.title("Reference")
            plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.samples_dir, f'epoch_{epoch+1}.png'))
        plt.close()
    
        def process_dataset(self):
            """Process all enhanced images with the trained generator"""
            # Load all enhanced images
            enhanced_files = [f for f in os.listdir(self.enhanced_dir) 
                            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            
            total_processed = 0
            metrics = []
            
            for filename in tqdm(enhanced_files, desc="Processing with GAN"):
                try:
                    # Setup paths
                    enhanced_path = os.path.join(self.enhanced_dir, filename)
                    reference_path = os.path.join(self.reference_dir, filename)
                    output_path = os.path.join(self.gan_output_dir, filename)
                    
                    # Load and preprocess image
                    img = cv2.imread(enhanced_path)
                    if img is None:
                        print(f"Warning: Could not read image {enhanced_path}")
                        continue
                    
                    # Resize for the network
                    img_resized = cv2.resize(img, (self.img_size, self.img_size))
                    img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
                    
                    # Generate enhanced image
                    input_tensor = tf.expand_dims(img_normalized, 0)
                    generated = self.generator(input_tensor, training=False)
                    generated = ((generated[0].numpy() + 1) * 127.5).astype(np.uint8)
                    
                    # Resize back to original size if needed
                    if img.shape[:2] != (self.img_size, self.img_size):
                        generated = cv2.resize(generated, (img.shape[1], img.shape[0]))
                    
                    # Save the result
                    cv2.imwrite(output_path, generated)
                    
                    # Calculate metrics if reference exists
                    if os.path.exists(reference_path):
                        reference = cv2.imread(reference_path)
                        if reference is not None:
                            # Calculate PSNR and SSIM
                            ref_float = reference.astype(np.float32)/255
                            gen_float = generated.astype(np.float32)/255
                            
                            psnr_value = psnr(ref_float, gen_float, data_range=1.0)
                            ssim_value = ssim(ref_float, gen_float, data_range=1.0, channel_axis=2)
                            
                            metrics.append({
                                'filename': filename,
                                'psnr': psnr_value,
                                'ssim': ssim_value
                            })
                    
                    total_processed += 1
                    
                except Exception as e:
                    print(f"Error processing {filename}: {str(e)}")
            
            # Print summary
            print(f"\nGAN Processing complete:")
            print(f"Total Processed Images: {total_processed}")
            
            if metrics:
                avg_psnr = np.mean([m['psnr'] for m in metrics])
                avg_ssim = np.mean([m['ssim'] for m in metrics])
                print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
                print(f"Average SSIM: {avg_ssim:.3f}")
            
            return metrics

    def load_best_model(self):
        """Load the best saved model"""
        best_model_path = os.path.join(self.checkpoint_dir, 'best_generator.h5')
        if os.path.exists(best_model_path):
            self.generator = tf.keras.models.load_model(best_model_path)
            print(f"Loaded best model from {best_model_path}")
        else:
            print("No saved model found. Using initialized model.")


class DirectUnderwaterGAN(UnderwaterGAN):
    def __init__(self, dataset_path, img_size=128, batch_size=2):
        # Initialize with different directories
        self.dataset_path = dataset_path
        self.original_dir = os.path.join(dataset_path, 'trainA')  # Changed from enhanced_results to trainA
        self.reference_dir = os.path.join(dataset_path, 'trainB')
        self.gan_output_dir = os.path.join(dataset_path, 'direct_gan_results')  # Changed name
        self.checkpoint_dir = os.path.join(dataset_path, 'direct_checkpoints')  # Changed name
        self.samples_dir = os.path.join(dataset_path, 'direct_samples')  # Changed name
        
        os.makedirs(self.gan_output_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.samples_dir, exist_ok=True)
        
        self.img_size = img_size
        self.batch_size = batch_size
        
        # Build and compile the models
        self.generator = self.build_generator()
        self.discriminator = self.build_discriminator()
        self.gan = self.build_gan()
        
        # Compile models
        self.discriminator.compile(
            loss='binary_crossentropy',
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5),
            metrics=['accuracy']
        )
        
        self.discriminator.trainable = False
        self.gan.compile(
            loss=['binary_crossentropy', 'mae'],
            loss_weights=[1, 100],  # L1 loss weight is higher
            optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=2e-4, beta_1=0.5)
        )
        
        self.discriminator.trainable = True
    
    def create_dataset(self):
        """Create training dataset from original and reference images"""
        # Load all file paths
        original_paths = [os.path.join(self.original_dir, f) for f in os.listdir(self.original_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        reference_paths = [os.path.join(self.reference_dir, f) for f in os.listdir(self.reference_dir) 
                        if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        # Find matching pairs
        original_filenames = [os.path.basename(f) for f in original_paths]
        reference_filenames = [os.path.basename(f) for f in reference_paths]
        common_files = set(original_filenames).intersection(set(reference_filenames))
        
        # Create matched arrays
        matched_original_paths = []
        matched_reference_paths = []
        for f in common_files:
            original_path = os.path.join(self.original_dir, f)
            reference_path = os.path.join(self.reference_dir, f)
            matched_original_paths.append(original_path)
            matched_reference_paths.append(reference_path)
        
        print(f"Found {len(matched_original_paths)} matching image pairs")
        
        # Load and preprocess all images at once
        original_images = []
        reference_images = []
        for o_path, r_path in zip(matched_original_paths, matched_reference_paths):
            o_img = cv2.imread(o_path)
            r_img = cv2.imread(r_path)
            if o_img is not None and r_img is not None:
                o_img = cv2.resize(o_img, (self.img_size, self.img_size))
                r_img = cv2.resize(r_img, (self.img_size, self.img_size))
                original_images.append((o_img.astype(np.float32) / 127.5) - 1)
                reference_images.append((r_img.astype(np.float32) / 127.5) - 1)
        
        original_array = np.array(original_images)
        reference_array = np.array(reference_images)
        
        # Create dataset from numpy arrays
        dataset = tf.data.Dataset.from_tensor_slices((original_array, reference_array))
        
        # Split into training and validation
        val_size = int(len(original_array) * 0.1)  # 10% for validation
        train_ds = dataset.skip(val_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        val_ds = dataset.take(val_size).batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        
        return train_ds, val_ds
    
    def process_dataset(self):
        """Process all original images with the trained generator"""
        # Load all original images
        original_files = [f for f in os.listdir(self.original_dir) 
                         if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        total_processed = 0
        metrics = []
        
        for filename in tqdm(original_files, desc="Processing with Direct GAN"):
            try:
                # Setup paths
                original_path = os.path.join(self.original_dir, filename)
                reference_path = os.path.join(self.reference_dir, filename)
                output_path = os.path.join(self.gan_output_dir, filename)
                
                # Load and preprocess image
                img = cv2.imread(original_path)
                if img is None:
                    print(f"Warning: Could not read image {original_path}")
                    continue
                
                # Resize for the network
                img_resized = cv2.resize(img, (self.img_size, self.img_size))
                img_normalized = (img_resized.astype(np.float32) / 127.5) - 1
                
                # Generate enhanced image
                input_tensor = tf.expand_dims(img_normalized, 0)
                generated = self.generator(input_tensor, training=False)
                generated = ((generated[0].numpy() + 1) * 127.5).astype(np.uint8)
                
                # Resize back to original size if needed
                if img.shape[:2] != (self.img_size, self.img_size):
                    generated = cv2.resize(generated, (img.shape[1], img.shape[0]))
                
                # Save the result
                cv2.imwrite(output_path, generated)
                
                # Calculate metrics if reference exists
                if os.path.exists(reference_path):
                    reference = cv2.imread(reference_path)
                    if reference is not None:
                        # Calculate PSNR and SSIM
                        ref_float = reference.astype(np.float32)/255
                        gen_float = generated.astype(np.float32)/255
                        
                        psnr_value = psnr(ref_float, gen_float, data_range=1.0)
                        ssim_value = ssim(ref_float, gen_float, data_range=1.0, channel_axis=2)
                        
                        # Calculate contrast improvement and brightness change
                        orig_contrast = np.std(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        gen_contrast = np.std(cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY))
                        contrast_improvement = (gen_contrast - orig_contrast) / orig_contrast
                        
                        orig_brightness = np.mean(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
                        gen_brightness = np.mean(cv2.cvtColor(generated, cv2.COLOR_BGR2GRAY))
                        brightness_change = (gen_brightness - orig_brightness) / 255.0
                        
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
        print(f"\nDirect GAN Processing complete:")
        print(f"Total Processed Images: {total_processed}")
        
        if metrics:
            avg_psnr = np.mean([m['psnr'] for m in metrics])
            avg_ssim = np.mean([m['ssim'] for m in metrics])
            avg_contrast = np.mean([m['contrast_improvement'] for m in metrics])
            avg_brightness = np.mean([m['brightness_change'] for m in metrics])
            
            print(f"\nAverage PSNR: {avg_psnr:.2f} dB")
            print(f"Average SSIM: {avg_ssim:.3f}")
            print(f"Average Contrast Improvement: {avg_contrast:.3f}")
            print(f"Average Brightness Change: {avg_brightness:.3f}")
            
            # Save metrics to CSV
            metrics_df = pd.DataFrame(metrics)
            metrics_df.to_csv(os.path.join(self.dataset_path, 'direct_gan_metrics.csv'), index=False)
        
        return metrics


def main():
    # Configuration
    dataset_path = "/Users/shashwatsharv/Dev/Project 2/code/dataset"  # Update this path
    
    # Initialize and train Direct GAN (trainA to trainB)
    gan = DirectUnderwaterGAN(dataset_path)
    
    # Train the model
    print("Starting Direct GAN training...")
    gan.train(epochs=25, save_interval=5)
    
    # Process the entire dataset
    print("\nProcessing dataset with trained Direct GAN...")
    metrics = gan.process_dataset()
    
    print(f"\nDirect GAN enhancement complete. Enhanced images saved to: {gan.gan_output_dir}")

if __name__ == "__main__":
    main()


# Average PSNR: 23.25 dB
# Average SSIM: 0.746
# Average Contrast Improvement: 0.056
# Average Brightness Change: -0.028