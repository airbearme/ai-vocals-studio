#!/usr/bin/env python3
"""
Advanced Data Augmentation for Voice Cloning Training
Implements sophisticated audio augmentation techniques for robust model training
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import random
import warnings
warnings.filterwarnings('ignore')

class VoiceDataAugmentor:
    """
    Advanced data augmentation for voice cloning training
    Increases dataset diversity and model robustness
    """
    
    def __init__(self, target_sr=22050, seed=42):
        self.target_sr = target_sr
        np.random.seed(seed)
        random.seed(seed)
        
        # Augmentation parameters
        self.augmentation_config = {
            'time_stretch': {'range': (0.8, 1.2), 'probability': 0.7},
            'pitch_shift': {'range': (-4, 4), 'probability': 0.7},
            'volume_change': {'range': (0.7, 1.3), 'probability': 0.6},
            'add_noise': {'snr_range': (10, 30), 'probability': 0.5},
            'reverb': {'intensity_range': (0.1, 0.4), 'probability': 0.4},
            'frequency_mask': {'max_mask': 0.1, 'probability': 0.3},
            'time_mask': {'max_mask': 0.1, 'probability': 0.3},
            'spec_augment': {'probability': 0.4}
        }
    
    def augment_audio(self, y, sr, num_augmentations=3):
        """
        Generate multiple augmented versions of audio
        """
        augmented_versions = [y]  # Include original
        
        for i in range(num_augmentations):
            augmented = self._apply_random_augmentation(y, sr)
            augmented_versions.append(augmented)
        
        return augmented_versions
    
    def _apply_random_augmentation(self, y, sr):
        """
        Apply a random combination of augmentations
        """
        y_aug = y.copy()
        
        # Time stretching
        if random.random() < self.augmentation_config['time_stretch']['probability']:
            rate = random.uniform(*self.augmentation_config['time_stretch']['range'])
            y_aug = self.time_stretch(y_aug, rate)
        
        # Pitch shifting
        if random.random() < self.augmentation_config['pitch_shift']['probability']:
            steps = random.uniform(*self.augmentation_config['pitch_shift']['range'])
            y_aug = self.pitch_shift(y_aug, sr, steps)
        
        # Volume change
        if random.random() < self.augmentation_config['volume_change']['probability']:
            factor = random.uniform(*self.augmentation_config['volume_change']['range'])
            y_aug = self.change_volume(y_aug, factor)
        
        # Add noise
        if random.random() < self.augmentation_config['add_noise']['probability']:
            snr = random.uniform(*self.augmentation_config['add_noise']['snr_range'])
            y_aug = self.add_noise(y_aug, snr)
        
        # Add reverb
        if random.random() < self.augmentation_config['reverb']['probability']:
            intensity = random.uniform(*self.augmentation_config['reverb']['intensity_range'])
            y_aug = self.add_reverb(y_aug, sr, intensity)
        
        # Spectral augmentation
        if random.random() < self.augmentation_config['spec_augment']['probability']:
            y_aug = self.spec_augment(y_aug, sr)

        # Ensure the augmented clip keeps the same length as the input so that
        # downstream batched training / balanced datasets stay frame-aligned.
        if len(y_aug) != len(y):
            y_aug = librosa.util.fix_length(y_aug, size=len(y))

        return y_aug
    
    def time_stretch(self, y, rate):
        """
        Time stretching without changing pitch
        """
        try:
            y_stretched = librosa.effects.time_stretch(y, rate=rate)
            return y_stretched
        except:
            return y
    
    def pitch_shift(self, y, sr, steps):
        """
        Pitch shifting without changing speed
        """
        try:
            y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=steps)
            return y_shifted
        except:
            return y
    
    def change_volume(self, y, factor):
        """
        Change volume by a factor
        """
        return y * factor
    
    def add_noise(self, y, snr_db):
        """
        Add Gaussian noise with specified SNR
        """
        signal_power = np.mean(y ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = np.random.normal(0, np.sqrt(noise_power), len(y))
        return y + noise
    
    def add_reverb(self, y, sr, intensity=0.3):
        """
        Add artificial reverb
        """
        try:
            # Create impulse response for reverb
            duration = int(0.5 * sr)  # 0.5 second reverb tail
            impulse = np.random.exponential(scale=intensity * sr, size=duration)
            impulse = impulse / np.max(impulse)
            
            # Convolve with impulse response
            y_reverb = signal.convolve(y, impulse, mode='same')
            
            # Mix original and reverb
            y_mixed = y * (1 - intensity) + y_reverb * intensity
            
            return y_mixed
        except:
            return y
    
    def spec_augment(self, y, sr):
        """
        SpecAugment: frequency and time masking
        """
        try:
            # Compute spectrogram
            n_fft = 2048
            hop_length = 512
            D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
            magnitude = np.abs(D)
            phase = np.angle(D)
            
            # Frequency masking
            if random.random() < self.augmentation_config['frequency_mask']['probability']:
                max_mask = int(self.augmentation_config['frequency_mask']['max_mask'] * magnitude.shape[0])
                num_masks = random.randint(1, 3)
                
                for _ in range(num_masks):
                    f = random.randint(0, max_mask)
                    f0 = random.randint(0, magnitude.shape[0] - f)
                    magnitude[f0:f0+f, :] = 0
            
            # Time masking
            if random.random() < self.augmentation_config['time_mask']['probability']:
                max_mask = int(self.augmentation_config['time_mask']['max_mask'] * magnitude.shape[1])
                num_masks = random.randint(1, 3)
                
                for _ in range(num_masks):
                    t = random.randint(0, max_mask)
                    t0 = random.randint(0, magnitude.shape[1] - t)
                    magnitude[:, t0:t0+t] = 0
            
            # Reconstruct
            D_augmented = magnitude * np.exp(1j * phase)
            y_augmented = librosa.istft(D_augmented, hop_length=hop_length)
            
            # Ensure same length
            if len(y_augmented) > len(y):
                y_augmented = y_augmented[:len(y)]
            elif len(y_augmented) < len(y):
                y_augmented = np.pad(y_augmented, (0, len(y) - len(y_augmented)))
            
            return y_augmented
            
        except:
            return y
    
    def add_background_noise(self, y, noise_file, snr_db=15):
        """
        Add background noise from another audio file
        """
        try:
            # Load noise
            noise, _ = librosa.load(noise_file, sr=self.target_sr, mono=True)
            
            # Adjust noise length
            if len(noise) < len(y):
                noise = np.tile(noise, int(np.ceil(len(y) / len(noise))))[:len(y)]
            else:
                noise = noise[:len(y)]
            
            # Calculate required noise power
            signal_power = np.mean(y ** 2)
            noise_power = np.mean(noise ** 2)
            desired_noise_power = signal_power / (10 ** (snr_db / 10))
            
            # Scale noise
            noise = noise * np.sqrt(desired_noise_power / noise_power)
            
            return y + noise
        except:
            return y
    
    def telephone_effect(self, y, sr):
        """
        Simulate telephone bandwidth (300-3400 Hz)
        """
        try:
            # Design bandpass filter
            nyquist = sr / 2
            low = 300 / nyquist
            high = 3400 / nyquist
            b, a = butter(4, [low, high], btype='band')
            
            # Apply filter
            y_filtered = filtfilt(b, a, y)
            
            return y_filtered
        except:
            return y
    
    def equalize_audio(self, y, sr):
        """
        Apply EQ to enhance voice frequencies
        """
        try:
            # Design EQ filters
            nyquist = sr / 2
            
            # Boost low-mids (voice fundamentals)
            low_mid_low = 200 / nyquist
            low_mid_high = 800 / nyquist
            b1, a1 = butter(2, [low_mid_low, low_mid_high], btype='band')
            
            # Boost presence (2-4 kHz)
            presence_low = 2000 / nyquist
            presence_high = 4000 / nyquist
            b2, a2 = butter(2, [presence_low, presence_high], btype='band')
            
            # Apply filters
            y_low_mid = filtfilt(b1, a1, y)
            y_presence = filtfilt(b2, a2, y)
            
            # Mix
            y_eq = y * 0.7 + y_low_mid * 0.2 + y_presence * 0.1
            
            return y_eq
        except:
            return y
    
    def clip_distortion(self, y, intensity=0.5):
        """
        Add subtle clipping distortion
        """
        threshold = 1.0 - intensity
        y_clipped = np.clip(y, -threshold, threshold)
        y_clipped = y_clipped / threshold
        return y_clipped
    
    def random_crop(self, y, min_duration=2.0, max_duration=10.0):
        """
        Randomly crop audio to specified duration range
        """
        min_samples = int(min_duration * self.target_sr)
        max_samples = int(max_duration * self.target_sr)
        
        if len(y) < min_samples:
            return y
        
        # Random duration
        target_length = random.randint(min_samples, min(max_samples, len(y)))
        
        # Random start position
        start = random.randint(0, len(y) - target_length)
        
        return y[start:start + target_length]
    
    def mixaudio_augmentation(self, y1, y2, mix_ratio=0.5):
        """
        Mix two audio samples
        """
        # Ensure same length
        if len(y1) != len(y2):
            min_len = min(len(y1), len(y2))
            y1 = y1[:min_len]
            y2 = y2[:min_len]
        
        return y1 * mix_ratio + y2 * (1 - mix_ratio)
    
    def advanced_augmentation_pipeline(self, y, sr, augmentation_level='medium'):
        """
        Apply advanced augmentation pipeline based on level
        """
        if augmentation_level == 'light':
            num_augmentations = 2
            intensity = 0.3
        elif augmentation_level == 'medium':
            num_augmentations = 4
            intensity = 0.5
        elif augmentation_level == 'heavy':
            num_augmentations = 6
            intensity = 0.7
        else:
            num_augmentations = 3
            intensity = 0.5
        
        augmented_audios = []
        
        for i in range(num_augmentations):
            y_aug = y.copy()
            
            # Apply combinations
            if i % 2 == 0:
                # Pitch + Time
                y_aug = self.pitch_shift(y_aug, sr, random.uniform(-2, 2))
                y_aug = self.time_stretch(y_aug, random.uniform(0.9, 1.1))
            else:
                # Volume + Noise
                y_aug = self.change_volume(y_aug, random.uniform(0.8, 1.2))
                y_aug = self.add_noise(y_aug, random.uniform(15, 25))
            
            # Random additional augmentations
            if random.random() < intensity:
                y_aug = self.add_reverb(y_aug, sr, random.uniform(0.1, 0.3))
            
            if random.random() < intensity * 0.5:
                y_aug = self.telephone_effect(y_aug, sr)
            
            augmented_audios.append(y_aug)
        
        return augmented_audios
    
    def batch_augment_dataset(self, audio_files, output_dir, augmentations_per_file=4):
        """
        Augment entire dataset of audio files
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        augmentation_log = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                # Load audio
                y, sr = librosa.load(audio_file, sr=self.target_sr, mono=True)
                
                # Generate augmentations
                augmented_versions = self.augment_audio(y, sr, num_augmentations=augmentations_per_file)
                
                # Save augmented versions
                base_name = Path(audio_file).stem
                for j, augmented in enumerate(augmented_versions):
                    output_path = os.path.join(output_dir, f"{base_name}_aug_{j:02d}.wav")
                    sf.write(output_path, augmented, self.target_sr)
                
                augmentation_log.append({
                    'original': audio_file,
                    'augmentations': augmentations_per_file,
                    'status': 'success'
                })
                
                print(f"✅ Augmented {i+1}/{len(audio_files)}: {base_name}")
                
            except Exception as e:
                augmentation_log.append({
                    'original': audio_file,
                    'status': 'failed',
                    'error': str(e)
                })
                print(f"❌ Failed to augment {audio_file}: {e}")
        
        return augmentation_log
    
    def create_training_balanced_dataset(self, audio_files, output_dir, target_samples_per_class=50):
        """
        Create a balanced training dataset with augmentation
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Calculate how many augmentations needed per file
        num_files = len(audio_files)
        augmentations_per_file = max(1, target_samples_per_class // num_files)
        
        print(f"📊 Creating balanced dataset:")
        print(f"   Original files: {num_files}")
        print(f"   Target samples per class: {target_samples_per_class}")
        print(f"   Augmentations per file: {augmentations_per_file}")
        
        return self.batch_augment_dataset(audio_files, output_dir, augmentations_per_file)

class ConditionalAugmentor:
    """
    Conditional augmentation based on audio characteristics
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.general_augmentor = VoiceDataAugmentor(target_sr)
    
    def analyze_and_augment(self, y, sr):
        """
        Analyze audio characteristics and apply appropriate augmentation
        """
        # Analyze characteristics
        duration = len(y) / sr
        rms = librosa.feature.rms(y=y)[0]
        avg_rms = np.mean(rms)
        
        # Duration-based augmentation
        if duration < 2.0:
            # Short audio: apply time stretching to make it longer
            y = self.general_augmentor.time_stretch(y, 1.2)
        elif duration > 10.0:
            # Long audio: apply random cropping
            y = self.general_augmentor.random_crop(y, min_duration=3.0, max_duration=8.0)
        
        # Volume-based augmentation
        if avg_rms < 0.1:
            # Quiet audio: boost volume
            y = self.general_augmentor.change_volume(y, 1.5)
        elif avg_rms > 0.5:
            # Loud audio: reduce volume slightly
            y = self.general_augmentor.change_volume(y, 0.8)
        
        # Apply standard augmentations
        y = self.general_augmentor.pitch_shift(y, sr, random.uniform(-2, 2))
        y = self.general_augmentor.add_noise(y, random.uniform(15, 25))
        
        return y

def main():
    """Test the data augmentation system"""
    import os
    
    augmentor = VoiceDataAugmentor()
    
    # Test with a sample file
    test_file = "dataset/test_audio.wav"
    if os.path.exists(test_file):
        y, sr = librosa.load(test_file, sr=22050, mono=True)
        
        print("🎤 Testing Data Augmentation")
        print(f"📁 Original audio: {len(y)/sr:.2f}s")
        
        # Generate augmentations
        augmented = augmentor.augment_audio(y, sr, num_augmentations=3)
        
        print(f"✅ Generated {len(augmented)} augmented versions")
        
        # Save augmentations
        output_dir = "output/augmented_test"
        os.makedirs(output_dir, exist_ok=True)
        
        for i, aug_y in enumerate(augmented):
            output_path = os.path.join(output_dir, f"augmented_{i}.wav")
            sf.write(output_path, aug_y, sr)
            print(f"💾 Saved: {output_path}")
        
        print("🎉 Data augmentation test complete!")
    else:
        print("Please provide a test audio file")

if __name__ == "__main__":
    main()