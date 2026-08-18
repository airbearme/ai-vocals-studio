#!/usr/bin/env python3
"""
Advanced Audio Processor for Precision Voice Cloning
Implements state-of-the-art audio preprocessing for maximum voice cloning accuracy
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.signal import butter, filtfilt
import warnings
warnings.filterwarnings('ignore')

class AdvancedAudioProcessor:
    """
    Professional-grade audio preprocessing for voice cloning
    Features: noise reduction, VAD, spectral enhancement, and quality optimization
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.noise_threshold_db = -40
        self.vad_threshold = 0.5
        
    def preprocess_audio(self, audio_path, output_path=None):
        """
        Complete preprocessing pipeline for voice cloning
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            # Step 1: Noise reduction
            y = self.reduce_noise(y, sr)
            
            # Step 2: Voice Activity Detection
            y = self.apply_vad(y, sr)
            
            # Step 3: Spectral enhancement
            y = self.enhance_spectra(y, sr)
            
            # Step 4: Dynamic range compression
            y = self.compress_dynamic_range(y)
            
            # Step 5: Normalization
            y = self.normalize_audio(y)
            
            # Step 6: Quality assessment
            quality_score = self.assess_quality(y, sr)
            
            # Save if output path provided
            if output_path:
                sf.write(output_path, y, self.target_sr)
                
            return y, quality_score
            
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            raise
    
    def reduce_noise(self, y, sr):
        """
        Advanced noise reduction using spectral subtraction
        """
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        # Estimate noise from first 0.5 seconds
        noise_samples = int(0.5 * sr / hop_length)
        noise_spectrum = np.mean(np.abs(D[:, :noise_samples]), axis=1, keepdims=True)
        
        # Spectral subtraction with over-subtraction factor
        alpha = 2.0  # Over-subtraction factor
        beta = 0.5   # Spectral floor factor
        
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Apply spectral subtraction
        enhanced_magnitude = magnitude - alpha * noise_spectrum
        enhanced_magnitude = np.maximum(enhanced_magnitude, beta * noise_spectrum)
        
        # Reconstruct signal
        D_enhanced = enhanced_magnitude * np.exp(1j * phase)
        y_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
        
        # Ensure same length
        if len(y_enhanced) > len(y):
            y_enhanced = y_enhanced[:len(y)]
        elif len(y_enhanced) < len(y):
            y_enhanced = np.pad(y_enhanced, (0, len(y) - len(y_enhanced)))
            
        return y_enhanced
    
    def apply_vad(self, y, sr):
        """
        Voice Activity Detection using energy and spectral features
        """
        # Compute energy-based VAD
        frame_length = int(0.025 * sr)  # 25ms frames
        hop_length = int(0.010 * sr)    # 10ms hop
        
        # Calculate frame energy
        frames = librosa.util.frame(y, frame_length=frame_length, hop_length=hop_length)
        energy = np.sum(frames ** 2, axis=0)
        
        # Normalize energy
        energy_db = 10 * np.log10(energy + 1e-10)
        energy_db = energy_db - np.max(energy_db)
        
        # Energy threshold
        energy_threshold = self.noise_threshold_db
        
        # Create VAD mask
        vad_mask = energy_db > energy_threshold
        
        # Apply spectral flatness for better VAD
        spectral_flatness = librosa.feature.spectral_flatness(y=y, hop_length=hop_length)[0]
        # NOTE: spectral_flatness uses an STFT frame grid (n_fft=2048) that may differ
        # in length from the energy-based VAD mask. Resample it onto the same grid so
        # the two masks can be combined element-wise without a shape mismatch.
        if len(spectral_flatness) != len(vad_mask):
            spectral_flatness = np.interp(
                np.linspace(0.0, 1.0, len(vad_mask)),
                np.linspace(0.0, 1.0, len(spectral_flatness)),
                spectral_flatness,
            )
        spectral_mask = spectral_flatness < 0.3  # Speech has low spectral flatness
        
        # Combine masks
        combined_mask = vad_mask & spectral_mask
        
        # Apply mask to audio
        y_vad = np.zeros_like(y)
        for i, keep in enumerate(combined_mask):
            start = i * hop_length
            end = start + frame_length
            if end <= len(y) and keep:
                y_vad[start:end] = y[start:end]
                
        return y_vad
    
    def enhance_spectra(self, y, sr):
        """
        Spectral enhancement using adaptive filtering
        """
        # Compute STFT
        n_fft = 2048
        hop_length = 512
        D = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        
        magnitude = np.abs(D)
        phase = np.angle(D)
        
        # Apply spectral masking (like speech enhancement)
        # Enhance formant regions
        mel_spec = librosa.feature.melspectrogram(S=magnitude, sr=sr, n_mels=80)
        
        # Enhance speech bands (300-3400 Hz)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        speech_band = (freq_bins >= 300) & (freq_bins <= 3400)
        
        # Boost speech frequencies
        magnitude[speech_band, :] *= 1.2
        
        # Reconstruct
        D_enhanced = magnitude * np.exp(1j * phase)
        y_enhanced = librosa.istft(D_enhanced, hop_length=hop_length)
        
        # Ensure same length
        if len(y_enhanced) > len(y):
            y_enhanced = y_enhanced[:len(y)]
        elif len(y_enhanced) < len(y):
            y_enhanced = np.pad(y_enhanced, (0, len(y) - len(y_enhanced)))
            
        return y_enhanced
    
    def compress_dynamic_range(self, y):
        """
        Dynamic range compression for consistent loudness
        """
        # Soft clipping
        threshold = 0.8
        ratio = 4.0
        
        y_compressed = np.where(
            np.abs(y) > threshold,
            threshold + (y - threshold) / ratio,
            y
        )
        
        return y_compressed
    
    def normalize_audio(self, y):
        """
        Peak normalization with headroom
        """
        target_level = 0.95
        max_val = np.max(np.abs(y))
        
        if max_val > 0:
            y_normalized = y * (target_level / max_val)
        else:
            y_normalized = y
            
        return y_normalized
    
    def assess_quality(self, y, sr):
        """
        Assess audio quality for voice cloning suitability
        Returns quality score (0-1) and metrics
        """
        metrics = {}
        
        # SNR estimation
        signal_power = np.mean(y ** 2)
        noise_floor = np.percentile(np.abs(y), 10) ** 2
        snr = 10 * np.log10(signal_power / (noise_floor + 1e-10))
        metrics['snr_db'] = snr
        
        # Duration check
        duration = len(y) / sr
        metrics['duration_s'] = duration
        
        # Zero crossing rate (speech indicator)
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        metrics['avg_zcr'] = np.mean(zcr)
        
        # Spectral centroid (brightness)
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        metrics['avg_spectral_centroid'] = np.mean(spectral_centroid)
        
        # MFCC quality
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        metrics['mfcc_variance'] = np.var(mfcc)
        
        # Calculate overall quality score
        quality_score = self._calculate_quality_score(metrics)
        metrics['overall_quality'] = quality_score
        
        return metrics
    
    def _calculate_quality_score(self, metrics):
        """
        Calculate overall quality score from metrics
        """
        score = 0.0
        
        # SNR score (higher is better, max 30dB)
        snr_score = min(1.0, max(0.0, (metrics['snr_db'] + 10) / 40))
        score += snr_score * 0.3
        
        # Duration score (3-30 seconds is ideal)
        duration = metrics['duration_s']
        if 3 <= duration <= 30:
            duration_score = 1.0
        elif duration < 3:
            duration_score = duration / 3
        else:
            duration_score = max(0.0, 1.0 - (duration - 30) / 30)
        score += duration_score * 0.2
        
        # ZCR score (speech has moderate ZCR)
        zcr = metrics['avg_zcr']
        zcr_score = 1.0 - min(1.0, abs(zcr - 0.1) / 0.2)
        score += zcr_score * 0.2
        
        # Spectral centroid (speech typically 2000-4000 Hz)
        centroid = metrics['avg_spectral_centroid']
        centroid_score = 1.0 - min(1.0, abs(centroid - 3000) / 2000)
        score += centroid_score * 0.15
        
        # MFCC variance (higher variance = more speech content)
        mfcc_var = metrics['mfcc_variance']
        mfcc_score = min(1.0, mfcc_var / 100)
        score += mfcc_score * 0.15
        
        return score
    
    def batch_preprocess(self, audio_files, output_dir):
        """
        Batch process multiple audio files
        """
        import os
        from pathlib import Path
        
        os.makedirs(output_dir, exist_ok=True)
        results = []
        
        for i, audio_file in enumerate(audio_files):
            try:
                output_path = os.path.join(output_dir, f"processed_{i:04d}.wav")
                y, quality = self.preprocess_audio(audio_file, output_path)
                
                results.append({
                    'input': audio_file,
                    'output': output_path,
                    'quality': quality,
                    'success': True
                })
                
                print(f"✅ Processed {i+1}/{len(audio_files)}: {os.path.basename(audio_file)}")
                print(f"   Quality Score: {quality['overall_quality']:.2f}")
                
            except Exception as e:
                results.append({
                    'input': audio_file,
                    'output': None,
                    'quality': None,
                    'success': False,
                    'error': str(e)
                })
                print(f"❌ Failed to process {audio_file}: {e}")
        
        return results

def main():
    """Test the advanced audio processor"""
    processor = AdvancedAudioProcessor()
    
    # Test with a sample file
    test_file = "dataset/test_audio.wav"
    if os.path.exists(test_file):
        output_file = "output/processed_test.wav"
        y, quality = processor.preprocess_audio(test_file, output_file)
        
        print("🎤 Audio Processing Complete!")
        print(f"📊 Quality Metrics:")
        for key, value in quality.items():
            print(f"   {key}: {value}")
    else:
        print("Please provide a test audio file")

if __name__ == "__main__":
    import os
    main()