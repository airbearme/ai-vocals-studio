#!/usr/bin/env python3
"""
Advanced Voice Feature Extractor for Precision Voice Cloning
Captures comprehensive voice characteristics including timbre, formants, prosody, and emotional content
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.stats import skew, kurtosis
import json
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

class VoiceFeatureExtractor:
    """
    Extract comprehensive voice features for precision cloning
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.feature_ranges = self._initialize_feature_ranges()
    
    def _initialize_feature_ranges(self):
        """Initialize typical ranges for voice features"""
        return {
            'pitch_hz': (85, 400),  # Human voice pitch range
            'formant_f1': (200, 800),
            'formant_f2': (800, 2500),
            'formant_f3': (2000, 3500),
            'jitter': (0.01, 0.05),
            'shimmer': (0.03, 0.15),
            'energy_db': (-60, -5),
            'spectral_centroid': (500, 5000),
            'spectral_rolloff': (2000, 8000),
            'mfcc_mean': (-30, 30),
            'zero_crossing_rate': (0.01, 0.3)
        }
    
    def extract_comprehensive_features(self, audio_path: str) -> Dict:
        """
        Extract comprehensive voice features from audio file
        """
        try:
            # Load audio
            y, sr = librosa.load(audio_path, sr=self.target_sr, mono=True)
            
            features = {
                'basic_info': self._extract_basic_info(y, sr),
                'pitch_features': self._extract_pitch_features(y, sr),
                'formant_features': self._extract_formant_features(y, sr),
                'spectral_features': self._extract_spectral_features(y, sr),
                'prosodic_features': self._extract_prosodic_features(y, sr),
                'voice_quality': self._extract_voice_quality(y, sr),
                'emotional_features': self._extract_emotional_features(y, sr),
                'temporal_features': self._extract_temporal_features(y, sr),
                'articulation_features': self._extract_articulation_features(y, sr)
            }
            
            # Calculate feature importance scores
            features['importance_scores'] = self._calculate_importance_scores(features)
            
            return features
            
        except Exception as e:
            print(f"Error extracting features: {e}")
            raise
    
    def _extract_basic_info(self, y: np.ndarray, sr: int) -> Dict:
        """Extract basic audio information"""
        duration = len(y) / sr
        rms_energy = librosa.feature.rms(y=y)[0]
        
        return {
            'duration_s': duration,
            'sample_rate': sr,
            'max_amplitude': np.max(np.abs(y)),
            'mean_rms': np.mean(rms_energy),
            'std_rms': np.std(rms_energy),
            'dynamic_range_db': 20 * np.log10(np.max(np.abs(y)) / (np.mean(np.abs(y)) + 1e-10))
        }
    
    def _extract_pitch_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract pitch-related features"""
        # Extract pitch using multiple methods
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        
        # Get dominant pitch
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if not pitch_values:
            pitch_values = [0]
        
        pitch_array = np.array(pitch_values)
        
        # Statistical features
        features = {
            'mean_pitch_hz': float(np.mean(pitch_array)),
            'std_pitch_hz': float(np.std(pitch_array)),
            'min_pitch_hz': float(np.min(pitch_array)),
            'max_pitch_hz': float(np.max(pitch_array)),
            'pitch_range_hz': float(np.max(pitch_array) - np.min(pitch_array)),
            'median_pitch_hz': float(np.median(pitch_array)),
            'pitch_contour': self._analyze_pitch_contour(pitch_array)
        }
        
        # Pitch stability metrics
        if len(pitch_array) > 1:
            pitch_diffs = np.diff(pitch_array)
            features['pitch_stability'] = float(1.0 / (1.0 + np.std(pitch_diffs)))
            features['pitch_variability'] = float(np.std(pitch_diffs))
        else:
            features['pitch_stability'] = 0.0
            features['pitch_variability'] = 0.0
        
        return features
    
    def _analyze_pitch_contour(self, pitch_array: np.ndarray) -> Dict:
        """Analyze pitch contour patterns"""
        if len(pitch_array) < 3:
            return {'slope': 0.0, 'curvature': 0.0, 'pattern': 'flat'}
        
        # Calculate slope (linear trend)
        time_points = np.arange(len(pitch_array))
        slope, intercept = np.polyfit(time_points, pitch_array, 1)
        
        # Calculate curvature (quadratic fit)
        coeffs = np.polyfit(time_points, pitch_array, 2)
        curvature = 2 * coeffs[0]
        
        # Determine pattern
        if abs(slope) < 0.1:
            pattern = 'flat'
        elif slope > 0:
            pattern = 'rising'
        else:
            pattern = 'falling'
        
        return {
            'slope': float(slope),
            'curvature': float(curvature),
            'pattern': pattern
        }
    
    def _extract_formant_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract formant frequencies (vocal tract characteristics)"""
        # Use LPC to estimate formants
        try:
            # Pre-emphasis
            pre_emphasis = 0.97
            y_emphasized = np.append(y[0], y[1:] - pre_emphasis * y[:-1])
            
            # Split into frames
            frame_length = int(0.025 * sr)
            hop_length = int(0.010 * sr)
            
            formants_f1 = []
            formants_f2 = []
            formants_f3 = []
            
            for i in range(0, len(y_emphasized) - frame_length, hop_length):
                frame = y_emphasized[i:i + frame_length]
                
                # Apply window
                windowed = frame * np.hamming(len(frame))
                
                # LPC analysis
                order = int(2 + sr / 1000)  # LPC order based on sample rate
                try:
                    from scipy.signal import lfilter
                    a = librosa.lpc(windowed, order)
                    
                    # Find roots
                    roots = np.roots(a)
                    
                    # Keep only roots with positive imaginary part
                    roots = roots[np.imag(roots) > 0]
                    
                    # Convert to frequencies
                    angles = np.angle(roots)
                    freqs = angles * sr / (2 * np.pi)
                    
                    # Sort and take first 3 formants
                    freqs = np.sort(freqs)
                    
                    if len(freqs) >= 1:
                        formants_f1.append(freqs[0])
                    if len(freqs) >= 2:
                        formants_f2.append(freqs[1])
                    if len(freqs) >= 3:
                        formants_f3.append(freqs[2])
                        
                except:
                    continue
            
            features = {}
            if formants_f1:
                features['formant_f1_mean'] = float(np.mean(formants_f1))
                features['formant_f1_std'] = float(np.std(formants_f1))
            if formants_f2:
                features['formant_f2_mean'] = float(np.mean(formants_f2))
                features['formant_f2_std'] = float(np.std(formants_f2))
            if formants_f3:
                features['formant_f3_mean'] = float(np.mean(formants_f3))
                features['formant_f3_std'] = float(np.std(formants_f3))
            
            # Formant ratios (important for voice quality)
            if formants_f1 and formants_f2:
                features['formant_ratio_f2_f1'] = float(np.mean(formants_f2) / np.mean(formants_f1))
            
            return features if features else {'error': 'Could not extract formants'}
            
        except Exception as e:
            return {'error': f'Formant extraction failed: {str(e)}'}
    
    def _extract_spectral_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract spectral characteristics"""
        # Spectral centroid
        spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        
        # Spectral rolloff
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)[0]
        
        # Spectral bandwidth
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)[0]
        
        # Spectral contrast
        spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)
        
        # Spectral flatness
        spectral_flatness = librosa.feature.spectral_flatness(y=y)[0]
        
        # MFCCs
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        
        # Chroma features
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        
        features = {
            'spectral_centroid_mean': float(np.mean(spectral_centroids)),
            'spectral_centroid_std': float(np.std(spectral_centroids)),
            'spectral_rolloff_mean': float(np.mean(spectral_rolloff)),
            'spectral_rolloff_std': float(np.std(spectral_rolloff)),
            'spectral_bandwidth_mean': float(np.mean(spectral_bandwidth)),
            'spectral_bandwidth_std': float(np.std(spectral_bandwidth)),
            'spectral_flatness_mean': float(np.mean(spectral_flatness)),
            'spectral_flatness_std': float(np.std(spectral_flatness)),
            'spectral_contrast_mean': float(np.mean(spectral_contrast)),
            'mfcc_means': [float(np.mean(mfcc)) for mfcc in mfccs],
            'mfcc_stds': [float(np.std(mfcc)) for mfcc in mfccs],
            'chroma_mean': float(np.mean(chroma)),
            'chroma_std': float(np.std(chroma))
        }
        
        # Timbre characteristics
        features['brightness'] = float(np.mean(spectral_centroids))
        features['warmth'] = float(np.mean(spectral_rolloff) - np.mean(spectral_centroids))
        
        return features
    
    def _extract_prosodic_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract prosodic features (rhythm, timing, stress)"""
        # Tempo and beats
        tempo, beats = librosa.beat.beat_track(y=y, sr=sr)
        
        # Onset detection
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        
        # Rhythm patterns
        if len(onsets) > 1:
            onset_intervals = np.diff(onsets) / sr
            rhythm_regularity = 1.0 / (1.0 + np.std(onset_intervals))
        else:
            rhythm_regularity = 0.0
        
        # Energy envelope
        energy_envelope = librosa.feature.rms(y=y)[0]
        energy_peaks = signal.find_peaks(energy_envelope, height=np.mean(energy_envelope))[0]
        
        features = {
            'tempo_bpm': float(tempo),
            'num_beats': len(beats),
            'num_onsets': len(onsets),
            'rhythm_regularity': float(rhythm_regularity),
            'onset_rate': float(len(onsets) / (len(y) / sr)),
            'energy_peak_rate': float(len(energy_peaks) / len(energy_envelope)),
            'stress_pattern': self._analyze_stress_pattern(energy_envelope)
        }
        
        return features
    
    def _analyze_stress_pattern(self, energy_envelope: np.ndarray) -> Dict:
        """Analyze stress patterns in speech"""
        # Find peaks and valleys
        peaks = signal.find_peaks(energy_envelope, height=np.mean(energy_envelope))[0]
        valleys = signal.find_peaks(-energy_envelope, height=-np.mean(energy_envelope))[0]
        
        if len(peaks) > 0 and len(valleys) > 0:
            stress_variation = np.std(energy_envelope[peaks]) / (np.mean(energy_envelope[peaks]) + 1e-10)
        else:
            stress_variation = 0.0
        
        return {
            'num_stressed_syllables': len(peaks),
            'stress_variation': float(stress_variation),
            'stress_pattern': 'variable' if stress_variation > 0.3 else 'consistent'
        }
    
    def _extract_voice_quality(self, y: np.ndarray, sr: int) -> Dict:
        """Extract voice quality metrics (jitter, shimmer, harmonics)"""
        # Harmonic-to-noise ratio
        harmonic, percussive = librosa.effects.hpss(y)
        hnr = 10 * np.log10(np.mean(harmonic**2) / (np.mean(percussive**2) + 1e-10))
        
        # Jitter (frequency perturbation)
        zero_crossings = librosa.zero_crossings(y)
        jitter = np.std(zero_crossings.astype(float)) if len(zero_crossings) > 0 else 0.0
        
        # Shimmer (amplitude perturbation)
        amplitude_envelope = librosa.feature.rms(y=y)[0]
        shimmer = np.std(amplitude_envelope) / (np.mean(amplitude_envelope) + 1e-10)
        
        # Breathiness (ratio of energy in high frequencies)
        D = librosa.stft(y)
        magnitude = np.abs(D)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        high_freq_energy = np.mean(magnitude[freq_bins > 5000, :])
        total_energy = np.mean(magnitude)
        breathiness = high_freq_energy / (total_energy + 1e-10)
        
        features = {
            'harmonic_to_noise_ratio_db': float(hnr),
            'jitter': float(jitter),
            'shimmer': float(shimmer),
            'breathiness': float(breathiness),
            'vocal_fry': self._detect_vocal_fry(y, sr),
            'creakiness': self._detect_creakiness(y, sr)
        }
        
        return features
    
    def _detect_vocal_fry(self, y: np.ndarray, sr: int) -> float:
        """Detect vocal fry (very low pitch, creaky voice)"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if not pitch_values:
            return 0.0
        
        # Vocal fry is characterized by very low pitch (< 70 Hz)
        low_pitch_ratio = np.mean(np.array(pitch_values) < 70)
        return float(low_pitch_ratio)
    
    def _detect_creakiness(self, y: np.ndarray, sr: int) -> float:
        """Detect voice creakiness (irregular pitch periods)"""
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if len(pitch_values) < 2:
            return 0.0
        
        # Creakiness = high pitch variability
        pitch_diffs = np.diff(pitch_values)
        creakiness = np.std(pitch_diffs) / (np.mean(pitch_diffs) + 1e-10)
        
        return float(min(1.0, creakiness))
    
    def _extract_emotional_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract emotional content from voice"""
        # Energy dynamics
        energy = librosa.feature.rms(y=y)[0]
        energy_dynamics = np.std(energy) / (np.mean(energy) + 1e-10)
        
        # Pitch dynamics
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_values = []
        for t in range(pitches.shape[1]):
            index = magnitudes[:, t].argmax()
            pitch = pitches[index, t]
            if pitch > 0:
                pitch_values.append(pitch)
        
        if pitch_values:
            pitch_dynamics = np.std(pitch_values) / (np.mean(pitch_values) + 1e-10)
        else:
            pitch_dynamics = 0.0
        
        # Speaking rate
        onsets = librosa.onset.onset_detect(y=y, sr=sr)
        speaking_rate = len(onsets) / (len(y) / sr) if len(y) > 0 else 0.0
        
        features = {
            'energy_dynamics': float(energy_dynamics),
            'pitch_dynamics': float(pitch_dynamics),
            'speaking_rate': float(speaking_rate),
            'arousal': self._estimate_arousal(energy_dynamics, pitch_dynamics),
            'valence': self._estimate_valence(pitch_values, energy),
            'dominance': self._estimate_dominance(energy, speaking_rate)
        }
        
        return features
    
    def _estimate_arousal(self, energy_dynamics: float, pitch_dynamics: float) -> float:
        """Estimate emotional arousal (activation level)"""
        arousal = (energy_dynamics + pitch_dynamics) / 2
        return float(min(1.0, arousal))
    
    def _estimate_valence(self, pitch_values: List, energy: np.ndarray) -> float:
        """Estimate emotional valence (positive/negative)"""
        if not pitch_values:
            return 0.5
        
        # Higher pitch and more energy variation = more positive
        mean_pitch = np.mean(pitch_values)
        energy_var = np.var(energy)
        
        # Normalize pitch (assuming 200-300 Hz is neutral)
        pitch_normalized = (mean_pitch - 250) / 100
        energy_normalized = min(1.0, energy_var / 0.01)
        
        valence = 0.5 + (pitch_normalized * 0.3) + (energy_normalized * 0.2)
        return float(max(0.0, min(1.0, valence)))
    
    def _estimate_dominance(self, energy: np.ndarray, speaking_rate: float) -> float:
        """Estimate dominance (power/confidence)"""
        mean_energy = np.mean(energy)
        rate_normalized = min(1.0, speaking_rate / 10)
        
        dominance = (mean_energy * 0.6) + (rate_normalized * 0.4)
        return float(min(1.0, dominance))
    
    def _extract_temporal_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract temporal features"""
        # Zero crossing rate
        zcr = librosa.feature.zero_crossing_rate(y)[0]
        
        # Autocorrelation
        autocorr = np.correlate(y, y, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        
        features = {
            'zero_crossing_rate_mean': float(np.mean(zcr)),
            'zero_crossing_rate_std': float(np.std(zcr)),
            'autocorrelation_peak': float(np.max(autocorr)),
            'autocorrelation_decay': self._calculate_autocorr_decay(autocorr)
        }
        
        return features
    
    def _calculate_autocorr_decay(self, autocorr: np.ndarray) -> float:
        """Calculate decay rate of autocorrelation"""
        if len(autocorr) < 2:
            return 0.0
        
        # Find time to first major decay
        peak_value = autocorr[0]
        half_peak = peak_value / 2
        
        for i, val in enumerate(autocorr):
            if val < half_peak:
                return float(i / len(autocorr))
        
        return 1.0
    
    def _extract_articulation_features(self, y: np.ndarray, sr: int) -> Dict:
        """Extract articulation features"""
        # Spectral flux (measure of spectral change)
        spectral_flux = librosa.onset.onset_strength(y=y, sr=sr)
        
        # Energy distribution across frequency bands
        D = librosa.stft(y)
        magnitude = np.abs(D)
        freq_bins = librosa.fft_frequencies(sr=sr, n_fft=2048)
        
        # Divide into frequency bands
        low_band = magnitude[(freq_bins >= 0) & (freq_bins < 500)]
        mid_band = magnitude[(freq_bins >= 500) & (freq_bins < 2000)]
        high_band = magnitude[(freq_bins >= 2000) & (freq_bins < 5000)]
        very_high_band = magnitude[freq_bins >= 5000]
        
        features = {
            'spectral_flux_mean': float(np.mean(spectral_flux)),
            'spectral_flux_std': float(np.std(spectral_flux)),
            'low_band_energy': float(np.mean(low_band)),
            'mid_band_energy': float(np.mean(mid_band)),
            'high_band_energy': float(np.mean(high_band)),
            'very_high_band_energy': float(np.mean(very_high_band)),
            'articulation_rate': float(np.mean(spectral_flux)),
            'articulation_clarity': float(np.mean(mid_band) / (np.mean(low_band) + 1e-10))
        }
        
        return features
    
    def _calculate_importance_scores(self, features: Dict) -> Dict:
        """Calculate importance scores for different feature categories"""
        scores = {}
        
        # Pitch features are most important for voice identity
        pitch_importance = 0.9
        scores['pitch_importance'] = pitch_importance
        
        # Formants define vocal tract shape
        formant_importance = 0.85
        scores['formant_importance'] = formant_importance
        
        # Spectral features define timbre
        spectral_importance = 0.8
        scores['spectral_importance'] = spectral_importance
        
        # Prosodic features define speaking style
        prosodic_importance = 0.7
        scores['prosodic_importance'] = prosodic_importance
        
        # Voice quality defines vocal health
        quality_importance = 0.75
        scores['quality_importance'] = quality_importance
        
        # Emotional features define expression
        emotional_importance = 0.6
        scores['emotional_importance'] = emotional_importance
        
        return scores
    
    def save_features(self, features: Dict, output_path: str):
        """Save features to JSON file"""
        with open(output_path, 'w') as f:
            json.dump(features, f, indent=2)
    
    def load_features(self, input_path: str) -> Dict:
        """Load features from JSON file"""
        with open(input_path, 'r') as f:
            return json.load(f)
    
    def compare_features(self, features1: Dict, features2: Dict) -> Dict:
        """Compare two feature sets and return similarity scores"""
        similarity_scores = {}
        
        # Compare pitch features
        if 'pitch_features' in features1 and 'pitch_features' in features2:
            pitch_sim = self._compare_pitch_features(
                features1['pitch_features'], 
                features2['pitch_features']
            )
            similarity_scores['pitch_similarity'] = pitch_sim
        
        # Compare spectral features
        if 'spectral_features' in features1 and 'spectral_features' in features2:
            spectral_sim = self._compare_spectral_features(
                features1['spectral_features'],
                features2['spectral_features']
            )
            similarity_scores['spectral_similarity'] = spectral_sim
        
        # Calculate overall similarity
        if similarity_scores:
            overall_sim = np.mean(list(similarity_scores.values()))
            similarity_scores['overall_similarity'] = float(overall_sim)
        
        return similarity_scores
    
    def _compare_pitch_features(self, pitch1: Dict, pitch2: Dict) -> float:
        """Compare pitch features"""
        try:
            mean_diff = abs(pitch1['mean_pitch_hz'] - pitch2['mean_pitch_hz'])
            std_diff = abs(pitch1['std_pitch_hz'] - pitch2['std_pitch_hz'])
            
            # Normalize differences
            mean_sim = 1.0 - min(1.0, mean_diff / 100)  # 100 Hz tolerance
            std_sim = 1.0 - min(1.0, std_diff / 50)     # 50 Hz tolerance
            
            return float((mean_sim + std_sim) / 2)
        except:
            return 0.0
    
    def _compare_spectral_features(self, spec1: Dict, spec2: Dict) -> float:
        """Compare spectral features"""
        try:
            centroid_diff = abs(spec1['spectral_centroid_mean'] - spec2['spectral_centroid_mean'])
            bandwidth_diff = abs(spec1['spectral_bandwidth_mean'] - spec2['spectral_bandwidth_mean'])
            
            centroid_sim = 1.0 - min(1.0, centroid_diff / 1000)  # 1000 Hz tolerance
            bandwidth_sim = 1.0 - min(1.0, bandwidth_diff / 1500)  # 1500 Hz tolerance
            
            return float((centroid_sim + bandwidth_sim) / 2)
        except:
            return 0.0

def main():
    """Test the voice feature extractor"""
    extractor = VoiceFeatureExtractor()
    
    # Test with a sample file
    test_file = "dataset/test_audio.wav"
    if os.path.exists(test_file):
        features = extractor.extract_comprehensive_features(test_file)
        
        print("🎤 Voice Feature Extraction Complete!")
        print(f"📊 Features extracted:")
        for category, category_features in features.items():
            if isinstance(category_features, dict):
                print(f"   {category}: {len(category_features)} features")
        
        # Save features
        output_path = "output/voice_features.json"
        extractor.save_features(features, output_path)
        print(f"💾 Features saved to {output_path}")
    else:
        print("Please provide a test audio file")

if __name__ == "__main__":
    import os
    main()