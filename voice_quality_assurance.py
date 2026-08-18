#!/usr/bin/env python3
"""
Voice Quality Assurance System for Precision Voice Cloning
Implements comprehensive similarity scoring and quality metrics
"""

import numpy as np
import librosa
import soundfile as sf
from scipy import signal
from scipy.spatial.distance import cosine, euclidean
from scipy.stats import pearsonr
import json
import time
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class VoiceQualityAssurance:
    """
    Comprehensive quality assurance system for voice cloning
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.quality_thresholds = {
            'excellent': 0.95,
            'good': 0.85,
            'acceptable': 0.75,
            'poor': 0.65
        }
        
        # Feature weights for similarity calculation
        self.feature_weights = {
            'pitch_similarity': 0.22,
            'timbre_similarity': 0.26,
            'rhythm_similarity': 0.12,
            'energy_similarity': 0.08,
            'spectral_similarity': 0.17,
            'waveform_similarity': 0.15
        }
    
    def comprehensive_similarity_score(self, original_audio: str, cloned_audio: str) -> Dict:
        """
        Calculate comprehensive similarity score between original and cloned audio
        """
        try:
            # Load audio files
            y_orig, sr_orig = librosa.load(original_audio, sr=self.target_sr, mono=True)
            y_clone, sr_clone = librosa.load(cloned_audio, sr=self.target_sr, mono=True)
            
            # Ensure same length
            min_len = min(len(y_orig), len(y_clone))
            y_orig = y_orig[:min_len]
            y_clone = y_clone[:min_len]
            
            # Calculate individual similarity scores
            scores = {
                'pitch_similarity': self._calculate_pitch_similarity(y_orig, y_clone),
                'timbre_similarity': self._calculate_timbre_similarity(y_orig, y_clone),
                'rhythm_similarity': self._calculate_rhythm_similarity(y_orig, y_clone),
                'energy_similarity': self._calculate_energy_similarity(y_orig, y_clone),
                'spectral_similarity': self._calculate_spectral_similarity(y_orig, y_clone),
                'waveform_similarity': self._calculate_waveform_similarity(y_orig, y_clone)
            }
            
            # Calculate weighted overall similarity
            overall_similarity = self._calculate_weighted_similarity(scores)
            scores['overall_similarity'] = overall_similarity
            
            # Determine quality rating
            scores['quality_rating'] = self._get_quality_rating(overall_similarity)
            
            # Additional metrics
            scores['additional_metrics'] = self._calculate_additional_metrics(y_orig, y_clone)
            
            return scores
            
        except Exception as e:
            print(f"Error calculating similarity: {e}")
            return {'error': str(e)}
    
    def _calculate_pitch_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """Calculate pitch similarity using F0 contour comparison"""
        try:
            # Extract pitch contours
            pitches_orig, _ = librosa.piptrack(y=y_orig, sr=self.target_sr)
            pitches_clone, _ = librosa.piptrack(y=y_clone, sr=self.target_sr)
            
            # Get dominant pitch values
            pitch_values_orig = []
            pitch_values_clone = []
            
            for t in range(pitches_orig.shape[1]):
                idx_orig = pitches_orig[:, t].argmax()
                idx_clone = pitches_clone[:, t].argmax()
                
                pitch_orig = pitches_orig[idx_orig, t]
                pitch_clone = pitches_clone[idx_clone, t]
                
                if pitch_orig > 0 and pitch_clone > 0:
                    pitch_values_orig.append(pitch_orig)
                    pitch_values_clone.append(pitch_clone)
            
            if not pitch_values_orig or not pitch_values_clone:
                return 0.0
            
            # Calculate correlation
            if len(pitch_values_orig) > 1 and len(pitch_values_clone) > 1:
                correlation, _ = pearsonr(pitch_values_orig, pitch_values_clone)
                return float(max(0.0, correlation))
            else:
                # Fallback to mean difference
                mean_orig = np.mean(pitch_values_orig)
                mean_clone = np.mean(pitch_values_clone)
                diff_ratio = abs(mean_orig - mean_clone) / (mean_orig + 1e-10)
                return float(max(0.0, 1.0 - diff_ratio))
                
        except Exception as e:
            print(f"Pitch similarity error: {e}")
            return 0.0
    
    def _calculate_timbre_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """Calculate timbre similarity using MFCC comparison"""
        try:
            # Extract MFCCs
            n_mfcc = 13
            mfcc_orig = librosa.feature.mfcc(y=y_orig, sr=self.target_sr, n_mfcc=n_mfcc)
            mfcc_clone = librosa.feature.mfcc(y=y_clone, sr=self.target_sr, n_mfcc=n_mfcc)
            
            # Ensure same length
            min_frames = min(mfcc_orig.shape[1], mfcc_clone.shape[1])
            mfcc_orig = mfcc_orig[:, :min_frames]
            mfcc_clone = mfcc_clone[:, :min_frames]
            
            # Calculate similarity for each MFCC coefficient
            mfcc_similarities = []
            for i in range(n_mfcc):
                # Calculate cosine similarity
                similarity = 1 - cosine(mfcc_orig[i], mfcc_clone[i])
                mfcc_similarities.append(max(0.0, similarity))
            
            # Weight lower MFCCs more heavily (they contain more timbre information)
            weights = np.linspace(1.0, 0.5, n_mfcc)
            weighted_similarity = np.average(mfcc_similarities, weights=weights)
            
            return float(weighted_similarity)
            
        except Exception as e:
            print(f"Timbre similarity error: {e}")
            return 0.0
    
    def _calculate_rhythm_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """Calculate rhythm similarity using onset patterns"""
        try:
            # Detect onsets
            onsets_orig = librosa.onset.onset_detect(y=y_orig, sr=self.target_sr)
            onsets_clone = librosa.onset.onset_detect(y=y_clone, sr=self.target_sr)
            
            if len(onsets_orig) < 2 or len(onsets_clone) < 2:
                return 0.5  # Neutral score if insufficient onsets
            
            # Calculate onset intervals
            intervals_orig = np.diff(onsets_orig) / self.target_sr
            intervals_clone = np.diff(onsets_clone) / self.target_sr
            
            # Compare interval distributions
            if len(intervals_orig) > 0 and len(intervals_clone) > 0:
                # Use dynamic time warping for sequence comparison
                try:
                    from scipy.spatial.distance import euclidean
                    try:
                        from fastdtw import fastdtw
                        
                        distance, _ = fastdtw(intervals_orig.reshape(-1, 1), 
                                           intervals_clone.reshape(-1, 1), 
                                           dist=euclidean)
                        
                        # Normalize distance
                        max_distance = max(len(intervals_orig), len(intervals_clone))
                        similarity = 1.0 - (distance / max_distance)
                        return float(max(0.0, min(1.0, similarity)))
                    except ImportError:
                        # Fallback to simple statistical comparison
                        mean_orig = np.mean(intervals_orig)
                        mean_clone = np.mean(intervals_clone)
                        std_orig = np.std(intervals_orig)
                        std_clone = np.std(intervals_clone)
                        
                        mean_diff = abs(mean_orig - mean_clone) / (mean_orig + 1e-10)
                        std_diff = abs(std_orig - std_clone) / (std_orig + 1e-10)
                        
                        similarity = 1.0 - (mean_diff + std_diff) / 2
                        return float(max(0.0, min(1.0, similarity)))
                except Exception as e:
                    print(f"Rhythm comparison error: {e}")
                    return 0.5
            
            return 0.5
            
        except Exception as e:
            print(f"Rhythm similarity error: {e}")
            return 0.5
    
    def _calculate_energy_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """Calculate energy profile similarity"""
        try:
            # Calculate energy envelopes
            energy_orig = librosa.feature.rms(y=y_orig)[0]
            energy_clone = librosa.feature.rms(y=y_clone)[0]
            
            # Ensure same length
            min_len = min(len(energy_orig), len(energy_clone))
            energy_orig = energy_orig[:min_len]
            energy_clone = energy_clone[:min_len]
            
            # Calculate correlation
            if len(energy_orig) > 1:
                correlation, _ = pearsonr(energy_orig, energy_clone)
                return float(max(0.0, correlation))
            else:
                return 0.5
                
        except Exception as e:
            print(f"Energy similarity error: {e}")
            return 0.5
    
    def _calculate_spectral_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """Calculate spectral similarity using chroma and spectral features"""
        try:
            # Extract spectral features
            chroma_orig = librosa.feature.chroma_stft(y=y_orig, sr=self.target_sr)
            chroma_clone = librosa.feature.chroma_stft(y=y_clone, sr=self.target_sr)
            
            spectral_centroid_orig = librosa.feature.spectral_centroid(y=y_orig, sr=self.target_sr)[0]
            spectral_centroid_clone = librosa.feature.spectral_centroid(y=y_clone, sr=self.target_sr)[0]
            
            # Calculate chroma similarity
            min_frames = min(chroma_orig.shape[1], chroma_clone.shape[1])
            chroma_orig = chroma_orig[:, :min_frames]
            chroma_clone = chroma_clone[:, :min_frames]
            
            chroma_similarity = 0.0
            for i in range(12):  # 12 chroma bins
                similarity = 1 - cosine(chroma_orig[i], chroma_clone[i])
                chroma_similarity += max(0.0, similarity)
            chroma_similarity /= 12
            
            # Calculate spectral centroid similarity
            min_centroid = min(len(spectral_centroid_orig), len(spectral_centroid_clone))
            spectral_centroid_orig = spectral_centroid_orig[:min_centroid]
            spectral_centroid_clone = spectral_centroid_clone[:min_centroid]
            
            if len(spectral_centroid_orig) > 1:
                centroid_correlation, _ = pearsonr(spectral_centroid_orig, spectral_centroid_clone)
                centroid_similarity = max(0.0, centroid_correlation)
            else:
                centroid_similarity = 0.5
            
            # Combine similarities
            combined_similarity = (chroma_similarity * 0.6) + (centroid_similarity * 0.4)
            
            return float(combined_similarity)
            
        except Exception as e:
            print(f"Spectral similarity error: {e}")
            return 0.5
    
    def _calculate_waveform_similarity(self, y_orig: np.ndarray, y_clone: np.ndarray) -> float:
        """
        Calculate sample-level waveform similarity using Pearson correlation.

        This is the strictest "indistinguishable" check: two genuinely identical
        (or bit-close) clones score near 1.0, while unrelated audio scores ~0.
        """
        try:
            min_len = min(len(y_orig), len(y_clone))
            if min_len < 2:
                return 0.5

            a = y_orig[:min_len]
            b = y_clone[:min_len]

            a_centered = a - a.mean()
            b_centered = b - b.mean()

            denom = float(np.sqrt(np.sum(a_centered ** 2) * np.sum(b_centered ** 2)))
            if denom < 1e-10:
                return 0.5

            correlation = float(np.sum(a_centered * b_centered) / denom)
            return float(max(0.0, min(1.0, correlation)))

        except Exception as e:
            print(f"Waveform similarity error: {e}")
            return 0.5

    def _calculate_weighted_similarity(self, scores: Dict) -> float:
        """Calculate weighted overall similarity score"""
        weighted_sum = 0.0
        total_weight = 0.0
        
        for feature, weight in self.feature_weights.items():
            if feature in scores:
                weighted_sum += scores[feature] * weight
                total_weight += weight
        
        if total_weight > 0:
            return float(weighted_sum / total_weight)
        return 0.0
    
    def _get_quality_rating(self, similarity_score: float) -> str:
        """Get quality rating based on similarity score"""
        if similarity_score >= self.quality_thresholds['excellent']:
            return 'excellent'
        elif similarity_score >= self.quality_thresholds['good']:
            return 'good'
        elif similarity_score >= self.quality_thresholds['acceptable']:
            return 'acceptable'
        elif similarity_score >= self.quality_thresholds['poor']:
            return 'poor'
        else:
            return 'unacceptable'
    
    def _calculate_additional_metrics(self, y_orig: np.ndarray, y_clone: np.ndarray) -> Dict:
        """Calculate additional quality metrics"""
        metrics = {}
        
        # Signal-to-noise ratio
        signal_power_orig = np.mean(y_orig ** 2)
        noise = y_clone - y_orig
        noise_power = np.mean(noise ** 2)
        snr = 10 * np.log10(signal_power_orig / (noise_power + 1e-10))
        metrics['snr_db'] = float(snr)
        
        # Harmonic distortion
        try:
            harmonic_orig, percussive_orig = librosa.effects.hpss(y_orig)
            harmonic_clone, percussive_clone = librosa.effects.hpss(y_clone)
            
            harmonic_ratio_orig = np.mean(harmonic_orig ** 2) / (np.mean(percussive_orig ** 2) + 1e-10)
            harmonic_ratio_clone = np.mean(harmonic_clone ** 2) / (np.mean(percussive_clone ** 2) + 1e-10)
            
            metrics['harmonic_distortion'] = float(abs(harmonic_ratio_orig - harmonic_ratio_clone))
        except:
            metrics['harmonic_distortion'] = 0.0
        
        # Dynamic range similarity
        dynamic_range_orig = np.max(np.abs(y_orig)) - np.min(np.abs(y_orig))
        dynamic_range_clone = np.max(np.abs(y_clone)) - np.min(np.abs(y_clone))
        metrics['dynamic_range_similarity'] = float(1.0 - abs(dynamic_range_orig - dynamic_range_clone) / (dynamic_range_orig + 1e-10))
        
        return metrics
    
    def batch_quality_assessment(self, original_files: List[str], cloned_files: List[str]) -> Dict:
        """
        Perform batch quality assessment on multiple file pairs
        """
        results = []
        
        for orig, clone in zip(original_files, cloned_files):
            try:
                score = self.comprehensive_similarity_score(orig, clone)
                results.append({
                    'original': orig,
                    'cloned': clone,
                    'score': score,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'original': orig,
                    'cloned': clone,
                    'error': str(e),
                    'status': 'failed'
                })
        
        # Calculate aggregate statistics
        successful_results = [r for r in results if r['status'] == 'success']
        
        if successful_results:
            overall_similarities = [r['score']['overall_similarity'] for r in successful_results]
            quality_ratings = [r['score']['quality_rating'] for r in successful_results]
            
            aggregate_stats = {
                'total_pairs': len(results),
                'successful_assessments': len(successful_results),
                'failed_assessments': len(results) - len(successful_results),
                'mean_similarity': float(np.mean(overall_similarities)),
                'std_similarity': float(np.std(overall_similarities)),
                'min_similarity': float(np.min(overall_similarities)),
                'max_similarity': float(np.max(overall_similarities)),
                'quality_distribution': {
                    'excellent': quality_ratings.count('excellent'),
                    'good': quality_ratings.count('good'),
                    'acceptable': quality_ratings.count('acceptable'),
                    'poor': quality_ratings.count('poor'),
                    'unacceptable': quality_ratings.count('unacceptable')
                }
            }
        else:
            aggregate_stats = {
                'total_pairs': len(results),
                'successful_assessments': 0,
                'failed_assessments': len(results),
                'error': 'No successful assessments'
            }
        
        return {
            'individual_results': results,
            'aggregate_statistics': aggregate_stats
        }
    
    def generate_quality_report(self, assessment_results: Dict, output_path: str):
        """Generate comprehensive quality report"""
        report = {
            'assessment_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'quality_thresholds': self.quality_thresholds,
            'feature_weights': self.feature_weights,
            'results': assessment_results,
            'recommendations': self._generate_recommendations(assessment_results)
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return report
    
    def _generate_recommendations(self, assessment_results: Dict) -> List[str]:
        """Generate improvement recommendations based on assessment results"""
        recommendations = []
        
        if 'aggregate_statistics' in assessment_results:
            stats = assessment_results['aggregate_statistics']
            
            if stats.get('mean_similarity', 0) < self.quality_thresholds['good']:
                recommendations.append("Overall similarity is below good threshold. Consider retraining with more data.")
            
            quality_dist = stats.get('quality_distribution', {})
            if quality_dist.get('unacceptable', 0) > 0:
                recommendations.append("Some clones are unacceptable. Review training data quality.")
            
            if quality_dist.get('poor', 0) > quality_dist.get('excellent', 0):
                recommendations.append("More poor than excellent results. Consider hyperparameter tuning.")
        
        return recommendations
    
    def real_time_quality_monitor(self, audio_buffer: np.ndarray, reference_features: Dict) -> Dict:
        """
        Real-time quality monitoring during generation
        """
        try:
            # Extract features from current audio buffer
            current_features = self._extract_quick_features(audio_buffer)
            
            # Compare with reference
            quick_similarity = self._quick_feature_comparison(current_features, reference_features)
            
            return {
                'similarity': quick_similarity,
                'quality_status': 'good' if quick_similarity > 0.8 else 'needs_improvement',
                'timestamp': time.time()
            }
        except Exception as e:
            return {'error': str(e)}
    
    def _extract_quick_features(self, audio: np.ndarray) -> Dict:
        """Extract quick features for real-time monitoring"""
        features = {}
        
        # RMS energy
        features['rms'] = float(np.mean(librosa.feature.rms(y=audio)[0]))
        
        # Spectral centroid
        features['spectral_centroid'] = float(np.mean(librosa.feature.spectral_centroid(y=audio, sr=self.target_sr)[0]))
        
        # Zero crossing rate
        features['zcr'] = float(np.mean(librosa.feature.zero_crossing_rate(audio)[0]))
        
        return features
    
    def _quick_feature_comparison(self, current_features: Dict, reference_features: Dict) -> float:
        """Quick feature comparison for real-time monitoring"""
        try:
            similarities = []
            
            for key in ['rms', 'spectral_centroid', 'zcr']:
                if key in current_features and key in reference_features:
                    current_val = current_features[key]
                    ref_val = reference_features[key]
                    
                    if ref_val > 0:
                        similarity = 1.0 - min(1.0, abs(current_val - ref_val) / ref_val)
                        similarities.append(similarity)
            
            return float(np.mean(similarities)) if similarities else 0.5
        except:
            return 0.5

class AudioAuthentication:
    """
    Audio authentication for verifying voice clone accuracy
    """
    
    def __init__(self, target_sr=22050):
        self.target_sr = target_sr
        self.qa_system = VoiceQualityAssurance(target_sr)
    
    def authenticate_voice_clone(self, original_audio: str, cloned_audio: str, threshold: float = 0.85) -> Dict:
        """
        Authenticate whether a cloned voice matches the original
        """
        similarity_scores = self.qa_system.comprehensive_similarity_score(original_audio, cloned_audio)
        
        authentication_result = {
            'is_authenticated': similarity_scores['overall_similarity'] >= threshold,
            'similarity_score': similarity_scores['overall_similarity'],
            'quality_rating': similarity_scores['quality_rating'],
            'threshold_used': threshold,
            'detailed_scores': similarity_scores,
            'confidence_level': self._calculate_confidence(similarity_scores['overall_similarity'])
        }
        
        return authentication_result
    
    def _calculate_confidence(self, similarity_score: float) -> str:
        """Calculate confidence level based on similarity score"""
        if similarity_score >= 0.95:
            return 'very_high'
        elif similarity_score >= 0.90:
            return 'high'
        elif similarity_score >= 0.80:
            return 'medium'
        elif similarity_score >= 0.70:
            return 'low'
        else:
            return 'very_low'

def main():
    """Test the quality assurance system"""
    import os
    import time
    
    qa = VoiceQualityAssurance()
    
    # Test with sample files
    original_file = "dataset/original.wav"
    cloned_file = "output/cloned.wav"
    
    if os.path.exists(original_file) and os.path.exists(cloned_file):
        print("🎤 Testing Voice Quality Assurance")
        
        # Calculate similarity
        scores = qa.comprehensive_similarity_score(original_file, cloned_file)
        
        print(f"📊 Similarity Scores:")
        print(f"   Overall: {scores['overall_similarity']:.4f}")
        print(f"   Quality Rating: {scores['quality_rating']}")
        print(f"   Pitch Similarity: {scores['pitch_similarity']:.4f}")
        print(f"   Timbre Similarity: {scores['timbre_similarity']:.4f}")
        print(f"   Rhythm Similarity: {scores['rhythm_similarity']:.4f}")
        
        # Save report
        report_path = "output/quality_report.json"
        qa.generate_quality_report({'individual_results': [{'score': scores}]}, report_path)
        print(f"💾 Quality report saved to {report_path}")
        
        print("🎉 Quality assurance test complete!")
    else:
        print("Please provide original and cloned audio files for testing")

if __name__ == "__main__":
    main()