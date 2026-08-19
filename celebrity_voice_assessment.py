#!/usr/bin/env python3
"""
Celebrity Voice Assessment System
Analyzes celebrity voice characteristics and predicts cloning feasibility
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import librosa
import soundfile as sf

class CelebrityVoiceAssessment:
    """
    Analyzes celebrity voice characteristics for cloning feasibility
    """
    
    def __init__(self, dataset_path="dataset"):
        self.dataset_path = Path(dataset_path)
        
        # Celebrity voice templates with known characteristics
        self.celebrity_templates = {
            "2Pac": {
                "voice_type": "deep_male_rapper",
                "distinctiveness": 0.95,  # Very distinct voice
                "pitch_range": [85, 220],
                "speaking_rate": 4.2,
                "emotional_range": ["aggressive", "storytelling", "confident"],
                "quality_requirements": "high",
                "min_training_duration": 15,  # minutes
                "recommended_engines": ["rvc", "sovits", "qwen3"],
                "best_engine": "rvc",  # RVC best for rap
                "cloneable": True,
                "expected_quality": 0.95
            },
            "Eminem": {
                "voice_type": "fast_male_rapper",
                "distinctiveness": 0.92,
                "pitch_range": [120, 280],
                "speaking_rate": 5.8,
                "emotional_range": ["aggressive", "emotional", "intense"],
                "quality_requirements": "high",
                "min_training_duration": 20,
                "recommended_engines": ["rvc", "sovits", "qwen3"],
                "best_engine": "rvc",
                "cloneable": True,
                "expected_quality": 0.93
            },
            "Snoop Dogg": {
                "voice_type": "smooth_male_rapper",
                "distinctiveness": 0.90,
                "pitch_range": [100, 180],
                "speaking_rate": 3.5,
                "emotional_range": ["relaxed", "storytelling", "laid_back"],
                "quality_requirements": "medium",
                "min_training_duration": 15,
                "recommended_engines": ["sovits", "qwen3", "rvc"],
                "best_engine": "sovits",
                "cloneable": True,
                "expected_quality": 0.90
            }
        }
    
    def assess_cloning_feasibility(self, celebrity_name: str, audio_samples: List[str]) -> Dict:
        """
        Assess celebrity voice cloning feasibility
        
        Returns:
            Assessment dict with feasibility score, quality prediction, and recommendations
        """
        print(f"🎤 Assessing cloning feasibility for {celebrity_name}...")
        
        # Check if celebrity is in template database
        celebrity_info = self.celebrity_templates.get(celebrity_name, {})
        
        if not celebrity_info:
            # Generic assessment for unknown celebrity
            return self._generic_assessment(celebrity_name, audio_samples)
        
        # Analyze available audio samples
        if not audio_samples:
            return {
                "celebrity": celebrity_name,
                "feasibility_score": 0.0,
                "cloneable": False,
                "reason": "No audio samples provided",
                "recommendation": "Provide audio samples for assessment"
            }
        
        # Analyze audio quality
        audio_quality = self._analyze_audio_quality(audio_samples)
        
        # Analyze voice distinctiveness
        voice_distinctiveness = self._analyze_voice_distinctiveness(audio_samples)
        
        # Calculate feasibility score
        feasibility_score = self._calculate_feasibility_score(
            celebrity_info, audio_quality, voice_distinctiveness
        )
        
        # Predict achievable quality
        predicted_quality = self._predict_achievable_quality(
            celebrity_info, audio_quality, len(audio_samples)
        )
        
        # Generate recommendations
        recommendations = self._generate_recommendations(
            celebrity_info, audio_quality, predicted_quality
        )
        
        return {
            "celebrity": celebrity_name,
            "feasibility_score": feasibility_score,
            "cloneable": feasibility_score > 0.7,
            "expected_quality": predicted_quality,
            "audio_quality": audio_quality,
            "voice_distinctiveness": voice_distinctiveness,
            "celebrity_info": celebrity_info,
            "recommendations": recommendations,
            "quality_assessment": self._detailed_quality_assessment(celebrity_info, audio_quality)
        }
    
    def _generic_assessment(self, celebrity_name: str, audio_samples: List[str]) -> Dict:
        """Assessment for celebrity not in template database"""
        if not audio_samples:
            return {
                "celebrity": celebrity_name,
                "feasibility_score": 0.0,
                "cloneable": False,
                "reason": "No audio samples provided",
                "recommendation": "Provide audio samples for assessment"
            }
        
        # Analyze the samples to determine voice characteristics
        voice_analysis = self._analyze_voice_characteristics(audio_samples)
        
        # Determine distinctiveness based on analysis
        distinctiveness = self._calculate_distinctiveness_score(voice_analysis)
        
        # Calculate feasibility
        feasibility_score = min(0.9, distinctiveness * 0.8 + 0.1)
        
        return {
            "celebrity": celebrity_name,
            "feasibility_score": feasibility_score,
            "cloneable": feasibility_score > 0.6,
            "expected_quality": feasibility_score * 0.9,
            "voice_analysis": voice_analysis,
            "distinctiveness": distinctiveness,
            "recommendation": "Voice appears to have " + 
                            ("high distinctiveness" if distinctiveness > 0.7 else 
                             "moderate distinctiveness" if distinctiveness > 0.4 else 
                             "low distinctiveness")
        }
    
    def _analyze_audio_quality(self, audio_files: List[str]) -> Dict:
        """Analyze quality of audio samples"""
        quality_scores = []
        
        for audio_file in audio_files:
            try:
                y, sr = librosa.load(audio_file, sr=22050)
                
                # Calculate various quality metrics
                snr = self._calculate_snr(y)
                dynamic_range = self._calculate_dynamic_range(y)
                zero_crossing_rate = self._calculate_zero_crossing_rate(y, sr)
                
                # Overall quality score (0-1)
                quality = (min(1.0, snr / 20) + min(1.0, dynamic_range / 60) + 
                          min(1.0, zero_crossing_rate / 5000)) / 3
                
                quality_scores.append(quality)
                
            except Exception as e:
                print(f"   ⚠️ Error analyzing {audio_file}: {e}")
                quality_scores.append(0.5)  # Default moderate quality
        
        avg_quality = np.mean(quality_scores) if quality_scores else 0.5
        
        return {
            "average_quality": avg_quality,
            "quality_category": "high" if avg_quality > 0.7 else "medium" if avg_quality > 0.4 else "low",
            "num_samples": len(audio_files),
            "individual_scores": quality_scores
        }
    
    def _analyze_voice_distinctiveness(self, audio_files: List[str]) -> Dict:
        """Analyze how distinct the voice is"""
        pitch_variances = []
        spectral_variances = []
        
        for audio_file in audio_files[:5]:  # Analyze first 5 files
            try:
                y, sr = librosa.load(audio_file, sr=22050, duration=30)
                
                # Pitch analysis
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                valid_pitches = pitches[pitches > 0]
                if len(valid_pitches) > 0:
                    pitch_variance = np.std(valid_pitches)
                    pitch_variances.append(pitch_variance)
                
                # Spectral analysis
                spectral_centroids = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
                spectral_variance = np.std(spectral_centroids)
                spectral_variances.append(spectral_variance)
                
            except Exception as e:
                print(f"   ⚠️ Error analyzing distinctiveness in {audio_file}: {e}")
        
        avg_pitch_variance = np.mean(pitch_variances) if pitch_variances else 0
        avg_spectral_variance = np.mean(spectral_variances) if spectral_variances else 0
        
        # Calculate distinctiveness score (higher variance = more distinct)
        distinctiveness = min(1.0, (avg_pitch_variance / 100) + (avg_spectral_variance / 2000))
        
        return {
            "distinctiveness_score": distinctiveness,
            "pitch_variance": avg_pitch_variance,
            "spectral_variance": avg_spectral_variance,
            "distinctiveness_category": "high" if distinctiveness > 0.7 else 
                                  "medium" if distinctiveness > 0.4 else "low"
        }
    
    def _analyze_voice_characteristics(self, audio_files: List[str]) -> Dict:
        """Analyze general voice characteristics"""
        characteristics = {
            "pitch_mean": [],
            "pitch_range": [],
            "energy_mean": [],
            "tempo": []
        }
        
        for audio_file in audio_files[:3]:
            try:
                y, sr = librosa.load(audio_file, sr=22050, duration=30)
                
                # Pitch characteristics
                pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
                valid_pitches = pitches[pitches > 0]
                if len(valid_pitches) > 0:
                    characteristics["pitch_mean"].append(np.mean(valid_pitches))
                    characteristics["pitch_range"].append(np.max(valid_pitches) - np.min(valid_pitches))
                
                # Energy characteristics
                energy = librosa.feature.rms(y=y)[0]
                characteristics["energy_mean"].append(np.mean(energy))
                
                # Tempo estimation
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                characteristics["tempo"].append(tempo)
                
            except Exception as e:
                print(f"   ⚠️ Error analyzing characteristics in {audio_file}: {e}")
        
        # Calculate averages
        avg_characteristics = {}
        for key, values in characteristics.items():
            if values:
                avg_characteristics[key] = np.mean(values)
        
        return avg_characteristics
    
    def _calculate_distinctiveness_score(self, voice_analysis: Dict) -> float:
        """Calculate distinctiveness score from voice analysis"""
        score = 0.5  # Base score
        
        # Pitch range contribution
        if "pitch_range" in voice_analysis:
            pitch_range = voice_analysis["pitch_range"]
            score += min(0.3, pitch_range / 200)  # Wider range = more distinct
        
        # Energy variation contribution
        if "energy_mean" in voice_analysis:
            # Check energy variation across samples
            score += 0.1  # Placeholder for energy variation analysis
        
        return min(1.0, score)
    
    def _calculate_feasibility_score(self, celebrity_info: Dict, 
                                     audio_quality: Dict, 
                                     voice_distinctiveness: Dict) -> float:
        """Calculate overall feasibility score"""
        base_score = celebrity_info.get("distinctiveness", 0.7)
        
        # Adjust based on audio quality
        quality_multiplier = audio_quality.get("average_quality", 0.5)
        
        # Adjust based on voice distinctiveness
        distinctiveness_multiplier = voice_distinctiveness.get("distinctiveness_score", 0.5)
        
        # Calculate final score
        feasibility = (base_score * 0.5 + 
                       quality_multiplier * 0.3 + 
                       distinctiveness_multiplier * 0.2)
        
        return min(1.0, feasibility)
    
    def _predict_achievable_quality(self, celebrity_info: Dict, 
                                   audio_quality: Dict, 
                                   num_samples: int) -> float:
        """Predict achievable quality based on data"""
        base_quality = celebrity_info.get("expected_quality", 0.85)
        
        # Adjust based on audio quality
        quality_adjustment = (audio_quality.get("average_quality", 0.5) - 0.5) * 0.3
        
        # Adjust based on number of samples
        samples_adjustment = min(0.1, (num_samples / 10) * 0.1)
        
        predicted_quality = base_quality + quality_adjustment + samples_adjustment
        
        return min(0.98, max(0.6, predicted_quality))
    
    def _generate_recommendations(self, celebrity_info: Dict, 
                               audio_quality: Dict, 
                               predicted_quality: float) -> List[str]:
        """Generate recommendations for training"""
        recommendations = []
        
        # Quality-based recommendations
        if audio_quality.get("average_quality", 0) < 0.5:
            recommendations.append("Consider audio enhancement before training")
        
        # Engine recommendations
        recommended_engines = celebrity_info.get("recommended_engines", ["qwen3", "sovits"])
        best_engine = celebrity_info.get("best_engine", "qwen3")
        recommendations.append(f"Best engine: {best_engine}")
        recommendations.append(f"Alternative engines: {', '.join(recommended_engines)}")
        
        # Training duration recommendations
        min_duration = celebrity_info.get("min_training_duration", 15)
        recommendations.append(f"Minimum training data: {min_duration} minutes recommended")
        
        # Quality expectations
        if predicted_quality > 0.9:
            recommendations.append("High-quality cloning achievable with good data")
        elif predicted_quality > 0.8:
            recommendations.append("Good quality cloning expected")
        else:
            recommendations.append("Moderate quality expected - consider more training data")
        
        return recommendations
    
    def _detailed_quality_assessment(self, celebrity_info: Dict, 
                                     audio_quality: Dict) -> Dict:
        """Generate detailed quality assessment"""
        return {
            "celebrity_cloneable": celebrity_info.get("cloneable", True),
            "expected_similarity": celebrity_info.get("expected_quality", 0.85),
            "current_data_quality": audio_quality.get("quality_category", "unknown"),
            "quality_gap": celebrity_info.get("expected_quality", 0.85) - audio_quality.get("average_quality", 0.5),
            "improvement_needed": max(0, celebrity_info.get("expected_quality", 0.85) - audio_quality.get("average_quality", 0.5))
        }
    
    def _calculate_snr(self, audio: np.ndarray) -> float:
        """Calculate Signal-to-Noise Ratio"""
        # Simple SNR estimation using RMS
        signal_power = np.mean(audio ** 2)
        noise_power = np.var(audio - np.mean(audio))
        if noise_power > 0:
            snr = 10 * np.log10(signal_power / noise_power)
            return min(30, max(0, snr))
        return 15  # Default moderate SNR
    
    def _calculate_dynamic_range(self, audio: np.ndarray) -> float:
        """Calculate dynamic range in dB"""
        max_amplitude = np.max(np.abs(audio))
        min_amplitude = np.min(np.abs(audio))
        if min_amplitude > 0:
            dynamic_range = 20 * np.log10(max_amplitude / min_amplitude)
            return min(80, max(20, dynamic_range))
        return 40  # Default moderate dynamic range
    
    def _calculate_zero_crossing_rate(self, audio: np.ndarray, sr: int) -> float:
        """Calculate zero crossing rate"""
        zero_crossings = np.where(np.diff(np.sign(audio)))[0]
        zcr = len(zero_crossings) / len(audio) * sr
        return zcr
    
    def recommend_engine(self, voice_type: str, data_characteristics: Dict) -> str:
        """Recommend best engine for specific voice type"""
        voice_type_lower = voice_type.lower()
        
        # Rap voices work best with RVC
        if "rap" in voice_type_lower or "rapper" in voice_type_lower:
            return "rvc"
        
        # Singing voices work well with SO-VITS
        if "sing" in voice_type_lower or "vocalist" in voice_type_lower:
            return "sovits"
        
        # General speech works well with Qwen3-TTS for speed
        if "speech" in voice_type_lower or "voiceover" in voice_type_lower:
            return "qwen3"
        
        # Default to Qwen3-TTS for versatility
        return "qwen3"
    
    def estimate_training_requirements(self, target_quality: float) -> Dict:
        """Estimate training requirements based on target quality"""
        if target_quality > 0.9:
            return {
                "min_duration_minutes": 30,
                "recommended_duration": 45,
                "engine": "sovits",
                "epochs": 10000,
                "expected_time_hours": 4
            }
        elif target_quality > 0.8:
            return {
                "min_duration_minutes": 15,
                "recommended_duration": 25,
                "engine": "qwen3",
                "epochs": 5000,
                "expected_time_hours": 2
            }
        else:
            return {
                "min_duration_minutes": 5,
                "recommended_duration": 10,
                "engine": "qwen3",
                "epochs": 1000,
                "expected_time_hours": 0.5
            }
    
    def generate_quality_report(self, celebrity_name: str) -> Dict:
        """Generate comprehensive quality report for celebrity"""
        celebrity_info = self.celebrity_templates.get(celebrity_name, {})
        
        if not celebrity_info:
            return {
                "celebrity": celebrity_name,
                "status": "unknown",
                "message": "Celebrity not in template database",
                "recommendation": "Provide audio samples for assessment"
            }
        
        return {
            "celebrity": celebrity_name,
            "status": "known",
            "cloneable": celebrity_info.get("cloneable", False),
            "expected_quality": celebrity_info.get("expected_quality", 0.0),
            "voice_type": celebrity_info.get("voice_type", "unknown"),
            "distinctiveness": celebrity_info.get("distinctiveness", 0.0),
            "best_engine": celebrity_info.get("best_engine", "qwen3"),
            "min_training_data": celebrity_info.get("min_training_duration", 10),
            "recommended_engines": celebrity_info.get("recommended_engines", ["qwen3"]),
            "requirements": self.estimate_training_requirements(
                celebrity_info.get("expected_quality", 0.8)
            )
        }

def main():
    """Test the celebrity voice assessment system"""
    assessor = CelebrityVoiceAssessment()
    
    # Test with 2Pac
    dataset_path = "/home/coden607/Desktop/Projects/ai-vocals-studio/dataset"
    
    # Find 2Pac audio files
    pac_files = []
    for ext in ['*.wav', '*.mp3']:
        pac_files.extend(Path(dataset_path).glob(f"*2Pac*{ext}"))
        pac_files.extend(Path(dataset_path).glob(f"*2pac*{ext}"))
    
    if pac_files:
        print(f"🎤 Found {len(pac_files)} 2Pac audio files")
        assessment = assessor.assess_cloning_feasibility("2Pac", [str(f) for f in pac_files[:5]])
        
        print(f"\n📊 Assessment Results:")
        print(f"   Feasibility Score: {assessment['feasibility_score']:.2f}")
        print(f"   Cloneable: {assessment['cloneable']}")
        print(f"   Expected Quality: {assessment['expected_quality']:.2f}")
        print(f"   Audio Quality: {assessment['audio_quality']['quality_category']}")
        print(f"   Voice Distinctiveness: {assessment['voice_distinctiveness']['distinctiveness_category']}")
        
        print(f"\n💡 Recommendations:")
        for rec in assessment['recommendations']:
            print(f"   • {rec}")
    else:
        print("❌ No 2Pac audio files found")
    
    # Generate quality report
    report = assessor.generate_quality_report("2Pac")
    print(f"\n📋 Quality Report:")
    print(f"   Celebrity: {report['celebrity']}")
    print(f"   Cloneable: {report['cloneable']}")
    print(f"   Expected Quality: {report['expected_quality']:.2f}")
    print(f"   Best Engine: {report['best_engine']}")
    print(f"   Min Training Data: {report['min_training_data']} minutes")

if __name__ == "__main__":
    main()