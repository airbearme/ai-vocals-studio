#!/usr/bin/env python3
"""
Test and Validate the Complete Precision Voice Cloning Pipeline
Comprehensive testing of all advanced components
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class PrecisionCloningTester:
    """
    Comprehensive testing system for precision voice cloning
    """
    
    def __init__(self):
        self.test_results = {}
        self.passed_tests = 0
        self.failed_tests = 0
        
    def run_all_tests(self):
        """Run comprehensive test suite"""
        print("🎤 Precision Voice Cloning - Comprehensive Test Suite")
        print("=" * 60)
        
        # Test 1: Component Initialization
        self.test_component_initialization()
        
        # Test 2: Audio Processing
        self.test_audio_processing()
        
        # Test 3: Feature Extraction
        self.test_feature_extraction()
        
        # Test 4: Data Augmentation
        self.test_data_augmentation()
        
        # Test 5: Quality Assurance
        self.test_quality_assurance()
        
        # Test 6: System Integration
        self.test_system_integration()
        
        # Test 7: Performance Validation
        self.test_performance_validation()
        
        # Print summary
        self.print_test_summary()
        
        return self.test_results
    
    def test_component_initialization(self):
        """Test initialization of all components"""
        print("\n📋 Test 1: Component Initialization")
        
        test_name = "Component Initialization"
        try:
            from advanced_audio_processor import AdvancedAudioProcessor
            from voice_feature_extractor import VoiceFeatureExtractor
            from data_augmentation import VoiceDataAugmentor
            from advanced_trainer import AdvancedVoiceTrainer
            from voice_quality_assurance import VoiceQualityAssurance
            from precision_voice_cloning_system import PrecisionVoiceCloningSystem
            
            # Test audio processor
            audio_proc = AdvancedAudioProcessor()
            assert audio_proc is not None, "Audio processor initialization failed"
            
            # Test feature extractor
            feature_ext = VoiceFeatureExtractor()
            assert feature_ext is not None, "Feature extractor initialization failed"
            
            # Test data augmentor
            data_aug = VoiceDataAugmentor()
            assert data_aug is not None, "Data augmentor initialization failed"
            
            # Test advanced trainer
            adv_trainer = AdvancedVoiceTrainer("dataset", "models")
            assert adv_trainer is not None, "Advanced trainer initialization failed"
            
            # Test quality assurance
            qa = VoiceQualityAssurance()
            assert qa is not None, "Quality assurance initialization failed"
            
            # Test precision system
            precision_sys = PrecisionVoiceCloningSystem()
            assert precision_sys is not None, "Precision system initialization failed"
            
            self.record_test_result(test_name, True, "All components initialized successfully")
            print("✅ PASSED: All components initialized successfully")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_audio_processing(self):
        """Test advanced audio processing"""
        print("\n📋 Test 2: Audio Processing")
        
        test_name = "Audio Processing"
        try:
            from advanced_audio_processor import AdvancedAudioProcessor
            
            processor = AdvancedAudioProcessor()
            
            # Create test audio
            test_audio = np.random.uniform(-0.5, 0.5, 22050 * 3)  # 3 seconds
            test_file = "test_audio.wav"
            import soundfile as sf
            sf.write(test_file, test_audio, 22050)
            
            # Test preprocessing
            processed_audio, quality = processor.preprocess_audio(test_file)
            
            assert processed_audio is not None, "Audio preprocessing failed"
            assert 'overall_quality' in quality, "Quality assessment failed"
            assert quality['overall_quality'] >= 0.0, "Quality score out of range"
            assert quality['overall_quality'] <= 1.0, "Quality score out of range"
            
            # Cleanup
            os.remove(test_file)
            
            self.record_test_result(test_name, True, f"Quality score: {quality['overall_quality']:.2f}")
            print(f"✅ PASSED: Audio processing works (Quality: {quality['overall_quality']:.2f})")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_feature_extraction(self):
        """Test voice feature extraction"""
        print("\n📋 Test 3: Feature Extraction")
        
        test_name = "Feature Extraction"
        try:
            from voice_feature_extractor import VoiceFeatureExtractor
            
            extractor = VoiceFeatureExtractor()
            
            # Create test audio
            test_audio = np.random.uniform(-0.5, 0.5, 22050 * 2)  # 2 seconds
            test_file = "test_audio_features.wav"
            import soundfile as sf
            sf.write(test_file, test_audio, 22050)
            
            # Test feature extraction
            features = extractor.extract_comprehensive_features(test_file)
            
            assert features is not None, "Feature extraction failed"
            assert 'pitch_features' in features, "Pitch features missing"
            assert 'spectral_features' in features, "Spectral features missing"
            assert 'prosodic_features' in features, "Prosodic features missing"
            
            # Verify feature structure
            assert 'mean_pitch_hz' in features['pitch_features'], "Mean pitch missing"
            assert 'spectral_centroid_mean' in features['spectral_features'], "Spectral centroid missing"
            
            # Cleanup
            os.remove(test_file)
            
            feature_count = sum(len(v) if isinstance(v, dict) else 1 for v in features.values())
            self.record_test_result(test_name, True, f"Extracted {feature_count} features")
            print(f"✅ PASSED: Feature extraction works ({feature_count} features extracted)")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_data_augmentation(self):
        """Test data augmentation"""
        print("\n📋 Test 4: Data Augmentation")
        
        test_name = "Data Augmentation"
        try:
            from data_augmentation import VoiceDataAugmentor
            
            augmentor = VoiceDataAugmentor()
            
            # Create test audio
            test_audio = np.random.uniform(-0.5, 0.5, 22050 * 2)  # 2 seconds
            test_file = "test_audio_aug.wav"
            import soundfile as sf
            sf.write(test_file, test_audio, 22050)
            
            # Test augmentation
            augmented_versions = augmentor.augment_audio(test_audio, 22050, num_augmentations=3)
            
            assert len(augmented_versions) == 4, "Wrong number of augmented versions"  # 3 + original
            assert all(len(v) == len(test_audio) for v in augmented_versions), "Augmentation changed audio length"
            
            # Cleanup
            os.remove(test_file)
            
            self.record_test_result(test_name, True, f"Generated {len(augmented_versions)} versions")
            print(f"✅ PASSED: Data augmentation works ({len(augmented_versions)} versions generated)")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_quality_assurance(self):
        """Test quality assurance system"""
        print("\n📋 Test 5: Quality Assurance")
        
        test_name = "Quality Assurance"
        try:
            from voice_quality_assurance import VoiceQualityAssurance
            
            qa = VoiceQualityAssurance()
            
            # Create test audio files
            test_audio1 = np.random.uniform(-0.5, 0.5, 22050 * 2)
            test_audio2 = np.random.uniform(-0.5, 0.5, 22050 * 2)
            
            test_file1 = "test_qa1.wav"
            test_file2 = "test_qa2.wav"
            
            import soundfile as sf
            sf.write(test_file1, test_audio1, 22050)
            sf.write(test_file2, test_audio2, 22050)
            
            # Test similarity calculation
            similarity_scores = qa.comprehensive_similarity_score(test_file1, test_file2)
            
            assert similarity_scores is not None, "Similarity calculation failed"
            assert 'overall_similarity' in similarity_scores, "Overall similarity missing"
            assert 'quality_rating' in similarity_scores, "Quality rating missing"
            assert 0.0 <= similarity_scores['overall_similarity'] <= 1.0, "Similarity out of range"
            
            # Cleanup
            os.remove(test_file1)
            os.remove(test_file2)
            
            self.record_test_result(test_name, True, f"Similarity: {similarity_scores['overall_similarity']:.2f}, Rating: {similarity_scores['quality_rating']}")
            print(f"✅ PASSED: Quality assurance works (Similarity: {similarity_scores['overall_similarity']:.2f}, Rating: {similarity_scores['quality_rating']})")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_system_integration(self):
        """Test complete system integration"""
        print("\n📋 Test 6: System Integration")
        
        test_name = "System Integration"
        try:
            from precision_voice_cloning_system import PrecisionVoiceCloningSystem
            
            system = PrecisionVoiceCloningSystem()
            
            # Test system status
            status = system.get_system_status()
            
            assert status is not None, "System status check failed"
            assert 'system_health' in status, "System health missing"
            assert 'components' in status, "Components status missing"
            assert 'available_models' in status, "Available models missing"
            
            # Verify structure
            assert isinstance(status['components'], dict), "Components should be dict"
            assert isinstance(status['available_models'], list), "Available models should be list"
            
            self.record_test_result(test_name, True, f"System health: {status['system_health']}")
            print(f"✅ PASSED: System integration works (Health: {status['system_health']})")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def test_performance_validation(self):
        """Test performance characteristics"""
        print("\n📋 Test 7: Performance Validation")
        
        test_name = "Performance Validation"
        try:
            from advanced_audio_processor import AdvancedAudioProcessor
            from voice_feature_extractor import VoiceFeatureExtractor
            
            # Create test audio
            test_audio = np.random.uniform(-0.5, 0.5, 22050 * 5)  # 5 seconds
            test_file = "test_perf.wav"
            import soundfile as sf
            sf.write(test_file, test_audio, 22050)
            
            # Test audio processing speed
            processor = AdvancedAudioProcessor()
            start_time = time.time()
            processed_audio, quality = processor.preprocess_audio(test_file)
            processing_time = time.time() - start_time
            
            # Test feature extraction speed
            extractor = VoiceFeatureExtractor()
            start_time = time.time()
            features = extractor.extract_comprehensive_features(test_file)
            extraction_time = time.time() - start_time
            
            # Cleanup
            os.remove(test_file)
            
            # Performance thresholds
            assert processing_time < 10.0, f"Audio processing too slow: {processing_time:.2f}s"
            assert extraction_time < 15.0, f"Feature extraction too slow: {extraction_time:.2f}s"
            
            self.record_test_result(test_name, True, f"Processing: {processing_time:.2f}s, Extraction: {extraction_time:.2f}s")
            print(f"✅ PASSED: Performance validation (Processing: {processing_time:.2f}s, Extraction: {extraction_time:.2f}s)")
            
        except Exception as e:
            self.record_test_result(test_name, False, str(e))
            print(f"❌ FAILED: {e}")
    
    def record_test_result(self, test_name, passed, message):
        """Record test result"""
        self.test_results[test_name] = {
            'passed': passed,
            'message': message,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        if passed:
            self.passed_tests += 1
        else:
            self.failed_tests += 1
    
    def print_test_summary(self):
        """Print comprehensive test summary"""
        print("\n" + "=" * 60)
        print("📊 TEST SUMMARY")
        print("=" * 60)
        
        total_tests = self.passed_tests + self.failed_tests
        success_rate = (self.passed_tests / total_tests * 100) if total_tests > 0 else 0
        
        print(f"Total Tests: {total_tests}")
        print(f"Passed: {self.passed_tests} ✅")
        print(f"Failed: {self.failed_tests} ❌")
        print(f"Success Rate: {success_rate:.1f}%")
        
        print("\n📋 Detailed Results:")
        for test_name, result in self.test_results.items():
            status = "✅ PASSED" if result['passed'] else "❌ FAILED"
            print(f"   {test_name}: {status}")
            print(f"      {result['message']}")
        
        print("\n" + "=" * 60)
        
        if success_rate >= 80:
            print("🎉 PRECISION VOICE CLONING SYSTEM: OPERATIONAL")
        elif success_rate >= 50:
            print("⚠️ PRECISION VOICE CLONING SYSTEM: PARTIALLY OPERATIONAL")
        else:
            print("❌ PRECISION VOICE CLONING SYSTEM: NEEDS ATTENTION")
        
        print("=" * 60)
        
        # Save test results
        self.save_test_results()
    
    def save_test_results(self):
        """Save test results to file"""
        results_file = "test_results_precision_cloning.json"
        
        with open(results_file, 'w') as f:
            json.dump({
                'test_results': self.test_results,
                'summary': {
                    'total_tests': self.passed_tests + self.failed_tests,
                    'passed_tests': self.passed_tests,
                    'failed_tests': self.failed_tests,
                    'success_rate': (self.passed_tests / (self.passed_tests + self.failed_tests) * 100) if (self.passed_tests + self.failed_tests) > 0 else 0
                },
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }, f, indent=2)
        
        print(f"💾 Test results saved to {results_file}")

def main():
    """Run the comprehensive test suite"""
    tester = PrecisionCloningTester()
    results = tester.run_all_tests()
    
    return results

if __name__ == "__main__":
    main()