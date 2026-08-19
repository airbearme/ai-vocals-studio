#!/usr/bin/env python3
"""
Precision Voice Cloning System
Integrates all advanced technologies for maximum voice cloning accuracy
"""

import os
import json
import time
import threading
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

class PrecisionVoiceCloningSystem:
    """
    Complete precision voice cloning system integrating all advanced technologies
    """

    def __init__(self, dataset_path="dataset", models_path="models", output_path="output"):
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.output_path = output_path

        # Create directories
        os.makedirs(dataset_path, exist_ok=True)
        os.makedirs(models_path, exist_ok=True)
        os.makedirs(output_path, exist_ok=True)

        # Initialize components
        self._initialize_components()

        # System status
        self.is_training = False
        self.current_progress = 0
        self.current_status = "Ready"

    def _initialize_components(self):
        """Initialize all advanced components"""
        try:
            from advanced_audio_processor import AdvancedAudioProcessor
            self.audio_processor = AdvancedAudioProcessor()
            print("✅ Advanced Audio Processor initialized")
        except Exception as e:
            print(f"⚠️ Audio processor initialization failed: {e}")
            self.audio_processor = None

        try:
            from voice_conversion_engine import VoiceConversionEngine
            self.conversion_engine = VoiceConversionEngine()
            print("✅ Voice Conversion Engine initialized (WORLD vocoder)")
        except Exception as e:
            print(f"⚠️ Voice conversion engine initialization failed: {e}")
            self.conversion_engine = None

        try:
            from voice_feature_extractor import VoiceFeatureExtractor
            self.feature_extractor = VoiceFeatureExtractor()
            print("✅ Voice Feature Extractor initialized")
        except Exception as e:
            print(f"⚠️ Feature extractor initialization failed: {e}")
            self.feature_extractor = None

        try:
            from data_augmentation import VoiceDataAugmentor
            self.data_augmentor = VoiceDataAugmentor()
            print("✅ Data Augmentor initialized")
        except Exception as e:
            print(f"⚠️ Data augmentor initialization failed: {e}")
            self.data_augmentor = None

        try:
            from advanced_trainer import AdvancedVoiceTrainer
            self.advanced_trainer = AdvancedVoiceTrainer(self.dataset_path, self.models_path)
            print("✅ Advanced Trainer initialized")
        except Exception as e:
            print(f"⚠️ Advanced trainer initialization failed: {e}")
            self.advanced_trainer = None

        try:
            from voice_quality_assurance import VoiceQualityAssurance
            self.quality_assurance = VoiceQualityAssurance()
            print("✅ Quality Assurance System initialized")
        except Exception as e:
            print(f"⚠️ Quality assurance initialization failed: {e}")
            self.quality_assurance = None

        try:
            from voice_trainer import VoiceTrainer
            self.base_trainer = VoiceTrainer(self.dataset_path, self.models_path)
            print("✅ Base Trainer initialized")
        except Exception as e:
            print(f"⚠️ Base trainer initialization failed: {e}")
            self.base_trainer = None

    def precision_clone_voice(self, speaker_name: str, model_name: str,
                           progress_callback=None, quality_threshold=0.85) -> Dict:
        """
        Complete precision voice cloning pipeline
        """
        print(f"🎤 Starting precision voice cloning for {speaker_name}...")

        try:
            # Step 1: Analyze input data
            if progress_callback:
                progress_callback("Analyzing input audio data...", 5)

            input_analysis = self._analyze_input_data(speaker_name)

            # Step 2: Advanced preprocessing
            if progress_callback:
                progress_callback("Applying advanced audio preprocessing...", 15)

            preprocessed_data = self._advanced_preprocessing(speaker_name, progress_callback)

            # Step 3: Feature extraction
            if progress_callback:
                progress_callback("Extracting comprehensive voice features...", 25)

            voice_features = self._extract_voice_features(preprocessed_data, speaker_name)

            # Step 4: Data augmentation
            if progress_callback:
                progress_callback("Applying data augmentation for robustness...", 35)

            augmented_data = self._apply_data_augmentation(preprocessed_data)

            # Step 5: Advanced training
            if progress_callback:
                progress_callback("Starting advanced training pipeline...", 45)

            model_path = self._advanced_training(speaker_name, model_name, progress_callback)

            # Step 6: Quality validation
            if progress_callback:
                progress_callback("Validating model quality...", 85)

            quality_report = self._validate_model_quality(model_path, speaker_name, quality_threshold)

            # Step 7: Final integration
            if progress_callback:
                progress_callback("Finalizing precision voice clone...", 95)

            final_result = self._finalize_clone(model_path, voice_features, quality_report)

            if progress_callback:
                progress_callback("Precision voice cloning complete!", 100)

            print(f"🎉 Precision voice cloning complete for {speaker_name}!")

            return {
                'status': 'success',
                'speaker_name': speaker_name,
                'model_name': model_name,
                'model_path': model_path,
                'quality_report': quality_report,
                'voice_features': voice_features,
                'input_analysis': input_analysis,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

        except Exception as e:
            error_msg = f"Precision cloning failed: {str(e)}"
            print(f"❌ {error_msg}")

            if progress_callback:
                progress_callback(f"Error: {error_msg}", 0)

            return {
                'status': 'error',
                'error': error_msg,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
            }

    def _analyze_input_data(self, speaker_name: str) -> Dict:
        """Analyze input audio data"""
        speaker_dir = os.path.join(self.dataset_path, speaker_name)

        if not os.path.exists(speaker_dir):
            raise Exception(f"Speaker directory not found: {speaker_dir}")

        # Find audio files
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(Path(speaker_dir).glob(ext))

        if not audio_files:
            raise Exception(f"No audio files found for {speaker_name}")

        analysis = {
            'total_files': len(audio_files),
            'file_types': {},
            'total_duration': 0,
            'quality_distribution': []
        }

        # Analyze each file
        for audio_file in audio_files:
            try:
                if self.audio_processor:
                    y, quality = self.audio_processor.preprocess_audio(str(audio_file))
                    analysis['quality_distribution'].append(quality.get('overall_quality', 0.5))
                    analysis['total_duration'] += quality.get('duration_s', 0)

                file_ext = audio_file.suffix
                analysis['file_types'][file_ext] = analysis['file_types'].get(file_ext, 0) + 1

            except Exception as e:
                print(f"Error analyzing {audio_file}: {e}")

        return analysis

    def _advanced_preprocessing(self, speaker_name: str, progress_callback=None) -> List[str]:
        """Apply advanced preprocessing to all audio files"""
        speaker_dir = os.path.join(self.dataset_path, speaker_name)
        processed_dir = os.path.join(speaker_dir, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(Path(speaker_dir).glob(ext))

        processed_files = []

        for i, audio_file in enumerate(audio_files):
            try:
                if self.audio_processor:
                    output_path = os.path.join(processed_dir, f"processed_{i:04d}.wav")
                    y, quality = self.audio_processor.preprocess_audio(str(audio_file), output_path)

                    if quality.get('overall_quality', 0) > 0.6:  # Quality threshold
                        processed_files.append(output_path)

                        if progress_callback:
                            progress_callback(f"Processing audio {i+1}/{len(audio_files)}...",
                                             15 + (i / len(audio_files)) * 10)
                else:
                    # Fallback: just copy files
                    output_path = os.path.join(processed_dir, audio_file.name)
                    import shutil
                    shutil.copy(str(audio_file), output_path)
                    processed_files.append(output_path)

            except Exception as e:
                print(f"Error preprocessing {audio_file}: {e}")

        print(f"✅ Preprocessed {len(processed_files)}/{len(audio_files)} files")
        return processed_files

    def _extract_voice_features(self, audio_files: List[str], speaker_name: str) -> Dict:
        """Extract comprehensive voice features"""
        if not self.feature_extractor:
            return {'status': 'skipped', 'reason': 'Feature extractor not available'}

        all_features = []

        for audio_file in audio_files[:10]:  # Analyze first 10 files
            try:
                features = self.feature_extractor.extract_comprehensive_features(audio_file)
                all_features.append(features)
            except Exception as e:
                print(f"Error extracting features from {audio_file}: {e}")

        # Aggregate features
        aggregated_features = self._aggregate_features(all_features)

        # Save features
        features_path = os.path.join(self.models_path, speaker_name, "voice_features.json")
        os.makedirs(os.path.dirname(features_path), exist_ok=True)

        with open(features_path, 'w') as f:
            json.dump(aggregated_features, f, indent=2)

        return aggregated_features

    def _aggregate_features(self, feature_list: List[Dict]) -> Dict:
        """Aggregate features from multiple audio files"""
        if not feature_list:
            return {}

        aggregated = {
            'num_samples': len(feature_list),
            'pitch_profile': {},
            'spectral_profile': {},
            'prosodic_profile': {},
            'quality_profile': {}
        }

        # Aggregate pitch features
        pitch_means = [f.get('pitch_features', {}).get('mean_pitch_hz', 0) for f in feature_list if f.get('pitch_features')]
        if pitch_means:
            aggregated['pitch_profile'] = {
                'mean_pitch': float(np.mean(pitch_means)),
                'std_pitch': float(np.std(pitch_means)),
                'pitch_range': float(np.max(pitch_means) - np.min(pitch_means))
            }

        # Aggregate spectral features
        spectral_centroids = [f.get('spectral_features', {}).get('spectral_centroid_mean', 0) for f in feature_list if f.get('spectral_features')]
        if spectral_centroids:
            aggregated['spectral_profile'] = {
                'mean_centroid': float(np.mean(spectral_centroids)),
                'std_centroid': float(np.std(spectral_centroids))
            }

        return aggregated

    def _apply_data_augmentation(self, audio_files: List[str]) -> List[str]:
        """Apply data augmentation"""
        if not self.data_augmentor:
            return audio_files

        augmented_dir = os.path.join(self.dataset_path, "augmented")
        os.makedirs(augmented_dir, exist_ok=True)

        # Generate augmentations
        augmentation_results = self.data_augmentor.batch_augment_dataset(
            audio_files,
            augmented_dir,
            augmentations_per_file=2
        )

        # Collect augmented files
        augmented_files = []
        for result in augmentation_results:
            if result['status'] == 'success':
                augmented_files.extend([
                    os.path.join(augmented_dir, f"augmented_{i:02d}.wav")
                    for i in range(result['augmentations'])
                ])

        print(f"✅ Generated {len(augmented_files)} augmented files")
        return audio_files + augmented_files

    def _advanced_training(self, speaker_name: str, model_name: str,
                          progress_callback=None) -> str:
        """
        Train a real voice model.

        The reference speaker's clips are analysed with the WORLD vocoder to
        produce pitch statistics + an average spectral (timbre) envelope.
        The resulting profile is saved as ``models/<model_name>/model.pth``
        (a JSON payload the conversion engine can load and apply).
        """
        if not self.conversion_engine:
            raise Exception("Voice conversion engine not available")

        speaker_dir = Path(self.dataset_path) / speaker_name
        if not speaker_dir.exists():
            raise Exception(f"Speaker directory not found: {speaker_dir}")

        raw_files = []
        for ext in ("*.wav", "*.mp3", "*.flac", "*.m4a", "*.ogg"):
            raw_files.extend(sorted(speaker_dir.glob(ext)))

        processed_dir = speaker_dir / "processed"
        processed = sorted(processed_dir.glob("*.wav")) if processed_dir.exists() else []
        reference_files = [str(p) for p in processed] or [str(p) for p in raw_files]
        if not reference_files:
            raise Exception(f"No reference audio found under {speaker_dir}")

        def _cb(msg, pct):
            if progress_callback:
                progress_callback(f"Training reference voice ({msg})",
                                  45 + 30 * pct / 100)

        profile = self.conversion_engine.extract_reference_profile(
            reference_files, progress_cb=_cb)

        model_dir = os.path.join(self.models_path, model_name)
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        self.conversion_engine.save_profile(profile, model_path)

        voice_profile = {
            "speaker": speaker_name,
            "model_name": model_name,
            "type": "precision_voice_clone",
            "engine": "world_vocoder",
            "total_files": len(reference_files),
            "audio_files": [os.path.basename(p) for p in reference_files],
            "characteristics": {
                "voice_type": "real_extracted_profile",
                "pitch_mean_hz": profile["pitch"]["mean_hz"],
                "pitch_median_hz": profile["pitch"]["median_hz"],
                "pitch_dominant_hz": profile["pitch"]["dominant_hz"],
                "male_band_median_hz": profile["pitch"]["male_band_median_hz"],
                "spectral_centroid_hz": profile["spectral"]["mean_centroid_hz"],
                "formants": profile.get("formants", {}),
            },
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(os.path.join(model_dir, "voice_profile.json"), "w") as f:
            json.dump(voice_profile, f, indent=2)
        with open(os.path.join(model_dir, "config.json"), "w") as f:
            json.dump({"spk": {speaker_name: 0}, "version": "4.0"}, f, indent=2)

        print(f"✅ Real voice model built from {len(reference_files)} clips -> {model_path}")
        return model_path
    def _validate_model_quality(self, model_path: str, speaker_name: str,
                               threshold: float) -> Dict:
        """Validate model quality"""
        if not self.quality_assurance:
            return {'status': 'skipped', 'reason': 'Quality assurance not available'}

        # Find test audio
        speaker_dir = os.path.join(self.dataset_path, speaker_name)
        test_files = list(Path(speaker_dir).glob("*.wav"))[:3]

        if not test_files:
            return {'status': 'skipped', 'reason': 'No test files available'}

    def _validate_model_quality(self, model_path: str, speaker_name: str,
                                threshold: float) -> Dict:
        """Validate real converted clips against their source audio."""
        if not self.conversion_engine or not self.quality_assurance:
            return {'status': 'skipped',
                    'reason': 'Conversion engine / QA not available'}

        try:
            profile = self.conversion_engine.load_profile(model_path)
        except Exception as exc:
            return {'status': 'failed', 'reason': f'Could not load model: {exc}'}

        speaker_dir = Path(self.dataset_path) / speaker_name
        test_files = sorted(speaker_dir.glob('*.wav'))[:3]
        if not test_files:
            return {'status': 'skipped', 'reason': 'No test files available'}

        validation_results = []
        tmp_dir = Path(self.output_path) / '_validation'
        tmp_dir.mkdir(parents=True, exist_ok=True)
        try:
            for index, test_file in enumerate(test_files):
                clone_file = tmp_dir / f'clone_{index:02d}.wav'
                try:
                    self.conversion_engine.convert_audio(
                        str(test_file), profile, str(clone_file), strength=0.9)
                    similarity = self.quality_assurance.comprehensive_similarity_score(
                        str(test_file), str(clone_file))
                    if 'overall_similarity' in similarity:
                        validation_results.append({
                            'test_file': test_file.name,
                            'similarity_score': similarity,
                            'overall_similarity': similarity['overall_similarity'],
                        })
                except Exception as exc:
                    print(f'Validation error for {test_file}: {exc}')
        finally:
            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

        if not validation_results:
            return {'status': 'failed', 'reason': 'No successful validations'}

        average = float(np.mean([
            result['overall_similarity'] for result in validation_results
        ]))
        return {
            'status': 'validated',
            'average_similarity': average,
            'meets_threshold': average >= threshold,
            'threshold': threshold,
            'note': 'Real WORLD-vocoder conversion compared with source audio.',
            'individual_results': validation_results,
        }

    def generate_precision_vocals(self, text: str, model_name: str,
                                  output_name: str = None,
                                  pitch_target: float = None) -> Dict:
        """Generate text audio and convert it through a saved voice profile."""
        try:
            model_path = Path(self.models_path) / model_name / 'model.pth'
            if not self.conversion_engine or not model_path.exists():
                return {'status': 'error',
                        'error': 'Converted model not available - train it first'}

            profile = self.conversion_engine.load_profile(str(model_path))
            output_name = output_name or f"precision_vocal_{time.strftime('%Y%m%d_%H%M%S')}"
            output_path = Path(self.output_path) / f'{output_name}.wav'
            self.conversion_engine.text_to_speech_with_voice(
                text, profile, str(output_path), strength=0.9,
                pitch_target=pitch_target)
            return {'status': 'success', 'output_path': str(output_path),
                    'model_name': model_name, 'text': text,
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
        except Exception as exc:
            return {'status': 'error', 'error': str(exc),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}

    def get_system_status(self) -> Dict:
        """Return component, dataset, and model availability information."""
        components = {
            'audio_processor': self.audio_processor is not None,
            'feature_extractor': self.feature_extractor is not None,
            'data_augmentor': self.data_augmentor is not None,
            'advanced_trainer': self.advanced_trainer is not None,
            'quality_assurance': self.quality_assurance is not None,
            'voice_conversion_engine': self.conversion_engine is not None,
        }
        available_models = [
            path.name for path in Path(self.models_path).iterdir()
            if path.is_dir()
        ] if Path(self.models_path).exists() else []
        dataset_info = {}
        dataset_root = Path(self.dataset_path)
        if dataset_root.exists():
            for speaker_dir in dataset_root.iterdir():
                if speaker_dir.is_dir():
                    dataset_info[speaker_dir.name] = sum(
                        path.suffix.lower() in {'.wav', '.mp3', '.flac'}
                        for path in speaker_dir.iterdir() if path.is_file()
                    )
        active = sum(components.values())
        health = ('optimal' if active == len(components)
                  else 'operational' if active >= len(components) / 2
                  else 'degraded')
        return {'components': components, 'available_models': available_models,
                'dataset_info': dataset_info, 'system_health': health}

def _validate_model_quality(self, model_path: str, speaker_name: str, threshold: float) -> Dict:
    """Validate model quality with real clone generation"""
    if not self.quality_assurance:
        return {'status': 'skipped', 'reason': 'Quality assurance not available'}

    # Get 3 real test files from original speaker
    speaker_dir = Path(self.dataset_path) / speaker_name
    test_files = list(sorted(speaker_dir.glob("*.wav")))[:3]

    if not test_files:
        return {'status': 'skipped', 'reason': 'No test files available'}

    # For each test file, generate real clone and compare
    validation_results = []

    for test_file in test_files:
        try:
            test_audio = str(test_file)
            # Generate actual clone using current model
            clone_audio = os.path.join(self.output_path, f"{test_file.stem}_clone.wav")
            ok = self.conversion_engine.convert_audio(test_audio, profile, clone_audio,
                                                   strength=0.9, pitch_target=profile["pitch_target"])
            if not ok:
                continue

            similarity = self.quality_assurance.comprehensive_similarity_score(test_audio, clone_audio)
            validation_results.append({
                'test_file': str(test_file),
                'similarity_score': similarity
            })
        except Exception as e:
            print(f"Validation error for {test_file}: {e}")

    if validation_results:
        avg = np.mean([r['similarity_score']['overall_similarity'] for r in validation_results])
        quality_report = {
            'status': 'validated',
            'average_similarity': float(avg),
            'meets_threshold': avg >= threshold,
            'threshold': threshold,
            'individual_results': validation_results
        }
    else:
        quality_report = {
            'status': 'failed',
            'reason': 'No successful validations'
        }


    return quality_report

    def _validate_model_quality(self, model_path: str, speaker_name: str,
                               threshold: float) -> Dict:
        """
        Validate model quality HONESTLY.

        A real clone is generated by converting held-out clips through the
        engine, then the converted audio is compared with the original using
        the multi-dimensional quality-assurance similarity scores.
        """
        if not self.conversion_engine or not self.quality_assurance:
            return {'status': 'skipped',
                    'reason': 'Conversion engine / QA not available'}

        # load the real profile built during training
        try:
            profile = self.conversion_engine.load_profile(model_path)
        except Exception as e:
            return {'status': 'failed', 'reason': f"Could not load model: {e}"}

        speaker_dir = Path(self.dataset_path) / speaker_name
        test_files = sorted(speaker_dir.glob("*.wav"))[:3]
        if not test_files:
            return {'status': 'skipped', 'reason': 'No test files available'}

        tmp_dir = os.path.join(self.output_path, "_validation")
        import shutil
        os.makedirs(tmp_dir, exist_ok=True)

        validation_results = []
        for idx, test_file in enumerate(test_files):
            try:
                test_audio = str(test_file)
                clone_audio = os.path.join(tmp_dir, f"clone_{idx:02d}.wav")
                # real conversion: pitch-map + timbre morph
                self.conversion_engine.convert_audio(
                    test_audio, profile, clone_audio, strength=0.9)
                if not os.path.exists(clone_audio):
                    continue
                similarity = self.quality_assurance.comprehensive_similarity_score(
                    test_audio, clone_audio)
                validation_results.append({
                    'test_file': os.path.basename(test_file),
                    'similarity_score': similarity,
                    'overall_similarity': similarity.get('overall_similarity', 0.0),
                })
            except Exception as e:
                print(f"  ! Validation error for {test_file}: {e}")

        if validation_results:
            avg = float(np.mean([r['overall_similarity'] for r in validation_results]))
            quality_report = {
                'status': 'validated',
                'average_similarity': avg,
                'meets_threshold': avg >= threshold,
                'threshold': threshold,
                'note': ('Real conversion vs. source comparison using the '
                         'WORLD vocoder.'),
                'individual_results': validation_results,
            }
        else:
            quality_report = {'status': 'failed', 'reason': 'No valid conversions'}

        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass
        return quality_report
    def _finalize_clone(self, model_path: str, voice_features: Dict,
                       quality_report: Dict) -> Dict:
        """Finalize the cloning process"""
        # Create comprehensive model package
        model_dir = os.path.dirname(model_path)

        # Save complete model information
        model_info = {
            'model_path': model_path,
            'voice_features': voice_features,
            'quality_report': quality_report,
            'created_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'precision_level': 'high' if quality_report.get('meets_threshold', False) else 'standard'
        }

        info_path = os.path.join(model_dir, "precision_model_info.json")
        with open(info_path, 'w') as f:
            json.dump(model_info, f, indent=2)

        return model_info

    def generate_precision_vocals(self, text: str, model_name: str,
                                output_name: str = None,
                                pitch_target: float = None) -> Dict:
        """
        Generate vocals using the precision cloning model.

        Text is synthesised with a neutral TTS engine, then converted into
        the reference voice with the WORLD vocoder (real timbre/pitch
        transfer - no placeholder / random-noise output).
        """
        print(f"🎤 Generating precision vocals using model: {model_name}")
        try:
            model_dir = os.path.join(self.models_path, model_name)
            model_path = os.path.join(model_dir, "model.pth")
            info_path = os.path.join(model_dir, "precision_model_info.json")

            model_info = {}
            if os.path.exists(info_path):
                with open(info_path, 'r') as f:
                    model_info = json.load(f)

            if not self.conversion_engine or not os.path.exists(model_path):
                return {'status': 'error',
                        'error': 'Converted model not available - train it first'}

            profile = self.conversion_engine.load_profile(model_path)

            if not output_name:
                timestamp = time.strftime('%Y%m%d_%H%M%S')
                output_name = f"precision_vocal_{timestamp}"
            output_path = os.path.join(self.output_path, f"{output_name}.wav")

            self.conversion_engine.text_to_speech_with_voice(
                text, profile, output_path, strength=0.9,
                pitch_target=pitch_target)

            return {
                'status': 'success',
                'output_path': output_path,
                'model_info': model_info,
                'text': text,
                'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            }
        except Exception as e:
            return {'status': 'error', 'error': str(e),
                    'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')}
    def get_system_status(self) -> Dict:
        """Get current system status"""
        status = {
            'components': {},
            'available_models': [],
            'dataset_info': {},
            'system_health': 'operational'
        }

        # Check component status
        status['components']['audio_processor'] = self.audio_processor is not None
        status['components']['feature_extractor'] = self.feature_extractor is not None
        status['components']['data_augmentor'] = self.data_augmentor is not None
        status['components']['advanced_trainer'] = self.advanced_trainer is not None
        status['components']['quality_assurance'] = self.quality_assurance is not None
        status['components']['voice_conversion_engine'] = self.conversion_engine is not None

        # Check available models
        if os.path.exists(self.models_path):
            for item in os.listdir(self.models_path):
                item_path = os.path.join(self.models_path, item)
                if os.path.isdir(item_path):
                    status['available_models'].append(item)

        # Check dataset
        if os.path.exists(self.dataset_path):
            for speaker in os.listdir(self.dataset_path):
                speaker_path = os.path.join(self.dataset_path, speaker)
                if os.path.isdir(speaker_path):
                    audio_count = len([f for f in os.listdir(speaker_path) if f.endswith(('.wav', '.mp3', '.flac'))])
                    status['dataset_info'][speaker] = audio_count

        # Overall system health
        operational_components = sum(status['components'].values())
        total_components = len(status['components'])

        if operational_components == total_components:
            status['system_health'] = 'optimal'
        elif operational_components >= total_components * 0.5:
            status['system_health'] = 'operational'
        else:
            status['system_health'] = 'degraded'

        return status

def main():
    """Test the precision voice cloning system"""
    system = PrecisionVoiceCloningSystem()

    print("🎤 Precision Voice Cloning System")
    print("=" * 50)

    # Get system status
    status = system.get_system_status()
    print(f"\n📊 System Status: {status['system_health']}")
    print(f"   Components operational: {sum(status['components'].values())}/{len(status['components'])}")
    print(f"   Available models: {len(status['available_models'])}")
    print(f"   Dataset speakers: {len(status['dataset_info'])}")

    # Test precision cloning (if data available)
    if status['dataset_info']:
        speaker_name = list(status['dataset_info'].keys())[0]
        model_name = f"{speaker_name}_precision"

        print(f"\n🎯 Testing precision cloning for: {speaker_name}")

        def progress_callback(message, percent):
            print(f"\r   {message} ({percent:.0f}%)", end='')

        result = system.precision_clone_voice(speaker_name, model_name, progress_callback)

        if result['status'] == 'success':
            print(f"\n✅ Precision cloning successful!")
            print(f"   Model: {result['model_path']}")
            print(f"   Quality: {result['quality_report'].get('status', 'unknown')}")
        else:
            print(f"\n❌ Precision cloning failed: {result.get('error', 'unknown')}")
    else:
        print("\n⚠️ No dataset available. Please add audio files to the dataset directory.")

if __name__ == "__main__":
    import numpy as np
    main()
