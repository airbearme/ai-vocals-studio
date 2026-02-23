import os, json, subprocess, shutil, glob
from pathlib import Path
import librosa
import numpy as np
import soundfile as sf
import json
from datetime import datetime

class VoiceTrainer:
    def __init__(self, dataset_path, models_path, config_path="training_config.json"):
        self.dataset_path = dataset_path
        self.models_path = models_path
        self.config_path = config_path
        self.training_log = []
        
    def prepare_training_data(self, speaker_name):
        """Prepare training data from audio files"""
        print(f"🎤 Preparing training data for {speaker_name}...")
        
        # Create speaker directory
        speaker_dir = os.path.join(self.dataset_path, speaker_name)
        os.makedirs(speaker_dir, exist_ok=True)
        
        # Get all audio files
        audio_files = []
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        
        for ext in audio_extensions:
            audio_files.extend(glob.glob(os.path.join(self.dataset_path, ext)))
        
        if not audio_files:
            raise Exception("No audio files found in dataset folder!")
        
        print(f"📁 Found {len(audio_files)} audio files")
        
        # Process and normalize audio files
        processed_files = []
        for i, audio_file in enumerate(audio_files):
            try:
                # Convert to WAV 22.05kHz (standard for voice cloning)
                output_file = os.path.join(speaker_dir, f"{speaker_name}_{i:04d}.wav")
                
                # Load and resample
                y, sr = librosa.load(audio_file, sr=22050)
                
                # Trim silence
                y, _ = librosa.effects.trim(y, top_db=20)
                
                # Normalize volume
                y = librosa.util.normalize(y)
                
                # Save processed file
                sf.write(output_file, y, 22050)
                processed_files.append(output_file)
                
                print(f"✅ Processed: {os.path.basename(audio_file)}")
                
            except Exception as e:
                print(f"❌ Error processing {audio_file}: {e}")
        
        if not processed_files:
            raise Exception("No audio files could be processed!")
        
        print(f"🎯 Successfully prepared {len(processed_files)} training files")
        return processed_files, speaker_dir
    
    def create_training_config(self, speaker_name, speaker_dir):
        """Create configuration for training"""
        config = {
            "train": {
                "log_interval": 200,
                "eval_interval": 1000,
                "seed": 1234,
                "epochs": 10000,
                "learning_rate": 2e-4,
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": 4,
                "fp16_run": True,
                "lr_decay": 0.999875,
                "segment_size": 8192,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0
            },
            "data": {
                "training_files": os.path.join(speaker_dir, "training_list.txt"),
                "validation_files": os.path.join(speaker_dir, "validation_list.txt"),
                "text_cleaners": ["cjke_cleaners2"],
                "max_wav_value": 32768.0,
                "sampling_rate": 22050,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": None,
                "add_blank": True,
                "n_speakers": 1,
                "cleaned_text": True,
                "spk2id": {speaker_name: 0}
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3,7,11],
                "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
                "upsample_rates": [8,8,2,2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16,16,4,4],
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "gin_channels": 256,
                "ssl_dim": 256,
                "n_speakers": 1,
                "sampling_rate": 22050
            }
        }
        
        config_file = os.path.join(speaker_dir, f"config_{speaker_name}.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def create_training_lists(self, speaker_dir, processed_files):
        """Create training and validation file lists"""
        # Split 90% training, 10% validation
        split_idx = int(len(processed_files) * 0.9)
        train_files = processed_files[:split_idx]
        val_files = processed_files[split_idx:]
        
        # Create training list
        train_list = os.path.join(speaker_dir, "training_list.txt")
        with open(train_list, 'w') as f:
            for file_path in train_files:
                # Extract filename without extension for text
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                f.write(f"{file_path}|{base_name}\n")
        
        # Create validation list
        val_list = os.path.join(speaker_dir, "validation_list.txt")
        with open(val_list, 'w') as f:
            for file_path in val_files:
                base_name = os.path.splitext(os.path.basename(file_path))[0]
                f.write(f"{file_path}|{base_name}\n")
        
        return train_list, val_list
    
    def train_model(self, speaker_name, config_file, output_model_name):
        """Train the voice model using so-vits-svc-fork"""
        print(f"🚀 Starting training for {speaker_name}...")
        print("⏳ This may take several hours depending on your data...")
        
        try:
            # Use so-vits-svc-fork training command
            cmd = [
                "svc", "train",
                "-c", config_file,
                "-n", output_model_name
            ]
            
            print(f"🔧 Running command: {' '.join(cmd)}")
            
            # Run training
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.dataset_path)
            
            if result.returncode == 0:
                print("✅ Training completed successfully!")
                
                # Find the generated model
                model_files = glob.glob(os.path.join(self.dataset_path, "logs", output_model_name, "G_*.pth"))
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    
                    # Copy to models folder
                    final_model = os.path.join(self.models_path, f"{output_model_name}.pth")
                    shutil.copy(latest_model, final_model)
                    
                    print(f"🎯 Model saved to: {final_model}")
                    return final_model
                else:
                    raise Exception("Model file not found after training")
            else:
                print(f"❌ Training failed: {result.stderr}")
                raise Exception(f"Training failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Training error: {e}")
            raise e
    
    def fine_tune_model(self, speaker_name, model_name, base_model_path):
        """Fine-tune a pre-trained model with custom voice data"""
        print(f"🔧 Fine-tuning {speaker_name} from pre-trained model...")
        
        try:
            # Step 1: Prepare training data
            processed_files, speaker_dir = self.prepare_training_data(speaker_name)
            
            # Step 2: Create fine-tuning configuration
            config_file = self.create_fine_tuning_config(speaker_name, speaker_dir, base_model_path)
            
            # Step 3: Create training lists
            train_list, val_list = self.create_training_lists(speaker_dir, processed_files)
            
            # Step 4: Copy base model to working directory
            working_model = os.path.join(speaker_dir, "base_model.pth")
            shutil.copy(base_model_path, working_model)
            
            # Step 5: Fine-tune the model
            model_path = self.train_model_with_base(speaker_name, config_file, model_name, working_model)
            
            print(f"🎯 Fine-tuning completed!")
            print(f"📁 Model saved as: {model_path}")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Fine-tuning error: {e}")
            raise e
    
    def create_fine_tuning_config(self, speaker_name, speaker_dir, base_model_path):
        """Create configuration for fine-tuning"""
        config = {
            "train": {
                "log_interval": 200,
                "eval_interval": 500,
                "seed": 1234,
                "epochs": 2000,  # Less epochs for fine-tuning
                "learning_rate": 1e-4,  # Lower learning rate for fine-tuning
                "betas": [0.8, 0.99],
                "eps": 1e-9,
                "batch_size": 4,
                "fp16_run": True,
                "lr_decay": 0.9999,
                "segment_size": 8192,
                "init_lr_ratio": 1,
                "warmup_epochs": 0,
                "c_mel": 45,
                "c_kl": 1.0
            },
            "data": {
                "training_files": os.path.join(speaker_dir, "training_list.txt"),
                "validation_files": os.path.join(speaker_dir, "validation_list.txt"),
                "text_cleaners": ["cjke_cleaners2"],
                "max_wav_value": 32768.0,
                "sampling_rate": 22050,
                "filter_length": 1024,
                "hop_length": 256,
                "win_length": 1024,
                "n_mel_channels": 80,
                "mel_fmin": 0.0,
                "mel_fmax": None,
                "add_blank": True,
                "n_speakers": 1,
                "cleaned_text": True,
                "spk2id": {speaker_name: 0}
            },
            "model": {
                "inter_channels": 192,
                "hidden_channels": 192,
                "filter_channels": 768,
                "n_heads": 2,
                "n_layers": 6,
                "kernel_size": 3,
                "p_dropout": 0.1,
                "resblock": "1",
                "resblock_kernel_sizes": [3,7,11],
                "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
                "upsample_rates": [8,8,2,2],
                "upsample_initial_channel": 512,
                "upsample_kernel_sizes": [16,16,4,4],
                "n_layers_q": 3,
                "use_spectral_norm": False,
                "gin_channels": 256,
                "ssl_dim": 256,
                "n_speakers": 1,
                "sampling_rate": 22050
            }
        }
        
        config_file = os.path.join(speaker_dir, f"finetune_config_{speaker_name}.json")
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        return config_file
    
    def train_model_with_base(self, speaker_name, config_file, output_model_name, base_model_path):
        """Train model starting from pre-trained weights"""
        print(f"🔧 Fine-tuning model from {base_model_path}")
        
        try:
            # Use so-vits-svc-fork fine-tuning command
            cmd = [
                "svc", "train",
                "-c", config_file,
                "-i", base_model_path,  # Initial model
                "-n", output_model_name
            ]
            
            print(f"🔧 Running fine-tuning command: {' '.join(cmd)}")
            
            # Run fine-tuning
            result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.dataset_path)
            
            if result.returncode == 0:
                print("✅ Fine-tuning completed successfully!")
                
                # Find the generated model
                model_files = glob.glob(os.path.join(self.dataset_path, "logs", output_model_name, "G_*.pth"))
                if model_files:
                    latest_model = max(model_files, key=os.path.getctime)
                    
                    # Copy to models folder
                    final_model = os.path.join(self.models_path, f"{output_model_name}.pth")
                    shutil.copy(latest_model, final_model)
                    
                    print(f"🎯 Fine-tuned model saved to: {final_model}")
                    return final_model
                else:
                    raise Exception("Fine-tuned model file not found")
            else:
                print(f"❌ Fine-tuning failed: {result.stderr}")
                raise Exception(f"Fine-tuning failed: {result.stderr}")
                
        except Exception as e:
            print(f"❌ Fine-tuning error: {e}")
            raise e

    def clone_voice(self, speaker_name, model_name):
        """Complete voice cloning pipeline"""
        try:
            print(f"🎤 Starting voice cloning for {speaker_name}...")
            
            # Step 1: Prepare training data
            processed_files, speaker_dir = self.prepare_training_data(speaker_name)
            
            # Step 2: Create training configuration
            config_file = self.create_training_config(speaker_name, speaker_dir)
            
            # Step 3: Create training lists
            train_list, val_list = self.create_training_lists(speaker_dir, processed_files)
            
            # Step 4: Train the model
            model_path = self.train_model(speaker_name, config_file, model_name)
            
            print(f"🎉 Voice cloning completed!")
            print(f"📁 Model saved as: {model_path}")
            print(f"🎯 You can now load this model in the main app!")
            
            return model_path
            
        except Exception as e:
            print(f"❌ Voice cloning failed: {e}")
            raise e

def quick_clone_voice(dataset_path, models_path, speaker_name, model_name):
    """Quick voice cloning function"""
    trainer = VoiceTrainer(dataset_path, models_path)
    return trainer.clone_voice(speaker_name, model_name)

if __name__ == "__main__":
    # Example usage
    dataset_path = "/home/coden809/ai-vocals-studio/dataset"
    models_path = "/home/coden809/ai-vocals-studio/models"
    
    speaker_name = input("Enter speaker name: ")
    model_name = input("Enter model name (e.g., 'my_voice'): ")
    
    try:
        model_path = quick_clone_voice(dataset_path, models_path, speaker_name, model_name)
        print(f"✅ Voice cloning complete! Model: {model_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
