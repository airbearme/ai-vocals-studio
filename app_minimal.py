#!/usr/bin/env python3
"""
AI Vocals Studio - Minimal Version
A simplified version that works without complex ML dependencies
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import json
import threading
import time
from pathlib import Path
import numpy as np
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import shutil

class AIVocalsStudioMinimal:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vocals Studio - Minimal")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set up directories
        self.DATA = Path("dataset")
        self.MODELS = Path("models")
        self.OUTPUT = Path("output")
        
        for dir_path in [self.DATA, self.MODELS, self.OUTPUT]:
            dir_path.mkdir(exist_ok=True)
        
        # Variables
        self.audio_path = tk.StringVar()
        self.model_var = tk.StringVar()
        self.current_operation = tk.StringVar(value="Ready")
        self.operation_progress = tk.IntVar(value=0)
        self.progress_text = tk.StringVar(value="0%")
        
        # Voice persona characteristics
        self.voice_personas = {
            "2pac": {
                "pitch_shift": -0.2,
                "speed": 1.1,
                "reverb": 0.3,
                "distortion": 0.1,
                "eq_low": 1.2,
                "eq_mid": 0.9,
                "eq_high": 1.1
            },
            "default": {
                "pitch_shift": 0,
                "speed": 1.0,
                "reverb": 0.1,
                "distortion": 0,
                "eq_low": 1.0,
                "eq_mid": 1.0,
                "eq_high": 1.0
            },
            "male": {
                "pitch_shift": -0.3,
                "speed": 1.0,
                "reverb": 0.2,
                "distortion": 0.05,
                "eq_low": 1.3,
                "eq_mid": 0.8,
                "eq_high": 0.9
            },
            "female": {
                "pitch_shift": 0.3,
                "speed": 1.0,
                "reverb": 0.15,
                "distortion": 0,
                "eq_low": 0.8,
                "eq_mid": 1.1,
                "eq_high": 1.2
            },
            "robot": {
                "pitch_shift": 0,
                "speed": 0.9,
                "reverb": 0.4,
                "distortion": 0.2,
                "eq_low": 1.5,
                "eq_mid": 0.7,
                "eq_high": 0.5
            }
        }
        
        # Create UI
        self.create_styles()
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
        # Auto-load models
        self.root.after(100, self.refresh_models)
    
    def create_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        # Configure styles
        style.configure('Title.TLabel', font=('Arial', 24, 'bold'))
        style.configure('Header.TLabel', font=('Arial', 12))
        style.configure('Operation.TLabel', font=('Arial', 10))
    
    def create_header(self):
        header_frame = ttk.Frame(self.root)
        header_frame.pack(fill='x', padx=10, pady=5)
        
        title_label = ttk.Label(header_frame, text="🎤 AI Vocals Studio", style='Title.TLabel')
        title_label.pack(side='left')
        
        status_label = ttk.Label(header_frame, text="Minimal Version - Basic Functions", style='Header.TLabel')
        status_label.pack(side='right')
    
    def create_main_content(self):
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Generation Tab
        self.create_generation_tab(self.notebook)
        
        # Models Tab
        self.create_models_tab(self.notebook)
        
        # Training Tab
        self.create_training_tab(self.notebook)
    
    def create_generation_tab(self, notebook):
        gen_frame = ttk.Frame(notebook)
        notebook.add(gen_frame, text="🎵 Generate")
        
        # Input Type Selection
        input_type_frame = ttk.LabelFrame(gen_frame, text="Input Type", padding=10)
        input_type_frame.pack(fill='x', padx=10, pady=5)
        
        self.input_type = tk.StringVar(value="audio")
        ttk.Radiobutton(input_type_frame, text="🎵 Audio File", variable=self.input_type, value="audio", command=self.toggle_input_type).pack(side='left', padx=10)
        ttk.Radiobutton(input_type_frame, text="📝 Text", variable=self.input_type, value="text", command=self.toggle_input_type).pack(side='left', padx=10)
        
        # Audio Input Section
        self.audio_frame = ttk.LabelFrame(gen_frame, text="Audio Input", padding=10)
        self.audio_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(self.audio_frame, text="Select audio file:").pack(anchor='w')
        
        audio_select_frame = ttk.Frame(self.audio_frame)
        audio_select_frame.pack(fill='x', pady=5)
        
        ttk.Entry(audio_select_frame, textvariable=self.audio_path, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(audio_select_frame, text="Browse", command=self.select_audio).pack(side='right', padx=(5, 0))
        
        # Text Input Section
        self.text_frame = ttk.LabelFrame(gen_frame, text="Text Input", padding=10)
        
        ttk.Label(self.text_frame, text="Enter text to convert to speech:").pack(anchor='w')
        
        self.text_input = tk.Text(self.text_frame, height=4, wrap='word')
        text_scrollbar = ttk.Scrollbar(self.text_frame, orient='vertical', command=self.text_input.yview)
        self.text_input.configure(yscrollcommand=text_scrollbar.set)
        
        self.text_input.pack(side='left', fill='both', expand=True, pady=5)
        text_scrollbar.pack(side='right', fill='y')
        
        # Text Settings
        text_settings_frame = ttk.Frame(self.text_frame)
        text_settings_frame.pack(fill='x', pady=5)
        
        ttk.Label(text_settings_frame, text="Voice:").pack(side='left')
        self.tts_voice = tk.StringVar(value="default")
        voice_combo = ttk.Combobox(text_settings_frame, textvariable=self.tts_voice, values=["default", "male", "female", "robot"], state='readonly', width=15)
        voice_combo.pack(side='left', padx=(5, 10))
        
        ttk.Label(text_settings_frame, text="Speed:").pack(side='left')
        self.tts_speed = tk.StringVar(value="1.0")
        speed_combo = ttk.Combobox(text_settings_frame, textvariable=self.tts_speed, values=["0.5", "0.8", "1.0", "1.2", "1.5"], state='readonly', width=10)
        speed_combo.pack(side='left', padx=(5, 0))
        
        # Model Selection
        model_frame = ttk.LabelFrame(gen_frame, text="Voice Model", padding=10)
        model_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(model_frame, text="Select model:").pack(anchor='w')
        self.model_combo = ttk.Combobox(model_frame, textvariable=self.model_var, state='readonly')
        self.model_combo.pack(fill='x', pady=5)
        
        # Generation Controls
        control_frame = ttk.LabelFrame(gen_frame, text="Generation", padding=10)
        control_frame.pack(fill='x', padx=10, pady=5)
        
        generate_btn = ttk.Button(control_frame, text="🎤 Generate Vocals", command=self.generate_vocals)
        generate_btn.pack(fill='x', pady=5)
        
        # Output Info
        output_frame = ttk.LabelFrame(gen_frame, text="Output", padding=10)
        output_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.output_text = tk.Text(output_frame, height=10, wrap='word')
        scrollbar = ttk.Scrollbar(output_frame, orient='vertical', command=self.output_text.yview)
        self.output_text.configure(yscrollcommand=scrollbar.set)
        
        self.output_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
    
    def create_models_tab(self, notebook):
        models_frame = ttk.Frame(notebook)
        notebook.add(models_frame, text="🤖 Models")
        
        # Model Management
        manage_frame = ttk.LabelFrame(models_frame, text="Model Management", padding=10)
        manage_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(manage_frame, text="🔄 Refresh Models", command=self.refresh_models).pack(fill='x', pady=2)
        ttk.Button(manage_frame, text="📁 Open Models Folder", command=self.open_models_folder).pack(fill='x', pady=2)
        
        # Available Models List
        list_frame = ttk.LabelFrame(models_frame, text="Available Models", padding=10)
        list_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.models_listbox = tk.Listbox(list_frame)
        models_scrollbar = ttk.Scrollbar(list_frame, orient='vertical', command=self.models_listbox.yview)
        self.models_listbox.configure(yscrollcommand=models_scrollbar.set)
        
        self.models_listbox.pack(side='left', fill='both', expand=True)
        models_scrollbar.pack(side='right', fill='y')
    
    def create_training_tab(self, notebook):
        training_frame = ttk.Frame(notebook)
        notebook.add(training_frame, text="🎓 Training")
        
        # Data Management
        data_frame = ttk.LabelFrame(training_frame, text="Training Data", padding=10)
        data_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(data_frame, text="📁 Import Single Clip", command=self.import_single_clip).pack(fill='x', pady=2)
        ttk.Button(data_frame, text="📂 Import Folder", command=self.import_folder).pack(fill='x', pady=2)
        ttk.Button(data_frame, text="📁 Open Data Folder", command=self.open_data_folder).pack(fill='x', pady=2)
        
        # Data Info
        info_frame = ttk.LabelFrame(training_frame, text="Data Information", padding=10)
        info_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        self.data_info_text = tk.Text(info_frame, height=15, wrap='word')
        data_scrollbar = ttk.Scrollbar(info_frame, orient='vertical', command=self.data_info_text.yview)
        self.data_info_text.configure(yscrollcommand=data_scrollbar.set)
        
        self.data_info_text.pack(side='left', fill='both', expand=True)
        data_scrollbar.pack(side='right', fill='y')
        
        # Update data info
        self.update_data_info()
    
    def create_footer(self):
        footer_frame = ttk.Frame(self.root)
        footer_frame.pack(fill='x', padx=10, pady=5)
        
        # Progress Bar
        self.progress_bar = ttk.Progressbar(footer_frame, variable=self.operation_progress, maximum=100)
        self.progress_bar.pack(fill='x', pady=(0, 5))
        
        # Status
        status_frame = ttk.Frame(footer_frame)
        status_frame.pack(fill='x')
        
        self.operation_label = ttk.Label(status_frame, textvariable=self.current_operation, style='Operation.TLabel')
        self.operation_label.pack(side='left')
        
        self.progress_label = ttk.Label(status_frame, textvariable=self.progress_text)
        self.progress_label.pack(side='right')
    
    def apply_voice_transformation(self, audio_segment, persona_name):
        """Apply voice persona characteristics to audio"""
        persona = self.voice_personas.get(persona_name.lower(), self.voice_personas["default"])
        
        # Apply pitch shift (simulated with speed change)
        if persona["pitch_shift"] != 0:
            speed_factor = 1.0 + persona["pitch_shift"]
            audio_segment = audio_segment._spawn(audio_segment.raw_data, overrides={
                "frame_rate": int(audio_segment.frame_rate * speed_factor)
            }).set_frame_rate(audio_segment.frame_rate)
        
        # Apply speed change
        if persona["speed"] != 1.0:
            audio_segment = audio_segment.speedup(playback_speed=persona["speed"])
        
        # Apply EQ (simulated with gain adjustments)
        if persona["eq_low"] != 1.0 or persona["eq_mid"] != 1.0 or persona["eq_high"] != 1.0:
            # Simple EQ simulation using gain
            gain = (persona["eq_low"] + persona["eq_mid"] + persona["eq_high"]) / 3.0
            audio_segment = audio_segment + (10 * (gain - 1.0))  # Convert to dB
        
        # Apply reverb (simulated with echo)
        if persona["reverb"] > 0:
            delay = int(100 * persona["reverb"])  # Delay in ms
            echo = audio_segment - 6  # Reduce volume for echo
            echo = echo.overlay(audio_segment, position=delay)
            audio_segment = audio_segment.overlay(echo)
        
        # Apply distortion (simulated with overdrive)
        if persona["distortion"] > 0:
            audio_segment = audio_segment + (6 * persona["distortion"])  # Add gain for distortion effect
        
        return audio_segment
    
    def generate_audio_from_text(self, text, voice="default", speed="1.0"):
        """Generate audio from text using gTTS"""
        try:
            # Generate speech using gTTS
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                tts.save(temp_file.name)
                
                # Convert to WAV for processing
                audio = AudioSegment.from_mp3(temp_file.name)
                
                # Apply voice transformation
                persona_name = voice if voice in self.voice_personas else "default"
                audio = self.apply_voice_transformation(audio, persona_name)
                
                # Apply speed adjustment
                speed_factor = float(speed)
                if speed_factor != 1.0:
                    audio = audio.speedup(playback_speed=speed_factor)
                
                # Export as WAV
                wav_path = temp_file.name.replace('.mp3', '.wav')
                audio.export(wav_path, format='wav', parameters=["-ar", "44100"])
                
                # Clean up temp MP3
                os.unlink(temp_file.name)
                
                return wav_path
                
        except Exception as e:
            raise Exception(f"Text-to-speech generation failed: {str(e)}")
    
    def process_audio_file(self, input_path, model_name):
        """Process existing audio file with voice transformation"""
        try:
            # Load audio file
            audio = AudioSegment.from_file(input_path)
            
            # Normalize audio
            audio = audio.normalize()
            
            # Apply voice transformation based on model
            persona_name = model_name.lower() if model_name.lower() in self.voice_personas else "default"
            audio = self.apply_voice_transformation(audio, persona_name)
            
            # Export as WAV
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio.export(temp_file.name, format='wav', parameters=["-ar", "44100"])
                return temp_file.name
                
        except Exception as e:
            raise Exception(f"Audio processing failed: {str(e)}")
    
    def toggle_input_type(self):
        if self.input_type.get() == "audio":
            self.text_frame.pack_forget()
            self.audio_frame.pack(fill='x', padx=10, pady=5)
        else:
            self.audio_frame.pack_forget()
            self.text_frame.pack(fill='x', padx=10, pady=5)
    
    def update_operation_progress(self, operation, progress, suggestion=""):
        self.current_operation.set(operation)
        self.operation_progress.set(progress)
        self.progress_text.set(f"{progress}%")
        
        if suggestion:
            self.add_output(f"💡 {suggestion}")
        
        self.root.update_idletasks()
    
    def add_output(self, message):
        timestamp = time.strftime("%H:%M:%S")
        self.output_text.insert('end', f"[{timestamp}] {message}\n")
        self.output_text.see('end')
        self.root.update_idletasks()
    
    def select_audio(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            self.audio_path.set(file_path)
            self.add_output(f"Selected audio: {os.path.basename(file_path)}")
    
    def refresh_models(self):
        self.update_operation_progress("🔍 Scanning for models...", 30, "Looking for available voice models")
        
        models = []
        
        # Check for .pth files in models directory and subdirectories
        for item in self.MODELS.iterdir():
            if item.is_file() and item.suffix == '.pth':
                models.append(item.stem)
            elif item.is_dir():
                for subitem in item.iterdir():
                    if subitem.is_file() and subitem.name == 'model.pth':
                        models.append(item.name)
        
        # Update combobox
        self.model_combo['values'] = models
        if models:
            self.model_var.set(models[0])
            self.add_output(f"Found {len(models)} models")
        
        # Update listbox
        self.models_listbox.delete(0, 'end')
        for model in models:
            self.models_listbox.insert('end', model)
        
        self.update_operation_progress("✅ Model scan complete", 100, f"Found {len(models)} models")
    
    def generate_vocals(self):
        # Check input type and validate
        if self.input_type.get() == "audio":
            if not self.audio_path.get():
                messagebox.showwarning("No Audio", "Please select an audio file first.")
                return
            input_source = self.audio_path.get()
            input_type_name = "audio file"
        else:
            text_content = self.text_input.get("1.0", "end-1c").strip()
            if not text_content:
                messagebox.showwarning("No Text", "Please enter some text to convert.")
                return
            input_source = text_content
            input_type_name = "text input"
        
        if not self.model_var.get():
            messagebox.showwarning("No Model", "Please select a voice model first.")
            return
        
        # Generate real audio
        def generation_worker():
            temp_file = None
            try:
                self.update_operation_progress("🎤 Starting vocal generation...", 10, f"Processing {input_type_name}")
                time.sleep(0.5)
                
                if self.input_type.get() == "text":
                    self.update_operation_progress("🔤 Converting text to speech...", 30, "Generating base audio from text")
                    time.sleep(1)
                    
                    # Generate audio from text
                    temp_file = self.generate_audio_from_text(
                        input_source, 
                        voice=self.tts_voice.get(), 
                        speed=self.tts_speed.get()
                    )
                    
                    self.add_output(f"📝 Generated speech from text: {len(input_source)} characters")
                    
                else:
                    self.update_operation_progress("🔄 Processing audio...", 30, "Loading and processing audio file")
                    time.sleep(1)
                    
                    # Process existing audio file
                    temp_file = self.process_audio_file(input_source, self.model_var.get())
                    
                    self.add_output(f"🎵 Processed audio file: {Path(input_source).name}")
                
                self.update_operation_progress("✨ Applying voice transformation...", 60, f"Applying {self.model_var.get()} voice persona")
                time.sleep(1)
                
                # Get model name for persona
                model_name = self.model_var.get().lower()
                if model_name not in self.voice_personas:
                    model_name = "default"
                
                # Apply voice persona transformation
                audio = AudioSegment.from_wav(temp_file)
                audio = self.apply_voice_transformation(audio, model_name)
                
                self.update_operation_progress("💾 Saving output...", 90, "Encoding to final audio format")
                time.sleep(0.5)
                
                # Create final output file
                if self.input_type.get() == "text":
                    input_name = "text_input"
                else:
                    input_name = Path(self.audio_path.get()).stem
                model_name_display = self.model_var.get()
                output_name = f"{input_name}_{model_name_display}_generated.wav"
                output_path = self.OUTPUT / output_name
                
                # Export with proper audio settings - simplified for maximum compatibility
                audio.export(str(output_path), format='wav')
                
                # Verify file was created and has content
                if output_path.exists() and output_path.stat().st_size > 1000:
                    self.update_operation_progress("✅ Generation complete!", 100, f"Output saved as {output_name}")
                    self.add_output(f"🎉 Successfully generated vocals from {input_type_name}: {output_name}")
                    self.add_output(f"📊 File size: {output_path.stat().st_size / 1024:.1f} KB")
                    self.add_output(f"🎵 Duration: {len(audio) / 1000:.1f} seconds")
                    
                    # Offer to play the file
                    if messagebox.askyesno("Success", f"Vocals generated successfully!\n\nOutput: {output_name}\n\nWould you like to play the audio file?"):
                        self.play_audio_file(str(output_path))
                else:
                    raise Exception("Generated file is empty or missing")
                
            except Exception as e:
                self.update_operation_progress("❌ Generation failed", 0, "Check error details")
                self.add_output(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Generation failed: {str(e)}")
            
            finally:
                # Clean up temporary file
                if temp_file and os.path.exists(temp_file):
                    try:
                        os.unlink(temp_file)
                    except:
                        pass
        
        # Run in thread
        thread = threading.Thread(target=generation_worker)
        thread.daemon = True
        thread.start()
    
    def play_audio_file(self, file_path):
        """Play audio file using system default player"""
        try:
            if os.name == 'nt':  # Windows
                os.startfile(file_path)
            elif os.name == 'posix':  # macOS and Linux
                if 'darwin' in os.sys.platform:  # macOS
                    os.system(f'open "{file_path}"')
                else:  # Linux
                    os.system(f'xdg-open "{file_path}"')
            self.add_output(f"🔊 Playing audio: {Path(file_path).name}")
        except Exception as e:
            self.add_output(f"❌ Failed to play audio: {str(e)}")
            messagebox.showwarning("Playback Error", f"Could not play audio file: {str(e)}")
    
    def import_single_clip(self):
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=[
                ("Audio Files", "*.wav *.mp3 *.flac *.m4a *.ogg"),
                ("All Files", "*.*")
            ]
        )
        if file_path:
            try:
                dest_path = self.DATA / Path(file_path).name
                import shutil
                shutil.copy2(file_path, dest_path)
                self.add_output(f"Imported: {Path(file_path).name}")
                self.update_data_info()
            except Exception as e:
                messagebox.showerror("Error", f"Failed to import file: {str(e)}")
    
    def import_folder(self):
        folder_path = filedialog.askdirectory(title="Select Folder with Audio Files")
        if folder_path:
            imported_count = 0
            for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
                for file_path in Path(folder_path).glob(ext):
                    try:
                        dest_path = self.DATA / file_path.name
                        import shutil
                        shutil.copy2(file_path, dest_path)
                        imported_count += 1
                    except:
                        pass
            
            self.add_output(f"Imported {imported_count} files from folder")
            self.update_data_info()
    
    def update_data_info(self):
        audio_files = []
        for ext in ['*.wav', '*.mp3', '*.flac', '*.m4a', '*.ogg']:
            audio_files.extend(self.DATA.glob(ext))
        
        self.data_info_text.delete('1.0', 'end')
        self.data_info_text.insert('end', f"Training Data Statistics\n")
        self.data_info_text.insert('end', f"{'='*30}\n\n")
        self.data_info_text.insert('end', f"Total audio files: {len(audio_files)}\n\n")
        
        if audio_files:
            self.data_info_text.insert('end', "Files:\n")
            for i, file_path in enumerate(audio_files[:20], 1):  # Show first 20
                self.data_info_text.insert('end', f"{i}. {file_path.name}\n")
            
            if len(audio_files) > 20:
                self.data_info_text.insert('end', f"... and {len(audio_files) - 20} more files\n")
    
    def open_models_folder(self):
        os.startfile(str(self.MODELS)) if os.name == 'nt' else os.system(f'xdg-open "{self.MODELS}"')
    
    def open_data_folder(self):
        os.startfile(str(self.DATA)) if os.name == 'nt' else os.system(f'xdg-open "{self.DATA}"')

def main():
    root = tk.Tk()
    app = AIVocalsStudioMinimal(root)
    root.mainloop()

if __name__ == "__main__":
    main()
