#!/usr/bin/env python3
"""
🎤 AI Vocals Studio - Minimal Working Version
A simplified version that works out of the box with voice cloning
"""
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import json
from pathlib import Path
import numpy as np
import soundfile as sf
from gtts import gTTS
from pydub import AudioSegment
import tempfile
import glob

# Try to import voice cloning engines
try:
    from qwen3_tts_engine import Qwen3TTSEngine
    HAS_QWEN = True
except ImportError:
    HAS_QWEN = False
    print("Qwen3-TTS not available. Install with: pip install qwen-tts")

# Configuration
BASE = os.path.expanduser("~/ai-vocals-studio")
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "dataset")
MODELS = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

class AIVocalsStudio:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 AI Vocals Studio - 2Pac Voice Cloning")
        self.root.geometry("1000x700")
        
        # Dark theme
        self.root.configure(bg='#1a1a1a')
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Initialize engines
        self.qwen_engine = Qwen3TTSEngine() if HAS_QWEN else None
        
        # Create UI
        self.create_widgets()
        self.load_2pac_data()
        
    def create_widgets(self):
        # Main container
        main_frame = tk.Frame(self.root, bg='#1a1a1a')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = tk.Label(main_frame, text="🎤 AI Vocals Studio", 
                               font=("Arial", 24, "bold"), fg='#00ff88', bg='#1a1a1a')
        title_label.pack(pady=(0, 10))
        
        subtitle = tk.Label(main_frame, text="2Pac Voice Cloning & Generation", 
                           font=("Arial", 14), fg='#ffffff', bg='#1a1a1a')
        subtitle.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Style notebook
        self.notebook.configure(style='Dark.TNotebook')
        self.style.configure('Dark.TNotebook', background='#2b2b2b', borderwidth=0)
        self.style.configure('Dark.TNotebook.Tab', background='#2b2b2b', foreground='white')
        self.style.map('Dark.TNotebook.Tab', background=[('selected', '#00ff88')])
        
        # Create tabs
        self.create_voice_clone_tab()
        self.create_tts_tab()
        self.create_models_tab()
        self.create_dataset_tab()
        
    def create_voice_clone_tab(self):
        # Voice Cloning Tab
        clone_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(clone_frame, text="🎭 Voice Clone")
        
        # Instructions
        instructions = tk.Label(clone_frame, 
                               text="Clone 2Pac's voice with just 3 seconds of audio!\n" +
                                    "1. Select a reference audio file\n" +
                                    "2. Enter what's said in the audio\n" +
                                    "3. Enter what you want 2Pac to say\n" +
                                    "4. Click 'Clone Voice'",
                               font=("Arial", 11), fg='#ffffff', bg='#2b2b2b',
                               justify=tk.LEFT)
        instructions.pack(pady=20, padx=20, anchor='w')
        
        # Reference audio selection
        ref_frame = tk.Frame(clone_frame, bg='#2b2b2b')
        ref_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(ref_frame, text="Reference Audio:", font=("Arial", 12, "bold"),
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        self.ref_audio_var = tk.StringVar()
        ref_entry = tk.Entry(ref_frame, textvariable=self.ref_audio_var, 
                            font=("Arial", 10), bg='#3a3a3a', fg='white')
        ref_entry.pack(fill=tk.X, pady=5)
        
        tk.Button(ref_frame, text="Browse Audio", command=self.browse_ref_audio,
                 bg='#00ff88', fg='black', font=("Arial", 10, "bold")).pack(anchor='w')
        
        # Reference text
        tk.Label(ref_frame, text="What's said in the audio:", font=("Arial", 12, "bold"),
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w', pady=(10, 5))
        
        self.ref_text_var = tk.StringVar(value="I'm gettin money")
        ref_text_entry = tk.Entry(ref_frame, textvariable=self.ref_text_var,
                                 font=("Arial", 10), bg='#3a3a3a', fg='white')
        ref_text_entry.pack(fill=tk.X, pady=5)
        
        # Target text
        tk.Label(ref_frame, text="What you want 2Pac to say:", font=("Arial", 12, "bold"),
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w', pady=(10, 5))
        
        self.target_text_var = tk.StringVar(value="Thug life baby, we keep it real")
        target_text_entry = tk.Entry(ref_frame, textvariable=self.target_text_var,
                                     font=("Arial", 10), bg='#3a3a3a', fg='white')
        target_text_entry.pack(fill=tk.X, pady=5)
        
        # Clone button
        clone_btn = tk.Button(clone_frame, text="🎭 Clone 2Pac Voice", 
                            command=self.clone_voice,
                            bg='#ff6b35', fg='white', font=("Arial", 14, "bold"),
                            height=2)
        clone_btn.pack(pady=20)
        
        # Progress
        self.clone_progress = ttk.Progressbar(clone_frame, mode='indeterminate')
        self.clone_progress.pack(fill=tk.X, padx=20, pady=10)
        
        # Status
        self.clone_status = tk.Label(clone_frame, text="Ready to clone voice",
                                   font=("Arial", 10), fg='#00ff88', bg='#2b2b2b')
        self.clone_status.pack()
        
    def create_tts_tab(self):
        # Text-to-Speech Tab
        tts_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(tts_frame, text="🗣️ Text-to-Speech")
        
        # Instructions
        instructions = tk.Label(tts_frame,
                               text="Generate speech using different TTS engines",
                               font=("Arial", 11), fg='#ffffff', bg='#2b2b2b')
        instructions.pack(pady=20)
        
        # Text input
        tk.Label(tts_frame, text="Enter text:", font=("Arial", 12, "bold"),
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w', padx=20)
        
        self.tts_text = tk.Text(tts_frame, height=5, font=("Arial", 10),
                               bg='#3a3a3a', fg='white')
        self.tts_text.pack(fill=tk.X, padx=20, pady=10)
        self.tts_text.insert('1.0', "Westside till we die!")
        
        # Engine selection
        engine_frame = tk.Frame(tts_frame, bg='#2b2b2b')
        engine_frame.pack(fill=tk.X, padx=20, pady=10)
        
        tk.Label(engine_frame, text="Engine:", font=("Arial", 12, "bold"),
                fg='#00ff88', bg='#2b2b2b').pack(side=tk.LEFT)
        
        self.tts_engine = tk.StringVar(value="gTTS")
        engines = ["gTTS (Free)"]
        if HAS_QWEN:
            engines.append("Qwen3-TTS (Advanced)")
        
        engine_menu = ttk.Combobox(engine_frame, textvariable=self.tts_engine,
                                  values=engines, state="readonly")
        engine_menu.pack(side=tk.LEFT, padx=10)
        
        # Generate button
        generate_btn = tk.Button(tts_frame, text="🎤 Generate Speech",
                               command=self.generate_tts,
                               bg='#00ff88', fg='black', font=("Arial", 12, "bold"))
        generate_btn.pack(pady=20)
        
    def create_models_tab(self):
        # Models Tab
        models_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(models_frame, text="🤖 Models")
        
        # Title
        tk.Label(models_frame, text="Available Voice Models",
                font=("Arial", 16, "bold"), fg='#00ff88', bg='#2b2b2b').pack(pady=20)
        
        # Models list
        models_text = tk.Text(models_frame, height=20, font=("Courier", 10),
                             bg='#3a3a3a', fg='white')
        models_text.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # Check available models
        models_info = self.check_available_models()
        models_text.insert('1.0', models_info)
        models_text.config(state=tk.DISABLED)
        
    def create_dataset_tab(self):
        # Dataset Tab
        dataset_frame = tk.Frame(self.notebook, bg='#2b2b2b')
        self.notebook.add(dataset_frame, text="📁 Dataset")
        
        # Title
        tk.Label(dataset_frame, text="2Pac Training Data",
                font=("Arial", 16, "bold"), fg='#00ff88', bg='#2b2b2b').pack(pady=20)
        
        # Dataset info
        self.dataset_info = tk.Text(dataset_frame, height=20, font=("Courier", 10),
                                    bg='#3a3a3a', fg='white')
        self.dataset_info.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
    def browse_ref_audio(self):
        filename = filedialog.askopenfilename(
            title="Select Reference Audio",
            filetypes=[("Audio Files", "*.wav *.mp3 *.m4a *.flac"), ("All Files", "*.*")]
        )
        if filename:
            self.ref_audio_var.set(filename)
    
    def clone_voice(self):
        if not HAS_QWEN:
            messagebox.showerror("Error", "Qwen3-TTS not installed!\nInstall with: pip install qwen-tts")
            return
        
        ref_audio = self.ref_audio_var.get()
        ref_text = self.ref_text_var.get()
        target_text = self.target_text_var.get()
        
        if not all([ref_audio, ref_text, target_text]):
            messagebox.showerror("Error", "Please fill in all fields")
            return
        
        if not os.path.exists(ref_audio):
            messagebox.showerror("Error", "Reference audio file not found")
            return
        
        # Start cloning in background
        self.clone_progress.start()
        self.clone_status.config(text="Cloning voice... This may take a few minutes")
        
        thread = threading.Thread(target=self._clone_voice_worker,
                                 args=(ref_audio, ref_text, target_text))
        thread.daemon = True
        thread.start()
    
    def _clone_voice_worker(self, ref_audio, ref_text, target_text):
        try:
            # Initialize engine if needed
            if not self.qwen_engine:
                self.qwen_engine = Qwen3TTSEngine()
            
            # Load model
            def progress_cb(msg, progress):
                self.root.after(0, lambda: self.clone_status.config(text=msg))
            
            if not self.qwen_engine.load_model(progress_cb=progress_cb):
                self.root.after(0, lambda: self.clone_status.config(text="Failed to load model"))
                self.root.after(0, self.clone_progress.stop)
                return
            
            # Clone voice
            output_path = self.qwen_engine.clone_voice(
                ref_audio=ref_audio,
                ref_text=ref_text,
                target_text=target_text,
                speaker_name="2Pac_Clone",
                progress_cb=progress_cb
            )
            
            if output_path:
                self.root.after(0, lambda: self.clone_status.config(
                    text=f"✅ Voice cloned! Saved to: {output_path}"))
                self.root.after(0, lambda: messagebox.showinfo("Success", 
                    f"2Pac voice cloned successfully!\nSaved to: {output_path}"))
            else:
                self.root.after(0, lambda: self.clone_status.config(text="❌ Cloning failed"))
                
        except Exception as e:
            self.root.after(0, lambda: self.clone_status.config(text=f"❌ Error: {str(e)}"))
        
        self.root.after(0, self.clone_progress.stop)
    
    def generate_tts(self):
        text = self.tts_text.get('1.0', tk.END).strip()
        engine = self.tts_engine.get()
        
        if not text:
            messagebox.showerror("Error", "Please enter text to generate")
            return
        
        try:
            if engine == "gTTS (Free)":
                # Use gTTS
                tts = gTTS(text=text, lang='en')
                output_path = os.path.join(OUT, "gtts_output.mp3")
                tts.save(output_path)
                messagebox.showinfo("Success", f"Audio saved to: {output_path}")
                
            elif HAS_QWEN and engine == "Qwen3-TTS (Advanced)":
                # Use Qwen3-TTS
                if not self.qwen_engine:
                    self.qwen_engine = Qwen3TTSEngine()
                
                # For now, use voice design with 2Pac description
                output_path = self.qwen_engine.design_voice(
                    text=text,
                    voice_description="Deep, aggressive rap voice like 2Pac with authentic Westside gangsta delivery and emotional intensity",
                    speaker_name="2Pac_TTS"
                )
                
                if output_path:
                    messagebox.showinfo("Success", f"Audio saved to: {output_path}")
                else:
                    messagebox.showerror("Error", "Failed to generate speech")
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate speech: {str(e)}")
    
    def check_available_models(self):
        info = "🤖 Available Voice Models\n" + "="*40 + "\n\n"
        
        # Check Qwen3-TTS
        if HAS_QWEN:
            info += "✅ Qwen3-TTS: Available\n"
            info += "   - 3-second rapid voice cloning\n"
            info += "   - Multi-language support\n"
            info += "   - Natural language voice control\n\n"
        else:
            info += "❌ Qwen3-TTS: Not installed\n"
            info += "   Install with: pip install qwen-tts\n\n"
        
        # Check gTTS
        info += "✅ gTTS: Available (Free)\n"
        info += "   - Basic text-to-speech\n"
        info += "   - Multiple languages\n\n"
        
        # Check local models
        if os.path.exists(MODELS):
            local_models = []
            for item in os.listdir(MODELS):
                if os.path.isdir(os.path.join(MODELS, item)):
                    local_models.append(item)
            
            if local_models:
                info += "📁 Local Models:\n"
                for model in local_models:
                    info += f"   - {model}\n"
            else:
                info += "📁 No local models found\n"
        
        return info
    
    def load_2pac_data(self):
        """Load and display 2Pac dataset info"""
        info = "📁 2Pac Training Dataset\n" + "="*40 + "\n\n"
        
        if os.path.exists(DATA):
            audio_files = []
            for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
                audio_files.extend(glob.glob(os.path.join(DATA, ext)))
            
            if audio_files:
                info += f"🎵 Found {len(audio_files)} audio files:\n\n"
                for i, file in enumerate(audio_files[:10]):  # Show first 10
                    filename = os.path.basename(file)
                    size = os.path.getsize(file) / (1024*1024)  # MB
                    info += f"   {i+1}. {filename} ({size:.1f} MB)\n"
                
                if len(audio_files) > 10:
                    info += f"   ... and {len(audio_files) - 10} more files\n"
                
                info += f"\n📊 Total dataset size: {sum(os.path.getsize(f) for f in audio_files) / (1024*1024):.1f} MB\n"
                info += "\n✅ Ready for voice cloning!\n"
            else:
                info += "❌ No audio files found in dataset\n"
                info += "Add 2Pac acapella files to get started\n"
        else:
            info += "❌ Dataset directory not found\n"
        
        self.dataset_info.insert('1.0', info)
        self.dataset_info.config(state=tk.DISABLED)

def main():
    root = tk.Tk()
    app = AIVocalsStudio(root)
    root.mainloop()

if __name__ == "__main__":
    main()
