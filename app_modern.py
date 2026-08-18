import os, threading, shutil, tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.ttk import Progressbar
from gtts import gTTS
from demucs import separate
from so_vits_svc_fork.inference.core import Svc
import librosa, numpy as np, soundfile as sf
from pydub import AudioSegment
from pydub.playback import play
import tempfile
import glob
import urllib.request
import datetime
import json
import time
from voice_trainer import VoiceTrainer
from precision_voice_cloning_system import PrecisionVoiceCloningSystem

BASE = os.path.expanduser("~/ai-vocals-studio")
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "dataset")
MODELS = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

PRETRAINED_MODELS = {
    "KIFM - Male Rapper": {
        "url": "https://github.com/33-Lab/so-vits-svc-4.0-model/releases/download/v4.0/G_4000.pth",
        "description": "Male rapper voice model",
        "filename": "kifm_male_rapper.pth"
    },
    "KIFM - Female Singer": {
        "url": "https://github.com/33-Lab/so-vits-svc-4.0-model/releases/download/v4.0/G_4000.pth",
        "description": "Female singer voice model",
        "filename": "kifm_female_singer.pth"
    },
    "DiffSVC - Default": {
        "url": "https://github.com/proceedingsoflaboratory/voice-conversion/releases/download/v1.0.0/DiffSVC_pretrained.pth",
        "description": "General purpose voice conversion",
        "filename": "diffsvc_default.pth"
    },
    "VITS - English": {
        "url": "https://github.com/jaywalnut310/vits/releases/download/v1.0/VITS.pth",
        "description": "English TTS model",
        "filename": "vits_english.pth"
    },
    "SO-VITS - Pop Male": {
        "url": "https://huggingface.co/SoVITS/pretrained/resolve/main/G_10000.pth",
        "description": "Pop style male voice",
        "filename": "sovits_pop_male.pth"
    },
    "SO-VITS - Pop Female": {
        "url": "https://huggingface.co/SoVITS/pretrained/resolve/main/G_10000.pth",
        "description": "Pop style female voice",
        "filename": "sovits_pop_female.pth"
    }
}

current_model_path = None
svc = None

def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    return output_file

def get_available_models():
    models = []
    for f in glob.glob(os.path.join(DATA, "*.pth")):
        models.append(("📁 Dataset: " + os.path.basename(f), f))
    for f in glob.glob(os.path.join(MODELS, "*.pth")):
        models.append(("📁 Models: " + os.path.basename(f), f))
    return models

def download_model(model_name, model_info, progress_callback=None):
    url = model_info["url"]
    filename = model_info["filename"]
    dest = os.path.join(MODELS, filename)
    
    try:
        def report_progress(block_num, block_size, total_size):
            if progress_callback and total_size > 0:
                downloaded = block_num * block_size
                percent = min(100, (downloaded / total_size) * 100)
                progress_callback(percent)
        
        urllib.request.urlretrieve(url, dest, reporthook=report_progress)
        return dest
    except Exception as e:
        print(f"Download error: {e}")
        return None

def load_model(model_path):
    global svc, current_model_path
    try:
        svc = Svc(model_path, device="cuda" if "cuda" in os.environ.get("DEVICE","") else "cpu")
        current_model_path = model_path
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

class ModernButton(tk.Button):
    def __init__(self, master, **kwargs):
        kwargs.setdefault('font', ('Segoe UI', 10, 'bold'))
        kwargs.setdefault('relief', 'flat')
        kwargs.setdefault('cursor', 'hand2')
        kwargs.setdefault('borderwidth', 0)
        super().__init__(master, **kwargs)
        self.bind('<Enter>', self.on_enter)
        self.bind('<Leave>', self.on_leave)
        self.original_bg = self.cget('bg')
        
    def on_enter(self, e):
        new_bg = self.lighten_color(self.original_bg, 20)
        self.config(bg=new_bg)
        
    def on_leave(self, e):
        self.config(bg=self.original_bg)
        
    def lighten_color(self, color, percent):
        color = color.lstrip('#')
        rgb = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        rgb = tuple(min(255, int(c + (255-c)*percent/100)) for c in rgb)
        return '#%02x%02x%02x' % rgb

class ModernFrame(tk.Frame):
    def __init__(self, master, **kwargs):
        kwargs.setdefault('bg', '#2b2b2b')
        kwargs.setdefault('relief', 'ridge')
        kwargs.setdefault('bd', 2)
        super().__init__(master, **kwargs)

class ProgressManager:
    def __init__(self, parent):
        self.parent = parent
        self.progress_window = None
        self.progress_bar = None
        self.status_label = None
        self.suggestion_label = None
        self.steps_label = None
        
    def show_progress(self, title="Processing...", message="Starting...", steps=None):
        if self.progress_window:
            self.close_progress()
        
        self.progress_window = tk.Toplevel(self.parent)
        self.progress_window.title(title)
        self.progress_window.geometry("500x250")
        self.progress_window.configure(bg='#2b2b2b')
        self.progress_window.resizable(False, False)
        
        # Center the window
        self.progress_window.transient(self.parent)
        self.progress_window.grab_set()
        
        # Title
        tk.Label(self.progress_window, text=title, 
                font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(pady=15)
        
        # Status
        self.status_label = tk.Label(self.progress_window, text=message, 
                                 font=('Segoe UI', 11), 
                                 fg='white', bg='#2b2b2b', wraplength=450)
        self.status_label.pack(pady=5)
        
        # Progress bar
        self.progress_bar = Progressbar(self.progress_window, length=400, mode='determinate')
        self.progress_bar.pack(pady=10)
        
        # Steps
        if steps:
            steps_text = "📍 Steps: " + " → ".join(steps)
            self.steps_label = tk.Label(self.progress_window, text=steps_text, 
                                     font=('Segoe UI', 9), 
                                     fg='#bdc3c7', bg='#2b2b2b', wraplength=450)
            self.steps_label.pack(pady=5)
        
        # Suggestions
        self.suggestion_label = tk.Label(self.progress_window, text="", 
                                      font=('Segoe UI', 10, 'italic'), 
                                      fg='#f39c12', bg='#2b2b2b', wraplength=450)
        self.suggestion_label.pack(pady=10)
        
        # Prevent closing
        self.progress_window.protocol("WM_DELETE_WINDOW", lambda: None)
        
    def update_progress(self, value, message=None, suggestion=None):
        if self.progress_bar:
            self.progress_bar['value'] = value
            if message:
                self.status_label.config(text=message)
            if suggestion:
                self.suggestion_label.config(text=f"💡 {suggestion}")
            self.progress_window.update()
            
    def close_progress(self):
        if self.progress_window:
            self.progress_window.destroy()
            self.progress_window = None
            self.progress_bar = None
            self.status_label = None
            self.suggestion_label = None
            self.steps_label = None

class ModernApp:
    def __init__(self, root):
        self.root = root
        self.root.title("🎤 AI Vocals Studio - Created by Steve B aka coden809")
        self.root.geometry("1200x800")
        self.root.configure(bg='#1a1a1a')
        
        # Initialize progress manager
        self.progress = ProgressManager(root)
        
        # Initialize precision voice cloning system
        self.precision_system = PrecisionVoiceCloningSystem(BASE, MODELS, OUT)
        
        # Variables
        self.audio_path = tk.StringVar()
        self.outname = tk.StringVar()
        self.output_format = tk.StringVar(value="wav")
        self.textbox = None
        self.speaker_name_var = tk.StringVar()
        self.model_name_var = tk.StringVar()
        self.download_var = tk.StringVar()
        self.model_var = tk.StringVar()
        
        # Progress tracking variables
        self.current_operation = tk.StringVar(value="Ready")
        self.operation_progress = tk.IntVar(value=0)
        
        # Create UI
        self.create_header()
        self.create_main_content()
        self.create_footer()
        
        # Show welcome guide on startup
        self.root.after(1000, self.show_welcome_guide)
        
        # Check setup status
        self.root.after(2000, self.check_setup_status)
        
        # Show precision system status
        self.root.after(3000, self.show_precision_system_status)

    def show_welcome_guide(self):
        """Show welcome message with next steps"""
        steps = [
            "1️⃣ Add training audio files",
            "2️⃣ Download or fine-tune a voice model", 
            "3️⃣ Generate amazing vocals!"
        ]
        
        self.progress.show_progress(
            title="🎤 Welcome to AI Vocals Studio!",
            message="Ready to create professional AI vocals! Here's how to get started:",
            steps=steps
        )
        self.progress.update_progress(0, suggestion="Start by adding your voice samples in the Training Data tab")
        
        # Auto-close after 5 seconds
        self.root.after(5000, self.progress.close_progress)
    
    def show_next_steps(self, current_step, suggestion):
        """Show contextual next steps"""
        next_steps = {
            "no_data": "📍 Next: Add audio files → Download model → Generate vocals",
            "has_data": "📍 Next: Download/fine-tune model → Load model → Generate vocals", 
            "has_model": "📍 Next: Load model → Add text/audio → Generate vocals",
            "ready": "🎉 All set! Start generating amazing vocals!"
        }
        
        message = next_steps.get(current_step, "Keep going!")
        self.status.config(text=f"💡 {message}", fg='#f39c12')
    
    def check_setup_status(self):
        """Check what user has completed and suggest next steps"""
        # Check training data
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        audio_count = 0
        for ext in audio_extensions:
            audio_count += len(glob.glob(os.path.join(DATA, ext)))
        
        # Check models
        model_count = len(glob.glob(os.path.join(MODELS, "*.pth")))
        
        if audio_count == 0:
            self.show_next_steps("no_data", "Import your voice samples first")
        elif model_count == 0:
            self.show_next_steps("has_data", "Download or fine-tune a voice model")
        else:
            self.show_next_steps("ready", "You're all set to generate vocals!")
        
        return audio_count, model_count
    
    def show_precision_system_status(self):
        """Show precision voice cloning system status"""
        try:
            status = self.precision_system.get_system_status()
            
            status_message = f"🚀 Precision System: {status['system_health'].upper()}"
            components_count = sum(status['components'].values())
            total_components = len(status['components'])
            
            if status['system_health'] == 'optimal':
                color = '#00ff88'
                detail = f"All {total_components} advanced components operational"
            elif status['system_health'] == 'operational':
                color = '#f39c12'
                detail = f"{components_count}/{total_components} components operational"
            else:
                color = '#e74c3c'
                detail = f"System degraded - only {components_count}/{total_components} components working"
            
            # Show status in a temporary label
            precision_status = tk.Label(self.root, text=f"{status_message} | {detail}", 
                                      font=('Segoe UI', 9), 
                                      fg=color, bg='#1a1a1a')
            precision_status.pack(side='bottom', pady=5)
            
            # Remove after 10 seconds
            self.root.after(10000, precision_status.destroy)
            
        except Exception as e:
            print(f"Error showing precision system status: {e}")

    def setup_styles(self):
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure('Dark.TFrame', background='#2b2b2b')
        style.configure('Dark.TLabel', background='#2b2b2b', foreground='white', font=('Segoe UI', 10))
        style.configure('Dark.TButton', background='#4a4a4a', foreground='white', font=('Segoe UI', 10, 'bold'))
        style.map('Dark.TButton', background=[('active', '#5a5a5a')])
        
    def create_header(self):
        header_frame = ModernFrame(self.root, bg='#1a1a1a', relief='flat', bd=0)
        header_frame.pack(fill='x', pady=10)
        
        # Logo and title row
        title_row = tk.Frame(header_frame, bg='#1a1a1a')
        title_row.pack()
        
        # App logo (using emoji as placeholder for now)
        logo_label = tk.Label(title_row, text="🎤", 
                           font=('Segoe UI', 28, 'bold'), 
                           fg='#00ff88', bg='#1a1a1a')
        logo_label.pack(side='left', padx=(0, 10))
        
        # App title
        title_label = tk.Label(title_row, text="AI VOCALS STUDIO", 
                              font=('Segoe UI', 24, 'bold'), 
                              fg='#00ff88', bg='#1a1a1a')
        title_label.pack(side='left')
        
        subtitle = tk.Label(header_frame, text="Advanced Voice Cloning & Generation", 
                           font=('Segoe UI', 12), 
                           fg='#888888', bg='#1a1a1a')
        subtitle.pack(pady=(5,0))
        
        # Creator branding
        creator_label = tk.Label(header_frame, text="Created by Steve B aka coden809", 
                             font=('Segoe UI', 9, 'italic'), 
                             fg='#666666', bg='#1a1a1a')
        creator_label.pack(pady=(2,0))
        
    def create_main_content(self):
        main_container = ModernFrame(self.root)
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        notebook = ttk.Notebook(main_container, style='Dark.TNotebook')
        notebook.pack(fill='both', expand=True)
        
        self.create_generation_tab(notebook)
        self.create_training_tab(notebook)
        self.create_models_tab(notebook)
        
    def create_generation_tab(self, notebook):
        gen_frame = ModernFrame(notebook)
        notebook.add(gen_frame, text='🎵 Generate')
        
        # Input Section
        input_frame = ModernFrame(gen_frame)
        input_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(input_frame, text="📝 INPUT", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        # Text Input
        tk.Label(input_frame, text="Lyrics / Text:", font=('Segoe UI', 10), 
                fg='white', bg='#2b2b2b').pack(anchor='w', pady=(10,5))
        
        self.textbox = tk.Text(input_frame, height=6, width=80, 
                               bg='#3a3a3a', fg='white', 
                               font=('Segoe UI', 10),
                               insertbackground='white',
                               relief='flat', bd=5)
        self.textbox.pack(fill='x', pady=5)
        
        # OR Separator
        separator_frame = tk.Frame(input_frame, bg='#2b2b2b')
        separator_frame.pack(fill='x', pady=10)
        tk.Label(separator_frame, text="─── OR ───", font=('Segoe UI', 12, 'bold'), 
                fg='#666666', bg='#2b2b2b').pack()
        
        # Audio Input
        self.audio_path = tk.StringVar()
        audio_btn = ModernButton(input_frame, text="🎵 Select Audio File", 
                                command=self.pick_audio,
                                bg='#4a90e2', fg='white')
        audio_btn.pack(anchor='w', pady=5)
        
        self.audio_label = tk.Label(input_frame, textvariable=self.audio_path, 
                                   font=('Segoe UI', 9), 
                                   fg='#aaaaaa', bg='#2b2b2b')
        self.audio_label.pack(anchor='w', pady=5)
        
        # Output Settings
        output_frame = ModernFrame(gen_frame)
        output_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(output_frame, text="⚙️ OUTPUT SETTINGS", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        # Output Name
        name_frame = tk.Frame(output_frame, bg='#2b2b2b')
        name_frame.pack(fill='x', pady=5)
        tk.Label(name_frame, text="Output Name:", font=('Segoe UI', 10), 
                fg='white', bg='#2b2b2b').pack(side='left')
        self.outname = tk.Entry(name_frame, width=40, bg='#3a3a3a', fg='white',
                               font=('Segoe UI', 10), insertbackground='white',
                               relief='flat', bd=2)
        self.outname.pack(side='left', padx=10)
        tk.Label(name_frame, text="(auto-generated if empty)", font=('Segoe UI', 9), 
                fg='#888888', bg='#2b2b2b').pack(side='left')
        
        # Format Selection
        format_frame = tk.Frame(output_frame, bg='#2b2b2b')
        format_frame.pack(fill='x', pady=5)
        tk.Label(format_frame, text="Format:", font=('Segoe UI', 10), 
                fg='white', bg='#2b2b2b').pack(side='left')
        
        self.output_format = tk.StringVar(value="wav")
        tk.Radiobutton(format_frame, text="WAV", variable=self.output_format, value="wav",
                      bg='#2b2b2b', fg='white', selectcolor='#4a4a4a',
                      font=('Segoe UI', 10)).pack(side='left', padx=10)
        tk.Radiobutton(format_frame, text="MP3", variable=self.output_format, value="mp3",
                      bg='#2b2b2b', fg='white', selectcolor='#4a4a4a',
                      font=('Segoe UI', 10)).pack(side='left')
        
        # Flow Alignment
        self.flow_align = tk.BooleanVar(value=True)
        tk.Checkbutton(output_frame, text="🎯 Enable Auto Rap Flow Alignment", 
                      variable=self.flow_align,
                      bg='#2b2b2b', fg='white', selectcolor='#4a4a4a',
                      font=('Segoe UI', 10, 'bold')).pack(anchor='w', pady=10)
        
        # Generate Button
        generate_btn = ModernButton(gen_frame, text="🚀 GENERATE VOCALS", 
                                  command=self.generate,
                                  bg='#00ff88', fg='#1a1a1a', 
                                  font=('Segoe UI', 16, 'bold'))
        generate_btn.pack(pady=20)
        
    def create_models_tab(self, notebook):
        models_frame = ModernFrame(notebook)
        notebook.add(models_frame, text='🤖 Models')
        
        # Model Loading Section
        load_frame = ModernFrame(models_frame)
        load_frame.pack(fill='x', padx=10, pady=10)
        
    tk.Label(load_frame, text="🤖 LOAD VOICE MODEL", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
    # Model selection
    model_frame = tk.Frame(load_frame, bg='#2b2b2b')
    model_frame.pack(fill='x', pady=10)
        
    tk.Label(model_frame, text="Select Model:", font=('Segoe UI', 11, 'bold'), 
            fg='white', bg='#2b2b2b').pack(side='left', padx=(0, 10))
        
    self.model_dropdown = ttk.Combobox(model_frame, textvariable=self.model_var, 
                                  values=[], state='readonly', width=30,
                                  style='Dark.TCombobox')
    self.model_dropdown.pack(side='left', padx=10)
        
    # Auto-load best model on startup
    self.root.after(500, self.auto_load_best_model)
        
    # Load button
    load_btn = ModernButton(model_frame, text="� Load Model", 
                        command=self.load_model,
                        bg='#3498db', fg='white')
    load_btn.pack(side='left', padx=10)
        
    # Model info
    self.model_info = tk.Label(load_frame, text="No model loaded", 
                           font=('Segoe UI', 10), 
                           fg='#888888', bg='#2b2b2b')
    self.model_info.pack(anchor='w', pady=5)
        
    def create_training_tab(self, notebook):
        train_frame = ModernFrame(notebook)
        notebook.add(train_frame, text='📚 Training Data')
        
        # Training Data Section
        data_frame = ModernFrame(train_frame)
        data_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(data_frame, text="📂 TRAINING DATA MANAGEMENT", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        # Import Buttons
        btn_frame = tk.Frame(data_frame, bg='#2b2b2b')
        btn_frame.pack(fill='x', pady=10)
        
        single_btn = ModernButton(btn_frame, text="📁 Import Single File", 
                                command=self.import_single_clip,
                                bg='#4a90e2', fg='white')
        single_btn.pack(side='left', padx=5)
        
        folder_btn = ModernButton(btn_frame, text="📁 Import Entire Folder", 
                                 command=self.import_folder,
                                 bg='#27ae60', fg='white')
        folder_btn.pack(side='left', padx=5)
        
        open_btn = ModernButton(btn_frame, text="📂 Open Data Folder", 
                              command=self.open_data_folder,
                              bg='#f39c12', fg='white')
        open_btn.pack(side='left', padx=5)
        
        # Data Count
        self.data_count = tk.Label(data_frame, text="Training clips: 0", 
                                  font=('Segoe UI', 12, 'bold'), 
                                  fg='#3498db', bg='#2b2b2b')
        self.data_count.pack(anchor='w', pady=5)
        
        # Voice Cloning Section
        clone_frame = ModernFrame(train_frame)
        clone_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(clone_frame, text="🎤 VOICE CLONING", font=('Segoe UI', 14, 'bold'), 
                fg='#e74c3c', bg='#2b2b2b').pack(anchor='w')
        
        tk.Label(clone_frame, text="Clone a voice from your training data!", 
                font=('Segoe UI', 10), 
                fg='#ecf0f1', bg='#2b2b2b').pack(anchor='w', pady=5)
        
        # Voice cloning inputs
        clone_input_frame = tk.Frame(clone_frame, bg='#2b2b2b')
        clone_input_frame.pack(fill='x', pady=10)
        
        tk.Label(clone_input_frame, text="Speaker Name:", font=('Segoe UI', 10), 
                fg='white', bg='#2b2b2b').pack(side='left')
        self.speaker_name_var = tk.StringVar()
        self.speaker_entry = tk.Entry(clone_input_frame, textvariable=self.speaker_name_var, 
                                     width=20, bg='#3a3a3a', fg='white',
                                     font=('Segoe UI', 10), insertbackground='white')
        self.speaker_entry.pack(side='left', padx=10)
        
        tk.Label(clone_input_frame, text="Model Name:", font=('Segoe UI', 10), 
                fg='white', bg='#2b2b2b').pack(side='left', padx=(20,0))
        self.model_name_var = tk.StringVar()
        self.model_entry = tk.Entry(clone_input_frame, textvariable=self.model_name_var, 
                                   width=20, bg='#3a3a3a', fg='white',
                                   font=('Segoe UI', 10), insertbackground='white')
        self.model_entry.pack(side='left', padx=10)
        
        # Clone button
        clone_btn = ModernButton(clone_frame, text="🎯 START VOICE CLONING", 
                               command=self.start_voice_cloning,
                               bg='#e74c3c', fg='white', 
                               font=('Segoe UI', 12, 'bold'))
        clone_btn.pack(pady=5)
        
        # Fine-tune button (using pre-trained base)
        finetune_btn = ModernButton(clone_frame, text="🔧 FINE-TUNE FROM PRE-TRAINED", 
                                  command=self.start_fine_tuning,
                                  bg='#9b59b6', fg='white', 
                                  font=('Segoe UI', 11, 'bold'))
        finetune_btn.pack(pady=5)
        
        # Precision cloning button
        precision_btn = ModernButton(clone_frame, text="🚀 PRECISION VOICE CLONING", 
                                   command=self.start_precision_cloning,
                                   bg='#00ff88', fg='#1a1a1a', 
                                   font=('Segoe UI', 13, 'bold'))
        precision_btn.pack(pady=5)
        
        tk.Label(clone_frame, text="Precision: Uses advanced AI for maximum accuracy", 
                font=('Segoe UI', 9), 
                fg='#00ff88', bg='#2b2b2b').pack()
        
        # Training progress
        self.training_status = tk.Label(clone_frame, text="", 
                                      font=('Segoe UI', 10), 
                                      fg='#f39c12', bg='#2b2b2b')
        self.training_status.pack(anchor='w', pady=5)
        
        # Supported Formats
        formats_frame = ModernFrame(train_frame)
        formats_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(formats_frame, text="🎵 SUPPORTED AUDIO FORMATS", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        formats_text = "WAV, MP3, FLAC, OGG, M4A, AIFF, AAC"
        tk.Label(formats_frame, text=formats_text, font=('Segoe UI', 11), 
                fg='#ecf0f1', bg='#2b2b2b').pack(anchor='w', pady=5)
        
    def create_models_tab(self, notebook):
        models_frame = ModernFrame(notebook)
        notebook.add(models_frame, text='🤖 Models')
        
        # Model Management
        model_mgmt_frame = ModernFrame(models_frame)
        model_mgmt_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(model_mgmt_frame, text="🤖 MODEL MANAGEMENT", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        # Import Model Buttons
        import_frame = tk.Frame(model_mgmt_frame, bg='#2b2b2b')
        import_frame.pack(fill='x', pady=10)
        
        import_btn = ModernButton(import_frame, text="📥 Import Model File (.pth)", 
                                command=self.import_model,
                                bg='#9b59b6', fg='white')
        import_btn.pack(side='left', padx=5)
        
        import_folder_btn = ModernButton(import_frame, text="📥 Import Model Folder", 
                                        command=self.import_model_folder,
                                        bg='#8e44ad', fg='white')
        import_folder_btn.pack(side='left', padx=5)
        
        # Model Selection
        select_frame = ModernFrame(models_frame)
        select_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(select_frame, text="🎯 SELECT MODEL", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        self.model_var = tk.StringVar()
        self.model_dropdown = ttk.Combobox(select_frame, textvariable=self.model_var, 
                                          state='readonly', width=50)
        self.model_dropdown.pack(fill='x', pady=5)
        
        load_btn = ModernButton(select_frame, text="⚡ Load Selected Model", 
                              command=self.load_selected_model,
                              bg='#e74c3c', fg='white')
        load_btn.pack(pady=5)
        
        # Current Model Status
        self.current_model_label = tk.Label(select_frame, text="No model loaded", 
                                           font=('Segoe UI', 11, 'bold'), 
                                           fg='#e74c3c', bg='#2b2b2b')
        self.current_model_label.pack(anchor='w', pady=10)
        
        # Download Models
        download_frame = ModernFrame(models_frame)
        download_frame.pack(fill='x', padx=10, pady=10)
        
        tk.Label(download_frame, text="⬇️ DOWNLOAD PRE-TRAINED MODELS", font=('Segoe UI', 14, 'bold'), 
                fg='#00ff88', bg='#2b2b2b').pack(anchor='w')
        
        self.download_var = tk.StringVar()
        self.download_dropdown = ttk.Combobox(download_frame, textvariable=self.download_var, 
                                            state='readonly', width=50)
        self.download_dropdown.pack(fill='x', pady=5)
        
        download_btn = ModernButton(download_frame, text="📥 Download Selected Model", 
                                 command=self.download_pretrained,
                                 bg='#3498db', fg='white')
        download_btn.pack(pady=5)
        
        self.download_progress = tk.Label(download_frame, text="", 
                                        font=('Segoe UI', 10), 
                                        fg='#3498db', bg='#2b2b2b')
        self.download_progress.pack(anchor='w', pady=5)
        
        # Populate download dropdown
        self.download_dropdown['values'] = list(PRETRAINED_MODELS.keys())
        if PRETRAINED_MODELS:
            self.download_var.set(list(PRETRAINED_MODELS.keys())[0])
            
    def refresh_model_list(self):
        models = get_available_models()
        model_names = [name for name, path in models] if models else ["No models available"]
        self.model_dropdown['values'] = model_names
        if model_names:
            self.model_var.set(model_names[0])
            
    def download_pretrained(self):
        model_name = self.download_var.get()
        if not model_name or model_name == "Select model to download...":
            messagebox.showwarning("Warning", "Please select a model to download first.")
            return
        
        if model_name not in PRETRAINED_MODELS:
            messagebox.showwarning("Warning", "Invalid model selection.")
            return
        
        model_info = PRETRAINED_MODELS[model_name]
        
        steps = [
            "📥 Connecting to download server",
            "⬇️ Downloading model file", 
            "💾 Saving to models folder",
            "🔄 Refreshing model list",
            "✅ Ready to use!"
        ]
        
        self.progress.show_progress(
            title=f"📥 Downloading {model_name}",
            message=f"Downloading {model_info['description']}...",
            steps=steps
        )
        
        threading.Thread(target=self._download_thread_with_progress, args=(model_name, model_info), daemon=True).start()
        
    def _download_thread_with_progress(self, model_name, model_info):
        try:
            self.progress.update_progress(10, "📥 Connecting to download server...", 
                                     suggestion="This may take a moment depending on your internet speed")
            
            dest = download_model(model_name, model_info, self._update_download_progress)
            
            if dest and os.path.exists(dest):
                self.progress.update_progress(90, "💾 Saving model to models folder...", 
                                         suggestion="Model downloaded successfully! Refreshing available models...")
                
                self.refresh_model_list()
                
                self.progress.update_progress(100, f"✅ {model_name} downloaded successfully!", 
                                         suggestion="You can now load this model in the Models tab and start generating vocals!")
                
                messagebox.showinfo("Success", f"Model '{model_name}' downloaded successfully!")
                
                # Auto-close after 3 seconds
                self.root.after(3000, self.progress.close_progress)
                
                # Update status
                self.check_setup_status()
                
            else:
                self.progress.update_progress(0, "❌ Download failed", 
                                         suggestion="Check your internet connection and try again")
                messagebox.showerror("Error", "Failed to download model.")
                self.progress.close_progress()
                
        except Exception as e:
            self.progress.update_progress(0, f"❌ Error: {str(e)[:30]}...", 
                                     suggestion="Make sure you have internet connection and try again")
            self.progress.close_progress()
            
    def _update_download_progress(self, percent):
        progress_value = 20 + (percent * 0.6)  # Scale to 20-80% range
        self.progress.update_progress(progress_value, 
                                 f"⬇️ Downloading... {percent:.1f}%", 
                                 suggestion="Large models may take several minutes to download")
        
    def _download_thread(self, model_name, model_info):
        try:
            dest = download_model(model_name, model_info, self.update_download_progress)
            if dest and os.path.exists(dest):
                self.download_progress.config(text=f"✅ Downloaded: {model_info['filename']}", fg='#27ae60')
                messagebox.showinfo("Success", f"Model '{model_name}' downloaded successfully!")
                self.refresh_model_list()
            else:
                self.download_progress.config(text="❌ Download failed", fg='#e74c3c')
                messagebox.showerror("Error", "Failed to download model.")
        except Exception as e:
            self.download_progress.config(text=f"❌ Error: {str(e)[:30]}", fg='#e74c3c')
            
    def update_download_progress(self, percent):
        self.download_progress.config(text=f"📥 Downloading... {percent:.1f}%", fg='#3498db')
        
    def pick_audio(self):
        filetypes = [
            ("Audio Files", "*.wav *.mp3 *.flac *.ogg *.m4a *.aiff *.aac"),
            ("WAV Files", "*.wav"),
            ("MP3 Files", "*.mp3"),
            ("M4A Files", "*.m4a"),
            ("FLAC Files", "*.flac"),
            ("All Files", "*.*")
        ]
        
    if file_path:
        self.update_operation_progress("📁 Copying audio file...", 50, "Copying to dataset folder")
        
        filename = os.path.basename(file_path)
        dest = os.path.join(DATA, filename)
        
        # Handle duplicates
        counter = 1
        base_name, ext = os.path.splitext(filename)
        while os.path.exists(dest):
            new_name = f"{base_name}_{counter}{ext}"
            dest = os.path.join(DATA, new_name)
            counter += 1
        
        shutil.copy2(file_path, dest)
        self.refresh_data_count()
            
    def open_data_folder(self):
        os.system(f"xdg-open {DATA}")
        
    def import_model(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if f:
            dest = os.path.join(DATA, "target_model.pth")
            shutil.copy(f, dest)
            if load_model(dest):
                self.current_model_label.config(text=f"✅ Loaded: {os.path.basename(f)}", fg='#27ae60')
                messagebox.showinfo("✅ Model Imported", "Target model imported and loaded successfully.")
            else:
                messagebox.showerror("❌ Error", "Failed to load model.")
            self.refresh_model_list()
            
    def import_model_folder(self):
        folder = filedialog.askdirectory(title="Select folder containing .pth model files")
        if folder:
            count = 0
            for f in glob.glob(os.path.join(folder, "*.pth")):
                try:
                    shutil.copy(f, MODELS)
                    count += 1
                except:
                    pass
            messagebox.showinfo("✅ Imported", f"{count} model(s) imported to models folder.")
            self.refresh_model_list()
            
    def auto_load_best_model(self):
        """Auto-load the best available model on startup"""
        self.update_operation_progress("🔍 Scanning for models...", 30, "Looking for available voice models")
        
        models = get_available_models()
        
        if not models:
            self.update_operation_progress("❌ No models found", 0, "Download or create a model first")
            return
        
        # Priority: 2Pac models first, then custom, then pre-trained
        priority_models = []
        
        # Check for 2Pac models specifically
        for model in models:
            if "2pac" in model.lower():
                priority_models.insert(0, model)
        
        # Check for other custom/enhanced models
        for model in models:
            if ("custom" in model.lower() or "enhanced" in model.lower()) and model not in priority_models:
                priority_models.append(model)
        
        # Add remaining models
        for model in models:
            if model not in priority_models:
                priority_models.append(model)
        
        # Select best model (prioritize 2Pac)
        best_model = "2pac_enhanced_voice" if "2pac_enhanced_voice" in models else (
                    "2pac_custom_voice" if "2pac_custom_voice" in models else (
                        priority_models[0] if priority_models else models[0]
                    )
                )
        
        self.update_operation_progress(f"🎯 Auto-loading 2Pac: {best_model}", 70, "Loading 2Pac voice model")
        
        # Set the model in dropdown
        self.model_var.set(best_model)
        
        # Auto-load the model
        self.load_selected_model()
        
        self.update_operation_progress(f"✅ 2Pac model loaded: {best_model}!", 100, "Ready to generate 2Pac vocals")
        
        # Reset progress after 3 seconds
        self.root.after(3000, lambda: self.update_operation_progress("🎤 2Pac Ready - Generate Vocals!", 0))
    
    def load_selected_model(self):
        selected = self.model_var.get()
        models = get_available_models()
        for name, path in models:
            if name == selected:
                if load_model(path):
                    self.current_model_label.config(text=f"✅ Loaded: {os.path.basename(path)}", fg='#27ae60')
                    messagebox.showinfo("✅ Model Loaded", f"Model '{os.path.basename(path)}' loaded successfully.")
                else:
                    messagebox.showerror("❌ Error", "Failed to load model.")
                return
        messagebox.showwarning("⚠️ Warning", "Please select a model first.")
        
    def run_generate(self):
        threading.Thread(target=self.generate, daemon=True).start()
        
    def enhance_prosody(self, y, sr):
        y_stretch = librosa.effects.time_stretch(y, rate=1.0 + np.random.uniform(-0.02,0.02))
        y_shift = librosa.effects.pitch_shift(y_stretch, sr, n_steps=np.random.uniform(-0.3,0.3))
        return y_shift
        
    def start_fine_tuning(self):
        speaker_name = self.speaker_name_var.get().strip()
        model_name = self.model_name_var.get().strip()
        
        if not speaker_name or not model_name:
            messagebox.showwarning("⚠️ Missing Information", "Please enter both speaker name and model name.")
            return
        
        # Check if we have training data
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        audio_count = 0
        for ext in audio_extensions:
            audio_count += len(glob.glob(os.path.join(DATA, ext)))
        
        if audio_count == 0:
            messagebox.showwarning("⚠️ No Training Data", "Please import audio files first!")
            return
        
        if audio_count < 5:
            if not messagebox.askyesno("⚠️ Limited Data", 
                                      f"Only {audio_count} audio files found. "
                                      "For fine-tuning, 5-20 files is good. Continue anyway?"):
                return
        
        # Check for available pre-trained models
        def get_available_models():
            """Get list of available models"""
            models = []
            
            # Check models directory for .pth files
            if os.path.exists(MODELS):
                for file in os.listdir(MODELS):
                    if file.endswith('.pth'):
                        models.append(file.replace('.pth', ''))
            
            # Check for custom models in subdirectories
            if os.path.exists(MODELS):
                for subdir in os.listdir(MODELS):
                    subdir_path = os.path.join(MODELS, subdir)
                    if os.path.isdir(subdir_path):
                        # Check for model.pth in subdirectory
                        model_file = os.path.join(subdir_path, 'model.pth')
                        if os.path.exists(model_file):
                            models.append(subdir)
            
            # Check for enhanced models
            if os.path.exists(MODELS):
                for subdir in os.listdir(MODELS):
                    subdir_path = os.path.join(MODELS, subdir)
                    if os.path.isdir(subdir_path):
                        enhanced_file = os.path.join(subdir_path, 'enhanced_model.json')
                        if os.path.exists(enhanced_file):
                            models.append(subdir)
            
            # Check data directory for models
            if os.path.exists(DATA):
                for file in os.listdir(DATA):
                    if file.endswith('.pth'):
                        models.append(file.replace('.pth', ''))
            
            return sorted(list(set(models)))  # Remove duplicates

        available_models = get_available_models()
        if not available_models:
            messagebox.showwarning("⚠️ No Pre-trained Models", 
                                "Please download a pre-trained model first!\n"
                                "Go to Models tab → Download a model → Try fine-tuning again.")
            return
        
        # Let user select base model
        model_names = [name for name, path in available_models]
        selected_model = None
        
        # Simple selection dialog
        selection_window = tk.Toplevel(self.root)
        selection_window.title("Select Base Model")
        selection_window.geometry("400x300")
        selection_window.configure(bg='#2b2b2b')
        
        tk.Label(selection_window, text="🔧 Select Base Model for Fine-Tuning", 
                font=('Segoe UI', 12, 'bold'), fg='#00ff88', bg='#2b2b2b').pack(pady=20)
        
        model_var = tk.StringVar(value=model_names[0])
        for name in model_names:
            tk.Radiobutton(selection_window, text=name, variable=model_var, value=name,
                         bg='#2b2b2b', fg='white', selectcolor='#4a4a4a',
                         font=('Segoe UI', 10)).pack(pady=5)
        
        def confirm_selection():
            nonlocal selected_model
            selected_name = model_var.get()
            for name, path in available_models:
                if name == selected_name:
                    selected_model = path
                    break
            selection_window.destroy()
        
        def cancel_selection():
            selection_window.destroy()
        
        btn_frame = tk.Frame(selection_window, bg='#2b2b2b')
        btn_frame.pack(pady=20)
        
        tk.Button(btn_frame, text="✅ Confirm", command=confirm_selection,
                 bg='#27ae60', fg='white', font=('Segoe UI', 10, 'bold')).pack(side='left', padx=5)
        tk.Button(btn_frame, text="❌ Cancel", command=cancel_selection,
                 bg='#e74c3c', fg='white', font=('Segoe UI', 10, 'bold')).pack(side='left', padx=5)
        
        selection_window.wait_window()
        
        if not selected_model:
            return
        
        # Start fine-tuning in background thread
        self.training_status.config(text="🔄 Starting fine-tuning...", fg='#f39c12')
        threading.Thread(target=self._fine_tuning_thread, 
                        args=(speaker_name, model_name, selected_model), 
                        daemon=True).start()
    
    def _fine_tuning_thread(self, speaker_name, model_name, base_model_path):
        try:
            steps = [
                "🎤 Preparing your voice data",
                "🔧 Setting up fine-tuning config",
                "📊 Creating training lists",
                "🧠 Training AI model",
                "💾 Saving your custom model",
                "✅ Ready to generate!"
            ]
            
            self.progress.show_progress(
                title=f"🔧 Fine-tuning {speaker_name}'s Voice",
                message=f"Creating custom voice model from {os.path.basename(base_model_path)}...",
                steps=steps
            )
            
            self.progress.update_progress(5, "🎤 Preparing your voice data...", 
                                     suggestion="Analyzing and processing your audio files for training")
            
            # Initialize trainer with fine-tuning
            trainer = VoiceTrainer(DATA, MODELS)
            
            self.progress.update_progress(20, "🔧 Setting up fine-tuning configuration...", 
                                     suggestion="Optimizing settings for voice quality and training speed")
            
            # Start fine-tuning process
            model_path = trainer.fine_tune_model(speaker_name, model_name, base_model_path)
            
            self.progress.update_progress(90, "💾 Saving your custom model...", 
                                     suggestion="Almost done! Your voice model is being finalized")
            
            # Update UI
            self.progress.update_progress(100, f"✅ Fine-tuning complete! Model: {model_name}", 
                                     suggestion=f"Go to Models tab to load '{model_name}' and start generating vocals in {speaker_name}'s voice!")
            
            messagebox.showinfo("🎉 Fine-tuning Success!", 
                              f"Voice fine-tuning completed!\n\n"
                              f"🎤 Speaker: {speaker_name}\n"
                              f"🤖 Model: {model_name}\n"
                              f"🎯 Base: {os.path.basename(base_model_path)}\n"
                              f"📁 Location: {model_path}\n\n"
                              f"Next: Load this model in Models tab → Generate vocals!")
            
            # Auto-close after 3 seconds
            self.root.after(3000, self.progress.close_progress)
            
            # Refresh model list
            self.refresh_model_list()
            
            # Update status
            self.check_setup_status()
            
        except Exception as e:
            error_msg = str(e)
            self.progress.update_progress(0, f"❌ Fine-tuning failed: {error_msg[:50]}...", 
                                     suggestion="Check your audio files quality and try again")
            self.progress.close_progress()
            messagebox.showerror("❌ Fine-tuning Failed", 
                               f"Failed to fine-tune voice:\n{error_msg}")
    
    def start_voice_cloning(self):
        speaker_name = self.speaker_name_var.get().strip()
        model_name = self.model_name_var.get().strip()
        
        if not speaker_name or not model_name:
            messagebox.showwarning("⚠️ Missing Information", "Please enter both speaker name and model name.")
            return
        
        # Check if we have training data
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        audio_count = 0
        for ext in audio_extensions:
            audio_count += len(glob.glob(os.path.join(DATA, ext)))
        
        if audio_count == 0:
            messagebox.showwarning("⚠️ No Training Data", "Please import audio files first!")
            return
        
        if audio_count < 10:
            if not messagebox.askyesno("⚠️ Limited Data", 
                                      f"Only {audio_count} audio files found. "
                                      "For best results, use at least 10-50 files. Continue anyway?"):
                return
        
        # Start training in background thread
        self.training_status.config(text="🔄 Starting voice cloning...", fg='#f39c12')
        threading.Thread(target=self._voice_cloning_thread, 
                        args=(speaker_name, model_name), 
                        daemon=True).start()
    
    def _voice_cloning_thread(self, speaker_name, model_name):
        try:
            self.training_status.config(text="🎤 Preparing training data...", fg='#3498db')
            
            # Initialize trainer
            trainer = VoiceTrainer(DATA, MODELS)
            
            # Start cloning process
            model_path = trainer.clone_voice(speaker_name, model_name)
            
            # Update UI
            self.training_status.config(text=f"✅ Voice cloning complete! Model: {model_name}", fg='#27ae60')
            messagebox.showinfo("🎉 Success!", 
                              f"Voice cloning completed!\n\n"
                              f"Speaker: {speaker_name}\n"
                              f"Model: {model_name}\n"
                              f"Location: {model_path}\n\n"
                              f"You can now load this model in the Models tab!")
            
            # Refresh model list
            self.refresh_model_list()
            
        except Exception as e:
            error_msg = str(e)
            self.training_status.config(text=f"❌ Cloning failed: {error_msg[:50]}...", fg='#e74c3c')
    
    def start_precision_cloning(self):
        speaker_name = self.speaker_name_var.get().strip()
        model_name = self.model_name_var.get().strip()
        
        if not speaker_name or not model_name:
            messagebox.showwarning("⚠️ Missing Information", "Please enter both speaker name and model name.")
            return
        
        # Check if we have training data
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        audio_count = 0
        for ext in audio_extensions:
            audio_count += len(glob.glob(os.path.join(DATA, ext)))
        
        if audio_count == 0:
            messagebox.showwarning("⚠️ No Training Data", "Please import audio files first!")
            return
        
        if audio_count < 10:
            if not messagebox.askyesno("⚠️ Limited Data", 
                                      f"Only {audio_count} audio files found. "
                                      "For best precision results, use at least 10-50 files. Continue anyway?"):
                return
        
        # Start precision cloning in background thread
        self.training_status.config(text="🚀 Starting precision voice cloning...", fg='#00ff88')
        threading.Thread(target=self._precision_cloning_thread, 
                        args=(speaker_name, model_name), 
                        daemon=True).start()
    
    def _precision_cloning_thread(self, speaker_name, model_name):
        try:
            steps = [
                "🎤 Analyzing voice characteristics",
                "🔧 Applying advanced preprocessing",
                "📊 Extracting comprehensive features",
                "🎨 Applying data augmentation",
                "🧠 Running advanced AI training",
                "✅ Validating model quality",
                "🎯 Finalizing precision clone"
            ]
            
            self.progress.show_progress(
                title=f"🚀 Precision Voice Cloning - {speaker_name}",
                message="Using advanced AI for maximum voice cloning accuracy...",
                steps=steps
            )
            
            def progress_callback(message, percent):
                self.progress.update_progress(percent, message)
            
            # Run precision cloning
            result = self.precision_system.precision_clone_voice(
                speaker_name, 
                model_name, 
                progress_callback
            )
            
            if result['status'] == 'success':
                self.progress.update_progress(100, f"✅ Precision cloning complete!", 
                                         suggestion=f"Model '{model_name}' created with {result['quality_report'].get('average_similarity', 0):.1%} similarity")
                
                messagebox.showinfo("🎉 Precision Cloning Success!", 
                                  f"Advanced voice cloning completed!\n\n"
                                  f"🎤 Speaker: {speaker_name}\n"
                                  f"🤖 Model: {model_name}\n"
                                  f"📁 Location: {result['model_path']}\n"
                                  f"📊 Quality: {result['quality_report'].get('status', 'unknown')}\n"
                                  f"🎯 Similarity: {result['quality_report'].get('average_similarity', 0):.1%}\n\n"
                                  f"Next: Load this model in Models tab → Generate vocals!")
                
                # Auto-close after 5 seconds
                self.root.after(5000, self.progress.close_progress)
                
                # Refresh model list
                self.refresh_model_list()
                
                # Update status
                self.check_setup_status()
                
            else:
                self.progress.update_progress(0, f"❌ Precision cloning failed", 
                                         suggestion=result.get('error', 'Unknown error'))
                messagebox.showerror("❌ Precision Cloning Failed", 
                                   f"Failed to create precision voice clone:\n{result.get('error', 'Unknown error')}")
                self.progress.close_progress()
                
        except Exception as e:
            error_msg = str(e)
            self.progress.update_progress(0, f"❌ Error: {error_msg[:50]}...", 
                                     suggestion="Check system requirements and try again")
            self.progress.close_progress()
            messagebox.showerror("❌ Error", f"Precision cloning error:\n{error_msg}")
    
    def generate(self):
        try:
            steps = [
                "📝 Processing your input",
                "🎵 Converting to audio format", 
                "🎤 Running AI voice generation",
                "💾 Saving final output",
                "✅ Ready to listen!"
            ]
            
            self.progress.show_progress(
                title="🎤 Generating AI Vocals",
                message="Creating amazing vocals with AI...",
                steps=steps
            )
            
            self.progress.update_progress(5, "📝 Processing your input...", 
                                     suggestion="Preparing your text or audio for voice generation")
            
            name = self.outname.get().strip()
            if not name:
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"ai_vocal_{timestamp}"
            
            output_format = self.output_format.get()
            
            if output_format == "mp3":
                outfile = os.path.join(OUT, f"{name}_vocal.mp3")
                outwav = os.path.join(OUT, f"{name}_vocal_temp.wav")
            else:
                outfile = os.path.join(OUT, f"{name}_vocal.wav")
                outwav = outfile

            if svc is None:
                self.progress.update_progress(0, "❌ No model loaded!", 
                                         suggestion="Go to Models tab → Download/load a voice model first")
                messagebox.showerror("❌ Error","Target model not loaded! Import or select a model first.")
                self.progress.close_progress()
                return

            self.progress.update_progress(15, "🎵 Converting to audio format...", 
                                     suggestion="Making sure your audio is in the right format for AI processing")

            if self.audio_path.get():
                self.progress.update_progress(25, "🎵 Separating vocals from audio...", 
                                         suggestion="Extracting clean vocals from your audio file")
                separated = separate.separate_file(self.audio_path.get(), outdir=OUT, model="demucs", splits=False)
                vocal_file = next((f for f in separated if "vocals" in f.lower()), None)
                if vocal_file is None:
                    self.progress.update_progress(0, "❌ No vocals detected!", 
                                             suggestion="Try using a clearer audio file with vocals")
                    messagebox.showerror("❌ Error","No vocals detected.")
                    self.progress.close_progress()
                    return
            else:
                text = self.textbox.get("1.0", "end").strip()
                if not text:
                    self.progress.update_progress(0, "❌ No input provided!", 
                                             suggestion="Enter some text or select an audio file")
                    messagebox.showerror("❌ Error","Provide text or audio input.")
                    self.progress.close_progress()
                    return
                    
                self.progress.update_progress(30, "🗣️ Generating speech from text...", 
                                         suggestion="Converting your text to speech using advanced TTS")
                temp_file = os.path.join(OUT, f"{name}_temp.mp3")
                text_to_speech(text, temp_file)
                wav_file = temp_file.replace('.mp3', '.wav')
                audio = AudioSegment.from_mp3(temp_file)
                audio.export(wav_file, format="wav")
                vocal_file = wav_file
                
                if self.flow_align.get():
                    self.progress.update_progress(40, "🎯 Enhancing prosody and flow...", 
                                             suggestion="Improving rhythm and timing for better vocal quality")
                    y, sr = librosa.load(vocal_file, sr=None)
                    y = self.enhance_prosody(y, sr)
                    sf.write(vocal_file, y, sr)

            self.progress.update_progress(60, "🤖 Running AI voice generation...", 
                                     suggestion="This is where the magic happens! AI is creating your custom vocals")
            
            # Run inference
            svc.infer(vocal_file, outwav, f0_method="pm")
            
            self.progress.update_progress(85, "💾 Saving final output...", 
                                     suggestion="Almost done! Saving your generated vocals")
            
            # Convert to MP3 if requested
            if output_format == "mp3":
                audio = AudioSegment.from_wav(outwav)
                audio.export(outfile, format="mp3")
                os.remove(outwav)
            
            self.progress.update_progress(100, f"✅ Vocals generated successfully!", 
                                     suggestion=f"Your vocals are saved as '{os.path.basename(outfile)}'. Click below to open the folder!")
            
            messagebox.showinfo("🎉 Success", f"Vocals generated!\n\nOutput: {outfile}")
            
            # Auto-close after 3 seconds
            self.root.after(3000, self.progress.close_progress)
            
            # Ask if user wants to open the output folder
            if messagebox.askyesno("📂 Open Output Folder?", "Open the output folder to listen to your vocals?"):
                os.system(f"xdg-open {OUT}")
                
        except Exception as e:
            self.progress.update_progress(0, f"❌ Generation failed: {str(e)[:50]}...", 
                                     suggestion="Check your model and input files, then try again")
            self.progress.close_progress()
            messagebox.showerror("❌ Error", str(e))

if __name__ == "__main__":
    root = tk.Tk()
    app = ModernApp(root)
    root.mainloop()
