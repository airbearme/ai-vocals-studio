import os, threading, shutil, tkinter as tk
from tkinter import filedialog, messagebox
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

BASE = os.path.expanduser("~/ai-vocals-studio")
OUT = os.path.join(BASE, "outputs")
DATA = os.path.join(BASE, "dataset")
MODELS = os.path.join(BASE, "models")
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Pre-trained models available for download
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

# Output format selection
OUTPUT_FORMAT = "wav"  # or "mp3"

# Load TTS
def text_to_speech(text, output_file):
    tts = gTTS(text=text, lang='en', slow=False)
    tts.save(output_file)
    return output_file

# Find all available models
def get_available_models():
    models = []
    # Check dataset folder
    for f in glob.glob(os.path.join(DATA, "*.pth")):
        models.append(("📁 Dataset: " + os.path.basename(f), f))
    # Check models folder
    for f in glob.glob(os.path.join(MODELS, "*.pth")):
        models.append(("📁 Models: " + os.path.basename(f), f))
    return models

# Download a pretrained model
def download_model(model_name, model_info, progress_callback=None):
    url = model_info["url"]
    filename = model_info["filename"]
    dest = os.path.join(MODELS, filename)
    
    try:
        # Download with progress
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

# Current loaded model
current_model_path = None
svc = None

def load_model(model_path):
    global svc, current_model_path
    try:
        svc = Svc(model_path, device="cuda" if "cuda" in os.environ.get("DEVICE","") else "cpu")
        current_model_path = model_path
        return True
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

class ModelManager:
    def __init__(self):
        self.available_models = []
        self.loaded_model = None
        self.refresh_models()
    
    def refresh_models(self):
        self.available_models = get_available_models()
        if not self.available_models:
            self.available_models = [("No models available", None)]
    
    def get_model_info(self, model_name):
        for name, path in self.available_models:
            if name == model_name:
                return path
        return None
    
    def load_model(self, model_name):
        model_path = self.get_model_info(model_name)
        if model_path:
            return load_model(model_path)
        return False
    
    def get_model_metadata(self, model_name):
        for name, path in self.available_models:
            if name == model_name:
                if path:
                    return {
                        "name": name,
                        "path": path,
                        "size": os.path.getsize(path) if os.path.exists(path) else 0,
                        "description": self._get_model_description(name)
                    }
        return None
    
    def _get_model_description(self, model_name):
        for name, info in PRETRAINED_MODELS.items():
            if name == model_name:
                return info["description"]
        return "Custom model"

class App:
    def __init__(self, root):
        root.title("AI Vocals Studio Plug-n-Play")
        
        # Lyrics/Text input
        tk.Label(root, text="Lyrics / Text").pack()
        self.textbox = tk.Text(root, height=8, width=60)
        self.textbox.pack(padx=5, pady=5)

        # Audio input selection
        self.audio_path = tk.StringVar()
        tk.Button(root, text="Select Audio Input", command=self.pick_audio).pack(pady=4)
        tk.Label(root, textvariable=self.audio_path).pack()

        # Output name
        tk.Label(root, text="Output Name (leave empty for auto-generated)").pack()
        self.outname = tk.Entry(root, width=40)
        self.outname.insert(0, "")
        self.outname.pack(pady=4)

        # Output format selection
        tk.Label(root, text="Output Format:").pack()
        self.output_format = tk.StringVar(value="wav")
        format_frame = tk.Frame(root)
        format_frame.pack(pady=4)
        tk.Radiobutton(format_frame, text="WAV", variable=self.output_format, value="wav").pack(side=tk.LEFT, padx=5)
        tk.Radiobutton(format_frame, text="MP3", variable=self.output_format, value="mp3").pack(side=tk.LEFT, padx=5)

        # Flow alignment option
        self.flow_align = tk.BooleanVar(value=True)
        tk.Checkbutton(root, text="Enable Auto Rap Flow (text only)", variable=self.flow_align).pack(pady=4)

        # Separator
        tk.Frame(root, height=2, bg="gray").pack(fill='x', pady=10)

        # Training Data section
        tk.Label(root, text="=== TRAINING DATA ===", font=("Arial", 10, "bold")).pack(pady=5)
        
        tk.Button(root, text="Import Single Audio File", command=self.import_single_clip).pack(pady=2)
        tk.Button(root, text="📂 Import Entire Folder of Audio Files", command=self.import_folder, bg="#4CAF50", fg="white").pack(pady=4)
        tk.Button(root, text="Open Training Data Folder", command=self.open_data_folder).pack(pady=2)
        
        # Show count of training files
        self.data_count = tk.Label(root, text="Training clips: 0", fg="blue")
        self.data_count.pack(pady=2)
        self.update_data_count()

        # Separator
        tk.Frame(root, height=2, bg="gray").pack(fill='x', pady=10)

        # Model section
        tk.Label(root, text="=== MODEL SELECTION ===", font=("Arial", 10, "bold")).pack(pady=5)
        
        tk.Button(root, text="Import Model File (.pth)", command=self.import_model).pack(pady=2)
        tk.Button(root, text="Import Model Folder", command=self.import_model_folder).pack(pady=2)
        
        # Download pretrained models section
        tk.Label(root, text="⬇️ Download Pre-trained Models:", font=("Arial", 9, "bold")).pack(pady=5)
        
        # Model selection dropdown
        tk.Label(root, text="Select Model:").pack()
        self.model_var = tk.StringVar()
        self.model_dropdown = tk.OptionMenu(root, self.model_var, "No models available")
        self.model_dropdown.pack(pady=2)
        
        # Download button with model selection
        self.download_var = tk.StringVar()
        self.download_dropdown = tk.OptionMenu(root, self.download_var, "Select model to download...")
        self.download_dropdown.pack(pady=2)
        
        # Populate download dropdown
        menu = self.download_dropdown["menu"]
        for name in PRETRAINED_MODELS:
            menu.add_command(label=name, command=lambda n=name: self.download_var.set(n))
        
        tk.Button(root, text="Download Selected Model", command=self.download_pretrained).pack(pady=2)
        tk.Button(root, text="Load Selected Model", command=self.load_selected_model).pack(pady=2)
        
        # Download progress
        self.download_progress = tk.Label(root, text="", fg="blue")
        self.download_progress.pack(pady=2)
        
        # Current model label
        self.current_model_label = tk.Label(root, text="No model loaded", fg="red")
        self.current_model_label.pack(pady=2)
        
        # Separator
        tk.Frame(root, height=2, bg="gray").pack(fill='x', pady=10)

        # Generate button
        tk.Button(root, text="🎤 Generate Vocals", command=self.run_generate, bg="green", fg="white", font=("Arial", 12, "bold")).pack(pady=10)

        # Status
        self.status = tk.Label(root, text="Ready.", fg="green")
        self.status.pack(pady=6)
        
        # Refresh model list
        self.refresh_model_list()

    def update_data_count(self):
        audio_count = len(glob.glob(os.path.join(DATA, "*.wav"))) + len(glob.glob(os.path.join(DATA, "*.mp3")))
        self.data_count.config(text=f"Training clips: {audio_count}")

    def refresh_model_list(self):
        models = get_available_models()
        if models:
            self.model_var.set(models[0][0])
            menu = self.model_dropdown["menu"]
            menu.delete(0, "end")
            for name, path in models:
                menu.add_command(label=name, command=lambda n=name, p=path: self.model_var.set(n))
        else:
            self.model_var.set("No models available")

    def download_pretrained(self):
        model_name = self.download_var.get()
        if not model_name or model_name == "Select model to download...":
            messagebox.showwarning("Warning", "Please select a model to download first.")
            return
        
        if model_name not in PRETRAINED_MODELS:
            messagebox.showwarning("Warning", "Invalid model selection.")
            return
        
        model_info = PRETRAINED_MODELS[model_name]
        
        # Start download in thread
        self.download_progress.config(text="Downloading...", fg="orange")
        threading.Thread(target=self._download_thread, args=(model_name, model_info), daemon=True).start()

    def _download_thread(self, model_name, model_info):
        try:
            dest = download_model(model_name, model_info, self.update_download_progress)
            if dest and os.path.exists(dest):
                self.download_progress.config(text=f"✓ Downloaded: {model_info['filename']}", fg="green")
                messagebox.showinfo("Success", f"Model '{model_name}' downloaded successfully!")
                self.refresh_model_list()
            else:
                self.download_progress.config(text="Download failed", fg="red")
                messagebox.showerror("Error", "Failed to download model.")
        except Exception as e:
            self.download_progress.config(text=f"Error: {str(e)[:30]}", fg="red")

    def update_download_progress(self, percent):
        self.download_progress.config(text=f"Downloading... {percent:.1f}%", fg="blue")

    def pick_audio(self):
        f = filedialog.askopenfilename(filetypes=[("Audio", "*.wav *.mp3")])
        if f:
            self.audio_path.set(f)

    def import_single_clip(self):
        files = filedialog.askopenfilenames(filetypes=[("Audio", "*.wav *.mp3")])
        count = 0
        for f in files:
            shutil.copy(f, DATA)
            count += 1
        if count > 0:
            messagebox.showinfo("Imported", f"{count} clip(s) imported to dataset.")
            self.update_data_count()

    def import_folder(self):
        """Import ALL audio files from a selected folder (including subfolders)"""
        folder = filedialog.askdirectory(title="Select FOLDER containing audio files to import")
        if not folder:
            return
        
        count = 0
        # Audio extensions to look for
        audio_extensions = ["*.wav", "*.mp3", "*.flac", "*.ogg", "*.m4a", "*.aiff", "*.aac"]
        
        # Walk through folder and all subfolders
        for root_dir, subdirs, files in os.walk(folder):
            for ext in audio_extensions:
                for f in glob.glob(os.path.join(root_dir, ext)):
                    try:
                        filename = os.path.basename(f)
                        dest_path = os.path.join(DATA, filename)
                        # Handle duplicate filenames
                        if os.path.exists(dest_path):
                            name, ext = os.path.splitext(filename)
                            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                            filename = f"{name}_{timestamp}{ext}"
                            dest_path = os.path.join(DATA, filename)
                        shutil.copy(f, dest_path)
                        count += 1
                    except Exception as e:
                        print(f"Error copying {f}: {e}")
        
        if count > 0:
            messagebox.showinfo("✅ Import Complete", f"Successfully imported {count} audio files!\n\nFiles saved to:\n{DATA}")
            self.update_data_count()
        else:
            messagebox.showwarning("⚠️ No Files Found", "No audio files found in the selected folder.")

    def open_data_folder(self):
        os.system(f"xdg-open {DATA}")

    def import_model(self):
        f = filedialog.askopenfilename(filetypes=[("PyTorch Model", "*.pth")])
        if f:
            dest = os.path.join(DATA, "target_model.pth")
            shutil.copy(f, dest)
            if load_model(dest):
                self.current_model_label.config(text=f"Loaded: {os.path.basename(f)}", fg="green")
                messagebox.showinfo("Model Imported", "Target model imported and loaded successfully.")
            else:
                messagebox.showerror("Error", "Failed to load model.")
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
            messagebox.showinfo("Imported", f"{count} model(s) imported to models folder.")
            self.refresh_model_list()

    def load_selected_model(self):
        models = get_available_models()
        selected = self.model_var.get()
        for name, path in models:
            if name == selected:
                if load_model(path):
                    self.current_model_label.config(text=f"Loaded: {os.path.basename(path)}", fg="green")
                    messagebox.showinfo("Model Loaded", f"Model '{os.path.basename(path)}' loaded successfully.")
                else:
                    messagebox.showerror("Error", "Failed to load model.")
                return
        messagebox.showwarning("Warning", "Please select a model first.")

    def run_generate(self):
        threading.Thread(target=self.generate, daemon=True).start()

    def enhance_prosody(self, y, sr):
        y_stretch = librosa.effects.time_stretch(y, rate=1.0 + np.random.uniform(-0.02,0.02))
        y_shift = librosa.effects.pitch_shift(y_stretch, sr, n_steps=np.random.uniform(-0.3,0.3))
        return y_shift

    def generate(self):
        try:
            self.status.config(text="Processing...", fg="orange")
            
            # Generate output name if empty
            name = self.outname.get().strip()
            if not name:
                # Create auto-generated name with timestamp
                timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                name = f"ai_vocal_{timestamp}"
            
            # Get output format
            output_format = self.output_format.get()
            
            # Create output path
            if output_format == "mp3":
                outfile = os.path.join(OUT, f"{name}_vocal.mp3")
                outwav = os.path.join(OUT, f"{name}_vocal_temp.wav")
            else:
                outfile = os.path.join(OUT, f"{name}_vocal.wav")
                outwav = outfile

            if svc is None:
                messagebox.showerror("Error","Target model not loaded! Import or select a model first.")
                return

            if self.audio_path.get():
                separated = separate.separate_file(self.audio_path.get(), outdir=OUT, model="demucs", splits=False)
                vocal_file = next((f for f in separated if "vocals" in f.lower()), None)
                if vocal_file is None:
                    messagebox.showerror("Error","No vocals detected.")
                    return
            else:
                text = self.textbox.get("1.0", "end").strip()
                if not text:
                    messagebox.showerror("Error","Provide text or audio input.")
                    return
                temp_file = os.path.join(OUT, f"{name}_temp.mp3")
                text_to_speech(text, temp_file)
                # Convert MP3 to WAV
                wav_file = temp_file.replace('.mp3', '.wav')
                audio = AudioSegment.from_mp3(temp_file)
                audio.export(wav_file, format="wav")
                vocal_file = wav_file
                if self.flow_align.get():
                    y, sr = librosa.load(vocal_file, sr=None)
                    y = self.enhance_prosody(y, sr)
                    sf.write(vocal_file, y, sr)

            # Run inference
            svc.infer(vocal_file, outwav, f0_method="pm")
            
            # Convert to MP3 if requested
            if output_format == "mp3":
                audio = AudioSegment.from_wav(outwav)
                audio.export(outfile, format="mp3")
                # Remove temp WAV
                os.remove(outwav)
            
            self.status.config(text=f"✅ Done → {outfile}", fg="green")
            messagebox.showinfo("Success", f"Vocals generated!\n\nOutput: {outfile}")
            
            # Ask if user wants to open the output folder
            if messagebox.askyesno("Open Output Folder?", "Open the output folder?"):
                os.system(f"xdg-open {OUT}")
                
        except Exception as e:
            self.status.config(text=f"Error: {e}", fg="red")
            messagebox.showerror("Error", str(e))

root = tk.Tk()
App(root)
root.mainloop()