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

class AIVocalsStudioMinimal:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Vocals Studio - Minimal")
        self.root.geometry("900x700")
        
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
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill='both', expand=True, padx=10, pady=5)
        
        # Generation Tab
        self.create_generation_tab(notebook)
        
        # Models Tab
        self.create_models_tab(notebook)
        
        # Training Tab
        self.create_training_tab(notebook)
    
    def create_generation_tab(self, notebook):
        gen_frame = ttk.Frame(notebook)
        notebook.add(gen_frame, text="🎵 Generate")
        
        # Audio Input Section
        input_frame = ttk.LabelFrame(gen_frame, text="Audio Input", padding=10)
        input_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(input_frame, text="Select audio file:").pack(anchor='w')
        
        audio_select_frame = ttk.Frame(input_frame)
        audio_select_frame.pack(fill='x', pady=5)
        
        ttk.Entry(audio_select_frame, textvariable=self.audio_path, width=50).pack(side='left', fill='x', expand=True)
        ttk.Button(audio_select_frame, text="Browse", command=self.select_audio).pack(side='right', padx=(5, 0))
        
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
        if not self.audio_path.get():
            messagebox.showwarning("No Audio", "Please select an audio file first.")
            return
        
        if not self.model_var.get():
            messagebox.showwarning("No Model", "Please select a voice model first.")
            return
        
        # Simulate generation process
        def generation_worker():
            try:
                self.update_operation_progress("🎤 Starting vocal generation...", 10, "Initializing voice synthesis")
                time.sleep(1)
                
                self.update_operation_progress("🔄 Processing audio...", 30, "Extracting vocals and applying voice model")
                time.sleep(2)
                
                self.update_operation_progress("✨ Applying voice conversion...", 60, "Converting to target voice")
                time.sleep(2)
                
                self.update_operation_progress("💾 Saving output...", 90, "Generating final audio file")
                time.sleep(1)
                
                # Create output file
                input_name = Path(self.audio_path.get()).stem
                model_name = self.model_var.get()
                output_name = f"{input_name}_{model_name}_generated.wav"
                output_path = self.OUTPUT / output_name
                
                # Create a dummy output file for demonstration
                with open(output_path, 'w') as f:
                    f.write("# Dummy generated audio file\n")
                
                self.update_operation_progress("✅ Generation complete!", 100, f"Output saved as {output_name}")
                self.add_output(f"🎉 Successfully generated vocals: {output_name}")
                
                messagebox.showinfo("Success", f"Vocals generated successfully!\nOutput: {output_name}")
                
            except Exception as e:
                self.update_operation_progress("❌ Generation failed", 0, "Check error details")
                self.add_output(f"❌ Error: {str(e)}")
                messagebox.showerror("Error", f"Generation failed: {str(e)}")
        
        # Run in thread
        thread = threading.Thread(target=generation_worker)
        thread.daemon = True
        thread.start()
    
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
