#!/usr/bin/env python3
"""
AI Vocals Studio - Streamlit Web Version
Professional consent-based voice cloning interface for web deployment.
"""
import streamlit as st
import os
import tempfile
import threading
import time
from pathlib import Path
from gtts import gTTS
import base64

# Try to import voice cloning engines
try:
    from qwen3_tts_engine import Qwen3TTSEngine
    HAS_QWEN_ENGINE = True
except ImportError:
    HAS_QWEN_ENGINE = False

# Configuration
BASE = os.environ.get("APP_DATA_DIR", os.path.expanduser("~/ai-vocals-studio"))
OUT = os.environ.get("OUTPUT_DIR", os.path.join(BASE, "outputs"))
DATA = os.environ.get("DATASET_DIR", os.path.join(BASE, "dataset"))
MODELS = os.environ.get("MODEL_CACHE_DIR", os.path.join(BASE, "models"))
os.makedirs(OUT, exist_ok=True)
os.makedirs(DATA, exist_ok=True)
os.makedirs(MODELS, exist_ok=True)

# Page config
st.set_page_config(
    page_title="AI Vocals Studio - Voice Cloning",
    page_icon=":studio_microphone:",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(135deg, #1a1a1a 0%, #2b2b2b 100%);
        color: white;
    }
    .stButton>button {
        background: linear-gradient(90deg, #00ff88, #00cc6a);
        color: black;
        font-weight: bold;
        border: none;
        border-radius: 8px;
        padding: 10px 20px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(90deg, #00cc6a, #00ff88);
        transform: translateY(-2px);
    }
    .stTextInput>div>div>input {
        background: #3a3a3a;
        color: white;
        border: 1px solid #00ff88;
    }
    .stTextArea>div>div>textarea {
        background: #3a3a3a;
        color: white;
        border: 1px solid #00ff88;
    }
    .stSelectbox>div>div>select {
        background: #3a3a3a;
        color: white;
    }
    .stProgress .progress-bar {
        background: linear-gradient(90deg, #00ff88, #00cc6a);
    }
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        background: linear-gradient(90deg, #00ff88, #00cc6a);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 2rem;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """Initialize session state variables"""
    if 'qwen_engine' not in st.session_state:
        st.session_state.qwen_engine = Qwen3TTSEngine() if HAS_QWEN_ENGINE else None
    if 'has_qwen_runtime' not in st.session_state:
        st.session_state.has_qwen_runtime = bool(
            st.session_state.qwen_engine and st.session_state.qwen_engine.can_clone()
        )
    if 'cloning_progress' not in st.session_state:
        st.session_state.cloning_progress = 0
    if 'cloning_status' not in st.session_state:
        st.session_state.cloning_status = "Ready"

def load_voice_dataset():
    """Load and display local reference dataset information."""
    dataset_info = {
        'total_files': 0,
        'total_size_mb': 0,
        'files': []
    }

    if os.path.exists(DATA):
        for ext in ['*.wav', '*.mp3', '*.m4a', '*.flac']:
            files = list(Path(DATA).glob(ext))
            dataset_info['files'].extend(files)

        dataset_info['total_files'] = len(dataset_info['files'])
        dataset_info['total_size_mb'] = sum(f.stat().st_size for f in dataset_info['files']) / (1024*1024)

    return dataset_info

def clone_voice_worker(ref_audio_path, ref_text, target_text, speaker_name):
    """Background worker for voice cloning"""
    try:
        if not st.session_state.qwen_engine:
            st.session_state.cloning_status = "Qwen3-TTS not available"
            return

        # Update progress
        st.session_state.cloning_status = "Loading model..."
        st.session_state.cloning_progress = 20

        # Load model
        def progress_cb(msg, progress):
            st.session_state.cloning_status = msg
            st.session_state.cloning_progress = progress

        if not st.session_state.qwen_engine.load_model(progress_cb=progress_cb):
            st.session_state.cloning_status = "Failed to load model"
            return

        st.session_state.cloning_status = "Cloning voice..."
        st.session_state.cloning_progress = 50

        # Clone voice
        output_path = st.session_state.qwen_engine.clone_voice(
            ref_audio=ref_audio_path,
            ref_text=ref_text,
            target_text=target_text,
            speaker_name=speaker_name,
            progress_cb=progress_cb
        )

        if output_path:
            st.session_state.cloning_status = f"Success! Saved to: {output_path}"
            st.session_state.cloning_progress = 100
            st.session_state.last_output = output_path
        else:
            st.session_state.cloning_status = "Cloning failed"

    except Exception as e:
        st.session_state.cloning_status = f"Error: {str(e)}"
        st.session_state.cloning_progress = 0

def audio_player(audio_path):
    """Create audio player for file"""
    try:
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
        base64_audio = base64.b64encode(audio_bytes).decode()
        audio_html = f"""
        <audio controls>
            <source src="data:audio/wav;base64,{base64_audio}" type="audio/wav">
        </audio>
        """
        st.markdown(audio_html, unsafe_allow_html=True)
    except Exception as e:
        st.error(f"Could not play audio: {e}")

def main():
    init_session_state()

    # Header
    st.markdown('<h1 class="main-header">:studio_microphone: AI Vocals Studio</h1>', unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #00ff88;'>Consent-Based Voice Cloning & Generation</h2>", unsafe_allow_html=True)
    st.warning(
        "Permission required: only upload or clone a voice you own, created yourself, "
        "or have explicit written permission/license to use. Do not clone artists, "
        "celebrities, public figures, private people, or copyrighted recordings "
        "without authorization."
    )

    # Sidebar
    st.sidebar.markdown("## :gear: Settings")

    # Engine selection
    available_engines = ["gTTS (Free)"]
    if st.session_state.has_qwen_runtime:
        available_engines.append("Qwen3-TTS (Advanced)")

    selected_engine = st.sidebar.selectbox("Voice Engine", available_engines)

    # Dataset info
    st.sidebar.markdown("## :file_folder: Dataset")
    dataset_info = load_voice_dataset()
    st.sidebar.metric("Reference Audio Files", dataset_info['total_files'])
    st.sidebar.metric("Dataset Size", f"{dataset_info['total_size_mb']:.1f} MB")

    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs([":studio_microphone: Voice Clone", ":speaking_head: Text-to-Speech", ":robot_face: Models", ":bar_chart: Analytics"])

    with tab1:
        st.markdown("### :studio_microphone: 3-Second Voice Cloning")
        st.markdown("Clone an authorized voice with a short reference sample.")
        st.info(
            "By continuing, you confirm the reference audio and target voice are authorized "
            "for this use, including any artist-style, label, estate, or rights-holder approval "
            "that may be required."
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Reference audio upload
            st.markdown("#### Step 1: Upload Reference Audio")
            ref_audio = st.file_uploader("Choose authorized reference audio", type=['wav', 'mp3', 'm4a', 'flac'])

            if ref_audio:
                # Save uploaded file temporarily
                with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
                    tmp_file.write(ref_audio.read())
                    tmp_audio_path = tmp_file.name

                st.success(f"Uploaded: {ref_audio.name}")
                audio_player(tmp_audio_path)

            # Reference text
            st.markdown("#### Step 2: What's said in the audio?")
            ref_text = st.text_input("Reference text", value="This is my reference voice sample",
                                   help="Enter exactly what's said in the reference audio")

            # Target text
            st.markdown("#### Step 3: What should the voice say?")
            target_text = st.text_area("Target text", value="This is a generated vocal take from AI Vocals Studio",
                                     help="Enter the words to synthesize")
            speaker_name = st.text_input("Voice label", value="Authorized_Voice",
                                       help="Use a project-safe label for the cloned voice")

        with col2:
            st.markdown("#### Step 4: Clone Voice")
            has_permission = st.checkbox(
                "I own this voice or have explicit written permission/license to clone it.",
                value=False
            )

            if st.button(":studio_microphone: Clone Voice", type="primary", use_container_width=True):
                if not st.session_state.has_qwen_runtime:
                    st.error("Qwen3-TTS not installed! Install with: pip install qwen-tts")
                elif not ref_audio:
                    st.error("Please upload a reference audio file")
                elif not ref_text:
                    st.error("Please enter the reference text")
                elif not target_text:
                    st.error("Please enter the target text")
                elif not has_permission:
                    st.error("Confirm that you own the voice or have explicit permission before cloning.")
                else:
                    # Start cloning in background
                    st.session_state.cloning_progress = 10
                    st.session_state.cloning_status = "Starting cloning..."

                    thread = threading.Thread(
                        target=clone_voice_worker,
                        args=(tmp_audio_path, ref_text, target_text, speaker_name.strip() or "Authorized_Voice")
                    )
                    thread.daemon = True
                    thread.start()

            # Progress
            if st.session_state.cloning_progress > 0:
                st.progress(st.session_state.cloning_progress)
                st.info(st.session_state.cloning_status)

                if st.session_state.cloning_progress == 100 and 'last_output' in st.session_state:
                    st.success("Voice cloning completed!")
                    audio_player(st.session_state.last_output)

    with tab2:
        st.markdown("### :speaking_head: Text-to-Speech")
        st.markdown("Generate speech using different TTS engines")

        # Text input
        text_input = st.text_area("Enter text to generate", value="Westside till we die!", height=100)

        col1, col2 = st.columns([3, 1])

        with col1:
            # Voice settings
            if selected_engine == "Qwen3-TTS (Advanced)":
                voice_description = st.text_input("Voice description",
                                                value="Warm, expressive rap vocal with confident delivery",
                                                help="Describe tone and delivery. Use only licensed artist references.")

        with col2:
            if st.button("Generate Speech", type="primary", use_container_width=True):
                if not text_input:
                    st.error("Please enter text to generate")
                else:
                    try:
                        if selected_engine == "gTTS (Free)":
                            # Use gTTS
                            tts = gTTS(text=text_input, lang='en')
                            output_path = os.path.join(OUT, f"gtts_output_{int(time.time())}.mp3")
                            tts.save(output_path)
                            st.success("Speech generated with gTTS!")
                            audio_player(output_path)

                        elif st.session_state.has_qwen_runtime and selected_engine == "Qwen3-TTS (Advanced)":
                            # Use Qwen3-TTS
                            if not st.session_state.qwen_engine:
                                st.session_state.qwen_engine = Qwen3TTSEngine()

                            output_path = st.session_state.qwen_engine.design_voice(
                                text=text_input,
                                voice_description=voice_description,
                                speaker_name="Designed_Voice"
                            )

                            if output_path:
                                st.success("Speech generated with Qwen3-TTS!")
                                audio_player(output_path)
                            else:
                                st.error("Failed to generate speech")

                    except Exception as e:
                        st.error(f"Error generating speech: {e}")

    with tab3:
        st.markdown("### :robot_face: Available Models")

        models_info = []

        # Qwen3-TTS
        if st.session_state.has_qwen_runtime:
            models_info.append({
                "name": "Qwen3-TTS",
                "status": "Available",
                "features": "3-second cloning, multi-language, natural language control"
            })
        else:
            models_info.append({
                "name": "Qwen3-TTS",
                "status": "Not Installed",
                "features": "Install with: pip install qwen-tts"
            })

        # gTTS
        models_info.append({
            "name": "gTTS",
            "status": "Available",
            "features": "Free TTS, multiple languages"
        })

        # Local models
        if os.path.exists(MODELS):
            for item in os.listdir(MODELS):
                if os.path.isdir(os.path.join(MODELS, item)):
                    models_info.append({
                        "name": item,
                        "status": "Local",
                        "features": "Custom trained model"
                    })

        # Display models
        for model in models_info:
            status_emoji = "Available" if model["status"] == "Available" else "Not Available" if model["status"] == "Not Installed" else "Local"
            st.markdown(f"**{model['name']}** - {status_emoji}")
            st.markdown(f"*{model['features']}*")
            st.markdown("---")

    with tab4:
        st.markdown("### :bar_chart: Analytics")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Clones", "0")
            st.metric("Success Rate", "0%")
            st.metric("Avg Processing Time", "0s")

        with col2:
            st.metric("Models Loaded", "1" if st.session_state.qwen_engine else "0")
            st.metric("Advanced Engine", "Available" if st.session_state.has_qwen_runtime else "Not installed")
            st.metric("Cache Size", "0 MB")

        # Usage chart
        st.markdown("#### Usage Over Time")
        st.info("Analytics will be available after more usage")

if __name__ == "__main__":
    main()
