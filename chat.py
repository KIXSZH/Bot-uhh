import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html
import google.generativeai as genai

# Additional Google Cloud libraries for multi-modal support
from google.cloud import translate_v2 as translate
from google.cloud import speech
from google.cloud import texttospeech
from google.cloud import videointelligence

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key is missing! Set it in the .env file.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=api_key)

# Configure Streamlit page
st.set_page_config(
    page_title="AgriChat 🌱 - Smart Farming Assistant",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced agriculture theme
st.markdown("""
<style>
    :root {
        --primary-color: #2e7d32;
        --secondary-color: #81c784;
    }
    .stChatInput textarea {
        border: 2px solid var(--primary-color) !important;
        border-radius: 10px !important;
    }
    .stChatInput button {
        background: var(--primary-color) !important;
        color: white !important;
        border-radius: 8px !important;
    }
    .stChatMessage {
        padding: 1rem;
        margin: 0.5rem 0;
        border-radius: 15px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    [data-testid="stSidebar"] {
        
        border-right: 2px solid var(--primary-color);
    }
    .sidebar-title {
        color: var(--primary-color);
        font-size: 1.5rem;
        font-weight: bold;
        margin-bottom: 1rem;
    }
    .example-question {
        padding: 0.8rem;
        margin: 0.5rem 0;
        background: white;
        border-radius: 10px;
        cursor: pointer;
        transition: all 0.3s;
    }
    .example-question:hover {
        transform: translateX(5px);
        box-shadow: 0 3px 6px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# --------- Additional Functions for Multi-language Features ----------

def translate_text(text, target_language="ta"):
    """Translate text to the target language using Google Cloud Translation API."""
    client = translate.Client()
    result = client.translate(text, target_language=target_language)
    return result["translatedText"]

def transcribe_audio(audio_bytes, language_code="ta-IN"):
    """Convert Tamil (or other language) speech in audio bytes to text using Google Speech-to-Text."""
    client = speech.SpeechClient()
    audio = speech.RecognitionAudio(content=audio_bytes)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,  # Ensure your audio meets this format
        sample_rate_hertz=16000,
        language_code=language_code,
    )
    response = client.recognize(config=config, audio=audio)
    transcript = ""
    for result in response.results:
        transcript += result.alternatives[0].transcript + " "
    return transcript.strip()

def synthesize_speech(text, language_code="en-US"):
    """Generate audio content from text using Google Text-to-Speech API."""
    client = texttospeech.TextToSpeechClient()
    synthesis_input = texttospeech.SynthesisInput(text=text)
    voice = texttospeech.VoiceSelectionParams(
        language_code=language_code,
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )
    return response.audio_content

def analyze_video_speech(video_path, language_code="ta-IN"):
    """Analyze video file to extract spoken Tamil (or specified language) transcript using Video Intelligence API."""
    client = videointelligence.VideoIntelligenceServiceClient()
    features = [videointelligence.Feature.SPEECH_TRANSCRIPTION]
    speech_config = videointelligence.SpeechTranscriptionConfig(language_code=language_code)
    video_context = videointelligence.VideoContext(speech_transcription_config=speech_config)
    
    with open(video_path, "rb") as video_file:
        input_content = video_file.read()
        
    operation = client.annotate_video(
        request={
            "features": features,
            "input_content": input_content,
            "video_context": video_context,
        }
    )
    result = operation.result(timeout=180)
    transcript = ""
    for annotation in result.annotation_results:
        for transcription in annotation.speech_transcriptions:
            transcript += transcription.alternatives[0].transcript + " "
    return transcript.strip()

def chat_with_gemini(prompt, lang="en"):
    """Call the Gemini API to generate a response. If lang=='ta', request the response in Tamil."""
    try:
        model = genai.GenerativeModel("gemini-pro")
        if lang == "ta":
            # Prompt in Tamil: You are an agricultural expert providing advice in Tamil.
            tg_prompt = f"நீங்கள் ஒரு விவசாய நிபுணர். கீழ்காணும் கேள்விக்கு விரிவான, நடைமுறை ஆலோசனைகளை தமிழில் வழங்கவும்: {prompt}"
            response = model.generate_content(tg_prompt)
        else:
            response = model.generate_content(
                f"You are an agricultural expert. Provide detailed, practical advice for: {prompt}"
            )
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please try again."

# ------------------ Sidebar Configuration ------------------

with st.sidebar:
    st.markdown('<div class="sidebar-title">🌾 AgriChat Features</div>', unsafe_allow_html=True)
    
    st.markdown("### Quick Actions")
    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.experimental_rerun()

    st.markdown("---")
    st.markdown("### Example Questions")
    examples = [
        "Best practices for organic tomato farming?",
        "How to prevent pest infestation in rice crops?",
        "Ideal irrigation methods for arid regions",
        "Latest trends in vertical farming technology",
        "How to improve soil fertility naturally?"
    ]
    for example in examples:
        if st.button(example, key=example):
            st.session_state.messages.append({"role": "user", "content": example})
            st.experimental_rerun()

    st.markdown("---")
    # Mode selections for input and language
    input_mode = st.radio("Select Input Mode", ["Text", "Audio", "Video"])
    language_option = st.radio("Select Language", ["English", "Tamil"])
    # Map language selections for internal use
    lang = "ta" if language_option == "Tamil" else "en"

    st.markdown("📘 **Tip:** Ask about specific crops, regional farming techniques, or climate adaptation strategies!")
    st.caption("⚠️ Note: Responses are AI-generated and should be verified with local experts")

# ------------------ Chat History Initialization ------------------

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": "🌱 Welcome to AgriChat! I'm your smart farming assistant. Ask me about crops, livestock, weather impacts, or sustainable practices!"
    }]

# ------------------ Chat Container ------------------

chat_container = st.container()

with chat_container:
    for message in st.session_state.messages:
        avatar = "🌾" if message["role"] == "assistant" else "👤"
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
    # Auto-scroll to the bottom
    html("<script>window.scrollTo(0, document.body.scrollHeight);</script>")

# ------------------ Input Handling ------------------

user_prompt = None

if input_mode == "Text":
    # Standard text input
    user_prompt = st.chat_input("Ask your agriculture question...")
elif input_mode == "Audio":
    # Audio file uploader for speech input
    audio_file = st.file_uploader("Upload an audio file (WAV format recommended)", type=["wav", "mp3"])
    if audio_file is not None:
        audio_bytes = audio_file.read()
        # For simplicity, we assume LINEAR16 WAV; adjust encoding/sample_rate as needed.
        st.info("Transcribing audio...")
        user_prompt = transcribe_audio(audio_bytes, language_code="ta-IN" if lang=="ta" else "en-US")
        st.markdown(f"**Transcribed Text:** {user_prompt}")
elif input_mode == "Video":
    # Video file uploader for video input
    video_file = st.file_uploader("Upload a video file (MP4)", type=["mp4"])
    if video_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
            tmp.write(video_file.read())
            tmp_path = tmp.name
        st.info("Analyzing video for speech transcription...")
        user_prompt = analyze_video_speech(tmp_path, language_code="ta-IN" if lang=="ta" else "en-US")
        st.markdown(f"**Extracted Transcript:** {user_prompt}")

if user_prompt:
    # Append user prompt to chat history
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_prompt)
    
    with st.spinner("🌱 Analyzing your query..."):
        response = chat_with_gemini(user_prompt, lang=lang)
    
    # Append and display the assistant's response
    st.session_state.messages.append({"role": "assistant", "content": response})
    with st.chat_message("assistant", avatar="🌾"):
        st.markdown(response)
        st.caption("Was this helpful? 👍 👎")
    
    # Option to play audio of the response
    if st.button("🔊 Listen to Response"):
        tts_language = "ta-IN" if lang == "ta" else "en-US"
        audio_content = synthesize_speech(response, language_code=tts_language)
        st.audio(audio_content, format="audio/mp3")

# ------------------ Footer ------------------

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: var(--primary-color); padding: 1rem;">
    🌍 Sustainable Farming Assistant • Version 1.2 • 
    <a href="#" style="color: var(--primary-color);">Privacy Policy</a>
</div>
""", unsafe_allow_html=True)
