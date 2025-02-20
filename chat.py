import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from streamlit.components.v1 import html
import google.generativeai as genai
import numpy as np
import tensorflow as tf
from tensorflow_tts import AutoProcessor, AutoModel
from vosk import Model, KaldiRecognizer
from pydub import AudioSegment
import io
import soundfile as sf

# ------------------ Text-to-Speech Setup ------------------
TTS_MODELS = {
    "en": {
        "processor": "tensorspeech/tts-tacotron2-ljspeech-en",
        "model": "tensorspeech/tts-tacotron2-ljspeech-en",
        "vocoder": "tensorspeech/tts-mb_melgan-ljspeech-en"
    },
    "ta": {
        "processor": "tensorspeech/tts-tacotron2-tamil",
        "model": "tensorspeech/tts-tacotron2-tamil",
        "vocoder": "tensorspeech/tts-mb_melgan-tamil"
    }
}

# ------------------ Speech-to-Text Setup ------------------
STT_MODELS = {
    "en": Model(lang="en-us"),
    "ta": Model(lang="tamil")
}

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key is missing! Set it in the .env file.")
    st.stop()

# Configure the Gemini API
genai.configure(api_key=api_key)

# ------------------ Audio Processing Functions ------------------
def convert_audio_to_wav(audio_data, sample_rate=16000):
    """Convert any audio format to 16kHz mono WAV format"""
    audio = AudioSegment.from_file(io.BytesIO(audio_data))
    audio = audio.set_frame_rate(sample_rate).set_channels(1)
    buffer = io.BytesIO()
    audio.export(buffer, format="wav")
    return buffer.getvalue()

def transcribe_audio(audio_bytes, language="en"):
    """Offline speech-to-text using Vosk"""
    audio_data = convert_audio_to_wav(audio_bytes)
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmpfile:
        tmpfile.write(audio_data)
        with sf.SoundFile(tmpfile.name, 'r') as sf_file:
            audio = sf_file.read(dtype='int16')
            recognizer = KaldiRecognizer(STT_MODELS[language], sf_file.samplerate)
            
            results = []
            chunk_size = 4000
            for i in range(0, len(audio), chunk_size):
                chunk = audio[i:i+chunk_size].tobytes()
                if recognizer.AcceptWaveform(chunk):
                    results.append(recognizer.Result())
            
            final = recognizer.FinalResult()
            results.append(final)
            
    return " ".join([json.loads(r)["text"] for r in results if r])

def synthesize_speech(text, language="en"):
    """Offline text-to-speech using TensorFlow TTS"""
    processor = AutoProcessor.from_pretrained(TTS_MODELS[language]["processor"])
    model = AutoModel.from_pretrained(TTS_MODELS[language]["model"])
    vocoder = AutoModel.from_pretrained(TTS_MODELS[language]["vocoder"])

    input_ids = processor.text_to_sequence(text)
    _, mel_outputs, _, _ = model.inference(input_ids)
    audio = vocoder.inference(mel_outputs)
    
    buffer = io.BytesIO()
    sf.write(buffer, audio.numpy().flatten(), 22050, format='WAV')
    return buffer.getvalue()

# ------------------ Updated Video Analysis ------------------
def analyze_video_speech(video_path, language="en"):
    """Extract audio from video and transcribe using offline STT"""
    video = AudioSegment.from_file(video_path)
    audio_data = video.set_frame_rate(16000).set_channels(1)
    buffer = io.BytesIO()
    audio_data.export(buffer, format="wav")
    return transcribe_audio(buffer.getvalue(), language)

# ------------------ Updated Chat Function ------------------
def chat_with_gemini(prompt, lang="en"):
    """Multilingual chat with Gemini"""
    try:
        model = genai.GenerativeModel("gemini-pro")
        if lang == "ta":
            sys_prompt = "நீங்கள் ஒரு விவசாய நிபுணர். கீழ்காணும் கேள்விக்கு விரிவான, நடைமுறை ஆலோசனைகளை தமிழில் வழங்கவும்: "
            response = model.generate_content(sys_prompt + prompt)
        else:
            response = model.generate_content(
                f"You are an agricultural expert. Provide detailed, practical advice for: {prompt}"
            )
        return response.text
    except Exception as e:
        return f"⚠️ Error: {str(e)}. Please try again."

# ------------------ UI Configuration (Same as before) ------------------
# [Keep all the existing Streamlit UI configuration and theming from original code]

# ------------------ Updated Input Handling ------------------
if input_mode == "Text":
    user_prompt = st.chat_input("Ask your agriculture question...")
elif input_mode == "Audio":
    audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "ogg"])
    if audio_file is not None:
        st.info("Transcribing audio...")
        user_prompt = transcribe_audio(audio_file.read(), lang)
elif input_mode == "Video":
    video_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if video_file is not None:
        st.info("Analyzing video...")
        user_prompt = analyze_video_speech(video_file, lang)

# [Rest of the existing chat handling and UI code remains the same]