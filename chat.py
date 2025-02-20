import streamlit as st
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    st.error("API key is missing! Set it in the .env file.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)

# Function to interact with Gemini API
def chat_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        full_prompt = f"You are an agriculture expert. Provide answers related to farming, crops, soil health, fertilizers, pest control, and weather. User query: {prompt}"
        response = model.generate_content(full_prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Streamlit UI
st.title("🌱 Agriculture Helper Bot 🚜")
st.write("Ask me anything about farming, crops, soil, pest control, and more!")

# Chat history using Streamlit session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ✅ Fix: Add a unique `key` to `st.chat_input()`
user_input = st.chat_input("Ask me about farming...", key="user_input")

if user_input:
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get Gemini response
    bot_response = chat_with_gemini(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

    # Display bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)
