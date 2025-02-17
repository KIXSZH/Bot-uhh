import google.generativeai as genai
import os
from dotenv import load_dotenv
import streamlit as st

# Load API key from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

if not api_key:
    raise ValueError("API key is missing! Set it in the .env file.")

# Configure the Gemini API
genai.configure(api_key=api_key)

# Configure Streamlit page
st.set_page_config(
    page_title="AgriChat 🌱",
    page_icon="🚜",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Custom CSS for agriculture theme
st.markdown("""
<style>
    
    
    .css-1q1n0ol {
        background-color: #2e7d32 !important;
        color: white !important;
    }
    .css-1q1n0ol:hover {
        background-color: #1b5e20 !important;
    }
    .sidebar .sidebar-content {
        background-color: #a5d6a7 !important;
    }
    .css-1hynsf2 {
        background-color: #81c784 !important;
    }
</style>
""", unsafe_allow_html=True)

def chat_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-pro")
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error: {e}"

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "🌾 Welcome to AgriChat! How can I help you with agriculture-related questions today? 🚜"}]

# Sidebar with agricultural theme
with st.sidebar:
    st.title("AgriChat 🌻")
    st.markdown("""
    **Your Farming Assistant**

    Ask about:
    - Crop rotation 🌱
    - Pest control 🐞
    - Soil health 🌍
    - Weather patterns ☔
    - Market trends 📈
    """)
    st.image("https://cdn-icons-png.flaticon.com/512/3079/3079158.png", width=200)

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your agriculture question..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response from Gemini
    response = chat_with_gemini(prompt)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})
    # Display assistant response
    with st.chat_message("assistant"):
        st.markdown(response)
