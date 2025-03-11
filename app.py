import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import google.generativeai as genai
from dotenv import load_dotenv
import speech_recognition as sr
from werkzeug.utils import secure_filename
from PIL import Image

# Load environment variables from .env file
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Configure Gemini API
genai.configure(api_key=API_KEY)

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///chat.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

db = SQLAlchemy(app)

# Model to store chat messages
class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    role = db.Column(db.String(10))
    message = db.Column(db.Text)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

with app.app_context():
    db.create_all()

# Helper function to call Gemini API
def chat_with_gemini(prompt):
    try:
        model = genai.GenerativeModel("gemini-1.5-pro")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"
# Restrict questions to agriculture-related topics
ALLOWED_KEYWORDS = [
    'hello', 'hi', 'hey','vanakam', 'greetings', 'morning', 'afternoon', 'evening', 'hola', 'namaste','crop', 'soil', 'fertilizer', 'irrigation', 'harvest', 'pest', 'disease', 'yield', 
    'plant', 'seed', 'weather', 'climate', 'agriculture', 'farming', 'organic', 'genetics',
    'greenhouse', 'compost', 'mulch', 'pruning', 'spraying', 'weeding', 'grafting', 'pollination',
    'soil health', 'irrigation system', 'drip irrigation', 'sprinkler', 'organic farming', 
    'hydroponics', 'aquaponics', 'permaculture', 'sustainable', 'crop rotation', 'monoculture', 
    'cover crop', 'pesticide', 'herbicide', 'insecticide', 'fungicide', 'weed control', 
    'agrochemicals', 'GMOs', 'genetically modified', 'fertilizer management', 'composting', 
    'microorganisms', 'biological pest control', 'soil erosion', 'soil fertility', 'water management', 
    'plant disease', 'fungal disease', 'bacterial disease', 'viral disease', 'crop disease', 
    'leaf blight', 'rust', 'aphids', 'fungus', 'nematodes', 'mildew', 'fungal spores', 'insects', 
    'moths', 'termites', 'rodents', 'birds', 'snails', 'slugs', 'antioxidants', 'biomass', 
    'biodiversity', 'irrigation technique', 'drought', 'flood', 'temperature', 'humidity', 'wind', 
    'frost', 'farming method', 'conventional farming', 'mechanization', 'agricultural technology', 
    'robotics', 'harvest timing', 'grain', 'rice', 'wheat', 'corn', 'soybean', 'cotton', 'barley', 
    'oats', 'sorghum', 'millet', 'peas', 'lentils', 'chickpeas', 'tomato', 'cucumber', 'carrot', 
    'onion', 'garlic', 'lettuce', 'spinach', 'kale', 'potato', 'sweet potato', 'pumpkin', 
    'watermelon', 'broccoli', 'cauliflower', 'green beans', 'beetroot', 'corn stalk', 'cotton boll', 
    'tobacco', 'cacao', 'coffee', 'citrus', 'apple', 'mango', 'banana', 'pineapple', 'cherry', 
    'avocado', 'grapes', 'peach', 'pear', 'plum', 'kiwi', 'berry', 'blackberry', 'blueberry', 
    'strawberry', 'raspberry', 'agricultural policy', 'supply chain', 'farm economics', 'market price', 
    'food security', 'food waste', 'crop insurance', 'food quality', 'food safety', 'yield prediction', 
    'soil moisture', 'nutrient deficiencies', 'agronomy', 'agriculture research', 'agri-tech', 'biotech', 
    'farm machinery', 'tractor', 'plow', 'harvester', 'combine', 'planting depth', 'seedling', 
    'agriculture extension', 'field management', 'labor costs', 'global warming', 'carbon footprint', 
    'sustainability', 'soil testing', 'crop growth', 'biodynamic farming', 'agricultural education', 
    'wildlife conservation', 'conservation tillage', 'integrated pest management', 'natural pesticides', 
    'pollution', 'land degradation', 'overgrazing', 'deforestation', 'ecosystem', 'carbon sequestration', 
    'renewable energy', 'solar farming', 'wind farming', 'bioenergy', 'biomass energy', 'agroforestry', 
    'land restoration', 'agriculture subsidies', 'international trade', 'agribusiness', 'farmer cooperatives', 
    'rural development', 'farming cooperatives', 'fair trade', 'organic certification', 'food labeling', 
    'export', 'import', 'supply chain management', 'food processing', 'distribution', 'market trends'
]


def is_valid_question(question):
    # Convert question to lowercase and check if any keyword is present
    question = question.lower()
    return any(keyword in question for keyword in ALLOWED_KEYWORDS)
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        user_message = request.form.get("message")
        
        if user_message:
            if not is_valid_question(user_message):
                # Invalid question -> Save as error message
                bot_response = "Sorry, I can only answer agriculture-related questions."
            else:
                # Valid question -> Call Gemini API
                bot_response = chat_with_gemini(user_message)

            # Save User Message
            user_chat = ChatMessage(role="user", message=user_message)
            db.session.add(user_chat)
            
            # Save Bot Response
            bot_chat = ChatMessage(role="bot", message=bot_response)
            db.session.add(bot_chat)
            
            db.session.commit()
            return redirect(url_for("index"))

    # Load all chat messages
    chats = ChatMessage.query.order_by(ChatMessage.timestamp.asc()).all()
    return render_template("index.html", chats=chats)


# Handle image upload
@app.route("/upload", methods=["POST"])
def upload():
    if 'file' in request.files:
        file = request.files['file']
        if file.filename != '' and file.filename.split('.')[-1].lower() in ALLOWED_EXTENSIONS:
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Process the image if needed (e.g., classification)
            img = Image.open(file_path)
            img = img.resize((256, 256))

            # Add to chat as a message
            message = f"Image uploaded: {filename}"
            chat = ChatMessage(role="user", message=message)
            db.session.add(chat)
            db.session.commit()

    return redirect(url_for('index'))

# Handle clearing all messages
@app.route("/clear", methods=["POST"])
def clear():
    ChatMessage.query.delete()
    db.session.commit()
    return redirect(url_for('index'))

# Handle audio input (using SpeechRecognition)
@app.route("/audio", methods=["POST"])
def audio():
    recognizer = sr.Recognizer()
    file = request.files['audio']
    if file:
        with sr.AudioFile(file) as source:
            audio = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio)
                user_chat = ChatMessage(role="user", message=text)
                db.session.add(user_chat)
                db.session.commit()

                # Generate Gemini response
                bot_response = chat_with_gemini(text)
                bot_chat = ChatMessage(role="bot", message=bot_response)
                db.session.add(bot_chat)
                db.session.commit()
            except sr.UnknownValueError:
                pass

    return redirect(url_for("index"))

if __name__ == "__main__":
    app.run(debug=True)
