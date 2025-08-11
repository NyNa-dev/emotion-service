from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import pipeline
import re

app = FastAPI()

# Enable CORS to allow requests from your Node.js backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your Node.js app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the emotion detection model
emotion_classifier = pipeline("text-classification", model="bhadresh-savani/distilbert-base-uncased-emotion", top_k=None)

# Basic text preprocessing
def preprocess_text(text: str) -> str:
    text = text.lower().strip()
    # Handle common emojis (extend as needed)
    text = re.sub(r':\)', ' happy ', text)
    text = re.sub(r':\(', ' sad ', text)
    return text

@app.post("/analyze-emotion")
async def analyze_emotion(message: str):
    # Preprocess the message
    processed_message = preprocess_text(message)
    # Run emotion detection
    emotions = emotion_classifier(processed_message)
    return {"emotions": emotions}