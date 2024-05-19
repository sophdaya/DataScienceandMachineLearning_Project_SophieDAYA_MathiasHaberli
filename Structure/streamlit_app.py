import streamlit as st
import pandas as pd
from transformers import CamembertTokenizer, CamembertForSequenceClassification
import torch
import numpy as np

# Load the tokenizer and model
tokenizer = CamembertTokenizer.from_pretrained('camembert-base')
model = CamembertForSequenceClassification.from_pretrained('BestModels', num_labels=6)

# Define difficulty levels
difficulty_levels = {0: 'A1', 1: 'A2', 2: 'B1', 3: 'B2', 4: 'C1', 5: 'C2'}

# Function to predict difficulty
def predict_difficulty(text):
    # Tokenize the text
    encodings = tokenizer(text, truncation=True, padding=True, max_length=512, return_tensors="pt")
    # Make prediction
    with torch.no_grad():
        outputs = model(**encodings)
    logits = outputs.logits
    prediction = torch.argmax(logits, axis=1).item()
    return difficulty_levels[prediction]

# Streamlit app
st.title("French Text Difficulty Predictor")
st.write("Enter a French text to predict its difficulty level (A1 to C2).")

# Text input
text_input = st.text_area("Enter French text here:")

# Predict button
if st.button("Predict Difficulty"):
    if text_input.strip() == "":
        st.write("Please enter a text.")
    else:
        # Predict the difficulty
        difficulty = predict_difficulty(text_input)
        st.write(f"The predicted difficulty level is: **{difficulty}**")


