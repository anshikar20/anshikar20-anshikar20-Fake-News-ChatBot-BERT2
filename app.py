import streamlit as st
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast
import torch

# Load the saved model and tokenizer
model = DistilBertForSequenceClassification.from_pretrained("./model")
tokenizer = DistilBertTokenizerFast.from_pretrained("./model")

# Title
st.title("📰 Fake News Detector Chatbot")
st.write("Enter a news article or snippet below and I’ll tell you if it’s likely true or fake!")

# Input
user_input = st.text_area("Paste your news snippet here 👇")

if st.button("Detect"):
    inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs).item()
        confidence = torch.max(probs).item() * 100

    label = "🟢 Likely True News" if pred == 1 else "🔴 Likely Fake News"
    st.markdown(f"## Prediction: {label}")
    st.markdown(f"**Confidence:** {confidence:.2f}%")

