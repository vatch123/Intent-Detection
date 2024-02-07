import streamlit as st
import pandas as pd
from transformers import pipeline

# This loading will take some time when it is downloading the pipeline for the first time
@st.cache_resource
def load_model():
  classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
  return classifier

classifier = load_model()

st.title("Intent recogntion using Machine learning!")

# Change this list to a set of intents you want to classify
default_labels = ['Accounts', 'Savings', 'Cheque', 'Credit Card', 'Mortgage', 'Close', 'Open']
# default_labels = ['Space & Cosmos', 'Physics', 'Chemistry', 'Robotics', 'Mathematics']

labels = st.multiselect(label="List of possible intents", options=default_labels, default=default_labels)
input = st.text_input("Enter your text input:")

if input:
  with st.spinner("Analyzing the intent..."):
    result = classifier(input, labels)
    df = pd.DataFrame.from_dict(result)
    st.bar_chart(df, x='labels', y='scores')
