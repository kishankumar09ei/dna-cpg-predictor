import streamlit as st
import torch
import numpy as np

# Load model (ensure the model is trained and saved as 'model.pth')
from model import CpGPredictor  # Import your LSTM model class & dictionary

alphabet = 'NACGT'
dna2int = {a: i for a, i in zip(alphabet, range(1, 6))}
dna2int.update({"pad": 0})  # Padding token


# Load trained model
@st.cache_resource
def load_model():
    model = CpGPredictor()
    model.load_state_dict(torch.load("best_model_2.pth"))
    model.eval()
    return model

model = load_model()



def preprocess_sequence(dna_seq: str, max_len=128):
    """ Converts a DNA sequence to an integer tensor, with padding if needed. """
    seq_int = [dna2int.get(base, 0) for base in dna_seq]  # Convert bases to integers
    seq_tensor = torch.tensor(seq_int, dtype=torch.long).unsqueeze(0)  # Add batch dimension
    length = torch.tensor([len(seq_int)])  # Store original length
    
    return seq_tensor, length

# Function to predict CpG sites
def predict_cpg_count(dna_seq: str):
    """ Takes a DNA sequence string, processes it, and returns the predicted CpG count. """
    seq_tensor, length = preprocess_sequence(dna_seq)
    with torch.no_grad():
        prediction = model(seq_tensor, length).item()
    
    return round(prediction, 2)

# Streamlit UI
st.title("DNA CpG Site Predictor")

# User Input
dna_input = st.text_input("Enter DNA Sequence:", "ACGTACGTACGCGTACG")

if st.button("Predict"):
    if len(dna_input) == 0:
        new_var = st.warning("Please enter a valid DNA sequence.")
    else:
        prediction = predict_cpg_count(dna_input)
        st.success(f"Predicted CpG Count: **{prediction:.2f}**")
