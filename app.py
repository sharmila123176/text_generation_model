import os
import streamlit as st
from transformers import pipeline

# Disable TensorFlow/Keras inside transformers
os.environ["USE_TF"] = "0"
os.environ["TRANSFORMERS_NO_TF"] = "1"

# Load the GPT-2 model
@st.cache_resource
def load_generator():
    return pipeline('text-generation', model='gpt2', framework="pt")

generator = load_generator()

# Streamlit UI
st.title("🧠 Text Generator using GPT-2")
st.write("Generate coherent paragraphs on any topic!")

# User input
prompt = st.text_area("Enter your topic or prompt:", "The impact of artificial intelligence on education is")

max_tokens = st.slider("Select number of new tokens to generate:", min_value=50, max_value=500, value=150)

if st.button("Generate Text"):
    with st.spinner("Generating..."):
        result = generator(prompt, max_new_tokens=max_tokens)[0]["generated_text"]
        st.success("Generated Text:")
        st.write(result)
