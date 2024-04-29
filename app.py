import streamlit as st 
import pandas as pd
from pandasai import SmartDataframe
from langchain_community.llms import Ollama
import os
from PIL import Image

# Initialize Ollama model
llm = Ollama(model="mistral")

## Streamlit app initialization
st.set_page_config(page_title="Talk to Your Data")

# Display Iridium logo
logo_path = "Images/IridiumAILogo.png"
if os.path.exists(logo_path):
    iridium_logo = Image.open(logo_path)
    st.image(iridium_logo, use_column_width=False)

# Main title and file uploader
#st.header("AI-Powered ASK YOUR DATA")
st.title("Ask Your Data")

# File uploader
uploader_file = st.file_uploader("Upload a CSV file", type=["csv"])

# Check for file upload
if uploader_file is not None:
    data = pd.read_csv(uploader_file)
    st.write(data.head(3))
    
    # Initialize SmartDataframe with LLM model
    df = SmartDataframe(data, config={"llm": llm})
    
    # Prompt input
    prompt = st.text_area("Enter your prompt:")
    
    # Generate response button
    if st.button("Generate"):
        if prompt:
            with st.spinner("Generating response..."):
                response = df.chat(prompt)
                st.write(response)
        else:
            st.warning("Please enter a prompt!")
