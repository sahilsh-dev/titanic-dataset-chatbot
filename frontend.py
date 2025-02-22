import streamlit as st
import requests
import base64
from PIL import Image
from io import BytesIO

BACKEND_URL = "http://localhost:8000/ask"

st.title("Titanic Dataset Chatbot 🚢")
st.write("Ask questions about the Titanic passenger dataset!")

question = st.text_input("Your Question:", key="input")

if question:
    with st.spinner("Analyzing..."):
        response = requests.post(BACKEND_URL, json={"question": question})

        if response.status_code == 200:
            result = response.json()
            st.subheader("Answer:")
            st.write(result["text"])

            if result["plot"]:
                img = Image.open(BytesIO(base64.b64decode(result["plot"])))
                st.image(img, caption="Visualization", use_container_width=True)
        else:
            st.error(f"Error: {response.json().get('detail', 'Unknown error')}")
