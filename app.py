# app.py

import sys
import types
import re
import streamlit as st
from rag_engine import get_meal_suggestions
from langchain.callbacks.base import BaseCallbackHandler

# 🛠️ Patch torch to avoid Streamlit crash
sys.modules['torch.classes'] = types.SimpleNamespace()

# 🌐 Streamlit callback to stream output token by token
class StreamlitCallbackHandler(BaseCallbackHandler):
    def __init__(self, container):
        self.container = container
        self.tokens = ""

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens += token
        self.container.markdown(format_meal_response(self.tokens + "▌"))

# 🎨 Format LLM response
def format_meal_response(text):
    text = re.sub(r"(\d+\.\s)(.*?)(:)", r"\1**\2**\3", text)  # Bold dish names
    return text.replace("\n", "\n\n")  # Add spacing

# 🚀 UI Setup
st.set_page_config(page_title="FridgeGPT", layout="centered")
st.title("👨‍🍳 FridgeGPT")
st.subheader("Your ingredients turned into quick meals")

# 📝 Input
ingredients = st.text_area("Enter your ingredients (comma-separated):", "")
mode = st.selectbox("Choose your cooking mood", ["normal", "student", "vegan", "hangover"])

# 🍳 Generate meal ideas
if st.button("🍳 Cook Something!"):
    if not ingredients.strip():
        st.warning("Please enter some ingredients.")
    else:
        with st.spinner("receipe is getting cooked..."):
            try:
                container = st.empty()
                stream_handler = StreamlitCallbackHandler(container)
                get_meal_suggestions(ingredients, mode, stream_handler)
            except Exception as e:
                st.error(f"Error: {e}")
