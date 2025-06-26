import os
import streamlit as st
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv('Api_Key')

# Set page config
st.set_page_config(page_title="LLM Chatbot", layout="centered")

# Initialize client
client = Groq(api_key=api_key)
# Title
st.title("💬 🤖 GroqTalk: LLM Playground")
# Models dropdown
st.sidebar.title("LLM Chatbot Settings")
model = st.sidebar.selectbox('Choose a model', ['compound-beta', 'compound-beta-mini'])




# Initialize history
if "history" not in st.session_state:
    st.session_state.history = []

# Display conversation history
for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User input
user_input = st.chat_input("Type your message...")
if user_input:
    # Display user message
    st.chat_message("user").markdown(user_input)
    st.session_state.history.append({"role": "user", "content": user_input})

    # Call LLM
    with st.spinner("Thinking..."):
        chat = client.chat.completions.create(
            messages=st.session_state.history,
            model=model,
        )
        response = chat.choices[0].message.content

    # Display assistant message
    st.chat_message("assistant").markdown(response)
    st.session_state.history.append({"role": "assistant", "content": response})
