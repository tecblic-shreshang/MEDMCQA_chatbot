import streamlit as st
from medmcq_chatbot import initialize_chatbot

# Set up the Streamlit page
st.set_page_config(page_title="🩺 MedMCQA Chatbot", layout="wide")
st.title("🩺 MedMCQA Medical Chatbot")
st.markdown("Ask any medical question. Powered by the MedMCQA dataset.")

# Initialize the chatbot only once
@st.cache_resource(show_spinner="Loading chatbot...")
def load_chatbot():
    return initialize_chatbot()

chatbot = load_chatbot()

# Chat input
user_query = st.text_input("Your Question:", placeholder="E.g., What is the normal blood pressure?")

if user_query:
    with st.spinner("Thinking..."):
        response = chatbot.chat(user_query)
        st.markdown("---")
        st.markdown("### 🤖 Response")
        st.markdown(response)

st.markdown("---")
st.caption("⚠️ This chatbot uses educational datasets and does not substitute professional medical advice.")
