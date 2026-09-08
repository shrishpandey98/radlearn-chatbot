"""
streamlit_app.py
────────────────
Main entry point for the RadLearn MVP UI Phase 4B.
"""
import streamlit as st
from ui.styles import apply_styles
from ui.sidebar import render_sidebar
from ui.chat import render_chat

# 1. Page Configuration
st.set_page_config(
    page_title="RadLearn | Radiology CME",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. Inject CSS
apply_styles()

# 3. Render Sidebar
render_sidebar()

# 4. Render Main Chat Interface
render_chat()
