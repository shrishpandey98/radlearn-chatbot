"""
streamlit_app.py
────────────────
Main entry point for the RadLearn MVP UI Phase 4B.
"""
# Streamlit Cloud uses an older SQLite version by default; ChromaDB requires >= 3.35.
# This overrides the default sqlite3 with the newer pysqlite3-binary.
import sys
__import__('pysqlite3')
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

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
