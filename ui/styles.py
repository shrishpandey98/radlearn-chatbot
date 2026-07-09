"""
ui/styles.py
────────────
Injects 'MEDORA AI' styled dark-themed CSS into the Streamlit application.
"""
import streamlit as st

def apply_styles():
    st.markdown("""
        <style>
        /* MEDORA AI theme - sleek dark interface */
        
        /* Chat message bubbles */
        [data-testid="stChatMessage"] {
            background-color: transparent !important;
            border-radius: 12px;
            padding: 1.5rem;
            margin-bottom: 1rem;
        }

        /* AI chat bubble */
        [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(even) {
            background-color: #0b1320 !important; /* Secondary dark */
            border: 1px solid #1e293b;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        }

        /* User chat bubble */
        [data-testid="stChatMessage"][data-baseweb="block"]:nth-child(odd) {
            background-color: transparent !important;
            border-left: 4px solid #00e5ff; /* Vibrant Cyan accent */
        }
        
        /* Typography adjustments for readability */
        p, li {
            color: #ffffff; /* Pure white text */
            line-height: 1.6;
        }
        
        /* Expanders */
        .streamlit-expanderHeader {
            background-color: #0b1320 !important;
            border-radius: 8px;
            border: 1px solid #1e293b;
            color: #e2e8f0;
            font-weight: 600;
        }
        .streamlit-expanderContent {
            border-left: 1px solid #1e293b;
            border-right: 1px solid #1e293b;
            border-bottom: 1px solid #1e293b;
            background-color: #040814;
            border-bottom-left-radius: 8px;
            border-bottom-right-radius: 8px;
            padding: 1rem;
        }

        /* Citation markers */
        a.citation-marker {
            background-color: #0f172a;
            color: #00e5ff;
            border-radius: 4px;
            padding: 2px 6px;
            font-size: 0.8em;
            font-weight: bold;
            vertical-align: super;
            border: 1px solid #00e5ff;
            margin: 0 2px;
            text-decoration: none;
            transition: all 0.2s ease;
            cursor: pointer;
        }
        a.citation-marker:hover {
            background-color: #00e5ff;
            color: #040814;
            text-decoration: none;
            box-shadow: 0 0 8px rgba(0, 229, 255, 0.5);
        }

        /* Source cards */
        .source-card {
            background-color: #0b1320;
            border: 1px solid #1e293b;
            border-radius: 8px;
            padding: 16px;
            margin-bottom: 16px;
            box-shadow: 0 2px 5px rgba(0,0,0,0.2);
        }
        .source-card h4 {
            margin-top: 0;
            color: #00e5ff;
            font-size: 1rem;
            margin-bottom: 8px;
        }
        .source-card p {
            font-size: 0.9rem;
            color: #94a3b8;
            margin-bottom: 4px;
        }
        
        /* Debug chunk cards */
        .chunk-card {
            background-color: #0f172a;
            border: 1px dashed #334155;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 12px;
        }
        .chunk-card h5 {
            margin-top: 0;
            color: #cbd5e1;
            font-size: 0.95rem;
        }

        /* Score badges */
        .score-badge {
            display: inline-block;
            background-color: rgba(0, 229, 255, 0.1);
            color: #00e5ff;
            border: 1px solid rgba(0, 229, 255, 0.3);
            font-size: 0.75rem;
            padding: 4px 8px;
            border-radius: 12px;
            font-weight: bold;
            margin-top: 8px;
        }

        /* Metrics */
        .metric-container {
            display: flex;
            gap: 16px;
            margin-top: 16px;
            margin-bottom: 16px;
        }
        .metric-pill {
            background-color: #0b1320;
            border: 1px solid #1e293b;
            border-radius: 16px;
            padding: 6px 16px;
            font-size: 0.85rem;
            color: #e2e8f0;
            display: flex;
            align-items: center;
            gap: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        }

        /* Chat input text and placeholder */
        [data-testid="stChatInputTextArea"] {
            color: #ffffff !important;
        }
        [data-testid="stChatInputTextArea"]::placeholder {
            color: #ffffff !important;
            opacity: 0.8;
        }

        /* All main content button text black and bold */
        section.main div.stButton button p {
            color: #000000 !important;
            font-weight: 600 !important;
        }

        /* Sidebar buttons should remain white */
        section[data-testid="stSidebar"] div.stButton button p {
            color: #ffffff !important;
            font-weight: 500 !important;
        }

        /* Upload button styling (positioning handled safely by JS) */
        div[data-testid="stPopover"] > div > button {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            color: #94a3b8 !important;
            font-size: 1.5rem !important;
            width: 45px !important;
            height: 45px !important;
            display: flex;
            align-items: center;
            justify-content: center;
            box-shadow: none !important;
        }
        div[data-testid="stPopover"] > div > button p {
            color: #94a3b8 !important;
            font-size: 1.5rem !important;
        }
        div[data-testid="stPopover"] > div > button:hover {
            color: #ffffff !important;
        }
        div[data-testid="stPopover"] > div > button:hover p {
            color: #ffffff !important;
        }
        div[data-testid="stChatInput"] {
            padding-left: 55px !important;
        }
        </style>
    """, unsafe_allow_html=True)
    
    import streamlit.components.v1 as components
    components.html("""
        <script>
        function updatePopoverPosition() {
            // ONLY select popovers in the main content area, ignoring sidebar 3-dot menus
            const mainPopovers = Array.from(window.parent.document.querySelectorAll('section.main div[data-testid="stPopover"]'));
            const chatInputs = window.parent.document.querySelectorAll('div[data-testid="stChatInput"]');
            
            // Find the specific popover that contains the 📎 icon
            const uploadPopover = mainPopovers.find(p => p.textContent.includes("📎"));
            
            if (uploadPopover && chatInputs.length > 0) {
                const popover = uploadPopover;
                const chatInput = chatInputs[0];
                
                const rect = chatInput.getBoundingClientRect();
                
                // Set absolute positioning via inline styles so React doesn't crash
                popover.style.position = 'fixed';
                popover.style.left = (rect.left + 5) + 'px';
                // Center vertically inside the chat input box (height ~45px)
                popover.style.top = (rect.top + (rect.height / 2) - 22) + 'px';
                popover.style.zIndex = '9999';
            }
        }
        setInterval(updatePopoverPosition, 100);
        </script>
    """, height=0, width=0)

