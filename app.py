import streamlit as st
from chatbot.mindease_ai import MindEaseAI
import time
from utils.vectorstore_manager import VectorStoreManager
import utils.vectorstore_manager as vsm

st.set_page_config(
    page_title="Calmera AI - friendly companion for peace of mind",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="expanded"
)


st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #2E7D32;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .crisis-warning {
        background-color: #FFF3E0;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #FF6F00;
        margin: 1rem 0;
    }
    .sentiment-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .positive { background-color: #E8F5E9; color: #2E7D32; }
    .negative { background-color: #FFEBEE; color: #C62828; }
    .neutral { background-color: #F5F5F5; color: #616161; }
</style>
""", unsafe_allow_html=True)


def init_session_state():
    """Initialize session state variables"""
    if "initialized" not in st.session_state:
        st.session_state.initialized = False
    
    if not st.session_state.initialized:
        with st.spinner("🌱 Initializing Calmera AI..."):
            try:
                
                
                vectorstore_manager = VectorStoreManager()
                vectorstore_manager.create_vectorstore()
                
                mindease = MindEaseAI(vectorstore_manager=vectorstore_manager)
                
                st.session_state.mindease = mindease
                st.session_state.messages = []
                st.session_state.show_sentiment = False
                st.session_state.initialized = True
                
                time.sleep(1)
                st.rerun()
                
            except Exception as e:
                st.error(f"❌ Initialization failed: {str(e)}")
                st.write("Full error:", e)
                st.stop()

init_session_state()


with st.sidebar:
    st.markdown("### 🌱 Calmera")
    st.markdown("*Your compassionate wellness companion*")
    st.divider()
    
    
    if st.session_state.mindease.rag_enabled:
        st.success("📚 Wellness Guides: Loaded")
        
        if hasattr(st.session_state.mindease, 'vectorstore_manager'):
            st.caption("Wellness Guides available")
    else:
        st.warning("📚 Wellness Guides: Not available")
        st.caption("Add PDF guides to `data/guides/` to enable")
    
    st.divider()
    
   
    st.markdown("### ⚙️ Settings")
    st.session_state.show_sentiment = st.checkbox(
        "Show emotional analysis",
        value=st.session_state.show_sentiment
    )
    
  
    if st.session_state.messages:
        emotional_state = st.session_state.mindease.get_emotional_summary()
        st.info(f"💭 Recent mood: {emotional_state}")
    
    st.divider()
    
  
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.mindease.clear_conversation()
        st.session_state.messages = []
        st.rerun()
    
    if st.button("📖 About Calmera", use_container_width=True):
        st.info("""
        MindEase uses LangChain and Cohere to provide:
        - Empathetic conversation
        - Crisis detection
        - Evidence-based wellness guidance
        - Emotional awareness
        
        Remember: This is not a replacement for professional mental health care.
        """)
      
        
        
        st.divider()
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 0.75rem;'>
        <p>© 2025 All Rights Reserved</p>
        <p>Developed by <strong>Samadhi Jagathsiri</strong></p>
    </div>
    """, unsafe_allow_html=True)
    
    


st.markdown('<h1 class="main-header">🌱 Calmera AI</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">A safe space for your thoughts and feelings</p>', unsafe_allow_html=True)


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
        if st.session_state.show_sentiment and "sentiment" in message:
            sentiment = message["sentiment"]
            polarity = sentiment.get("polarity", 0)
            
            if polarity > 0.1:
                badge_class = "positive"
                emoji = "😊"
            elif polarity < -0.1:
                badge_class = "negative"
                emoji = "😔"
            else:
                badge_class = "neutral"
                emoji = "😐"
            
            st.markdown(
                f'<span class="sentiment-badge {badge_class}">{emoji} {sentiment.get("label", "neutral")}</span>',
                unsafe_allow_html=True
            )


if prompt := st.chat_input("Share what's on your mind..."):
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = st.session_state.mindease.process_message(prompt)
                
                response = result["response"]
                sentiment = result["sentiment"]
                crisis = result["crisis_detected"]
                used_rag = result["used_rag"]
                
                st.markdown(response)
                
               
                if crisis:
                    st.markdown('<div class="crisis-warning">⚠️ Crisis resources provided</div>', unsafe_allow_html=True)
                
                if used_rag:
                    st.caption("📚 Response enhanced with wellness guides")
                
               
                if st.session_state.show_sentiment:
                    polarity = sentiment.get("polarity", 0)
                    
                    if polarity > 0.1:
                        badge_class = "positive"
                        emoji = "😊"
                    elif polarity < -0.1:
                        badge_class = "negative"
                        emoji = "😔"
                    else:
                        badge_class = "neutral"
                        emoji = "😐"
                    
                    st.markdown(
                        f'<span class="sentiment-badge {badge_class}">{emoji} {sentiment.get("label", "neutral")}</span>',
                        unsafe_allow_html=True
                    )
                
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": response,
                    "sentiment": sentiment,
                    "crisis": crisis,
                    "used_rag": used_rag
                })
                
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.write("Please try again or start a new conversation.")                                                                                                     


st.divider()
st.caption("⚠️ Calmera is a supportive tool, not a substitute for professional mental health care. If you're in crisis, please contact emergency services or a crisis helpline.")


