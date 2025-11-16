import nest_asyncio
nest_asyncio.apply()

import streamlit as st
import os
from datetime import datetime
from docbot import DocBot

# Page configuration
st.set_page_config(
    page_title="Document Chatbot - Gemini AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #667eea;
    }
    
    .source-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
        margin: 0.5rem 0;
    }
    
    .error-box {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #dc3545;
    }
    
    .success-box {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 5px;
        border-left: 3px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

def initialize_chatbot():
    """Initialize the chatbot instance"""
    if 'chatbot' not in st.session_state:
        try:
            st.session_state.chatbot = DocBot()
            st.session_state.chatbot_initialized = True
        except Exception as e:
            st.error(f"Failed to initialize chatbot: {str(e)}")
            st.session_state.chatbot_initialized = False
    
    return st.session_state.get('chatbot_initialized', False)

def handle_file_upload():
    """Handle document upload and processing"""
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'txt', 'md'],
        help="Supported formats: PDF, TXT, Markdown",
        key="file_uploader"
    )
    
    if uploaded_file is not None:
        # Check if this is a new file
        file_key = f"{uploaded_file.name}_{uploaded_file.size}"
        
        if st.session_state.get('current_file_key') != file_key:
            with st.spinner("🔄 Processing document..."):
                try:
                    # Process the document
                    result = st.session_state.chatbot.doc_loader(
                        uploaded_file.getvalue(),
                        uploaded_file.name,
                        uploaded_file.type
                    )
                    
                    if result["success"]:
                        st.session_state.current_file_key = file_key
                        st.session_state.current_filename = uploaded_file.name
                        st.session_state.processing_result = result
                        
                        st.markdown(f"""
                        <div class="success-box">
                            ✅ <strong>Document processed successfully!</strong><br>
                            📄 File: {uploaded_file.name}<br>
                            📊 Chunks: {result['chunk_count']}<br>
                            📝 Characters: {result.get('total_characters', 'Unknown')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Clear previous chat when new document is loaded
                        st.session_state.messages = []
                        
                    else:
                        st.markdown(f"""
                        <div class="error-box">
                            ❌ <strong>Failed to process document</strong><br>
                            Error: {result.get('error', 'Unknown error')}
                        </div>
                        """, unsafe_allow_html=True)
                        
                except Exception as e:
                    st.error(f"Unexpected error: {str(e)}")
    
    return uploaded_file is not None

def display_chat_interface():
    """Display the main chat interface"""
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat history
    chat_container = st.container()
    
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                
                # Show sources if available
                if message["role"] == "assistant" and "sources" in message and message["sources"]:
                    with st.expander("📚 View Sources", expanded=False):
                        for source in message["sources"]:
                            st.markdown(f"""
                            <div class="source-box">
                                <strong>Source {source['index']}:</strong><br>
                                {source['content']}
                            </div>
                            """, unsafe_allow_html=True)
    
    # Chat input
    if prompt := st.chat_input("Ask a question about your document..."):
        if not st.session_state.get('current_filename'):
            st.warning("⚠️ Please upload a document first!")
            return
        
        # Add user message to chat history
        st.session_state.messages.append({
            "role": "user", 
            "content": prompt,
            "timestamp": datetime.now()
        })
        
        # Display user message
        with st.chat_message("user"):
            st.write(prompt)
        
        # Get AI response
        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):
                response = st.session_state.chatbot.ask_question(prompt)
                
                if response["success"]:
                    st.write(response["answer"])
                    
                    # Show sources
                    if response["sources"]:
                        with st.expander("📚 Sources Used", expanded=False):
                            for source in response["sources"]:
                                st.markdown(f"""
                                <div class="source-box">
                                    <strong>Source {source['index']}:</strong><br>
                                    {source['content']}
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Add assistant message to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["answer"],
                        "sources": response["sources"],
                        "timestamp": datetime.now()
                    })
                else:
                    error_msg = response["answer"]
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": f"❌ {error_msg}",
                        "timestamp": datetime.now()
                    })

def display_sidebar():
    """Display sidebar with controls and information"""
    with st.sidebar:
        st.header("🤖 Document Chatbot")
        st.markdown("*Powered by Google Gemini*")
        
        # File upload section
        st.markdown("### 📁 Upload Document")
        file_uploaded = handle_file_upload()
        
        # Document info
        if st.session_state.get('current_filename'):
            st.markdown("---")
            st.markdown("### 📊 Current Document")
            
            summary = st.session_state.chatbot.get_document_summary()
            if summary["loaded"]:
                st.info(f"""
                **File:** {st.session_state.current_filename}  
                **Chunks:** {summary['chunk_count']}  
                **Characters:** {summary['total_characters']:,}  
                **Est. Tokens:** {summary['estimated_tokens']:,}  
                **Status:** ✅ Ready
                """)
            
            # Clear document button
            if st.button("🗑️ Clear Document", use_container_width=True):
                st.session_state.chatbot.clear_document()
                for key in ['current_file_key', 'current_filename', 'processing_result', 'messages']:
                    if key in st.session_state:
                        del st.session_state[key]
                st.rerun()
        
        # Chat controls
        st.markdown("---")
        st.markdown("### 💬 Chat Controls")
        
        if st.button("🔄 Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
        
        # Usage tips
        st.markdown("---")
        st.markdown("### 💡 Tips")
        st.markdown("""
        **Good questions to ask:**
        - "What is this document about?"
        - "Summarize the main points"
        - "What does [term] mean?"
        - "List the key findings"
        - "Explain [concept] from the document"
        
        **File size limits:**
        - PDFs: Up to 200MB
        - Text files: Up to 50MB
        """)
        
        # Model information
        st.markdown("---")
        st.markdown("### ⚙️ Model Info")
        st.markdown("""
        **LLM:** Gemini 1.5 Pro  
        **Embeddings:** Gemini Embeddings  
        **Vector Store:** FAISS  
        **Max Context:** 2M tokens
        """)

def main():
    """Main application function"""
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🤖 Document Chatbot</h1>
        <p>Ask questions about your documents using Google Gemini AI</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Check API key
    if not os.getenv("GOOGLE_API_KEY"):
        st.error("❌ Google API key not found. Please set GOOGLE_API_KEY in your .env file.")
        st.stop()
    
    # Initialize chatbot
    if not initialize_chatbot():
        st.error("❌ Failed to initialize chatbot. Please check your API key and try again.")
        st.stop()
    
    # Layout
    col1, col2 = st.columns([3, 1])
    
    with col1:
        display_chat_interface()
    
    with col2:
        display_sidebar()

if __name__ == "__main__":
    main()