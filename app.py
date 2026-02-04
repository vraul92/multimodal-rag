"""
Multi-Modal RAG: Interactive Web Application
Gradio interface optimized for Hugging Face Spaces

Author: Rahul Vuppalapati
GitHub: https://github.com/vraul92
"""

import gradio as gr
import os
import sys
import numpy as np
from PIL import Image
import io
import base64

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Global instances
RAG_SYSTEM = None
EMBEDDING_ENGINE = None
VECTOR_STORE = None


def initialize_system():
    """Initialize RAG components (lazy loading)."""
    global RAG_SYSTEM, EMBEDDING_ENGINE, VECTOR_STORE
    
    if RAG_SYSTEM is not None:
        return RAG_SYSTEM
    
    print("🔄 Initializing Multi-Modal RAG...")
    
    from embedding_engine import MultiModalEmbedding, VectorStore
    from rag_pipeline import MultiModalRetriever, AnswerGenerator, MultiModalRAG
    
    # Initialize components
    EMBEDDING_ENGINE = MultiModalEmbedding()
    VECTOR_STORE = VectorStore(dimension=512)
    retriever = MultiModalRetriever(VECTOR_STORE, EMBEDDING_ENGINE)
    
    # Use mock generator for HF Spaces (no API key needed)
    generator = AnswerGenerator(model="mock")
    
    RAG_SYSTEM = MultiModalRAG(
        EMBEDDING_ENGINE,
        VECTOR_STORE,
        retriever,
        generator
    )
    
    print("✅ System ready!")
    return RAG_SYSTEM


def process_document(file_obj):
    """
    Process uploaded PDF and index it.
    
    Args:
        file_obj: Gradio file object
        
    Returns:
        Status message and document info
    """
    if file_obj is None:
        return "❌ No file uploaded", None
    
    try:
        rag = initialize_system()
        
        # Get file path
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        
        print(f"📄 Processing: {os.path.basename(file_path)}")
        
        # Index document
        success = rag.index_document(file_path)
        
        if success:
            num_chunks = len(VECTOR_STORE.metadata)
            return (
                f"✅ Document indexed successfully!\n"
                f"📊 Extracted {num_chunks} chunks (text + images)",
                file_path
            )
        else:
            return "❌ Failed to process document", None
            
    except Exception as e:
        print(f"❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None


def chat_with_document(message, history, document_state):
    """
    Handle chat interaction.
    
    Args:
        message: User message
        history: Chat history
        document_state: Currently loaded document
        
    Returns:
        Updated history and visible elements
    """
    if document_state is None:
        history.append((message, "❌ Please upload a document first!"))
        return history, "Please upload a PDF document to start chatting."
    
    if not message.strip():
        return history, "Type a question about the document."
    
    try:
        rag = initialize_system()
        
        # Query
        result = rag.query(message, k=5)
        
        # Format answer
        answer = result['answer']
        
        # Add citations info
        citations = []
        for chunk in result.get('retrieved_chunks', []):
            meta = chunk['metadata']
            if meta.get('chunk_type') == 'image':
                citations.append(f"📷 Page {meta.get('page_num', '?')}")
            else:
                citations.append(f"📄 Page {meta.get('page_num', '?')}")
        
        if citations:
            answer += f"\n\n**Sources:** {', '.join(set(citations))}"
        
        history.append((message, answer))
        
        # Return with thinking indicator cleared
        return history, "Ask another question or upload a different document."
        
    except Exception as e:
        error_msg = f"❌ Error processing query: {str(e)}"
        history.append((message, error_msg))
        return history, error_msg


def clear_chat():
    """Clear chat history."""
    return [], "Chat cleared. Ready for new conversation."


# CSS for better styling
custom_css = """
.gradio-container {
    max-width: 1200px !important;
}
.chatbot {
    height: 500px;
}
.footer {
    text-align: center;
    margin-top: 20px;
    color: #666;
    font-size: 0.9em;
}
"""

# Build Gradio interface
with gr.Blocks(
    title="Multi-Modal RAG: Documents That See",
    theme=gr.themes.Soft(),
    css=custom_css
) as demo:
    
    gr.Markdown("""
    # 🎯 Multi-Modal RAG
    ### Chat with documents containing text, charts, and images
    
    **Upload a PDF** → **Ask questions** about text AND visuals → **Get cited answers**
    """)
    
    # State to track document
    document_state = gr.State(None)
    
    with gr.Row():
        # Left column - Document upload
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Document")
            
            file_upload = gr.File(
                label="Upload PDF",
                file_types=[".pdf"],
                type="filepath"
            )
            
            upload_btn = gr.Button("🔍 Process Document", variant="primary")
            
            status_text = gr.Textbox(
                label="Status",
                value="Upload a PDF to get started",
                interactive=False,
                lines=3
            )
            
            gr.Markdown("""
            #### 💡 Example Questions:
            - "What does Figure 3 show?"
            - "Explain the trend in the bar chart"
            - "Summarize the key findings"
            - "What architecture is described?"
            """)
            
            gr.Markdown("""
            #### ℹ️ About:
            - ✅ Understands text + charts + diagrams
            - ✅ Cites sources with page numbers
            - ✅ Zero setup - runs in browser
            """)
        
        # Right column - Chat
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat with Document")
            
            chatbot = gr.Chatbot(
                label="Conversation",
                height=500,
                bubble_full_width=False
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about the document...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear Chat")
            
            info_text = gr.Textbox(
                value="Upload a PDF document to start chatting.",
                interactive=False,
                show_label=False
            )
    
    # Footer
    gr.Markdown("""
    ---
    
    <div class="footer">
    
    **Multi-Modal RAG** by <a href="https://github.com/vraul92" target="_blank">Rahul Vuppalapati</a> | 
    <a href="https://github.com/vraul92/multimodal-rag" target="_blank">GitHub</a> | 
    <a href="https://linkedin.com/in/vrc7" target="_blank">LinkedIn</a>
    
    Built with ❤️ using Gradio + PyTorch + CLIP
    
    </div>
    """)
    
    # Event handlers
    upload_btn.click(
        fn=process_document,
        inputs=[file_upload],
        outputs=[status_text, document_state]
    )
    
    # Also trigger on file upload directly
    file_upload.change(
        fn=process_document,
        inputs=[file_upload],
        outputs=[status_text, document_state]
    )
    
    # Chat submission
    submit_btn.click(
        fn=chat_with_document,
        inputs=[msg_input, chatbot, document_state],
        outputs=[chatbot, info_text]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )
    
    # Enter key in message box
    msg_input.submit(
        fn=chat_with_document,
        inputs=[msg_input, chatbot, document_state],
        outputs=[chatbot, info_text]
    ).then(
        fn=lambda: "",
        outputs=[msg_input]
    )
    
    # Clear chat
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, info_text]
    )


if __name__ == "__main__":
    # Get port from environment (HF Spaces sets this)
    port = int(os.environ.get("PORT", 7860))
    
    print("🚀 Starting Multi-Modal RAG...")
    print(f"📱 App will be available at: http://localhost:{port}")
    print("")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        quiet=False
    )
