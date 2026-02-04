"""
Multi-Modal RAG: Standalone Demo Version
All code in one file for Hugging Face Spaces compatibility

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
import fitz  # PyMuPDF
from typing import List, Dict, Tuple, Optional, Union

# Mock implementations for demo (no heavy dependencies)
print("🔄 Loading Multi-Modal RAG Demo...")

class SimpleDocumentProcessor:
    """Simple PDF processor for demo."""
    
    def process_pdf(self, pdf_path: str) -> Dict:
        """Extract text and images from PDF."""
        doc = fitz.open(pdf_path)
        pages = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            
            # Get page as image
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            
            pages.append({
                'page_num': page_num,
                'text': text,
                'page_image': img
            })
        
        doc.close()
        return {'pages': pages, 'metadata': {'total_pages': len(pages)}}
    
    def create_chunks(self, document: Dict, chunk_size: int = 1000) -> List[Dict]:
        """Create chunks from document."""
        chunks = []
        for page in document['pages']:
            text = page['text']
            # Simple chunking
            for i in range(0, len(text), chunk_size):
                chunk_text = text[i:i+chunk_size]
                if chunk_text.strip():
                    chunks.append({
                        'text': chunk_text,
                        'page_num': page['page_num'],
                        'chunk_type': 'text'
                    })
            
            # Add image reference
            chunks.append({
                'text': f"[Image: Page {page['page_num']} contains visual content]",
                'page_num': page['page_num'],
                'chunk_type': 'image'
            })
        
        return chunks


class SimpleRAGSystem:
    """Simple RAG system for demo."""
    
    def __init__(self):
        self.chunks = []
        self.document_name = ""
    
    def index_document(self, file_path: str) -> bool:
        """Index a PDF document."""
        try:
            processor = SimpleDocumentProcessor()
            doc = processor.process_pdf(file_path)
            self.chunks = processor.create_chunks(doc)
            self.document_name = os.path.basename(file_path)
            print(f"✅ Indexed {len(self.chunks)} chunks from {self.document_name}")
            return True
        except Exception as e:
            print(f"❌ Error indexing: {e}")
            return False
    
    def query(self, question: str, k: int = 3) -> Dict:
        """Query the document."""
        if not self.chunks:
            return {
                'answer': "Please upload a document first!",
                'sources': []
            }
        
        # Simple keyword matching for demo
        question_lower = question.lower()
        relevant_chunks = []
        
        for chunk in self.chunks:
            score = 0
            chunk_text_lower = chunk['text'].lower()
            
            # Check for keyword overlap
            question_words = set(question_lower.split())
            chunk_words = set(chunk_text_lower.split())
            overlap = len(question_words & chunk_words)
            score = overlap
            
            if score > 0 or len(relevant_chunks) < k:
                relevant_chunks.append({
                    'chunk': chunk,
                    'score': score
                })
        
        # Sort by score and take top k
        relevant_chunks.sort(key=lambda x: x['score'], reverse=True)
        top_chunks = relevant_chunks[:k]
        
        # Generate demo response
        sources = []
        for rc in top_chunks:
            meta = rc['chunk']
            sources.append(f"Page {meta['page_num'] + 1}")
        
        # Create contextual response
        if top_chunks:
            preview = top_chunks[0]['chunk']['text'][:200] if top_chunks[0]['chunk']['text'] else "Visual content"
            answer = f"""Based on your document **{self.document_name}**, here's what I found:

**Regarding your question:** *{question}*

From the retrieved content (Pages: {', '.join(set(sources))}):

• The document contains relevant information about this topic
• Key sections discuss related concepts in {len(set(sources))} page(s)
• Preview from most relevant section:
  \"{preview}...\"

---

**Note:** This is a **demo version** with simplified retrieval. For production use with:
- ✨ Semantic search with embeddings
- 🖼️  Visual understanding of charts/diagrams  
- 🧠 GPT-4/Claude integration

Deploy with full dependencies (see GitHub for complete version)."""
        else:
            answer = f"I couldn't find specific information about '{question}' in the document. Try asking about topics mentioned in the text."
        
        return {
            'answer': answer,
            'sources': list(set(sources)),
            'chunks_found': len(top_chunks)
        }


# Global RAG instance
RAG = SimpleRAGSystem()


def process_document(file_obj):
    """Process uploaded PDF."""
    if file_obj is None:
        return "❌ No file uploaded", None
    
    try:
        file_path = file_obj.name if hasattr(file_obj, 'name') else file_obj
        success = RAG.index_document(file_path)
        
        if success:
            return (
                f"✅ Document processed successfully!\n"
                f"📄 {RAG.document_name}\n"
                f"📊 {len(RAG.chunks)} chunks extracted",
                file_path
            )
        else:
            return "❌ Failed to process document", None
            
    except Exception as e:
        import traceback
        traceback.print_exc()
        return f"❌ Error: {str(e)}", None


def chat_with_document(message, history, document_state):
    """Handle chat."""
    if document_state is None:
        history.append((message, "❌ Please upload a document first!"))
        return history, "Upload a PDF to start."
    
    if not message.strip():
        return history, "Type a question."
    
    try:
        result = RAG.query(message)
        answer = result['answer']
        
        if result['sources']:
            answer += f"\n\n**📍 Sources:** {', '.join(result['sources'])}"
        
        history.append((message, answer))
        return history, "Ready for next question."
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        error_msg = f"❌ Error: {str(e)}"
        history.append((message, error_msg))
        return history, error_msg


def clear_chat():
    """Clear chat."""
    return [], "Chat cleared. Upload a document to start."


# CSS
custom_css = """
.gradio-container { max-width: 1200px !important; }
.chatbot { height: 500px; }
.footer { text-align: center; margin-top: 20px; color: #666; font-size: 0.9em; }
"""

# Build UI
with gr.Blocks(title="Multi-Modal RAG Demo") as demo:
    
    gr.Markdown("""
    # 🎯 Multi-Modal RAG Demo
    ### Chat with PDF documents (Demo Version)
    
    **Upload a PDF** → **Ask questions** → **Get answers with page citations**
    
    🌐 **Live Demo** | 📁 [GitHub](https://github.com/vraul92/multimodal-rag)
    """)
    
    document_state = gr.State(None)
    
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload Document")
            
            file_upload = gr.File(label="Upload PDF", file_types=[".pdf"], type="filepath")
            upload_btn = gr.Button("🔍 Process Document", variant="primary")
            
            status_text = gr.Textbox(
                label="Status",
                value="Upload a PDF to start",
                interactive=False,
                lines=3
            )
            
            gr.Markdown("""
            #### 💡 Example Questions:
            - "What is this document about?"
            - "Summarize the main points"
            - "What does Figure 1 show?"
            - "Explain the methodology"
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 💬 Chat")
            
            chatbot = gr.Chatbot(label="Conversation", height=500)
            
            with gr.Row():
                msg_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Ask about the document...",
                    scale=4
                )
                submit_btn = gr.Button("Send", variant="primary", scale=1)
            
            clear_btn = gr.Button("🗑️ Clear Chat")
    
    gr.Markdown("""
    ---
    <div class="footer">
    <b>Multi-Modal RAG Demo</b> by <a href="https://github.com/vraul92">Rahul Vuppalapati</a> | 
    <a href="https://linkedin.com/in/vrc7">LinkedIn</a>
    </div>
    """)
    
    # Events
    upload_btn.click(fn=process_document, inputs=[file_upload], outputs=[status_text, document_state])
    file_upload.change(fn=process_document, inputs=[file_upload], outputs=[status_text, document_state])
    
    submit_btn.click(fn=chat_with_document, inputs=[msg_input, chatbot, document_state], outputs=[chatbot, status_text])
    msg_input.submit(fn=chat_with_document, inputs=[msg_input, chatbot, document_state], outputs=[chatbot, status_text])
    clear_btn.click(fn=clear_chat, outputs=[chatbot, status_text])


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    print("🚀 Starting Multi-Modal RAG Demo...")
    
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=False,
        show_error=True,
        theme=gr.themes.Soft(),
        css=custom_css
    )
