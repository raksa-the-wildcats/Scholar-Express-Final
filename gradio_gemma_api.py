"""
Standalone RAG Chatbot with Gemini API
A simple PDF chatbot using Retrieval-Augmented Generation with Google's Gemini API
"""

import gradio as gr
import os
import numpy as np
import pymupdf  # PyMuPDF for PDF processing

# RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    import google.generativeai as genai
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    RAG_AVAILABLE = False

# Global variables
embedding_model = None
gemini_model = None
document_chunks = []
document_embeddings = None
processed_text = ""

def initialize_models():
    """Initialize embedding model and Gemini API"""
    global embedding_model, gemini_model
    
    if not RAG_AVAILABLE:
        return False, "Required dependencies not installed"
    
    try:
        # Initialize embedding model (CPU to save resources)
        if embedding_model is None:
            print("Loading embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("✅ Embedding model loaded successfully")
        
        # Configure Gemini API
        if gemini_model is None:
            api_key = os.getenv('GEMINI_API_KEY')
            if not api_key:
                return False, "GEMINI_API_KEY not found in environment variables"
            
            print("Configuring Gemini API...")
            genai.configure(api_key=api_key)
            gemini_model = genai.GenerativeModel('gemma-3n-e4b-it')
            print("✅ Gemini model initialized successfully")
        
        return True, "All models ready"
        
    except Exception as e:
        print(f"Error initializing: {e}")
        import traceback
        traceback.print_exc()
        return False, f"Error: {str(e)}"

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        if isinstance(pdf_file, str):
            # File path
            pdf_document = pymupdf.open(pdf_file)
        else:
            # File object
            pdf_bytes = pdf_file.read()
            pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        text_content = ""
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            text_content += f"\n--- Page {page_num + 1} ---\n"
            text_content += page.get_text()
        
        pdf_document.close()
        return text_content
        
    except Exception as e:
        raise Exception(f"Error extracting text from PDF: {str(e)}")

def chunk_text(text, chunk_size=500, overlap=50):
    """Split text into overlapping chunks"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embeddings(chunks):
    """Create embeddings for text chunks"""
    if embedding_model is None:
        return None
    
    try:
        print(f"Creating embeddings for {len(chunks)} chunks...")
        embeddings = embedding_model.encode(chunks, show_progress_bar=True)
        return np.array(embeddings)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=3):
    """Retrieve most relevant chunks for a question"""
    if embedding_model is None or embeddings is None:
        return chunks[:top_k]
    
    try:
        question_embedding = embedding_model.encode([question])
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return chunks[:top_k]

def process_pdf(pdf_file, progress=gr.Progress()):
    """Process uploaded PDF and prepare for Q&A"""
    global document_chunks, document_embeddings, processed_text
    
    if pdf_file is None:
        return "❌ Please upload a PDF file first"
    
    try:
        # Extract text from PDF
        progress(0.2, desc="Extracting text from PDF...")
        text = extract_text_from_pdf(pdf_file)
        
        if not text.strip():
            return "❌ No text found in PDF"
        
        processed_text = text
        
        # Create chunks
        progress(0.4, desc="Creating text chunks...")
        document_chunks = chunk_text(text)
        
        # Create embeddings
        progress(0.6, desc="Creating embeddings...")
        document_embeddings = create_embeddings(document_chunks)
        
        if document_embeddings is None:
            return "❌ Failed to create embeddings"
        
        progress(1.0, desc="PDF processed successfully!")
        return f"✅ PDF processed successfully! Created {len(document_chunks)} chunks. You can now ask questions about the document."
        
    except Exception as e:
        return f"❌ Error processing PDF: {str(e)}"

def chat_with_pdf(message, history):
    """Generate response using RAG with Gemini API"""
    global gemini_model
    
    if not message.strip():
        return history
    
    if not processed_text:
        return history + [[message, "❌ Please upload and process a PDF first"]]
    
    # Check if model is initialized
    if gemini_model is None:
        print("Model not initialized, attempting to initialize...")
        success, error_msg = initialize_models()
        if not success:
            return history + [[message, f"❌ Failed to initialize: {error_msg}"]]
    
    try:
        # Retrieve relevant chunks
        if document_chunks and document_embeddings is not None:
            relevant_chunks = retrieve_relevant_chunks(message, document_chunks, document_embeddings)
            context = "\n\n".join(relevant_chunks)
        else:
            # Fallback to truncated text
            context = processed_text[:2000] + "..." if len(processed_text) > 2000 else processed_text
        
        # Create prompt for Gemini
        prompt = f"""You are a helpful assistant that answers questions about documents. Use the provided context to answer questions accurately and concisely.

Context:
{context}

Question: {message}

Please provide a clear and helpful answer based on the context provided."""
        
        # Generate response using Gemini API
        response = gemini_model.generate_content(prompt)
        
        response_text = response.text if hasattr(response, 'text') else str(response)
        
        return history + [[message, response_text]]
        
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        print(f"Full error: {e}")
        import traceback
        traceback.print_exc()
        return history + [[message, error_msg]]

def clear_chat():
    """Clear chat history and processed data"""
    global document_chunks, document_embeddings, processed_text
    document_chunks = []
    document_embeddings = None
    processed_text = ""
    
    return [], "Ready to process a new PDF"

def get_model_status():
    """Get current model loading status"""
    global gemini_model, embedding_model
    
    statuses = []
    
    if embedding_model is not None:
        statuses.append("✅ Embedding model loaded")
    else:
        statuses.append("❌ Embedding model not loaded")
    
    if gemini_model is not None:
        statuses.append("✅ Gemini model ready")
    else:
        statuses.append("❌ Gemini model not initialized")
    
    return " | ".join(statuses)

# Initialize models on startup
model_status = "⏳ Initializing models..."
if RAG_AVAILABLE:
    success, message = initialize_models()
    model_status = "✅ Models ready" if success else f"❌ {message}"
else:
    model_status = "❌ Dependencies not installed"

# Create Gradio interface
with gr.Blocks(
    title="RAG Chatbot with Gemini API",
    theme=gr.themes.Soft(),
    css="""
    .main-container { max-width: 1200px; margin: 0 auto; }
    .status-box { padding: 15px; margin: 10px 0; border-radius: 8px; }
    .chat-container { height: 500px; }
    """
) as demo:
    
    gr.Markdown("# 🤖 RAG Chatbot with Gemini API")
    gr.Markdown("### Upload a PDF and ask questions about it using Retrieval-Augmented Generation powered by Google's Gemini API")
    
    with gr.Row():
        status_display = gr.Markdown(f"**Status:** {model_status}")
        
        # Add refresh button for status
        refresh_btn = gr.Button("♾️ Refresh Status", size="sm")
        
        def update_status():
            return get_model_status()
        
        refresh_btn.click(
            fn=update_status,
            outputs=[status_display]
        )
    
    with gr.Row():
        # Left column - PDF upload
        with gr.Column(scale=1):
            gr.Markdown("## 📄 Upload PDF")
            
            pdf_input = gr.File(
                file_types=[".pdf"],
                label="Upload PDF Document"
            )
            
            process_btn = gr.Button(
                "🔄 Process PDF",
                variant="primary",
                size="lg"
            )
            
            status_output = gr.Markdown(
                "Upload a PDF to get started",
                elem_classes="status-box"
            )
            
            clear_btn = gr.Button(
                "🗑️ Clear All",
                variant="secondary"
            )
        
        # Right column - Chat
        with gr.Column(scale=2):
            gr.Markdown("## 💬 Ask Questions")
            
            chatbot = gr.Chatbot(
                value=[],
                height=400,
                elem_classes="chat-container"
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about your PDF...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
    
    # Event handlers
    process_btn.click(
        fn=process_pdf,
        inputs=[pdf_input],
        outputs=[status_output],
        show_progress=True
    )
    
    send_btn.click(
        fn=chat_with_pdf,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    msg_input.submit(
        fn=chat_with_pdf,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    clear_btn.click(
        fn=clear_chat,
        outputs=[chatbot, status_output]
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )