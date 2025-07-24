"""
Standalone RAG Chatbot with Gemma 3n
A simple PDF chatbot using Retrieval-Augmented Generation
"""

import gradio as gr
import torch
import os
import io
import numpy as np
from PIL import Image
import pymupdf  # PyMuPDF for PDF processing

# RAG dependencies
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    from transformers import Gemma3nForConditionalGeneration, AutoProcessor
    RAG_AVAILABLE = True
except ImportError as e:
    print(f"Missing dependencies: {e}")
    RAG_AVAILABLE = False

# Global variables
embedding_model = None
chatbot_model = None
chatbot_processor = None
document_chunks = []
document_embeddings = None
processed_text = ""

def initialize_models():
    """Initialize embedding model and chatbot model"""
    global embedding_model, chatbot_model, chatbot_processor
    
    if not RAG_AVAILABLE:
        return False, "Required dependencies not installed"
    
    try:
        # Initialize embedding model (CPU to save GPU memory)
        if embedding_model is None:
            print("Loading embedding model...")
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("✅ Embedding model loaded successfully")
        
        # Initialize chatbot model
        if chatbot_model is None or chatbot_processor is None:
            hf_token = os.getenv('HF_TOKEN')
            if not hf_token:
                return False, "HF_TOKEN not found in environment"
            
            print("Loading Gemma 3n model...")
            chatbot_model = Gemma3nForConditionalGeneration.from_pretrained(
                "google/gemma-3n-e4b-it",
                device_map="auto",
                torch_dtype=torch.bfloat16,
                token=hf_token
            ).eval()
            
            chatbot_processor = AutoProcessor.from_pretrained(
                "google/gemma-3n-e4b-it",
                token=hf_token
            )
            
            print("✅ Gemma 3n model loaded successfully")
        
        return True, "All models loaded successfully"
        
    except Exception as e:
        print(f"Error loading models: {e}")
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
    """Generate response using RAG"""
    global chatbot_model, chatbot_processor
    
    if not message.strip():
        return history
    
    if not processed_text:
        return history + [[message, "❌ Please upload and process a PDF first"]]
    
    # Check if models are loaded
    if chatbot_model is None or chatbot_processor is None:
        print("Models not loaded, attempting to reload...")
        success, error_msg = initialize_models()
        if not success:
            return history + [[message, f"❌ Failed to load models: {error_msg}"]]
    
    try:
        # Retrieve relevant chunks
        if document_chunks and document_embeddings is not None:
            relevant_chunks = retrieve_relevant_chunks(message, document_chunks, document_embeddings)
            context = "\n\n".join(relevant_chunks)
        else:
            # Fallback to truncated text
            context = processed_text[:2000] + "..." if len(processed_text) > 2000 else processed_text
        
        # Create messages for Gemma
        messages = [
            {
                "role": "system",
                "content": [{"type": "text", "text": "You are a helpful assistant that answers questions about documents. Use the provided context to answer questions accurately and concisely."}]
            },
            {
                "role": "user",
                "content": [{"type": "text", "text": f"Context:\n{context}\n\nQuestion: {message}"}]
            }
        ]
        
        # Process with Gemma
        inputs = chatbot_processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to(chatbot_model.device)
        
        input_len = inputs["input_ids"].shape[-1]
        
        with torch.inference_mode():
            generation = chatbot_model.generate(
                **inputs,
                max_new_tokens=300,
                do_sample=False,
                temperature=0.7,
                pad_token_id=chatbot_processor.tokenizer.pad_token_id,
                use_cache=True
            )
            generation = generation[0][input_len:]
        
        response = chatbot_processor.decode(generation, skip_special_tokens=True)
        
        return history + [[message, response]]
        
    except Exception as e:
        error_msg = f"❌ Error generating response: {str(e)}"
        return history + [[message, error_msg]]

def clear_chat():
    """Clear chat history and processed data"""
    global document_chunks, document_embeddings, processed_text
    document_chunks = []
    document_embeddings = None
    processed_text = ""
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return [], "Ready to process a new PDF"

def get_model_status():
    """Get current model loading status"""
    global chatbot_model, chatbot_processor, embedding_model
    
    statuses = []
    
    if embedding_model is not None:
        statuses.append("✅ Embedding model loaded")
    else:
        statuses.append("❌ Embedding model not loaded")
    
    if chatbot_model is not None and chatbot_processor is not None:
        statuses.append("✅ Chatbot model loaded")
    else:
        statuses.append("❌ Chatbot model not loaded")
    
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
    title="RAG Chatbot with Gemma 3n",
    theme=gr.themes.Soft(),
    css="""
    .main-container { max-width: 1200px; margin: 0 auto; }
    .status-box { padding: 15px; margin: 10px 0; border-radius: 8px; }
    .chat-container { height: 500px; }
    """
) as demo:
    
    gr.Markdown("# 🤖 RAG Chatbot with Gemma 3n")
    gr.Markdown("### Upload a PDF and ask questions about it using Retrieval-Augmented Generation")
    
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