"""
DOLPHIN PDF Document AI - Local Gemma 3n Version
Optimized for powerful GPU deployment with local models
Features: AI-generated alt text for accessibility using local Gemma 3n
"""

import gradio as gr
import json
import markdown
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel, AutoModelForImageTextToText
import torch
try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity
    RAG_DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"RAG dependencies not available: {e}")
    print("Please install: pip install sentence-transformers scikit-learn")
    RAG_DEPENDENCIES_AVAILABLE = False
    SentenceTransformer = None
import os
import tempfile
import uuid
import base64
import io
from utils.utils import *
from utils.markdown_utils import MarkdownConverter

# Math extension is optional for enhanced math rendering
MATH_EXTENSION_AVAILABLE = False
try:
    from mdx_math import MathExtension
    MATH_EXTENSION_AVAILABLE = True
except ImportError:
    pass


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model optimized for powerful GPU"""
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            self.model = self.model.float()
        
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)"""
        is_batch = isinstance(image, list)
        
        if not is_batch:
            images = [image]
            prompts = [prompt]
        else:
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values
        
        if torch.cuda.is_available():
            batch_pixel_values = batch_pixel_values.half().to(self.device)
        else:
            batch_pixel_values = batch_pixel_values.to(self.device)
        
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt"
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=batch_pixel_values,
                decoder_input_ids=batch_prompt_ids,
                decoder_attention_mask=batch_attention_mask,
                min_length=1,
                max_length=2048,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
                do_sample=False,
                num_beams=1,
                repetition_penalty=1.1,
                temperature=1.0
            )
        
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
            
        if not is_batch:
            return results[0]
        return results


class Gemma3nModel:
    def __init__(self, model_id="google/gemma-3n-E4B-it"):
        """Initialize the Gemma 3n model for text generation and image description"""
        self.model_id = model_id
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype="auto",
            device_map="auto"
        )
        self.model.eval()
        print(f"✅ Gemma 3n loaded (Device: {self.model.device}, DType: {self.model.dtype})")
        
    def generate_alt_text(self, pil_image):
        """Generate alt text for an image using local Gemma 3n"""
        try:
            # Ensure image is in RGB mode
            if pil_image.mode != 'RGB':
                pil_image = pil_image.convert('RGB')
            
            # Create a detailed prompt for alt text generation
            prompt = """You are an accessibility expert creating alt text for images to help visually impaired users understand visual content. Analyze this image and provide a clear, concise description that captures the essential visual information.

Focus on:
- Main subject or content of the image
- Important details, text, or data shown
- Layout and structure if relevant (charts, diagrams, tables)
- Context that would help someone understand the image's purpose

Provide a descriptive alt text in 1-2 sentences that is informative but not overly verbose. Start directly with the description without saying "This image shows" or similar phrases."""
            
            # Prepare the message format
            message = {
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }
            
            # Apply chat template and generate
            input_ids = self.processor.apply_chat_template(
                [message],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_len = input_ids["input_ids"].shape[-1]
            
            input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=256,
                disable_compile=True,
                do_sample=False,
                temperature=0.1
            )
            
            text = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            alt_text = text[0].strip()
            
            # Clean up the alt text
            alt_text = alt_text.replace('\n', ' ').replace('\r', ' ')
            # Remove common prefixes if they appear
            prefixes_to_remove = ["This image shows", "The image shows", "This shows", "The figure shows"]
            for prefix in prefixes_to_remove:
                if alt_text.startswith(prefix):
                    alt_text = alt_text[len(prefix):].strip()
                    break
            
            return alt_text if alt_text else "Image description unavailable"
            
        except Exception as e:
            print(f"❌ Error generating alt text: {e}")
            import traceback
            traceback.print_exc()
            return "Image description unavailable"
    
    def chat(self, prompt, history=None):
        """Chat functionality using Gemma 3n for text-only conversations"""
        try:
            # Create message format
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt}
                ]
            }
            
            # If history exists, include it
            conversation = history if history else []
            conversation.append(message)
            
            # Apply chat template and generate
            input_ids = self.processor.apply_chat_template(
                conversation,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            input_len = input_ids["input_ids"].shape[-1]
            
            input_ids = input_ids.to(self.model.device, dtype=self.model.dtype)
            outputs = self.model.generate(
                **input_ids,
                max_new_tokens=1024,
                disable_compile=True,
                do_sample=True,
                temperature=0.7
            )
            
            text = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return text[0].strip()
            
        except Exception as e:
            print(f"❌ Error in chat: {e}")
            import traceback
            traceback.print_exc()
            return f"Error generating response: {str(e)}"


def convert_pdf_to_images_gradio(pdf_file):
    """Convert uploaded PDF file to list of PIL Images"""
    try:
        import pymupdf
        
        if isinstance(pdf_file, str):
            pdf_document = pymupdf.open(pdf_file)
        else:
            pdf_bytes = pdf_file.read()
            pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            mat = pymupdf.Matrix(2.0, 2.0)
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            pil_image = Image.open(io.BytesIO(img_data)).convert("RGB")
            images.append(pil_image)
        
        pdf_document.close()
        return images
        
    except Exception as e:
        raise Exception(f"Error converting PDF: {str(e)}")


def process_pdf_document(pdf_file, model, progress=gr.Progress()):
    """Process uploaded PDF file page by page"""
    if pdf_file is None:
        return "No PDF file uploaded", ""
    
    try:
        progress(0.1, desc="Converting PDF to images...")
        images = convert_pdf_to_images_gradio(pdf_file)
        
        if not images:
            return "Failed to convert PDF to images", ""
        
        all_results = []
        
        for page_idx, pil_image in enumerate(images):
            progress((page_idx + 1) / len(images) * 0.8 + 0.1, 
                    desc=f"Processing page {page_idx + 1}/{len(images)}...")
            
            layout_output = model.chat("Parse the reading order of this document.", pil_image)
            
            padded_image, dims = prepare_image(pil_image)
            recognition_results = process_elements_optimized(
                layout_output, 
                padded_image, 
                dims, 
                model, 
                max_batch_size=4
            )
            
            try:
                markdown_converter = MarkdownConverter()
                markdown_content = markdown_converter.convert(recognition_results)
            except:
                markdown_content = generate_fallback_markdown(recognition_results)
            
            page_result = {
                "page_number": page_idx + 1,
                "markdown": markdown_content
            }
            all_results.append(page_result)
        
        progress(1.0, desc="Processing complete!")
        
        combined_markdown = "\n\n---\n\n".join([
            f"# Page {result['page_number']}\n\n{result['markdown']}" 
            for result in all_results
        ])
        
        return combined_markdown, "processing_complete"
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        return error_msg, "error"


def process_elements_optimized(layout_results, padded_image, dims, model, max_batch_size=4):
    """Optimized element processing for powerful GPU"""
    layout_results = parse_layout_string(layout_results)
    
    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0
    
    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    pil_crop = crop_margin(pil_crop)
                    
                    # Generate alt text for accessibility using local Gemma 3n
                    alt_text = gemma_model.generate_alt_text(pil_crop)
                    
                    buffered = io.BytesIO()
                    pil_crop.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    data_uri = f"data:image/png;base64,{img_base64}"
                    
                    figure_results.append({
                        "label": label,
                        "text": f"![{alt_text}]({data_uri})\n\n*{alt_text}*",
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                        "alt_text": alt_text,
                    })
                else:
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    
                    if label == "tab":
                        table_elements.append(element_info)
                    else:
                        text_elements.append(element_info)
                        
            reading_order += 1
            
        except Exception as e:
            print(f"Error processing element {label}: {str(e)}")
            continue
    
    recognition_results = figure_results.copy()
    
    if text_elements:
        text_results = process_element_batch_optimized(
            text_elements, model, "Read text in the image.", max_batch_size
        )
        recognition_results.extend(text_results)
    
    if table_elements:
        table_results = process_element_batch_optimized(
            table_elements, model, "Parse the table in the image.", max_batch_size
        )
        recognition_results.extend(table_results)
    
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))
    return recognition_results


def process_element_batch_optimized(elements, model, prompt, max_batch_size=4):
    """Process elements in batches for powerful GPU"""
    results = []
    batch_size = min(len(elements), max_batch_size)
    
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)
        
        batch_results = model.chat(prompts_list, crops_list)
        
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
            
        del crops_list, batch_elements
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results


def generate_fallback_markdown(recognition_results):
    """Generate basic markdown if converter fails"""
    markdown_content = ""
    for element in recognition_results:
        if element["label"] == "tab":
            markdown_content += f"\n\n{element['text']}\n\n"
        elif element["label"] in ["para", "title", "sec", "sub_sec"]:
            markdown_content += f"{element['text']}\n\n"
        elif element["label"] == "fig":
            # Image should already have alt text from processing
            markdown_content += f"{element['text']}\n\n"
    return markdown_content


# Initialize models
model_path = "./hf_model"
if not os.path.exists(model_path):
    model_path = "ByteDance/DOLPHIN"

# Model paths and configuration
model_path = "./hf_model" if os.path.exists("./hf_model") else "ByteDance/DOLPHIN"
hf_token = os.getenv('HF_TOKEN')
gemma_model_id = "google/gemma-3n-E4B-it"

# Initialize models
print("Loading DOLPHIN model...")
dolphin_model = DOLPHIN(model_path)
print(f"✅ DOLPHIN model loaded (Device: {dolphin_model.device})")

print("Loading Gemma 3n model...")
gemma_model = Gemma3nModel(gemma_model_id)

model_status = "✅ Both models loaded successfully"

# Initialize embedding model
if RAG_DEPENDENCIES_AVAILABLE:
    try:
        print("Loading embedding model for RAG...")
        embedding_model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
        print("✅ Embedding model loaded successfully (CPU)")
    except Exception as e:
        print(f"❌ Error loading embedding model: {e}")
        embedding_model = None
else:
    print("❌ RAG dependencies not available")
    embedding_model = None


# Global state for managing tabs
processed_markdown = ""
show_results_tab = False
document_chunks = []
document_embeddings = None


def chunk_document(text, chunk_size=1024, overlap=100):
    """Split document into overlapping chunks for RAG"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks

def create_embeddings(chunks):
    """Create embeddings for document chunks"""
    if embedding_model is None:
        return None
    
    try:
        # Process in smaller batches on CPU
        batch_size = 32
        embeddings = []
        
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            batch_embeddings = embedding_model.encode(batch, show_progress_bar=False)
            embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    except Exception as e:
        print(f"Error creating embeddings: {e}")
        return None

def retrieve_relevant_chunks(question, chunks, embeddings, top_k=3):
    """Retrieve most relevant chunks for a question"""
    if embedding_model is None or embeddings is None:
        return chunks[:3]  # Fallback to first 3 chunks
    
    try:
        question_embedding = embedding_model.encode([question], show_progress_bar=False)
        similarities = cosine_similarity(question_embedding, embeddings)[0]
        
        # Get top-k most similar chunks
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        relevant_chunks = [chunks[i] for i in top_indices]
        
        return relevant_chunks
    except Exception as e:
        print(f"Error retrieving chunks: {e}")
        return chunks[:3]  # Fallback

def process_uploaded_pdf(pdf_file, progress=gr.Progress()):
    """Main processing function for uploaded PDF"""
    global processed_markdown, show_results_tab, document_chunks, document_embeddings
    
    if pdf_file is None:
        return "❌ No PDF uploaded", gr.Tabs(visible=False)
    
    try:
        # Process PDF
        progress(0.1, desc="Processing PDF...")
        combined_markdown, status = process_pdf_document(pdf_file, dolphin_model, progress)
        
        if status == "processing_complete":
            processed_markdown = combined_markdown
            
            # Create chunks and embeddings for RAG
            progress(0.9, desc="Creating document chunks for RAG...")
            document_chunks = chunk_document(processed_markdown)
            document_embeddings = create_embeddings(document_chunks)
            print(f"Created {len(document_chunks)} chunks")
            
            show_results_tab = True
            progress(1.0, desc="PDF processed successfully!")
            return "✅ PDF processed successfully! Chatbot is ready in the Chat tab.", gr.Tabs(visible=True)
        else:
            show_results_tab = False
            return combined_markdown, gr.Tabs(visible=False)
            
    except Exception as e:
        show_results_tab = False
        error_msg = f"❌ Error processing PDF: {str(e)}"
        return error_msg, gr.Tabs(visible=False)


def get_processed_markdown():
    """Return the processed markdown content"""
    global processed_markdown
    return processed_markdown if processed_markdown else "No document processed yet."


def clear_all():
    """Clear all data and hide results tab"""
    global processed_markdown, show_results_tab, document_chunks, document_embeddings
    processed_markdown = ""
    show_results_tab = False
    document_chunks = []
    document_embeddings = None
    
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return None, "", gr.Tabs(visible=False)


# Create Gradio interface
with gr.Blocks(
    title="DOLPHIN PDF AI - Local Gemma 3n", 
    theme=gr.themes.Soft(),
    css="""
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif !important;
    }
    
    .main-container { 
        max-width: 1000px; 
        margin: 0 auto; 
    }
    .upload-container { 
        text-align: center; 
        padding: 40px 20px;
        border: 2px dashed #e0e0e0;
        border-radius: 15px;
        margin: 20px 0;
    }
    .upload-button {
        font-size: 18px !important;
        padding: 15px 30px !important;
        margin: 20px 0 !important;
        font-weight: 600 !important;
    }
    .status-message {
        text-align: center;
        padding: 15px;
        margin: 10px 0;
        border-radius: 8px;
        font-weight: 500;
    }
    .chatbot-container {
        max-height: 600px;
    }
    h1, h2, h3 {
        font-weight: 700 !important;
    }
    #progress-container {
        margin: 10px 0;
        min-height: 20px;
    }
    """
) as demo:
    
    with gr.Tabs() as main_tabs:
        # Home Tab
        with gr.TabItem("🏠 Home", id="home"):
            embedding_status = "✅ RAG ready" if embedding_model else "❌ RAG not loaded"
            gr.Markdown(
                "# Scholar Express - Local Gemma 3n Version\n"
                "### Upload a research paper to get a web-friendly version with AI-generated alt text for accessibility. Includes an AI chatbot powered by local Gemma 3n.\n"
                f"**System:** {model_status}\n"
                f"**RAG System:** {embedding_status}\n"
                f"**DOLPHIN:** Local model for PDF processing\n"
                f"**Gemma 3n:** Local model for alt text generation and chat\n"
                f"**Alt Text:** Gemma 3n generates descriptive alt text for images\n"
                f"**GPU:** {'CUDA available' if torch.cuda.is_available() else 'CPU only'}"
            )
            
            with gr.Column(elem_classes="upload-container"):
                gr.Markdown("## 📄 Upload Your PDF Document")
                
                pdf_input = gr.File(
                    file_types=[".pdf"],
                    label="",
                    height=150,
                    elem_id="pdf_upload"
                )
                
                process_btn = gr.Button(
                    "🚀 Process PDF", 
                    variant="primary", 
                    size="lg",
                    elem_classes="upload-button"
                )
                
                clear_btn = gr.Button(
                    "🗑️ Clear", 
                    variant="secondary"
                )
            
            # Dedicated progress space
            progress_space = gr.HTML(
                value="",
                visible=False,
                elem_id="progress-container"
            )
            
            # Status output (hidden during processing)
            status_output = gr.Markdown(
                "",
                elem_classes="status-message"
            )
        
        # Results Tab (initially hidden)
        with gr.TabItem("📖 Document", id="results", visible=False) as results_tab:
            gr.Markdown("## Processed Document")
            
            markdown_display = gr.Markdown(
                value="",
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False}
                ],
                height=700
            )
        
        # Chatbot Tab (initially hidden)
        with gr.TabItem("💬 Chat", id="chat", visible=False) as chat_tab:
            gr.Markdown("## Ask Questions About Your Document")
            
            chatbot = gr.Chatbot(
                value=[],
                height=500,
                elem_classes="chatbot-container",
                placeholder="Your conversation will appear here once you process a document..."
            )
            
            with gr.Row():
                msg_input = gr.Textbox(
                    placeholder="Ask a question about the processed document...",
                    scale=4,
                    container=False
                )
                send_btn = gr.Button("Send", variant="primary", scale=1)
            
            gr.Markdown(
                "*Ask questions about your processed document. The AI uses RAG (Retrieval-Augmented Generation) with local Gemma 3n to find relevant sections and provide accurate answers.*",
                elem_id="chat-notice"
            )
    
    # Event handlers
    process_btn.click(
        fn=process_uploaded_pdf,
        inputs=[pdf_input],
        outputs=[status_output, results_tab],
        show_progress=True
    ).then(
        fn=get_processed_markdown,
        outputs=[markdown_display]
    ).then(
        fn=lambda: gr.TabItem(visible=True),
        outputs=[chat_tab]
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[pdf_input, status_output, results_tab]
    ).then(
        fn=lambda: gr.HTML(visible=False),
        outputs=[progress_space]
    ).then(
        fn=lambda: gr.TabItem(visible=False),
        outputs=[chat_tab]
    )
    
    # Chatbot functionality with local Gemma 3n
    def chatbot_response(message, history):
        if not message.strip():
            return history
        
        if not processed_markdown:
            return history + [[message, "❌ Please process a PDF document first before asking questions."]]
        
        try:
            # Use RAG to get relevant chunks from markdown
            if document_chunks and len(document_chunks) > 0:
                relevant_chunks = retrieve_relevant_chunks(message, document_chunks, document_embeddings, top_k=3)
                context = "\n\n".join(relevant_chunks)
                # Smart truncation: aim for ~6000 chars for local model
                if len(context) > 6000:
                    # Try to cut at sentence boundaries
                    sentences = context[:6000].split('.')
                    context = '.'.join(sentences[:-1]) + '...' if len(sentences) > 1 else context[:6000] + '...'
            else:
                # Fallback to truncated document if RAG fails
                context = processed_markdown[:6000] + "..." if len(processed_markdown) > 6000 else processed_markdown
            
            # Create prompt for Gemma 3n
            prompt = f"""You are a helpful assistant that answers questions about documents. Use the provided context to answer questions accurately and concisely.

Context from the document:
{context}

Question: {message}

Please provide a clear and helpful answer based on the context provided."""
            
            # Generate response using local Gemma 3n
            response_text = gemma_model.chat(prompt)
            return history + [[message, response_text]]
            
        except Exception as e:
            error_msg = f"❌ Error generating response: {str(e)}"
            print(f"Full error: {e}")
            import traceback
            traceback.print_exc()
            return history + [[message, error_msg]]
    
    send_btn.click(
        fn=chatbot_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )
    
    # Also allow Enter key to send message
    msg_input.submit(
        fn=chatbot_response,
        inputs=[msg_input, chatbot],
        outputs=[chatbot]
    ).then(
        lambda: "",
        outputs=[msg_input]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=4,
        inbrowser=False,
        quiet=True
    )