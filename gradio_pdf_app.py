"""
PDF Document Processing Gradio App for HuggingFace Spaces
Built on DOLPHIN model for document parsing and analysis
"""

import gradio as gr
import json
import markdown
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch
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
    # mdx_math is not available in standard PyPI, gracefully continue without it
    pass


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model optimized for HF Spaces
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(
            model_id_or_path,
            torch_dtype=torch.float16,  # Use half precision for memory efficiency
            device_map="auto" if torch.cuda.is_available() else None
        )
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if not torch.cuda.is_available():
            # Keep full precision on CPU
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
        
        # Prepare image
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values
        
        if torch.cuda.is_available():
            batch_pixel_values = batch_pixel_values.half().to(self.device)
        else:
            batch_pixel_values = batch_pixel_values.to(self.device)
        
        # Prepare prompt
        prompts = [f"<s>{p} <Answer/>" for p in prompts]
        batch_prompt_inputs = self.tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt"
        )

        batch_prompt_ids = batch_prompt_inputs.input_ids.to(self.device)
        batch_attention_mask = batch_prompt_inputs.attention_mask.to(self.device)
        
        # Generate text with memory-efficient settings
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values=batch_pixel_values,
                decoder_input_ids=batch_prompt_ids,
                decoder_attention_mask=batch_attention_mask,
                min_length=1,
                max_length=2048,  # Reduced for memory efficiency
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
        
        # Process output
        sequences = self.tokenizer.batch_decode(outputs.sequences, skip_special_tokens=False)
        
        # Clean prompt text from output
        results = []
        for i, sequence in enumerate(sequences):
            cleaned = sequence.replace(prompts[i], "").replace("<pad>", "").replace("</s>", "").strip()
            results.append(cleaned)
            
        if not is_batch:
            return results[0]
        return results


def convert_pdf_to_images_gradio(pdf_file):
    """Convert uploaded PDF file to list of PIL Images"""
    try:
        import pymupdf
        
        # Handle different file input types
        if isinstance(pdf_file, str):
            # If it's a file path (Gradio 5.x behavior)
            pdf_document = pymupdf.open(pdf_file)
        else:
            # If it's a file object with .read() method
            pdf_bytes = pdf_file.read()
            pdf_document = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        
        images = []
        for page_num in range(len(pdf_document)):
            page = pdf_document[page_num]
            
            # Render page to image with high DPI for better quality
            mat = pymupdf.Matrix(2.0, 2.0)  # 2x zoom for better quality
            pix = page.get_pixmap(matrix=mat)
            
            # Convert to PIL Image
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
        return "No PDF file uploaded", [], {}
    
    try:
        # Convert PDF to images
        progress(0.1, desc="Converting PDF to images...")
        images = convert_pdf_to_images_gradio(pdf_file)
        
        if not images:
            return "Failed to convert PDF to images", [], {}
        
        # Process each page
        all_results = []
        page_previews = []
        
        for page_idx, pil_image in enumerate(images):
            progress((page_idx + 1) / len(images) * 0.8 + 0.1, 
                    desc=f"Processing page {page_idx + 1}/{len(images)}...")
            
            # Stage 1: Layout parsing
            layout_output = model.chat("Parse the reading order of this document.", pil_image)
            
            # Stage 2: Element processing with memory optimization
            padded_image, dims = prepare_image(pil_image)
            recognition_results = process_elements_optimized(
                layout_output, 
                padded_image, 
                dims, 
                model, 
                max_batch_size=4  # Smaller batch size for memory efficiency
            )
            
            # Convert to markdown
            try:
                markdown_converter = MarkdownConverter()
                markdown_content = markdown_converter.convert(recognition_results)
            except:
                # Fallback markdown generation
                markdown_content = generate_fallback_markdown(recognition_results)
            
            # Store page results
            page_result = {
                "page_number": page_idx + 1,
                "layout_output": layout_output,
                "elements": recognition_results,
                "markdown": markdown_content
            }
            all_results.append(page_result)
            
            # Create page preview with results
            page_preview = {
                "image": pil_image,
                "page_num": page_idx + 1,
                "element_count": len(recognition_results),
                "markdown_preview": markdown_content[:500] + "..." if len(markdown_content) > 500 else markdown_content
            }
            page_previews.append(page_preview)
        
        progress(1.0, desc="Processing complete!")
        
        # Combine all markdown
        combined_markdown = "\n\n---\n\n".join([
            f"# Page {result['page_number']}\n\n{result['markdown']}" 
            for result in all_results
        ])
        
        # Create summary JSON
        summary_json = {
            "total_pages": len(images),
            "processing_status": "completed",
            "pages": all_results,
            "model_info": {
                "device": model.device,
                "total_elements": sum(len(page["elements"]) for page in all_results)
            }
        }
        
        return combined_markdown, page_previews, summary_json
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        return error_msg, [], {"error": error_msg}


def process_elements_optimized(layout_results, padded_image, dims, model, max_batch_size=4):
    """Optimized element processing for memory efficiency"""
    layout_results = parse_layout_string(layout_results)
    
    text_elements = []
    table_elements = []
    figure_results = []
    previous_box = None
    reading_order = 0
    
    # Collect elements to process
    for bbox, label in layout_results:
        try:
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )
            
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    # Convert to base64 for figure display
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    pil_crop = crop_margin(pil_crop)
                    
                    buffered = io.BytesIO()
                    pil_crop.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    data_uri = f"data:image/png;base64,{img_base64}"
                    
                    figure_results.append({
                        "label": label,
                        "text": f"![Figure {reading_order}]({data_uri})",
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
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
    
    # Process elements in small batches
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
    """Process elements in small batches for memory efficiency"""
    results = []
    batch_size = min(len(elements), max_batch_size)
    
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        prompts_list = [prompt] * len(crops_list)
        
        # Process batch
        batch_results = model.chat(prompts_list, crops_list)
        
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
            
        # Clear memory
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
            markdown_content += f"{element['text']}\n\n"
    return markdown_content


def create_page_gallery(page_previews):
    """Create a gallery view of processed pages"""
    if not page_previews:
        return "No pages processed yet."
    
    gallery_html = "<div style='display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px;'>"
    
    for preview in page_previews:
        gallery_html += f"""
        <div style='border: 1px solid #ddd; padding: 15px; border-radius: 8px;'>
            <h3>Page {preview['page_num']}</h3>
            <p><strong>Elements found:</strong> {preview['element_count']}</p>
            <div style='max-height: 200px; overflow-y: auto; background: #f5f5f5; padding: 10px; border-radius: 4px; font-size: 12px;'>
                {preview['markdown_preview']}
            </div>
        </div>
        """
    
    gallery_html += "</div>"
    return gallery_html


# Initialize model
model_path = "./hf_model"
if not os.path.exists(model_path):
    model_path = "ByteDance/DOLPHIN"

try:
    dolphin_model = DOLPHIN(model_path)
    print(f"Model loaded successfully from {model_path}")
    model_status = f"✅ Model loaded: {model_path} (Device: {dolphin_model.device})"
except Exception as e:
    print(f"Error loading model: {e}")
    dolphin_model = None
    model_status = f"❌ Model failed to load: {str(e)}"


def process_uploaded_pdf(pdf_file, progress=gr.Progress()):
    """Main processing function for uploaded PDF"""
    if dolphin_model is None:
        return "Model not loaded", "Model not loaded", {}, "Model not loaded"
    
    if pdf_file is None:
        return "No PDF uploaded", "No PDF uploaded", {}, "No PDF uploaded"
    
    try:
        # Process the PDF
        combined_markdown, page_previews, summary_json = process_pdf_document(
            pdf_file, dolphin_model, progress
        )
        
        # Create page gallery
        gallery_html = create_page_gallery(page_previews)
        
        return combined_markdown, combined_markdown, summary_json, gallery_html
        
    except Exception as e:
        error_msg = f"Error processing PDF: {str(e)}"
        return error_msg, error_msg, {"error": error_msg}, error_msg


def clear_all():
    """Clear all inputs and outputs"""
    return None, "", "", {}, ""


# Create Gradio interface optimized for HuggingFace Spaces
with gr.Blocks(
    title="DOLPHIN PDF Document AI", 
    theme=gr.themes.Soft(),
    css="""
    .main-container { max-width: 1200px; margin: 0 auto; }
    .status-box { padding: 10px; border-radius: 5px; margin: 10px 0; }
    .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
    .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
    """
) as demo:
    gr.Markdown("# 🐬 DOLPHIN PDF Document AI")
    gr.Markdown(
        "Upload a PDF document and process it page by page with the DOLPHIN model. "
        "Optimized for HuggingFace Spaces deployment."
    )
    
    # Model status
    gr.Markdown(f"**Model Status:** {model_status}")
    
    with gr.Row():
        # Left column: Upload and controls
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Upload PDF Document")
            pdf_input = gr.File(
                file_types=[".pdf"],
                label="Select PDF File",
                height=200
            )
            
            with gr.Row():
                process_btn = gr.Button("🚀 Process PDF", variant="primary", size="lg")
                clear_btn = gr.Button("🗑️ Clear All", variant="secondary")
        
        # Right column: Results tabs
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Processing Results")
            
            with gr.Tabs():
                with gr.TabItem("📖 Markdown Output"):
                    markdown_output = gr.Markdown(
                        label="Processed Document",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False}
                        ],
                        height=600
                    )
                
                with gr.TabItem("📝 Raw Markdown"):
                    raw_markdown = gr.Code(
                        label="Raw Markdown Text",
                        language="markdown",
                        lines=25
                    )
                
                with gr.TabItem("🔍 Page Gallery"):
                    page_gallery = gr.HTML(
                        label="Page Overview"
                    )
                
                with gr.TabItem("🔧 JSON Details"):
                    json_output = gr.JSON(
                        label="Processing Details",
                        height=600
                    )
    
    # Progress bar
    progress_bar = gr.HTML(visible=False)
    
    # Event handlers
    process_btn.click(
        fn=process_uploaded_pdf,
        inputs=[pdf_input],
        outputs=[markdown_output, raw_markdown, json_output, page_gallery],
        show_progress=True
    )
    
    clear_btn.click(
        fn=clear_all,
        outputs=[pdf_input, markdown_output, raw_markdown, json_output, page_gallery]
    )
    
    # Footer
    gr.Markdown(
        "---\n"
        "**Note:** This app is optimized for NVIDIA T4 deployment on HuggingFace Spaces. "
        "Processing time depends on document complexity and page count."
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
        max_threads=2,  # Limit threads for memory efficiency
        inbrowser=False,
        quiet=True
    )