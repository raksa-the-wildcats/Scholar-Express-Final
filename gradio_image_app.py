import gradio as gr
import json
import markdown
from markdown.extensions import codehilite
import cv2
import numpy as np
from PIL import Image
from transformers import AutoProcessor, VisionEncoderDecoderModel
import torch
import os
from utils.utils import *
from utils.markdown_utils import MarkdownConverter

try:
    from mdx_math import MathExtension
    MATH_EXTENSION_AVAILABLE = True
except ImportError:
    MATH_EXTENSION_AVAILABLE = False


class DOLPHIN:
    def __init__(self, model_id_or_path):
        """Initialize the Hugging Face model
        
        Args:
            model_id_or_path: Path to local model or Hugging Face model ID
        """
        self.processor = AutoProcessor.from_pretrained(model_id_or_path)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_id_or_path)
        self.model.eval()
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        if self.device == "cuda":
            self.model = self.model.half()
        
        self.tokenizer = self.processor.tokenizer
        
    def chat(self, prompt, image):
        """Process an image or batch of images with the given prompt(s)
        
        Args:
            prompt: Text prompt or list of prompts to guide the model
            image: PIL Image or list of PIL Images to process
            
        Returns:
            Generated text or list of texts from the model
        """
        # Check if we're dealing with a batch
        is_batch = isinstance(image, list)
        
        if not is_batch:
            # Single image, wrap it in a list for consistent processing
            images = [image]
            prompts = [prompt]
        else:
            # Batch of images
            images = image
            prompts = prompt if isinstance(prompt, list) else [prompt] * len(images)
        
        # Prepare image
        batch_inputs = self.processor(images, return_tensors="pt", padding=True)
        batch_pixel_values = batch_inputs.pixel_values
        if self.device == "cuda":
            batch_pixel_values = batch_pixel_values.half()
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
        
        # Generate text
        outputs = self.model.generate(
            pixel_values=batch_pixel_values,
            decoder_input_ids=batch_prompt_ids,
            decoder_attention_mask=batch_attention_mask,
            min_length=1,
            max_length=4096,
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
            
        # Return a single result for single image input
        if not is_batch:
            return results[0]
        return results


def render_markdown_with_math(markdown_content):
    """Convert markdown to HTML with MathJax support that works in Gradio"""
    import re
    
    # Convert basic markdown to HTML first
    html_content = markdown.markdown(markdown_content)
    
    # Create a complete HTML document with MathJax
    html_with_math = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 100%;
            margin: 0;
            padding: 20px;
        }}
        .math-container {{
            margin: 15px 0;
        }}
        .display-math {{
            text-align: center;
            margin: 20px 0;
        }}
        .inline-math {{
            display: inline;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #f2f2f2;
        }}
        pre {{
            background-color: #f5f5f5;
            padding: 10px;
            border-radius: 4px;
            overflow-x: auto;
        }}
        code {{
            background-color: #f5f5f5;
            padding: 2px 4px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
        }}
    </style>
    <script>
        window.MathJax = {{
            tex: {{
                inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
                displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']],
                processEscapes: true,
                processEnvironments: true,
                tags: 'ams',
                autoload: {{
                    color: [],
                    colorv2: ['color']
                }},
                packages: {{'[+]': ['noerrors']}}
            }},
            options: {{
                ignoreHtmlClass: 'tex2jax_ignore',
                processHtmlClass: 'tex2jax_process'
            }},
            loader: {{
                load: ['[tex]/noerrors']
            }}
        }};
        
        // Function to trigger MathJax processing after content loads
        function processMath() {{
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise().catch(function (err) {{
                    console.log('MathJax typeset failed: ' + err.message);
                }});
            }}
        }}
        
        // Process math when page loads
        document.addEventListener('DOMContentLoaded', function() {{
            setTimeout(processMath, 100);
        }});
        
        // Also process when MathJax loads
        window.addEventListener('load', function() {{
            setTimeout(processMath, 200);
        }});
    </script>
    <script type="text/javascript" id="MathJax-script" async
        src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"
        onload="processMath()">
    </script>
</head>
<body>
    <div class="tex2jax_process">
        {html_content}
    </div>
    <script>
        // Additional processing trigger
        setTimeout(function() {{
            if (window.MathJax && window.MathJax.typesetPromise) {{
                window.MathJax.typesetPromise();
            }}
        }}, 500);
    </script>
</body>
</html>
"""
    
    return html_with_math


def process_elements(layout_results, padded_image, dims, model, max_batch_size=16, save_dir=None, image_name="gradio_session"):
    """Parse all document elements with parallel decoding"""
    layout_results = parse_layout_string(layout_results)

    # Store text and table elements separately
    text_elements = []  # Text elements
    table_elements = []  # Table elements
    figure_results = []  # Image elements (saved as files)
    previous_box = None
    reading_order = 0

    # Setup output directories if save_dir is provided
    if save_dir:
        setup_output_dirs(save_dir)

    # Collect elements to process and group by type
    for bbox, label in layout_results:
        try:
            # Adjust coordinates
            x1, y1, x2, y2, orig_x1, orig_y1, orig_x2, orig_y2, previous_box = process_coordinates(
                bbox, padded_image, dims, previous_box
            )

            # Crop and parse element
            cropped = padded_image[y1:y2, x1:x2]
            if cropped.size > 0 and cropped.shape[0] > 3 and cropped.shape[1] > 3:
                if label == "fig":
                    # Convert cropped OpenCV image to PIL
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    
                    # Apply margin cropping to remove white space around the figure
                    pil_crop = crop_margin(pil_crop)
                    
                    # Convert to base64 for Gradio display (works better than file paths)
                    import base64
                    import io
                    buffered = io.BytesIO()
                    pil_crop.save(buffered, format="PNG")
                    img_base64 = base64.b64encode(buffered.getvalue()).decode()
                    
                    # Create data URI for direct embedding in markdown
                    data_uri = f"data:image/png;base64,{img_base64}"
                    
                    figure_results.append(
                        {
                            "label": label,
                            "text": data_uri,  # Pass base64 directly to _handle_figure
                            "figure_base64": data_uri,
                            "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                            "reading_order": reading_order,
                        }
                    )
                else:
                    # Prepare element for parsing
                    pil_crop = Image.fromarray(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))
                    element_info = {
                        "crop": pil_crop,
                        "label": label,
                        "bbox": [orig_x1, orig_y1, orig_x2, orig_y2],
                        "reading_order": reading_order,
                    }
                    
                    # Group by type
                    if label == "tab":
                        table_elements.append(element_info)
                    else:  # Text elements
                        text_elements.append(element_info)

            reading_order += 1

        except Exception as e:
            print(f"Error processing bbox with label {label}: {str(e)}")
            continue

    # Initialize results list
    recognition_results = figure_results.copy()
    
    # Process text elements (in batches)
    if text_elements:
        text_results = process_element_batch(text_elements, model, "Read text in the image.", max_batch_size)
        recognition_results.extend(text_results)
    
    # Process table elements (in batches)
    if table_elements:
        table_results = process_element_batch(table_elements, model, "Parse the table in the image.", max_batch_size)
        recognition_results.extend(table_results)

    # Sort elements by reading order
    recognition_results.sort(key=lambda x: x.get("reading_order", 0))

    return recognition_results


def process_element_batch(elements, model, prompt, max_batch_size=16):
    """Process elements of the same type in batches"""
    results = []
    
    # Determine batch size
    batch_size = len(elements)
    if max_batch_size is not None and max_batch_size > 0:
        batch_size = min(batch_size, max_batch_size)
    
    # Process in batches
    for i in range(0, len(elements), batch_size):
        batch_elements = elements[i:i+batch_size]
        crops_list = [elem["crop"] for elem in batch_elements]
        
        # Use the same prompt for all elements in the batch
        prompts_list = [prompt] * len(crops_list)
        
        # Batch inference
        batch_results = model.chat(prompts_list, crops_list)
        
        # Add results
        for j, result in enumerate(batch_results):
            elem = batch_elements[j]
            results.append({
                "label": elem["label"],
                "bbox": elem["bbox"],
                "text": result.strip(),
                "reading_order": elem["reading_order"],
            })
    
    return results


# Initialize model
model_path = "./hf_model"
if not os.path.exists(model_path):
    model_path = "ByteDance/DOLPHIN"

try:
    dolphin_model = DOLPHIN(model_path)
    print(f"Model loaded successfully from {model_path}")
except Exception as e:
    print(f"Error loading model: {e}")
    dolphin_model = None


def process_image(image, task_type):
    """Process uploaded image and return results in different formats"""
    if dolphin_model is None:
        return None, "Model not loaded", "Model not loaded", {"error": "Model not loaded"}
    
    if image is None:
        return None, "No image uploaded", "No image uploaded", {"error": "No image uploaded"}
    
    try:
        # Convert to PIL Image if needed
        if hasattr(image, 'convert'):
            pil_image = image.convert("RGB")
        else:
            pil_image = Image.fromarray(image).convert("RGB")
        
        if task_type == "Document Parsing":
            # Full document processing with two stages
            # Stage 1: Page-level layout and reading order parsing
            layout_output = dolphin_model.chat("Parse the reading order of this document.", pil_image)
            
            # Stage 2: Element-level content parsing with image extraction
            import tempfile
            import uuid
            
            # Create temporary directory for saving figures
            temp_dir = tempfile.mkdtemp()
            session_id = str(uuid.uuid4())[:8]
            
            padded_image, dims = prepare_image(pil_image)
            recognition_results = process_elements(
                layout_output, 
                padded_image, 
                dims, 
                dolphin_model, 
                max_batch_size=16,
                save_dir=temp_dir,
                image_name=f"session_{session_id}"
            )
            
            # Convert to markdown
            try:
                markdown_converter = MarkdownConverter()
                markdown_content = markdown_converter.convert(recognition_results)
            except:
                # Fallback if markdown converter fails
                markdown_content = ""
                for element in recognition_results:
                    if element["label"] == "tab":
                        markdown_content += f"\n\n{element['text']}\n\n"
                    elif element["label"] in ["para", "title", "sec", "sub_sec"]:
                        markdown_content += f"{element['text']}\n\n"
                    elif element["label"] == "fig":
                        markdown_content += f"{element['text']}\n\n"
            
            # Create structured JSON output
            json_output = {
                "task_type": task_type,
                "layout_parsing": layout_output,
                "recognition_results": recognition_results,
                "model_info": {
                    "device": dolphin_model.device,
                    "model_path": model_path
                },
                "temp_dir": temp_dir
            }
            
            # Return markdown content directly for Gradio's built-in LaTeX support
            return pil_image, markdown_content, markdown_content, json_output
            
        else:
            # Simple element-level processing for other tasks
            if task_type == "Table Extraction":
                prompt = "Parse the table in the image."
            elif task_type == "Text Reading":
                prompt = "Read text in the image."
            elif task_type == "Formula Recognition":
                prompt = "Read text in the image."
            else:
                prompt = "Read text in the image."
            
            # Process with model
            result = dolphin_model.chat(prompt, pil_image)
            
            # Create JSON output
            json_output = {
                "task_type": task_type,
                "prompt": prompt,
                "result": result,
                "model_info": {
                    "device": dolphin_model.device,
                    "model_path": model_path
                }
            }
            
            return pil_image, result, result, json_output
        
    except Exception as e:
        error_msg = f"Error processing image: {str(e)}"
        return None, error_msg, error_msg, {"error": error_msg}


def clear_all():
    """Clear all inputs and outputs"""
    return None, None, "", "", {}


# Create Gradio interface
with gr.Blocks(title="DOLPHIN Document AI", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🐬 DOLPHIN Document AI Interface")
    gr.Markdown("Upload an image and select a task to process with the DOLPHIN model")
    
    with gr.Row():
        # Column 1: Image Upload
        with gr.Column(scale=1):
            gr.Markdown("### 📤 Upload Image")
            image_input = gr.Image(
                type="pil", 
                label="Upload Image",
                height=600
            )
            
            task_type = gr.Dropdown(
                choices=["Document Parsing", "Table Extraction", "Text Reading", "Formula Recognition"],
                value="Document Parsing",
                label="Task Type"
            )
            
            with gr.Row():
                submit_btn = gr.Button("🚀 Submit", variant="primary")
                cancel_btn = gr.Button("❌ Clear", variant="secondary")
        
        # Column 2: Image Preview
        with gr.Column(scale=1):
            gr.Markdown("### 👁️ Image Preview")
            image_preview = gr.Image(
                type="pil",
                label="Uploaded Image",
                interactive=False,
                height=600
            )
        
        # Column 3: Results with Tabs
        with gr.Column(scale=1):
            gr.Markdown("### 📋 Results")
            with gr.Tabs():
                with gr.TabItem("📖 Markdown Preview"):
                    markdown_preview = gr.Markdown(
                        label="Rendered Markdown",
                        latex_delimiters=[
                            {"left": "$$", "right": "$$", "display": True},
                            {"left": "$", "right": "$", "display": False},
                            {"left": "\\(", "right": "\\)", "display": False},
                            {"left": "\\[", "right": "\\]", "display": True}
                        ],
                        container=True,
                        height=600
                    )
                
                with gr.TabItem("📝 Raw Markdown"):
                    raw_markdown = gr.Code(
                        label="Raw Markdown Text",
                        language="markdown",
                        container=True,
                        interactive=False,
                        lines=25
                    )
                
                with gr.TabItem("🔧 JSON"):
                    json_output = gr.JSON(
                        label="JSON Output",
                        height=600
                    )
    
    # Event handlers
    submit_btn.click(
        fn=process_image,
        inputs=[image_input, task_type],
        outputs=[image_preview, markdown_preview, raw_markdown, json_output]
    )
    
    cancel_btn.click(
        fn=clear_all,
        outputs=[image_input, image_preview, markdown_preview, raw_markdown, json_output]
    )
    
    # Auto-update preview when image is uploaded
    image_input.change(
        fn=lambda img: img if img is not None else None,
        inputs=[image_input],
        outputs=[image_preview]
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )