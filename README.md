# Scholar Express
## AI-Powered Accessible Academic Research Platform

Scholar Express is an innovative AI-powered platform that transforms inaccessible scientific research papers into interactive, screen-reader compatible documents. The system addresses critical accessibility barriers faced by students with disabilities in academic research, leveraging specialized AI models to make scientific literature truly inclusive.

## 🎯 Problem Statement
According to the U.S. National Center for Education Statistics, a significant portion of undergraduate students have disabilities:
- 18% of male undergraduate students
- 22% of female undergraduate students  
- 54% of nonbinary undergraduate students

These students face major barriers when conducting research, as scientific PDFs are fundamentally inaccessible to screen readers due to complex mathematical equations, figures, and diagrams lacking alt text descriptions.

## 🚀 Key Features

### Document Processing
- **OCR and layout analysis** optimized for scientific papers
- **Table and figure extraction** with proper formatting for research content
- **AI-generated alt text** specifically for scientific diagrams, charts, and equations
- **Structured markdown output** that preserves document hierarchy

### Interactive Features
- **RAG-powered chatbot** for scientific document Q&A
- **Real-time voice conversations** about research content
- **Multi-tab interface** optimized for research workflows

### Accessibility Focus  
- **Screen reader compatible** output
- **Descriptive alt text** for all figures following WCAG guidelines
- **Privacy-first design** with local processing

## 🏗️ System Architecture

### Core AI Models
The platform utilizes a specialized ensemble of AI models, each optimized for specific tasks:

- **Gemma 3n 4B**: Primary engine for alt text generation and document chatbot functionality
- **Gemma 3n 2B**: Specialized for real-time voice chat interactions  
- **DOLPHIN**: Handles PDF layout analysis and text extraction
- **SentenceTransformer**: Enables semantic search for Retrieval-Augmented Generation (RAG)

### Processing Pipeline

#### PDF Processing
```
PDF Upload → Image Conversion → Layout Analysis → Element Extraction → Alt Text Generation → Markdown Output
```

#### Chat System  
```
User Question → Document Search → Context Retrieval → AI Response (Gemma 3n 4B)
```

#### Voice System
```
Audio Input → Speech Detection → Voice Processing → Text Response → Speech Output
```

## 📁 Project Structure

```
Scholar-Express/
├── 📄 Core Application Files
│   ├── app.py                          # Main Gradio application with multi-tab interface
│   ├── chat.py                         # Document chat functionality
│   ├── gradio_final_app.py            # Final integrated Gradio application
│   └── gradio_local_gemma.py          # Local Gemma model integration
│
├── 🔧 Configuration & Dependencies
│   ├── requirements.txt                # Main project dependencies
│   ├── requirements_gemma.txt          # Gemma-specific dependencies
│   ├── requirements_voice_gemma.txt    # Voice chat dependencies
│   ├── requirements_hf_spaces.txt      # HuggingFace Spaces deployment
│   ├── pyproject.toml                  # Project configuration (Black formatting)
│   └── config/
│       └── Dolphin.yaml               # DOLPHIN model configuration
│
├── 🛠️ Utility Modules
│   └── utils/
│       ├── markdown_utils.py          # Markdown processing utilities
│       ├── model.py                   # AI model management
│       ├── processor.py               # Document processing utilities
│       └── utils.py                   # General utility functions
│
├── 🎤 Voice Chat System  
│   └── voice_chat/
│       ├── app.py                     # Voice chat Gradio interface
│       ├── gemma3n_inference.py       # Gemma 3n voice inference
│       ├── inference.py               # General inference utilities
│       ├── server.py                  # Voice chat server
│       ├── requirements.txt           # Voice-specific dependencies
│       ├── litgpt/                    # LitGPT integration
│       │   ├── config.py              # Model configuration
│       │   ├── model.py               # Model architecture
│       │   ├── tokenizer.py           # Tokenization utilities
│       │   └── generate/              # Text generation utilities
│       ├── utils/
│       │   ├── vad.py                 # Voice Activity Detection
│       │   ├── snac_utils.py          # Audio processing utilities
│       │   └── assets/
│       │       └── silero_vad.onnx    # Silero VAD model
│       └── data/samples/              # Audio sample outputs
│
├── 🤖 Pre-trained Models
│   └── hf_model/                      # HuggingFace model files
│       ├── config.json                # Model configuration
│       ├── model.safetensors          # Model weights
│       ├── tokenizer.json             # Tokenizer configuration
│       └── generation_config.json     # Generation parameters
│
├── 🧪 Development & Demo Files
│   ├── demo_element_hf.py             # Element extraction demo
│   ├── demo_page_hf.py                # Page processing demo
│   ├── gradio_pdf_app.py              # PDF processing demo
│   ├── gradio_image_app.py            # Image processing demo
│   ├── gradio_gemma.py                # Gemma integration demo
│   └── gradio_gemma_api.py            # Gemma API demo
│
└── 📚 Documentation
    ├── README.md                       # This comprehensive guide
    └── Scholar_Express_Technical_Write_Up.pdf  # Detailed technical documentation
```

### 🔑 Essential Files Explained

#### Core Application
- **`app.py`**: Main entry point with complete Gradio interface featuring PDF processing, document chat, and voice interaction tabs
- **`gradio_final_app.py`**: Integrated application combining all features
- **`chat.py`**: Implements RAG-based document question-answering functionality

#### Configuration & Dependencies  
- **`requirements.txt`**: Complete dependency list including PyTorch, Transformers, Gradio, PDF processing, and voice libraries
- **`requirements_voice_gemma.txt`**: Specialized dependencies for voice chat (LitGPT, SNAC, Whisper)
- **`config/Dolphin.yaml`**: Configuration file for DOLPHIN model parameters and settings

#### Utility Modules (`utils/`)
- **`model.py`**: AI model loading, initialization, and management functions
- **`processor.py`**: PDF processing, image extraction, and document parsing utilities  
- **`markdown_utils.py`**: Markdown generation and formatting for accessible output
- **`utils.py`**: General helper functions for file handling and data processing

#### Voice Chat System (`voice_chat/`)
- **`gemma3n_inference.py`**: Core Gemma 3n 2B inference engine for voice processing
- **`utils/vad.py`**: Voice Activity Detection using Silero VAD model
- **`utils/snac_utils.py`**: Audio preprocessing and formatting utilities
- **`litgpt/`**: Lightweight GPT implementation for efficient voice processing

#### Model Files (`hf_model/`)
- **`model.safetensors`**: Pre-trained model weights in SafeTensors format
- **`config.json`**: Model architecture and parameter configuration
- **`tokenizer.json`**: Tokenization rules and vocabulary

### 📋 Dependency Categories

The project uses multiple requirement files for different deployment scenarios:

| File | Purpose | Key Dependencies |
|------|---------|------------------|
| `requirements.txt` | Main application | PyTorch, Transformers, Gradio, PyMuPDF |
| `requirements_voice_gemma.txt` | Voice features | LitGPT, SNAC, Whisper, Librosa |
| `requirements_hf_spaces.txt` | HuggingFace deployment | Streamlined for cloud deployment |
| `requirements_gemma.txt` | Gemma-specific | Optimized for Gemma model usage |

### Key Components

#### PDF Processing (`app.py:convert_pdf_to_images_gradio`)
- Converts PDFs to high-quality images (2x scaling)
- Uses PyMuPDF for reliable extraction

#### Layout Analysis (`app.py:process_elements_optimized`) 
- DOLPHIN identifies text blocks, tables, figures, headers
- Maintains proper reading order for accessibility

#### Alt Text Generation
- Gemma 3n 4B processes images with accessibility-focused prompts
- Generates 1-2 sentence descriptions following WCAG guidelines
- Low temperature (0.1) for consistent, reliable output

#### RAG System
- **Document chunking**: Smart overlap-based chunking (1024 tokens, 100 overlap)
- **Semantic retrieval**: SentenceTransformer embeddings with cosine similarity
- **Context integration**: Top-3 relevant chunks for accurate responses

#### Voice Chat System
- **Gemma 3n 2B**: Optimized for real-time voice processing
- **Silero VAD**: Voice Activity Detection for speech vs silence
- **gTTS**: Google Text-to-Speech for audio responses
- **Audio preprocessing**: 16kHz mono, normalized amplitude

## 🛠️ Technology Stack

| Component | Technology |
|-----------|------------|
| Frontend | Gradio web interface with streaming capabilities |
| AI Models | Gemma 3n, DOLPHIN, SentenceTransformer |
| Document Processing | PyMuPDF, OpenCV, PIL |
| Voice Processing | Librosa, VAD, gTTS |
| Search | SentenceTransformers for semantic retrieval |

## 🎨 Architecture Philosophy

### Right Tool for Right Job
- **DOLPHIN** for PDF extraction and layout analysis
- **Gemma 3n 4B** for alt text generation and document chat
- **Gemma 3n 2B** for real-time voice interaction
- Each component matched to its optimal model and specialization

### Privacy-First Design
- All processing happens locally to protect sensitive academic content
- Meets institutional privacy requirements for research documents

### Accessibility Focus
- AI-generated alt text makes academic papers inclusive for visually impaired researchers
- Addresses a real gap in academic publishing accessibility

## 🚀 Getting Started

1. **Install dependencies**: The app uses Gradio, PyMuPDF, and various AI model libraries
2. **Run the application**: `python app.py`
3. **Access the interface**: Open the Gradio web interface
4. **Upload a PDF**: Use the document processing tab to convert research papers
5. **Interact**: Chat with documents or use voice features for hands-free research

## 💡 Design Challenges Solved

### Challenge 1: Narrowing Down Big Ideas
- Focused on three core applications: alt text, document chat, and voice interaction
- Chose accessibility as the primary value proposition
- Specialized each model variant (4B vs 2B) for optimal performance

### Challenge 2: Storage Limitations  
- Developed code-first approach with thorough review before testing
- Built comprehensive error handling upfront since debugging was expensive
- Improved documentation and commenting discipline

## 📈 Impact

Scholar Express bridges the accessibility gap in scientific research, ensuring that the 18-54% of students with disabilities can access the same research literature as their peers, while providing enhanced interaction capabilities for all users working with complex scientific content.
