import os
import tempfile
import torch
import librosa
import numpy as np
from transformers import AutoModelForImageTextToText, AutoProcessor
from huggingface_hub import login
import io
from pydub import AudioSegment
import base64
import traceback


class Gemma3nInference:
    def __init__(self, device='cuda:0'):
        self.device = device
        
        # Login to Hugging Face using token from environment
        hf_token = os.getenv('HF_TOKEN')
        if hf_token:
            login(token=hf_token)
        else:
            print("Warning: HF_TOKEN not found in environment variables")
        
        print("Loading Gemma 3n model...")
        try:
            # Try loading Gemma 3n E2B (2B effective params) using the correct class
            model_name = "google/gemma-3n-E2B-it"
            self.model = AutoModelForImageTextToText.from_pretrained(
                model_name,
                torch_dtype="auto",  # Let it auto-detect the best dtype
                device_map="auto",
                trust_remote_code=True
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            print(f"Gemma 3n E2B model loaded successfully on device: {self.model.device}")
            print(f"Model dtype: {self.model.dtype}")
        except Exception as e:
            print(f"Error loading Gemma 3n model: {e}")
            print("Trying alternative loading method...")
            try:
                # Try loading without vision components initially
                from transformers import AutoConfig
                config = AutoConfig.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                # Disable vision tower if causing issues
                if hasattr(config, 'vision_config'):
                    print("Attempting to load without problematic vision config...")
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_name,
                    torch_dtype="auto",
                    trust_remote_code=True,
                    ignore_mismatched_sizes=True
                ).to(self.device)
                self.processor = AutoProcessor.from_pretrained(
                    model_name,
                    trust_remote_code=True
                )
                print("Gemma 3n E2B model loaded with alternative method")
            except Exception as e2:
                print(f"Alternative loading also failed: {e2}")
                raise e2

    def preprocess_audio(self, audio_path):
        """Convert audio to Gemma 3n format: 16kHz mono float32 in range [-1, 1]"""
        try:
            # Load audio file
            audio, sr = librosa.load(audio_path, sr=16000, mono=True)
            
            # Ensure audio is in range [-1, 1]
            if audio.max() > 1.0 or audio.min() < -1.0:
                audio = audio / max(abs(audio.max()), abs(audio.min()))
            
            # Limit to 30 seconds as recommended
            max_samples = 30 * 16000
            if len(audio) > max_samples:
                audio = audio[:max_samples]
            
            return audio.astype(np.float32)
        except Exception as e:
            print(f"Error preprocessing audio: {e}")
            raise

    def create_multimodal_input(self, audio_path, text_prompt="Respond naturally to this audio input"):
        """Create multimodal input for Gemma 3n using the same format as the notebook"""
        try:
            # Preprocess audio
            audio_array = self.preprocess_audio(audio_path)
            
            # Create multimodal message format exactly like the notebook
            message = {
                "role": "user",
                "content": [
                    {"type": "audio", "audio": audio_path},  # Use path instead of array
                    {"type": "text", "text": text_prompt}
                ]
            }
            
            # Process with Gemma 3n processor using the notebook approach
            inputs = self.processor.apply_chat_template(
                [message],  # History is a list
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
            
            return inputs.to(self.device, dtype=self.model.dtype)
        except Exception as e:
            print(f"Error creating multimodal input: {e}")
            traceback.print_exc()
            raise

    def generate_response(self, audio_path, max_new_tokens=256):
        """Generate text response from audio input using notebook approach"""
        try:
            # Create multimodal input
            inputs = self.create_multimodal_input(audio_path)
            input_len = inputs["input_ids"].shape[-1]
            
            # Generate response exactly like the notebook
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    disable_compile=True
                )
            
            # Decode response exactly like the notebook
            text = self.processor.batch_decode(
                outputs[:, input_len:],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            return text[0].strip() if text else "No response generated"
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def stream_response(self, audio_path, max_new_tokens=512, temperature=0.9):
        """Generate streaming text response from audio input"""
        try:
            # Create multimodal input
            inputs = self.create_multimodal_input(audio_path)
            
            # Generate streaming response
            with torch.no_grad():
                # Use the model's generate method with streaming
                streamer = self.processor.tokenizer
                
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.processor.tokenizer.eos_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                    return_dict_in_generate=True,
                    output_scores=True
                )
                
                # Decode the full response
                response = self.processor.tokenizer.decode(
                    outputs.sequences[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                
                return response.strip()
                
        except Exception as e:
            print(f"Error in streaming response: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"

    def text_to_speech_simple(self, text):
        """Convert text to speech using gTTS"""
        try:
            from gtts import gTTS
            
            # Create TTS object
            tts = gTTS(text=text, lang='en', slow=False)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as tmp_file:
                tts.save(tmp_file.name)
                
                # Convert MP3 to WAV format that the system expects
                audio_segment = AudioSegment.from_mp3(tmp_file.name)
                
                # Convert to expected format (24kHz, mono, 16-bit)
                audio_segment = audio_segment.set_frame_rate(24000)
                audio_segment = audio_segment.set_channels(1)
                audio_segment = audio_segment.set_sample_width(2)
                
                # Export to WAV bytes
                audio_buffer = io.BytesIO()
                audio_segment.export(audio_buffer, format="wav")
                
                # Clean up temp file
                os.unlink(tmp_file.name)
                
                return audio_buffer.getvalue()
            
        except ImportError:
            print("gTTS not available, falling back to silence")
            # Fallback to silence if gTTS not installed
            duration_seconds = max(1, len(text) / 20)
            sample_rate = 24000
            samples = int(duration_seconds * sample_rate)
            audio_data = np.zeros(samples, dtype=np.int16)
            audio_segment = AudioSegment(
                audio_data.tobytes(),
                frame_rate=sample_rate,
                sample_width=2,
                channels=1
            )
            audio_buffer = io.BytesIO()
            audio_segment.export(audio_buffer, format="wav")
            return audio_buffer.getvalue()
            
        except Exception as e:
            print(f"Error in TTS: {e}")
            # Return minimal audio data on error
            return b'\x00' * 1024

    def process_audio_stream(self, audio_bytes):
        """Process audio stream and return response audio stream"""
        try:
            # Decode base64 audio
            audio_data = base64.b64decode(audio_bytes)
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                f.write(audio_data)
                temp_audio_path = f.name
            
            try:
                # Generate text response
                text_response = self.generate_response(temp_audio_path)
                print(f"Generated response: {text_response}")
                
                # Convert to speech (placeholder)
                audio_response = self.text_to_speech_simple(text_response)
                
                return audio_response
                
            finally:
                # Clean up temp file
                if os.path.exists(temp_audio_path):
                    os.unlink(temp_audio_path)
                    
        except Exception as e:
            print(f"Error processing audio stream: {e}")
            traceback.print_exc()
            # Return minimal audio data on error
            return b'\x00' * 1024

    def warm_up(self):
        """Warm up the model"""
        try:
            print("Warming up Gemma 3n model...")
            # Create a short dummy audio
            dummy_audio = np.zeros(16000, dtype=np.float32)  # 1 second of silence
            
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                # Save dummy audio
                import soundfile as sf
                sf.write(f.name, dummy_audio, 16000)
                
                # Generate a quick response
                response = self.generate_response(f.name, max_new_tokens=10)
                print(f"Warm-up response: {response}")
                
                # Clean up
                os.unlink(f.name)
            
            print("Gemma 3n warm-up complete")
        except Exception as e:
            print(f"Error during warm-up: {e}")
            # Don't fail startup on warm-up errors
            pass