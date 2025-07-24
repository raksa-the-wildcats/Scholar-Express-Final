import flask
import base64
import tempfile
import traceback
import os
from flask import Flask, Response, stream_with_context
from inference import OmniInference
from gemma3n_inference import Gemma3nInference


class OmniChatServer(object):
    def __init__(self, ip='0.0.0.0', port=60808, run_app=True,
                 ckpt_dir='./checkpoint', device='cuda:0', use_gemma3n=True) -> None:
        server = Flask(__name__)
        # CORS(server, resources=r"/*")
        # server.config["JSON_AS_ASCII"] = False

        self.use_gemma3n = use_gemma3n
        
        if self.use_gemma3n:
            print("Initializing Gemma 3n backend...")
            try:
                self.client = Gemma3nInference(device)
            except Exception as e:
                print(f"Failed to load Gemma 3n, falling back to original model: {e}")
                self.use_gemma3n = False
                self.client = OmniInference(ckpt_dir, device)
        else:
            print("Initializing original Omni backend...")
            self.client = OmniInference(ckpt_dir, device)
            
        self.client.warm_up()

        server.route("/chat", methods=["POST"])(self.chat)

        if run_app:
            server.run(host=ip, port=port, threaded=False)
        else:
            self.server = server

    def chat(self) -> Response:

        req_data = flask.request.get_json()
        try:
            data_buf = req_data["audio"].encode("utf-8")
            stream_stride = req_data.get("stream_stride", 4)
            max_tokens = req_data.get("max_tokens", 2048)

            if self.use_gemma3n:
                # Use Gemma 3n inference
                audio_response = self.client.process_audio_stream(data_buf)
                
                def generate_chunks():
                    # Stream the audio response in chunks
                    chunk_size = 4096
                    for i in range(0, len(audio_response), chunk_size):
                        yield audio_response[i:i + chunk_size]
                
                return Response(stream_with_context(generate_chunks()), mimetype="audio/wav")
            else:
                # Use original inference
                data_buf = base64.b64decode(data_buf)
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(data_buf)
                    audio_generator = self.client.run_AT_batch_stream(f.name, stream_stride, max_tokens)
                    return Response(stream_with_context(audio_generator), mimetype="audio/wav")
        except Exception as e:
            print(traceback.format_exc())


# CUDA_VISIBLE_DEVICES=1 gunicorn -w 2 -b 0.0.0.0:60808 'server:create_app()'
def create_app():
    server = OmniChatServer(run_app=False)
    return server.server


def serve(ip='0.0.0.0', port=60808, device='cuda:0', use_gemma3n=True):

    OmniChatServer(ip, port=port, run_app=True, device=device, use_gemma3n=use_gemma3n)


if __name__ == "__main__":
    import fire
    fire.Fire(serve)
    
