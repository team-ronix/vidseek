import torch
import ffmpeg
import numpy as np
import json
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASR:
    def __init__(self, model_name="openai/whisper-small", video_path=None):
        self.video_path = video_path
        self.audio = None
        self.result = None
        self.model_id = model_name
        
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id, dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
        )
        self.model.to(device)
        processor = AutoProcessor.from_pretrained(self.model_id)
        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=torch_dtype,
            device=device,
        )
    
    def extract_audio_from_video(self):
        # whisper requires mono audio not stereo -> ac=1
        # whisper trained on 16kHz audio -> ar='16k'
        try:
            audio, err = (
                ffmpeg
                .input(self.video_path)
                .output('-', format='s16le', acodec='pcm_s16le', ac=1, ar='16k')
                .run(capture_stdout=True, capture_stderr=True)
            )
            if err and err.strip():
                print(f"FFmpeg stderr: {err.decode('utf-8')}")
        except ffmpeg.Error as e:
            print(f"Error extracting audio: {e.stderr.decode('utf-8')}")
            raise
        # / 32768.0 -> for normalizaiton to range -1 to 1
        self.audio = np.frombuffer(audio, np.int16).flatten().astype(np.float32) / 32768.0
        
    def get_audio(self):
        if self.audio is None:
            self.extract_audio_from_video()
        return self.audio
    
    def transcribe(self):
        audio = self.get_audio()
        self.result = self.pipe(
            audio,
            chunk_length_s=30,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={"task": "translate"}
        )
            
    def get_text(self):
        if self.result is None:
            self.transcribe()
        return self.result['text']
    
    def get_chunks(self):
        if self.result is None:
            self.transcribe()
        return self.result['chunks']
    
    def save_transcription(self, output_path='transcription.json'):
        if self.result is None:
            self.transcribe()
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(self.result, f, indent=2, ensure_ascii=False)