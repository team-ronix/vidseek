import torch
import ffmpeg
import numpy as np
import json
import os
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


class ASR:
    def __init__(self, model_name="openai/whisper-small", video_path=None):
        self.video_path = video_path
        self.audio = None
        self.result = None
        self.model_id = model_name

        os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.model = AutoModelForSpeechSeq2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True
        ).to(self.device)

        self.model.eval() 

        processor = AutoProcessor.from_pretrained(self.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=self.model_id,  
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=torch_dtype,
            device=0 if self.device == "cuda:0" else -1,
        )

    def _filter_characters(self, text):
        allowed_characters = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 .,!?;:'\"-()[]{}")
        filtered_text = ''.join(c for c in text if c in allowed_characters)
        return filtered_text
    
    def extract_audio_from_video(self):
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

        self.audio = np.frombuffer(audio, np.int16).astype(np.float32) / 32768.0

    def get_audio(self):
        if self.audio is None:
            self.extract_audio_from_video()
        return self.audio

    def transcribe(self, task="transcribe"):
        audio = self.get_audio()

        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=(self.device == "cuda:0")):
                self.result = self.pipe(
                    audio,
                    chunk_length_s=20,  
                    batch_size=2,       
                    return_timestamps=True,
                    generate_kwargs={"task": task}
                )

        torch.cuda.empty_cache()
        self.result['text'] = self._filter_characters(self.result['text'])
        for chunk in self.result['chunks']:
            chunk['text'] = self._filter_characters(chunk['text'])

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
            
    def load_transcription(self, input_path='transcription.json'):
        with open(input_path, 'r', encoding='utf-8') as f:
            self.result = json.load(f)