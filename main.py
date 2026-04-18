import os
import tomllib
import uuid
import time
import subprocess

import docker
import torch
import requests
import cv2
import numpy as np
from moviepy import VideoFileClip
from sklearn.metrics.pairwise import cosine_similarity

from embeddings import SimpleEmbedding
from get_video import get_video_pair


class ProcessVideo:
    def __init__(
        self,
        video_path,
        description,
        config_file='config.toml'
    ):
        self.video_path = video_path
        self.video_description = description
        self._read_config_file(config_file)
        self.embedding = SimpleEmbedding(self.embed_url, self.embed_model)

    def _read_config_file(self, config_file) -> None:
        with open(config_file, 'rb') as f:
            data = tomllib.load(f)
            self.whisper_url = data['config']['whisper_url']
            self.qwen_video_url = data['config']['qwen_video_url']
            self.embed_url = data['config']['embed_url']

            self.video_model = data['model']['video']
            self.embed_model = data['model']['embed']

            self.a = data['param']['video']
            self.b = data['param']['transcribe']
            self.c = data['param']['ocr']

    def _clear_gpu_memory(self) -> None:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
    
    def _stop_service(self, service_name: str) -> None:
        subprocess.run(["docker-compose", "stop", service_name], capture_output=True)
        self._clear_gpu_memory()
        time.sleep(2)

    def _start_service(self, service_name: str) -> None:
        subprocess.run(["docker-compose", "start", service_name], capture_output=True)
        time.sleep(5)

    def _wait_for_service(self, url: str, max_retries: int = 30) -> None:
        for i in range(max_retries):
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    return True
            except:
                pass
            time.sleep(2)
        return False

    def _extract_audio(self, video_path: str) -> bytes:
        video = VideoFileClip(video_path)
        audioname = f"{uuid.uuid4()}.mp3"
        audio = video.audio
        audio.write_audiofile(audioname)
        return audioname

    def _process_audio(self) -> str:
        self._audio_path = self._extract_audio(self.video_path)

        with open(self._audio_path, 'rb') as f:
            files = {'audio_file': f, "task": "transcribe", "output": "json"}
            transcript = requests.post(self.whisper_url, files=files).text
        #self._stop_service('whisper')
        return transcript

    def _process_video_sense(self) -> str:
        vllm_payload = {
            #"model": "Qwen/Qwen3-VL-4B-Instruct",
            "model": self.video_model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this video"},
                        {"type": "video_url", "video_url": {"url": f"file:////data/data/{self.video_path}"}, "max_pixels": 360 * 420, "fps": 1.0, "max_tokens": 1024},
                    ]
                }
            ]
        }

        visual_description = requests.post(self.qwen_video_url, json=vllm_payload).json()
        #self._stop_service('vllm-qwen')
        return visual_description['choices'][0]['message']['content']

    def _process_ocr(self) -> str:
        result = []
        cap = cv2.VideoCapture('data/' + self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if count % 10 == 0:
                if frame is not None:
                    _, img_encoded = cv2.imencode('.jpg', frame)

                    try:
                        response = requests.post(
                            "http://0.0.0.0:8010/ocr",
                            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
                        )
                        result.append(" ".join(response.json()))
                    except Exception as e:
                        print(e)
            count += 1
        
        result = " ".join(result)

        payload = {
            "model": self.video_model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a professional editor and analyst with expertise in OCR error correction."
                },
                {
                    "role": "user",
                    "content": f"Clean:\n1.Analyze the provided OCR text and correct typos, character misrecognitions (e.g., '0' instead of 'O', '1' instead of 'l'), and punctuation errors. Ensure the text flows logically.\n2.Translate (if needed): If the text is not in English, translate the corrected version into fluent English.\n3.Summarize: Provide a concise summary of the key points in English.\n\nConstraints:\n1.Do not make up facts; if a word is completely illegible, mark it as [unintelligible].\n2.The summary should be structured (bullet points or a short paragraph).",
                    "max_tokens": 1024
                }
            ]
        }

        ocr_description = requests.post(self.qwen_video_url, json=payload).json()
        print(ocr_description['choices'][0]['message']['content'])

        cap.release()

    def process_video(self):
        self._process_ocr()
       # print(self._get_embeddings())
       # os.remove(self._audio_path)

    def _get_embeddings(self):
        #self._start_service("embedding")
        self._wait_for_service(self.embed_url.replace("/v1/embeddings", "health"))

        video_sense_embedding = self.embedding.get_embedding(self._process_video_sense())
        audio_sense_embedding = self.embedding.get_embedding(self._process_audio())
        description_embedding = self.embedding.get_embedding(self.video_description)

        weighted = self.a * video_sense_embedding + self.b * audio_sense_embedding

        similarity = cosine_similarity([weighted], [description_embedding])[0][0]
        self._stop_service("embedding")
        return similarity


def main():
    video_path, description = get_video_pair()

    handler = ProcessVideo(video_path, description)
    handler.process_video()

if __name__ == "__main__":
    main()
