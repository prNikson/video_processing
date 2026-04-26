import os
import tomllib
import uuid
import time
import subprocess
import json

import docker
import torch
import requests
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor

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
        self.prompt = self._load_prompt()

    def _read_config_file(self, config_file) -> None:
        with open(config_file, 'rb') as f:
            data = tomllib.load(f)
            self.whisper_url = data['config']['whisper_url']
            self.qwen_video_url = data['config']['qwen_video_url']
            self.rerank_url = data['config']['rerank_url']
            self.paddle_url = data['config']['paddle_url']

            self.video_model = data['model']['video']
            self.embed_model = data['model']['embed']

            self.a = data['param']['video']
            self.b = data['param']['transcribe']
            self.c = data['param']['ocr']

            self.prompt_path = data['prompt']['main_prompt']

    def _load_prompt(self) -> None:
        with open(self.prompt_path, 'r') as f:
            return f.read()

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
        audioname = f"data/{uuid.uuid4()}.mp3"
        subprocess.run([
            "ffmpeg",
            "-y",
            "-i", video_path,
            "-vn",
            "-acodec", "mp3",
            audioname,
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        return audioname

    def _process_audio(self) -> str:
        self._audio_path = self._extract_audio('data/' + self.video_path)

        with open(self._audio_path, 'rb') as f:
            files = {'audio_file': f, "task": "transcribe", "output": "json"}
            transcript = requests.post(self.whisper_url, files=files).text
        #self._stop_service('whisper')
        return transcript

    def _process_video_sense(self) -> str:
        vllm_payload = {
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
        return visual_description['choices'][0]['message']['content']

    def _process_ocr(self) -> str:
        result = []
        cap = cv2.VideoCapture('data/' + self.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)

        if int(fps) == 30:
            divider = 10
        else:
            divider = 30

        count = 0

        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            if count % divider == 0:
                if frame is not None:
                    _, img_encoded = cv2.imencode('.jpg', frame)

                    try:
                        response = requests.post(
                            self.paddle_url,
                            files={"file": ("frame.jpg", img_encoded.tobytes(), "image/jpeg")}
                        )
                        result.append(" ".join(response.json()))
                    except Exception as e:
                        print(e)
            count += 1
        
        result = " ".join(result)

        cap.release()

        return result

    def _summarize_all(self, transcript: str, visual_desc: str, ocr_text: str) -> np.ndarray:
        payload = {
            "model": self.video_model,
            "messages": [
                {
                    "role": "user",
                    "content": self.prompt.format(visual_description=visual_desc, ocr_text=ocr_text, transcript=transcript)
                }
            ],
            "temperature": 0.2,
            "max_tokens": 800,
            "response_format": {"type": "json_object"}
        }

        response = requests.post(self.qwen_video_url, json=payload).json()

        result = response["choices"][0]["message"]["content"]

        return result

    def _normalize_weights(self, confidence) -> None:
        visual = confidence['visual']
        speech = confidence['speech']
        ocr = confidence['ocr']

        total = visual + speech + ocr

        if total > 0:
            self.visual_norm = visual / total
            self.speech_norm = speech / total
            self.ocr_norm = ocr / total
        else:
            self.visual_norm = self.a
            self.speech_norm = self.b
            self.ocr_norm = self.c

    def process_video(self):
        with ThreadPoolExecutor(max_workers=3) as executor:
            audio_future = executor.submit(self._process_audio)
            video_future = executor.submit(self._process_video_sense)
            ocr_future = executor.submit(self._process_ocr)

            transcript = audio_future.result()
            visual_desc = video_future.result()
            ocr_text = ocr_future.result()

        summary = json.loads(self._summarize_all(transcript, visual_desc, ocr_text))
        self._normalize_weights(summary['confidence'])

        self._get_embeddings(summary)

        os.remove(self._audio_path)

    def _prepare_description(self):
        print(self.video_description)
        #self.video_description = self.video_description.replace("#", "")

    def _get_embeddings(self, summary: dict[str, str]) -> float:
        print(summary)
        print(self.video_description)
        return
        video_sense_embedding = self.embedding.get_embedding(summary['video_summary'])
        audio_sense_embedding = self.embedding.get_embedding(summary['spoken_topics'])
        ocr_embedding = self.embedding.get_embedding(summary['subtitle_meaning'])

        overall_embedding = self.embedding.get_embedding(summary['overall_context'])

        self._prepare_description()
        description_embedding = self.embedding.get_embedding(self.video_description)

        weighted = self.visual_norm * video_sense_embedding +\
        self.speech_norm * audio_sense_embedding +\
        self.ocr_norm * ocr_embedding

        print(summary)

    # weighted similarity
        weighted_similarity = cosine_similarity([weighted], [description_embedding])[0][0]
        print("Weighted similarity:", weighted_similarity)

    # only video similarity
        print("Video similarity:", cosine_similarity([video_sense_embedding], [description_embedding])[0][0])

    # only audio similarity
        print("Audio similarity:", cosine_similarity([audio_sense_embedding], [description_embedding])[0][0])

    #only ocr similarity
        print("OCR similarity:", cosine_similarity([ocr_embedding], [description_embedding])[0][0])

    # tags similarity
        print("Overall context similarity:", cosine_similarity([overall_embedding], [description_embedding])[0][0])

