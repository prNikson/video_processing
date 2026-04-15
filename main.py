import requests
from moviepy import VideoFileClip


WHISPER_URL = "http://0.0.0.0:9000/asr"
VLLM_URL = "http://localhost:8000/v1/chat/completions"

def extract_audio(video_path) -> str:
    audioname = "1.mp3"
    video = VideoFileClip(video_path)
    audio = video.audio
    audio.write_audiofile(audioname)
    return audioname

def process_audio(video_path):
    audio_path = extract_audio(video_path)

    with open(audio_path, 'rb') as f:
        files = {'audio_file': f, "task": "transcribe", "output": "json"}
        transcript = requests.post(WHISPER_URL, files=files).text
    return transcript

def process_video(video_path):
    vllm_payload = {
        "model": "Qwen/Qwen3-VL-4B-Instruct",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this video"},
                    {"type": "video_url", "video_url": {"url": f"file:////data/{video_path}"}, "max_pixels": 360 * 420, "fps": 1.0, "max_tokens": 512},
                ]
            }
        ]
    }

    visual_description = requests.post(VLLM_URL, json=vllm_payload).json()
    return visual_description['choices'][0]['message']['content']
def main():
    video_path = "fhd.mp4"
    print(process_audio(video_path))
    print(process_video(video_path))


if __name__ == "__main__":
    main()
