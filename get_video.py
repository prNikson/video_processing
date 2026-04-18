import uuid
import requests
import pandas as pd
from random import randrange


df = pd.read_csv("output.csv")

link = df['link'].tolist()
description = df['description'].tolist()

def download_video(video_url: str) -> str | None:
    with requests.get(video_url, stream=True) as r:
        r.raise_for_status()
        video_name = str(uuid.uuid4())
        with open(f"data/{video_name}.mp4", "wb") as f:
            for chunk in r.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)
        return video_name
    return None

def get_video_pair() -> tuple[str, str]:
   # rand_number = randrange(len(link))
    rand_number = 19
    videoname = download_video(link[rand_number - 1])
    if videoname is not None:
        return f"{videoname}.mp4", description[rand_number - 1]
