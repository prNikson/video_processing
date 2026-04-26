from summary import ProcessVideo
from get_video import get_video_pair


def main():
    video_path, description = get_video_pair()

    processing = ProcessVideo(video_path, description)
    processing.process_video()


if __name__ == "__main__":
    main()
