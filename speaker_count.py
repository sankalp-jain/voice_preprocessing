from pyannote.audio import Pipeline
import sys

def count_speakers(audio_path: str, hf_token: str):
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization",
        use_auth_token=hf_token
    )
    diarization = pipeline(audio_path)
    speakers = set()
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.add(speaker)
    print(f"Estimated number of speakers: {len(speakers)}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python speaker_count.py path_to_audio_file HF_ACCESS_TOKEN")
        sys.exit(1)
    audio_file = sys.argv[1]
    token = sys.argv[2]
    count_speakers(audio_file, token)
