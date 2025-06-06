import os
import wave
import contextlib
import webrtcvad
import collections
import numpy as np
from pydub import AudioSegment
from scipy.io import wavfile
import tempfile

# Convert to 16kHz mono WAV
def convert_to_wav_16k_mono(input_path):
    sound = AudioSegment.from_file(input_path)
    sound = sound.set_channels(1).set_frame_rate(16000)
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    sound.export(temp_file.name, format="wav")
    return temp_file.name

# Frame generator for VAD
class Frame:
    def __init__(self, bytes, timestamp, duration):
        self.bytes = bytes
        self.timestamp = timestamp
        self.duration = duration

def frame_generator(frame_duration_ms, audio, sample_rate):
    n = int(sample_rate * (frame_duration_ms / 1000.0) * 2)
    offset = 0
    timestamp = 0.0
    duration = (float(n) / sample_rate) / 2.0
    while offset + n < len(audio):
        yield Frame(audio[offset:offset + n], timestamp, duration)
        timestamp += duration
        offset += n

def vad_collector(sample_rate, frame_duration_ms, padding_duration_ms, vad, frames):
    num_padding_frames = int(padding_duration_ms / frame_duration_ms)
    ring_buffer = collections.deque(maxlen=num_padding_frames)
    triggered = False
    voiced_segments = []
    voiced_frames = []

    for frame in frames:
        is_speech = vad.is_speech(frame.bytes, sample_rate)

        if not triggered:
            ring_buffer.append((frame, is_speech))
            num_voiced = len([f for f, speech in ring_buffer if speech])
            if num_voiced > 0.9 * ring_buffer.maxlen:
                triggered = True
                voiced_frames.extend([f for f, s in ring_buffer])
                ring_buffer.clear()
        else:
            voiced_frames.append(frame)
            ring_buffer.append((frame, is_speech))
            num_unvoiced = len([f for f, speech in ring_buffer if not speech])
            if num_unvoiced > 0.9 * ring_buffer.maxlen:
                triggered = False
                segment = b''.join([f.bytes for f in voiced_frames])
                voiced_segments.append(segment)
                voiced_frames = []
                ring_buffer.clear()

    if voiced_frames:
        segment = b''.join([f.bytes for f in voiced_frames])
        voiced_segments.append(segment)

    return voiced_segments

# Main processing function
def extract_best_voice_segments(input_file, output_file="clean_sample.wav", top_n_seconds=20):
    print("[INFO] Preprocessing audio...")

    # Step 1: Convert to 16kHz mono wav
    wav_path = convert_to_wav_16k_mono(input_file)

    with contextlib.closing(wave.open(wav_path, 'rb')) as wf:
        sample_rate = wf.getframerate()
        pcm_data = wf.readframes(wf.getnframes())

    vad = webrtcvad.Vad(3)  # Aggressiveness mode: 0 (least) to 3 (most aggressive)
    frames = list(frame_generator(30, pcm_data, sample_rate))
    segments = vad_collector(sample_rate, 30, 300, vad, frames)

    print(f"[INFO] Found {len(segments)} voiced segments")

    # Convert raw segments to numpy arrays and score them
    def segment_to_np(segment_bytes):
        return np.frombuffer(segment_bytes, dtype=np.int16)

    scored_segments = []
    for seg in segments:
        arr = segment_to_np(seg)
        energy = np.sum(arr ** 2) / len(arr)
        duration = len(arr) / sample_rate
        if duration > 0.5:  # ignore too short
            scored_segments.append((seg, energy, duration))

    # Sort by energy (you can add more scoring logic here)
    scored_segments.sort(key=lambda x: x[1], reverse=True)

    # Select top N seconds
    selected_segments = []
    total_duration = 0
    for seg, _, duration in scored_segments:
        if total_duration + duration <= top_n_seconds:
            selected_segments.append(seg)
            total_duration += duration
        else:
            break

    print(f"[INFO] Selected {total_duration:.2f} seconds of voiced audio")

    # Concatenate and save
    all_bytes = b''.join(selected_segments)
    output_wave = wave.open(output_file, 'wb')
    output_wave.setnchannels(1)
    output_wave.setsampwidth(2)
    output_wave.setframerate(16000)
    output_wave.writeframes(all_bytes)
    output_wave.close()

    print(f"[✅] Clean voice data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    input_path = "audio.mp3"  # Change to your 100MB file
    extract_best_voice_segments(input_path, output_file="clean_sample.wav", top_n_seconds=20)
