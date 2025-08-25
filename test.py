import torch
from watermark_engines import WatermarkEngine
from pydub import AudioSegment
import numpy as np

def load_audio_tensor(path, sample_rate=16000):
    audio = AudioSegment.from_file(path).set_channels(1).set_frame_rate(sample_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.tensor(samples).view(1, 1, -1)

# === Test ===
sample = load_audio_tensor("samples/03-01-06-01-02-02-01.wav")
engine = WatermarkEngine("AudioSeal")

watermarked = engine.embed(sample, watermark='1010101010101010')
detected = engine.detect(watermarked)

print(f"✅ AudioSeal detected watermark on clean audio? {detected}")
