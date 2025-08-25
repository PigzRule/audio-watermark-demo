import torch
import random
import numpy as np
import io
from pydub import AudioSegment
import scipy.signal

class AudioEffects:
    @staticmethod
    def speed(tensor, speed_range=(0.8, 1.2), sample_rate=16000):
        factor = random.uniform(*speed_range)
        indices = torch.arange(0, tensor.size(-1), factor, device=tensor.device)
        indices = indices.clamp(0, tensor.size(-1) - 1).long()
        return tensor[..., indices]

    @staticmethod
    def echo(tensor, volume_range=(0.1, 0.5), duration_range=(0.1, 0.3), sample_rate=16000):
        vol = random.uniform(*volume_range)
        delay = int(random.uniform(*duration_range) * sample_rate)
        echo = torch.zeros_like(tensor)
        echo[..., delay:] = tensor[..., :-delay] * vol
        return tensor + echo

    @staticmethod
    def random_noise(tensor, noise_std=0.01):
        noise = noise_std * torch.randn_like(tensor)
        return (tensor + noise).clamp(-1, 1)

    @staticmethod
    def pink_noise(tensor, noise_std=0.01):
        white = torch.randn_like(tensor)
        b, a = [0.049922, -0.095993, 0.050612, -0.004408], [1, -2.494956, 2.017265, -0.522189]
        pink = torch.from_numpy(scipy.signal.lfilter(b, a, white.squeeze().cpu().numpy())).to(tensor.device)
        return (tensor + pink.view(1, 1, -1) * noise_std).clamp(-1, 1)

    @staticmethod
    def lowpass_filter(tensor, cutoff_freq=3000, sample_rate=16000):
        b, a = scipy.signal.butter(5, cutoff_freq / (sample_rate / 2), btype='low')
        filtered = scipy.signal.lfilter(b, a, tensor.squeeze().cpu().numpy())
        return torch.from_numpy(filtered).view(1, 1, -1).to(tensor.device)

    @staticmethod
    def highpass_filter(tensor, cutoff_freq=500, sample_rate=16000):
        b, a = scipy.signal.butter(5, cutoff_freq / (sample_rate / 2), btype='high')
        filtered = scipy.signal.lfilter(b, a, tensor.squeeze().cpu().numpy())
        return torch.from_numpy(filtered).view(1, 1, -1).to(tensor.device)

    @staticmethod
    def smooth(tensor, window_size_range=(2, 5)):
        window_size = random.randint(*window_size_range)
        return torch.nn.functional.avg_pool1d(tensor, window_size, stride=1, padding=window_size//2)

    @staticmethod
    def shush(tensor, fraction=0.005):
        tensor = tensor.clone()
        length = tensor.size(-1)
        start = random.randint(0, length - int(fraction * length))
        tensor[..., start:start+int(fraction * length)] = 0
        return tensor

    @staticmethod
    def mp3_compression(tensor, sample_rate=16000, bitrate='64k'):
        audio_np = tensor.squeeze().cpu().numpy()
        audio_segment = AudioSegment(
            (audio_np * 32768.0).astype(np.int16).tobytes(),
            frame_rate=sample_rate,
            sample_width=2,
            channels=1
        )
        buffer = io.BytesIO()
        audio_segment.export(buffer, format='mp3', bitrate=bitrate)
        buffer.seek(0)
        compressed = AudioSegment.from_mp3(buffer)
        compressed_samples = np.array(compressed.get_array_of_samples()).astype(np.float32)
        compressed_samples = torch.from_numpy(compressed_samples).view(1, 1, -1) / 32768.0
        compressed_samples = compressed_samples.to(tensor.device)
        return compressed_samples / compressed_samples.abs().max()

    @staticmethod
    def bandstop_filter(tensor, sample_rate=16000, low_cut=1000, high_cut=3000):
        audio_np = tensor.squeeze().cpu().numpy()
        sos = scipy.signal.butter(10, [low_cut, high_cut], btype='bandstop', fs=sample_rate, output='sos')
        filtered = scipy.signal.sosfilt(sos, audio_np)
        return torch.from_numpy(filtered).view(1, 1, -1).to(tensor.device)

    @staticmethod
    def adversarial_noise(tensor, epsilon=0.002):
        noise = epsilon * torch.sign(torch.randn_like(tensor))
        attacked = tensor + noise
        attacked = attacked / attacked.abs().max()
        return attacked
    
    @staticmethod
    def targeted_bandpass_noise(audio, sample_rate):
        noise = torch.randn_like(audio) * 0.01

        sos = scipy.signal.butter(4, [3000, 6000], btype='bandpass', fs=sample_rate, output='sos')
        band_noise = scipy.signal.sosfilt(sos, noise.cpu().numpy())

        noisy_audio = audio + torch.from_numpy(band_noise).to(audio.device)
        return noisy_audio.clamp(-1, 1)

    @staticmethod
    def frequency_nulling(audio, sample_rate):
        spectrum = torch.fft.rfft(audio.clone(), dim=-1)

        start = int(0.25 * spectrum.size(-1))
        end = int(0.4 * spectrum.size(-1))
        spectrum[..., start:end] = 0

        return torch.fft.irfft(spectrum, n=audio.size(-1), dim=-1).clamp(-1, 1)