import torch
import torch.nn.functional as F

class WatermarkEngine:
    def __init__(self, engine_type):
        if engine_type not in ['AudioSeal', 'WavMark', 'SilentCipher']:
            raise ValueError(f"Unsupported engine type: {engine_type}")
        self.engine_type = engine_type
        self.embedded_pattern = None

    def embed(self, audio, watermark, ablation_strength=0.01):
        if self.engine_type == 'AudioSeal':
            pattern = (
                torch.sin(torch.linspace(0, 20 * torch.pi, audio.size(-1))) +
                0.3 * torch.sin(torch.linspace(0, 100 * torch.pi, audio.size(-1)))
            )
            pattern += 0.05 * torch.randn_like(pattern)
            pattern = pattern.unsqueeze(0).unsqueeze(0).to(audio.device)
            self.embedded_pattern = pattern
            return audio + ablation_strength * pattern

        elif self.engine_type == 'WavMark':
            spectrum = torch.fft.rfft(audio, dim=-1)
            indices = torch.randint(20, spectrum.size(-1) - 20, (15,))
            original = spectrum[..., indices].clone()


            scale_factors = torch.normal(mean=1.25, std=0.1, size=indices.shape).to(audio.device)
            spectrum[..., indices] *= scale_factors

            self.embedded_pattern = {
                "indices": indices,
                "original": original,
                "scale_factors": scale_factors
            }

            return torch.fft.irfft(spectrum, n=audio.size(-1), dim=-1)

        elif self.engine_type == 'SilentCipher':
            spectrum = torch.fft.rfft(audio, dim=-1)
            low_freq_range = 6
            perturbation = (1e-3 + torch.rand(low_freq_range, device=audio.device) * 1e-3)
            spectrum[..., :low_freq_range] += perturbation
            self.embedded_pattern = spectrum[..., :low_freq_range].clone()
            return torch.fft.irfft(spectrum, n=audio.size(-1), dim=-1)

    def detect(self, audio):
        if self.embedded_pattern is None:
            return False

        if self.engine_type == 'AudioSeal':
            min_len = min(audio.shape[-1], self.embedded_pattern.shape[-1])
            audio = audio[..., :min_len]
            pattern = self.embedded_pattern[..., :min_len]
            audio = audio + 0.01 * torch.randn_like(audio)
            correlation = F.cosine_similarity(audio.flatten(), pattern.flatten(), dim=0)
            return correlation.item() > (0.4 + 0.1 * torch.rand(1).item())

        elif self.engine_type == 'WavMark':
            if self.embedded_pattern is None:
                return False

            spectrum = torch.fft.rfft(audio, dim=-1)
            indices = self.embedded_pattern["indices"]
            original = self.embedded_pattern["original"]
            current = spectrum[..., indices]

            delta = torch.mean(torch.abs(torch.abs(current) - torch.abs(original)))

            threshold = 0.008 + 0.012 * torch.rand(1).item()

            return delta.item() > threshold

        elif self.engine_type == 'SilentCipher':
            spectrum = torch.fft.rfft(audio, dim=-1)
            low_freq = spectrum[..., :self.embedded_pattern.size(-1)]
            delta = torch.mean(torch.abs(low_freq - self.embedded_pattern))
            return delta.item() < (0.004 + 0.002 * torch.rand(1).item())
