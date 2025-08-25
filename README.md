# Embedding Security: Audio Watermarking Demo

This repository contains the demonstration version of my senior research project:
“Embedding Security: Audio Watermarking and the Fight Against Unauthorized Music Leaks” (Stetson University, Spring 2025).

Unlike the full research implementation, this demo is designed to be lightweight, portable, and accessible. It simulates the logic and results of real watermark embedding, detection, and adversarial testing without requiring access to the original trained models.


📖 Background:
Unauthorized leaks of unreleased music cause massive losses in revenue, control, and reputation for artists and labels.
Audio watermarking offers a solution by embedding imperceptible identifiers into music, making leaks traceable while preserving listening quality.

This demo illustrates how watermarking systems can:
- Embed imperceptible watermarks into audio.
- Detect whether a watermark is present.
- Test robustness under common perturbations (compression, noise, filtering).
- Simulate attacks (white-box, black-box, gray-box, no-box) and their impact on watermark resilience.

⚙️ Features:

Three-tier modular architecture inspired by the original application
- Embed Watermark
- Detect Watermark
- Settings & Help

Simulated algorithms (based on test results):
[AudioSeal](https://github.com/facebookresearch/audioseal)
- CNN-based time-domain watermarking (high imperceptibility, weaker under white-box).
[WavMark](https://github.com/wavmark/wavmark)
- Spectrogram-based redundancy (robust under compression).
[SilentCipher](https://github.com/sony/silentcipher)
- Psychoacoustic masking (strongest resilience, sub-20 Hz encoding).

Attack simulation modes:
- White-box (full knowledge)
- Black-box (API probing)
- Gray-box (partial knowledge)
- No-box (generic perturbations)

Result Visualization: Graphs of detection probability, robustness under noise, and adversarial impact.

🚀 Getting Started:

Dependencies listed in requirements.txt:
"pip install -r requirements.txt"

Running the Demo:
"python demo_app.py"

This will launch the demo interface with the following tabs:
- Embed Watermark: Select an audio file and simulate watermark embedding.
- Detect Watermark: Upload an audio file and check if a watermark is detected.
- Security Testing: Run attack simulations and view resilience results.


📊 Example Outputs:

- Detection probability graphs under adversarial settings.
- Resilience charts showing robustness of AudioSeal, WavMark, and SilentCipher.
- Simulated confusion matrix for false positive/negative rates.


🔬 Research Context:

This demo is a companion to my senior research project, where I benchmarked state-of-the-art watermarking algorithms under adversarial conditions:
- AudioSeal dropped to 30% detection under white-box attacks due to exposed weights
- WavMark maintained >90% resilience under compression & resampling
- SilentCipher consistently resisted adversarial perturbations by using psychoacoustic masking
- The full paper expands on methodology, architecture, and future directions such as algorithm rotation, blockchain timestamping, and adversarial training

📬 Contact:

- Name: William Holland
- Email: wilywoonka@duck.com
