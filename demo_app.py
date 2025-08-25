import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from pydub import AudioSegment
from attacks import AudioEffects
from watermark_engines import WatermarkEngine
import warnings
warnings.filterwarnings("ignore")

# Helper functions
def load_audio_file(filepath, target_sample_rate):
    audio = AudioSegment.from_file(filepath).set_channels(1).set_frame_rate(target_sample_rate)
    samples = np.array(audio.get_array_of_samples()).astype(np.float32) / 32768.0
    return torch.tensor(samples).view(1, 1, -1)

def save_tensor_to_wav(tensor, filepath, sample_rate):
    from pydub import AudioSegment
    tensor = tensor.squeeze().cpu().numpy()
    tensor = np.clip(tensor, -1, 1)
    audio_segment = AudioSegment(
        (tensor * 32768.0).astype(np.int16).tobytes(),
        frame_rate=sample_rate,
        sample_width=2,
        channels=1
    )
    audio_segment.export(filepath, format="wav")

def compute_snr(original, modified):
    original = original.squeeze().cpu().numpy()
    modified = modified.squeeze().cpu().numpy()
    min_len = min(len(original), len(modified))
    original, modified = original[:min_len], modified[:min_len]
    noise = original - modified
    snr = 10 * np.log10(np.sum(original**2) / (np.sum(noise**2) + 1e-10))
    return round(snr, 2)

# Algorithm-specific sample rates
algorithm_sample_rates = {
    'AudioSeal': 16000,
    'WavMark': 22050,
    'SilentCipher': 44100
}

# Attack functions
attack_functions = {
    'clean (control) (No-box)': lambda x, sr: x,
    'speed (No-box)': lambda x, sr: AudioEffects.speed(x, sample_rate=sr),
    'echo (No-box)': lambda x, sr: AudioEffects.echo(x, sample_rate=sr),
    'random_noise (Black-box)': lambda x, sr: AudioEffects.random_noise(x),
    'pink_noise (Black-box)': lambda x, sr: AudioEffects.pink_noise(x),
    'lowpass_filter (No-box)': lambda x, sr: AudioEffects.lowpass_filter(x, sample_rate=sr),
    'highpass_filter (No-box)': lambda x, sr: AudioEffects.highpass_filter(x, sample_rate=sr),
    'smooth (No-box)': lambda x, sr: AudioEffects.smooth(x),
    'shush (No-box)': lambda x, sr: AudioEffects.shush(x),
    'mp3_compression (Black-box)': lambda x, sr: AudioEffects.mp3_compression(x, sample_rate=sr),
    'bandstop_filter (Grey-box)': lambda x, sr: AudioEffects.bandstop_filter(x, sample_rate=sr),
    'adversarial_noise (White-box)': lambda x, sr: AudioEffects.adversarial_noise(x),
    "targeted_bandpass_noise (Grey-box)": AudioEffects.targeted_bandpass_noise,
    "frequency_nulling (White-box)": AudioEffects.frequency_nulling
}

algorithms = ['AudioSeal', 'WavMark', 'SilentCipher']
ablations = [0.005, 0.01, 0.02]

results = []

sample_files = [f"samples/{f}" for f in os.listdir("samples") if f.endswith(('.wav', '.mp3'))]

for algo_name in algorithms:
    sr = algorithm_sample_rates[algo_name]
    print(f"\n[Benchmarking {algo_name} @ {sr} Hz]")

    ablation_list = ablations

    for ablation_strength in ablation_list:

        for sample_file in sample_files:
            basename = os.path.splitext(os.path.basename(sample_file))[0]
            audio = load_audio_file(sample_file, sr)

            wm_engine = WatermarkEngine(algo_name)
            wm_audio = wm_engine.embed(audio, watermark='1010101010101010') if algo_name != 'AudioSeal' \
                        else wm_engine.embed(audio, watermark='1010101010101010', ablation_strength=ablation_strength)

            # Save watermarked audio
            os.makedirs("watermarked_outputs", exist_ok=True)
            save_tensor_to_wav(wm_audio, f"watermarked_outputs/{algo_name}_{basename}.wav", sample_rate=sr)

            snr_wm = compute_snr(audio, wm_audio)

            for attack_name, attack_fn in tqdm(attack_functions.items(), desc=f"{algo_name}-{basename}"):
                try:
                    attacked_audio = attack_fn(wm_audio.clone(), sr)
                    os.makedirs("attacked_outputs", exist_ok=True)
                    attack_safe = attack_name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
                    save_tensor_to_wav(attacked_audio, f"attacked_outputs/{algo_name}_{basename}_{attack_safe}.wav", sample_rate=sr)
                    snr_attacked = compute_snr(audio, attacked_audio)
                    detected = wm_engine.detect(attacked_audio)
                    results.append({
                        'Algorithm': algo_name,
                        'Strength': ablation_strength if algo_name == 'AudioSeal' else 'N/A',
                        'Attack': attack_name,
                        'Sample': basename,
                        'Detected': int(detected),
                        'SNR (Watermarked)': snr_wm,
                        'SNR (Attacked)': snr_attacked
                    })
                except Exception as e:
                    print(f"Error: {e}")
                    results.append({
                        'Algorithm': algo_name,
                        'Strength': ablation_strength if algo_name == 'AudioSeal' else 'N/A',
                        'Attack': attack_name,
                        'Sample': basename,
                        'Detected': 0,
                        'SNR (Watermarked)': snr_wm,
                        'SNR (Attacked)': 0
                    })

        # False Positive: unmarked input test
        for sample_file in sample_files:
            audio = load_audio_file(sample_file, sr)
            wm_engine = WatermarkEngine(algo_name)
            try:
                detected = wm_engine.detect(audio)
            except ValueError:
                detected = 0
            results.append({
                'Algorithm': algo_name,
                'Strength': ablation_strength if algo_name == 'AudioSeal' else 'N/A',
                'Attack': 'False Positive (Unmarked)',
                'Sample': os.path.basename(sample_file),
                'Detected': int(detected),
                'SNR (Watermarked)': 'N/A',
                'SNR (Attacked)': 'N/A'
            })

# Save and visualize
results_df = pd.DataFrame(results)
results_df.to_csv("benchmark_results.csv", index=False)

# Color map for threat types
attack_colors = {
    'No-box': 'skyblue',
    'Black-box': 'orange',
    'Grey-box': 'green',
    'White-box': 'red',
    'control': 'gray',
    'Unmarked': 'purple'
}

for algo in results_df['Algorithm'].unique():
    subset = results_df[results_df['Algorithm'] == algo]

    # Extract threat model and clean label
    subset['Threat Model'] = subset['Attack'].str.extract(r'\((.*?)\)')[0]
    subset['Attack Label'] = subset['Attack'].str.replace(r'\s*\(.*?\)$', '', regex=True)
    subset['Sort Key'] = subset['Threat Model'] + "_" + subset['Attack Label']

    # Group and compute detection average
    grouped = subset.groupby(['Threat Model', 'Attack Label'])['Detected'].mean().reset_index()
    grouped['Detected'] *= 100
    grouped['Combined Label'] = grouped.apply(lambda row: f"{row['Attack Label']} ({row['Threat Model']})", axis=1)
    grouped = grouped.sort_values(by=['Threat Model', 'Attack Label'])
    bar_colors = grouped['Threat Model'].map(attack_colors).fillna('gray').tolist()

    # Plotting
    plt.figure(figsize=(14, 6))
    bars = plt.bar(grouped['Combined Label'], grouped['Detected'], color=bar_colors, zorder=3)
    plt.ylabel("Detection Rate (%)")
    plt.title(f"Average Detection Rate per Attack - {algo}")
    plt.grid(axis='y', zorder=0)
    plt.ylim(0, 100)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(f"{algo}_detection_bar.png")
    plt.show()

print("Benchmark Complete. Results saved to benchmark_results.csv and bar charts generated.")