import os
import json
import numpy as np
import scipy.io.wavfile as wavfile

GENRES = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

GENRE_COLORS = {
    'blues': '#3b82f6',
    'classical': '#8b5cf6',
    'country': '#d97706',
    'disco': '#ec4899',
    'hiphop': '#10b981',
    'jazz': '#6366f1',
    'metal': '#ef4444',
    'pop': '#f43f5e',
    'reggae': '#14b8a6',
    'rock': '#f97316'
}

DATASET_INFO = {
    'name': 'GTZAN Genre Collection',
    'creator': 'George Tzanetakis & Perry Cook (2002)',
    'total_tracks': 1000,
    'genres_count': 10,
    'tracks_per_genre': 100,
    'raw_duration_sec': 30,
    'segment_duration_sec': 3,
    'total_segments': 10000,
    'sample_rate': 22050,
    'bit_depth': '16-bit Mono PCM',
    'train_val_test_split': '70% / 15% / 15%',
    'cnn_input_shape': '(128, 130, 1)',
    'feature_representation': 'Log-Mel Spectrogram (dB)'
}

def load_gtzan_summary():
    path = os.path.join(os.path.dirname(__file__), 'data', 'gtzan_summary.json')
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}

def synthesize_genre_sample(genre: str, sr=22050, duration=3.0):
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    sig = np.zeros_like(t)

    if genre == 'classical':
        # Rich harmonic strings/piano chords (C major: C4, E4, G4, C5)
        notes = [261.63, 329.63, 392.00, 523.25]
        for note in notes:
            sig += 0.25 * np.sin(2 * np.pi * note * t)
            sig += 0.12 * np.sin(4 * np.pi * note * t)
            sig += 0.05 * np.sin(6 * np.pi * note * t)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 0.33 * t)
        sig *= envelope

    elif genre == 'metal':
        # High gain distorted power chord
        chord = 0.4 * np.sin(2 * np.pi * 82.4 * t) + 0.3 * np.sin(2 * np.pi * 123.5 * t) + 0.2 * np.sin(2 * np.pi * 164.8 * t)
        sig = np.tanh(4.5 * chord)
        kick = np.exp(-15 * (t % 0.2)) * np.sin(2 * np.pi * 60 * t)
        noise = np.random.normal(0, 0.08, len(t)) * (t % 0.1 < 0.04)
        sig = 0.6 * sig + 0.3 * kick + 0.15 * noise

    elif genre == 'hiphop':
        # 808 Sub-bass pulse
        sub = np.sin(2 * np.pi * (50 - 15 * (t % 0.75)) * t) * np.exp(-3 * (t % 0.75))
        snare_env = np.exp(-25 * (t % 0.705))
        snare = snare_env * (0.5 * np.sin(2 * np.pi * 180 * t) + 0.5 * np.random.normal(0, 0.2, len(t)))
        hihat = np.random.normal(0, 0.06, len(t)) * (np.sin(2 * np.pi * 5.6 * t) > 0.6)
        sig = 0.55 * sub + 0.3 * snare + 0.15 * hihat

    elif genre == 'disco':
        # Four-on-the-floor kick drum (124 BPM)
        kick = np.exp(-20 * (t % 0.484)) * np.sin(2 * np.pi * (110 - 60 * (t % 0.484)) * t)
        bass = 0.3 * np.sin(2 * np.pi * 110 * t) * (np.sin(2 * np.pi * 4.13 * t) > 0)
        offbeat_hat = np.random.normal(0, 0.08, len(t)) * (np.sin(2 * np.pi * 4.13 * t + np.pi) > 0.7)
        sig = 0.5 * kick + 0.35 * bass + 0.15 * offbeat_hat

    elif genre == 'jazz':
        # Walking acoustic bass + minor 7th harmonics
        bass_freqs = [98.0, 110.0, 123.47, 130.81]
        step = (t // 0.52).astype(int) % 4
        current_freq = np.array([bass_freqs[s] for s in step])
        bass = 0.4 * np.sin(2 * np.pi * current_freq * t) * np.exp(-2.5 * (t % 0.52))
        chord = 0.15 * np.sin(2 * np.pi * 293.66 * t) + 0.12 * np.sin(2 * np.pi * 349.23 * t) + 0.1 * np.sin(2 * np.pi * 440.0 * t)
        ride = np.random.normal(0, 0.04, len(t)) * (np.sin(2 * np.pi * 3.8 * t) > 0.3)
        sig = 0.45 * bass + 0.35 * chord + 0.2 * ride

    elif genre == 'reggae':
        # Offbeat guitar skank + one-drop bass
        skank = (np.random.normal(0, 0.05, len(t)) + 0.25 * np.sin(2 * np.pi * 330 * t)) * (np.sin(2 * np.pi * 2.5 * t - np.pi/2) > 0.8)
        bass = 0.45 * np.sin(2 * np.pi * 65.4 * t) * (np.sin(2 * np.pi * 1.25 * t) > 0)
        sig = 0.55 * bass + 0.35 * skank + 0.1 * np.random.normal(0, 0.02, len(t))

    elif genre == 'rock':
        # Overdrive rock riff + 8th-note drums
        riff_step = (t // 0.375).astype(int) % 4
        riff_freq = 110 * np.where(riff_step == 2, 1.18, 1.0)
        riff = np.tanh(2.5 * (0.4 * np.sin(2 * np.pi * riff_freq * t) + 0.2 * np.sin(4 * np.pi * riff_freq * t)))
        kick = np.exp(-18 * (t % 0.5)) * np.sin(2 * np.pi * 80 * t) * (np.sin(2 * np.pi * 2 * t) > 0)
        snare = np.exp(-16 * (t % 0.5)) * np.random.normal(0, 0.15, len(t)) * (np.sin(2 * np.pi * 2 * t) <= 0)
        sig = 0.5 * riff + 0.3 * kick + 0.2 * snare

    elif genre == 'blues':
        shuffle = np.tanh(1.8 * (0.35 * np.sin(2 * np.pi * 110 * t) + 0.25 * np.sin(2 * np.pi * 116.54 * t)))
        organ = 0.15 * np.sin(2 * np.pi * 220 * t) + 0.1 * np.sin(2 * np.pi * 330 * t)
        sig = 0.55 * shuffle + 0.25 * organ + 0.1 * np.random.normal(0, 0.03, len(t))

    elif genre == 'country':
        bass_note = np.where((t // 0.5).astype(int) % 2 == 0, 98.0, 146.83)
        bass = 0.35 * np.sin(2 * np.pi * bass_note * t) * np.exp(-3 * (t % 0.5))
        acoustic = 0.25 * (np.sin(2 * np.pi * 392 * t) + np.sin(2 * np.pi * 493.88 * t) + np.sin(2 * np.pi * 587.33 * t)) * np.exp(-5 * (t % 0.25))
        sig = 0.45 * bass + 0.45 * acoustic + 0.05 * np.random.normal(0, 0.02, len(t))

    else: # pop
        synth = 0.3 * np.sin(2 * np.pi * 440 * t) + 0.2 * np.sin(2 * np.pi * 554.37 * t) + 0.15 * np.sin(2 * np.pi * 659.25 * t)
        kick = np.exp(-22 * (t % 0.46)) * np.sin(2 * np.pi * (95 - 40 * (t % 0.46)) * t)
        shimmer = np.sin(2 * np.pi * 1760 * t) * 0.08
        sig = 0.45 * synth + 0.4 * kick + 0.15 * shimmer

    max_val = np.max(np.abs(sig))
    if max_val > 0:
        sig = (sig / max_val) * 0.95

    return sig.astype(np.float32)

def generate_and_save_samples():
    samples_dir = os.path.join(os.path.dirname(__file__), 'data', 'samples')
    os.makedirs(samples_dir, exist_ok=True)
    sr = 22050
    for genre in GENRES:
        file_path = os.path.join(samples_dir, f'{genre}.wav')
        sig = synthesize_genre_sample(genre, sr=sr, duration=3.0)
        int16_sig = (sig * 32767).astype(np.int16)
        wavfile.write(file_path, sr, int16_sig)
        print(f'Generated sample for {genre}: {file_path}')

if __name__ == '__main__':
    generate_and_save_samples()
    print('Samples generated successfully.')
