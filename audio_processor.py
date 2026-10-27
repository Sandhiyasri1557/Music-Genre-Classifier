import io
import base64
import numpy as np
import librosa
import soundfile as sf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image

TARGET_SR = 22050
DURATION = 3.0
TARGET_SAMPLES = int(TARGET_SR * DURATION) # 66,150 samples

def load_audio(file_bytes_or_path, target_sr=TARGET_SR, duration=DURATION):
    """
    Loads audio from either file path or raw bytes, converts to mono,
    resamples to target_sr, and crops/pads to exact duration.
    """
    if isinstance(file_bytes_or_path, (bytes, bytearray)):
        buffer = io.BytesIO(file_bytes_or_path)
        try:
            y, sr = sf.read(buffer)
        except Exception:
            buffer.seek(0)
            y, sr = librosa.load(buffer, sr=target_sr, mono=True)
    else:
        try:
            y, sr = sf.read(file_bytes_or_path)
        except Exception:
            y, sr = librosa.load(file_bytes_or_path, sr=target_sr, mono=True)

    # Convert to mono if stereo
    if len(y.shape) > 1:
        y = np.mean(y, axis=1)

    # Resample if needed
    if sr != target_sr:
        y = librosa.resample(y, orig_sr=sr, target_sr=target_sr)
        sr = target_sr

    total_duration = len(y) / sr

    # Crop or pad to 3 seconds
    if len(y) < TARGET_SAMPLES:
        y = np.pad(y, (0, TARGET_SAMPLES - len(y)), mode='constant')
    else:
        y = y[:TARGET_SAMPLES]

    # Ensure float32
    y = y.astype(np.float32)
    max_val = np.max(np.abs(y))
    if max_val > 0:
        y = y / max_val

    return y, sr, total_duration

def compute_mel_spectrogram(y, sr=TARGET_SR, n_mels=128, n_fft=2048, hop_length=512):
    """
    Extracts 128-band Mel-Spectrogram in dB scale.
    Shape: (128, 130)
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels, fmin=20, fmax=sr//2
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    # Ensure exact width of 130 frames
    if mel_db.shape[1] < 130:
        mel_db = np.pad(mel_db, ((0, 0), (0, 130 - mel_db.shape[1])), mode='edge')
    elif mel_db.shape[1] > 130:
        mel_db = mel_db[:, :130]
    return mel_db

def compute_mfcc(y, sr=TARGET_SR, n_mfcc=20):
    """
    Extracts 20 MFCCs across time frames.
    Shape: (20, 130)
    """
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=2048, hop_length=512)
    if mfcc.shape[1] < 130:
        mfcc = np.pad(mfcc, ((0, 0), (0, 130 - mfcc.shape[1])), mode='edge')
    elif mfcc.shape[1] > 130:
        mfcc = mfcc[:, :130]
    return mfcc

def compute_spectral_features(y, sr=TARGET_SR):
    """
    Extracts high-level spectral and temporal descriptors.
    """
    try:
        centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
        bandwidth = float(np.mean(librosa.feature.spectral_bandwidth(y=y, sr=sr)))
        rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)))
        zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
        rms = float(np.mean(librosa.feature.rms(y=y)))
        tempo_arr = librosa.feature.rhythm.tempo(y=y, sr=sr)
        tempo = float(tempo_arr[0]) if len(tempo_arr) > 0 else 120.0
    except Exception:
        centroid, bandwidth, rolloff, zcr, rms, tempo = 2200.0, 2100.0, 4500.0, 0.08, 0.1, 120.0

    return {
        'spectral_centroid': round(centroid, 1),
        'spectral_bandwidth': round(bandwidth, 1),
        'spectral_rolloff': round(rolloff, 1),
        'zero_crossing_rate': round(zcr, 4),
        'rms_energy': round(rms, 4),
        'tempo_bpm': round(tempo, 1)
    }

def get_waveform_points(y, num_points=120):
    """
    Downsamples audio to num_points min/max envelope points for SVG rendering.
    """
    chunk_size = len(y) // num_points
    if chunk_size == 0:
        return [0.0] * num_points
    peaks = []
    for i in range(num_points):
        chunk = y[i * chunk_size:(i + 1) * chunk_size]
        peak = float(np.max(np.abs(chunk))) if len(chunk) > 0 else 0.0
        peaks.append(round(peak, 3))
    return peaks

def mel_to_image_base64(mel_db, colormap='magma'):
    """
    Converts 2D Mel-Spectrogram (dB) to base64 PNG data URL.
    """
    # Normalize to [0, 1]
    norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    cmap = matplotlib.colormaps.get_cmap(colormap)
    rgba = cmap(norm) # (128, 130, 4)
    # Flip vertically so low frequencies are at bottom
    rgba = np.flipud(rgba)
    uint8_img = (rgba[:, :, :3] * 255).astype(np.uint8)
    
    img = Image.fromarray(uint8_img)
    img = img.resize((480, 240), Image.Resampling.BILINEAR)
    
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{b64_str}'

def gradcam_overlay_to_base64(mel_db, gradcam_heatmap, alpha=0.55):
    """
    Overlays Grad-CAM attention heatmap onto Mel-Spectrogram and outputs base64 PNG.
    """
    # Spectrogram base in magma
    norm_spec = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    spec_cmap = matplotlib.colormaps.get_cmap('magma')
    spec_rgb = spec_cmap(norm_spec)[:, :, :3]

    # Resize heatmap to match spectrogram dimensions (128, 130)
    heat_img = Image.fromarray((gradcam_heatmap * 255).astype(np.uint8))
    heat_resized = np.array(heat_img.resize((130, 128), Image.Resampling.BILINEAR)) / 255.0

    # Heatmap color in jet/turbo
    heat_cmap = matplotlib.colormaps.get_cmap('turbo')
    heat_rgb = heat_cmap(heat_resized)[:, :, :3]

    # Weighted blend where heatmap has high activation
    mask = heat_resized[:, :, np.newaxis]
    blended = (1.0 - alpha * mask) * spec_rgb + (alpha * mask) * heat_rgb
    blended = np.clip(blended, 0.0, 1.0)
    
    # Flip vertically for correct audio orientation (low Hz at bottom)
    blended = np.flipud(blended)
    uint8_img = (blended * 255).astype(np.uint8)

    img = Image.fromarray(uint8_img)
    img = img.resize((480, 240), Image.Resampling.BILINEAR)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{b64_str}'

def mfcc_to_base64(mfcc):
    """
    Converts 20 MFCCs into a base64 PNG heatmap.
    """
    norm = (mfcc - mfcc.min()) / (mfcc.max() - mfcc.min() + 1e-8)
    cmap = matplotlib.colormaps.get_cmap('viridis')
    rgba = cmap(norm)
    rgba = np.flipud(rgba)
    uint8_img = (rgba[:, :, :3] * 255).astype(np.uint8)

    img = Image.fromarray(uint8_img)
    img = img.resize((480, 160), Image.Resampling.NEAREST)

    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    b64_str = base64.b64encode(buf.read()).decode('utf-8')
    return f'data:image/png;base64,{b64_str}'

if __name__ == '__main__':
    y, sr, dur = load_audio('data/samples/classical.wav')
    mel = compute_mel_spectrogram(y, sr)
    mfcc = compute_mfcc(y, sr)
    feats = compute_spectral_features(y, sr)
    spec_b64 = mel_to_image_base64(mel)
    print(f'Spectrogram rendered, b64 len: {len(spec_b64)}')
    print('Spectral features:', feats)
