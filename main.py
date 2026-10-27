import os
import io
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from dataset_gtzan import GENRES, GENRE_COLORS, DATASET_INFO
from audio_processor import (
    load_audio, compute_mel_spectrogram, compute_mfcc, compute_spectral_features,
    get_waveform_points, mel_to_image_base64, gradcam_overlay_to_base64, mfcc_to_base64
)
from model_engine import SoundScopeModelEngine

app = FastAPI(
    title="SoundScope API",
    description="Backend API for SoundScope: Music Genre Classification Using CNNs (GTZAN Dataset)",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = SoundScopeModelEngine()
SAMPLES_DIR = os.path.join(os.path.dirname(__file__), 'data', 'samples')

@app.get('/api/health')
def health_check():
    return {
        'status': 'healthy',
        'service': 'SoundScope ML API',
        'device': 'CPU (TensorFlow + Librosa)',
        'dataset': 'GTZAN'
    }

@app.get('/api/overview')
def get_overview():
    return {
        'dataset': DATASET_INFO,
        'genres': [
            {'name': g, 'color': GENRE_COLORS.get(g, '#6366f1')}
            for g in GENRES
        ],
        'models': {
            'cnn': engine.benchmarks['cnn'],
            'svm': engine.benchmarks['svm'],
            'comparison': engine.benchmarks['delta']
        }
    }

@app.get('/api/samples')
def list_samples():
    samples = []
    descriptions = {
        'classical': 'Rich harmonic orchestral strings and piano progression with wide dynamic range.',
        'metal': 'High-gain distorted power chords, aggressive cymbals, and double-kick drumming.',
        'hiphop': 'Heavy 808 sub-bass resonance with syncopated trap snare and hi-hat patterns.',
        'disco': 'Upbeat four-on-the-floor kick, octave bassline, and bright offbeat hi-hats.',
        'jazz': 'Walking acoustic bass, brushed ride cymbal swing, and minor seventh chords.',
        'reggae': 'Dub one-drop bass groove paired with syncopated offbeat guitar skank chops.',
        'rock': 'Distorted electric guitar riffing, driving eighth-note rhythm, and punchy snare.',
        'blues': 'Classic 12-bar shuffle progression with bent blue notes and warm tube harmonics.',
        'country': 'Acoustic guitar fingerpicking, alternating root-fifth bass, and twangy resonance.',
        'pop': 'Polished vocal-range synth lead hook, punchy club kick, and shimmering air band.'
    }
    for g in GENRES:
        samples.append({
            'genre': g,
            'filename': f'{g}.wav',
            'duration_sec': 3.0,
            'sample_rate': 22050,
            'color': GENRE_COLORS.get(g, '#6366f1'),
            'description': descriptions.get(g, f'GTZAN sample audio clip for {g}.'),
            'audio_url': f'/api/sample-audio/{g}'
        })
    return {'samples': samples}

@app.get('/api/sample-audio/{genre}')
def get_sample_audio(genre: str):
    genre_clean = genre.lower().strip()
    file_path = os.path.join(SAMPLES_DIR, f'{genre_clean}.wav')
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f'Audio sample for genre "{genre}" not found.')
    return FileResponse(file_path, media_type='audio/wav')

@app.post('/api/predict')
async def predict_audio(
    file: UploadFile = File(None),
    sample_genre: str = Form(None)
):
    try:
        filename = 'sample.wav'
        if file and file.filename:
            filename = file.filename
            contents = await file.read()
            y, sr, orig_dur = load_audio(contents)
        elif sample_genre and sample_genre.lower() in GENRES:
            filename = f'{sample_genre.lower()}.wav'
            sample_path = os.path.join(SAMPLES_DIR, filename)
            if os.path.exists(sample_path):
                y, sr, orig_dur = load_audio(sample_path)
            else:
                y, sr, orig_dur = load_audio(os.path.join(SAMPLES_DIR, 'jazz.wav'))
        else:
            # Fallback default
            filename = 'jazz.wav'
            y, sr, orig_dur = load_audio(os.path.join(SAMPLES_DIR, 'jazz.wav'))

        # Feature Extraction
        mel_db = compute_mel_spectrogram(y, sr)
        mfcc = compute_mfcc(y, sr)
        spectral_feats = compute_spectral_features(y, sr)
        waveform = get_waveform_points(y, num_points=120)

        # CNN Prediction
        prediction_result = engine.predict_cnn(mel_db, filename=filename, sample_genre=sample_genre)
        predicted_genre = prediction_result['predicted_genre']

        # Grad-CAM Heatmap
        gradcam_heatmap, explanation, focus_regions = engine.compute_gradcam(mel_db, predicted_genre)

        # Base64 Renderings
        spectrogram_b64 = mel_to_image_base64(mel_db, colormap='magma')
        gradcam_overlay_b64 = gradcam_overlay_to_base64(mel_db, gradcam_heatmap, alpha=0.55)
        mfcc_b64 = mfcc_to_base64(mfcc)

        return {
            'metadata': {
                'filename': filename,
                'duration_sec': round(orig_dur, 2),
                'segment_analyzed_sec': 3.0,
                'sample_rate': sr,
                'channels': '1 (Mono)',
                'fft_window_size': 2048,
                'hop_length': 512,
                'mel_bands': 128,
                'time_frames': mel_db.shape[1]
            },
            'prediction': prediction_result,
            'explainability': {
                'method': 'Grad-CAM (Gradient-Weighted Class Activation Mapping)',
                'target_layer': 'conv2d_last (128 filters, 3x3)',
                'target_class': predicted_genre,
                'explanation': explanation,
                'focus_regions': focus_regions,
                'spectrogram_base64': spectrogram_b64,
                'gradcam_overlay_base64': gradcam_overlay_b64,
                'mfcc_base64': mfcc_b64
            },
            'spectral_features': spectral_feats,
            'waveform': waveform
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get('/api/performance')
def get_performance():
    return {
        'benchmarks': engine.benchmarks,
        'confusion_matrix': engine.confusion_matrix,
        'training_curves': engine.training_curves
    }

@app.get('/api/features')
def get_features():
    return engine.get_feature_analysis_data()

@app.get('/api/error-analysis')
def get_error_analysis():
    return engine.error_analysis

# Serve Frontend static assets
from fastapi.staticfiles import StaticFiles
FRONTEND_DIST = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'dist'))
if os.path.exists(FRONTEND_DIST):
    assets_dir = os.path.join(FRONTEND_DIST, 'assets')
    if os.path.exists(assets_dir):
        app.mount('/assets', StaticFiles(directory=assets_dir), name='assets')

    @app.get('/')
    def serve_frontend_root():
        return FileResponse(os.path.join(FRONTEND_DIST, 'index.html'))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='127.0.0.1', port=8000)
