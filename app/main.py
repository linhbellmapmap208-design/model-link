import asyncio
from contextlib import asynccontextmanager
from html import escape
from importlib import resources
from importlib.metadata import version
import io
import logging
import os
from pathlib import Path
import tempfile
import threading

import moduleconf
import numpy as np
from pydub import AudioSegment
import soxr
import torch
import torch._inductor.config as inductor_config
from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel
from transkun.Data import writeMidi
import yt_dlp

# Optimize torch inductor for CPU code generation
inductor_config.cpp_wrapper = True
inductor_config.coordinate_descent_tuning = True

logger = logging.getLogger(__name__)
inference_lock = threading.Lock()
API_KEY = os.getenv('TRANSKUN_API_KEY', '')


def require_api_key(x_api_key: str = Header(default='')):
    if not API_KEY:
        return  # no key configured — open access (dev mode)
    if x_api_key != API_KEY:
        raise HTTPException(401, 'API key không hợp lệ. Gửi header X-API-Key.')
    return x_api_key


@asynccontextmanager
async def lifespan(app):
    torch.set_num_threads(os.cpu_count() or 4)
    package = resources.files('transkun')
    conf_manager = moduleconf.parseFromFile(str(package / 'pretrained/2.0.conf'))
    model_class = conf_manager['Model'].module.TransKun
    model = model_class(conf=conf_manager['Model'].config).to('cpu')
    checkpoint = torch.load(str(package / 'pretrained/2.0.pt'), map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint.get('best_state_dict', checkpoint.get('state_dict')), strict=False)
    model.eval()
    # Disable gradient checkpointing (only needed for training, adds overhead for inference)
    model.backbone.useGradientCheckpoint = False
    # Compile the bottleneck function for optimized CPU code generation
    model.processFramesBatch = torch.compile(model.processFramesBatch, mode='max-autotune')
    # Warm up the compiled model with a tiny dummy input so the first real request is fast
    dummy = torch.from_numpy(np.zeros((model.fs, 1), dtype=np.float32))
    with torch.inference_mode():
        model.transcribe(dummy, discardSecondHalf=False)
    app.state.model = model
    logger.info('TransKun %s loaded on CPU (compiled, %d threads)', version('transkun'), torch.get_num_threads())
    yield


app = FastAPI(title='TransKun Piano to MIDI', lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/', response_class=HTMLResponse)
def home():
    suffix = os.getenv('BASE44_PUBLIC_HOST_SUFFIX', '')
    public_url = f'https://3000-{suffix}' if suffix else ''
    return Path(__file__).with_name('index.html').read_text().replace('__PUBLIC_URL__', escape(public_url, quote=True))


@app.get('/health')
def health():
    return {'status': 'ready', 'model': 'TransKun', 'version': version('transkun'), 'device': 'cpu', 'optimized': 'torch.compile + 4 cores'}


def transcribe_audio(data: bytes, suffix: str) -> bytes:
    if not inference_lock.acquire(blocking=False):
        raise HTTPException(429, 'Model đang bận. Hãy thử lại sau.')
    try:
        with tempfile.TemporaryDirectory(prefix='transkun-') as directory:
            audio_path = Path(directory) / ('input' + suffix)
            audio_path.write_bytes(data)
            try:
                audio = AudioSegment.from_file(audio_path).set_sample_width(2)
            except Exception as exc:
                raise HTTPException(422, 'Không đọc được file âm thanh.') from exc
            if len(audio) == 0 or len(audio) > 2400000:
                raise HTTPException(422, 'Âm thanh phải dài từ hơn 0 đến 40 phút.')
            samples = np.asarray(audio.get_array_of_samples(), dtype=np.float32).reshape(-1, audio.channels) / 32768.0
            model = app.state.model
            if audio.frame_rate != model.fs:
                samples = soxr.resample(samples, audio.frame_rate, model.fs)
            with torch.inference_mode():
                notes = model.transcribe(torch.from_numpy(samples), discardSecondHalf=False)
            output = io.BytesIO()
            writeMidi(notes).write(output)
            return output.getvalue()
    finally:
        inference_lock.release()


class URLRequest(BaseModel):
    url: str


def download_audio(url: str) -> tuple[bytes, str]:
    """Download audio from a URL (SoundCloud, etc.) using yt-dlp. Returns (data, suffix)."""
    with tempfile.TemporaryDirectory(prefix='ytdl-') as directory:
        outtmpl = str(Path(directory) / 'audio.%(ext)s')
        ydl_opts = {
            'format': 'bestaudio/best',
            'outtmpl': outtmpl,
            'noplaylist': True,
            'quiet': True,
            'no_warnings': True,
            'extractaudio': True,
            'audioformat': 'mp3',
        }
        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise HTTPException(422, f'Không tải được audio từ link: {exc}') from exc
        downloaded = Path(ydl.prepare_filename(info))
        if not downloaded.exists():
            # yt-dlp may change extension when extracting audio
            candidates = list(Path(directory).glob('audio.*'))
            if not candidates:
                raise HTTPException(500, 'Tải audio xong nhưng không tìm thấy file.')
            downloaded = candidates[0]
        data = downloaded.read_bytes()
        if len(data) > 200 * 1024 * 1024:
            raise HTTPException(413, 'Audio quá lớn, tối đa 200 MB.')
        return data, downloaded.suffix.lower()


@app.post('/transcribe-url', responses={200: {'content': {'audio/midi': {}}}})
async def transcribe_url(request: URLRequest, _api_key: str = Depends(require_api_key)):
    url = request.url.strip()
    if not url.startswith(('http://', 'https://')):
        raise HTTPException(422, 'Link không hợp lệ. Cần bắt đầu bằng http:// hoặc https://')
    try:
        data, suffix = await asyncio.to_thread(download_audio, url)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Audio download failed')
        raise HTTPException(500, 'Không tải được audio từ link.') from exc
    try:
        midi = await asyncio.to_thread(transcribe_audio, data, suffix)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Transcription failed')
        raise HTTPException(500, 'Không chuyển đổi được âm thanh. Kiểm tra log dịch vụ.') from exc
    return Response(midi, media_type='audio/midi', headers={'Content-Disposition': 'attachment; filename="transcription.mid"'})


@app.post('/transcribe', responses={200: {'content': {'audio/midi': {}}}})
async def transcribe(file: UploadFile = File(...), _api_key: str = Depends(require_api_key)):
    suffix = Path(file.filename or '').suffix.lower()
    if suffix not in {'.wav', '.mp3', '.flac', '.ogg', '.m4a'}:
        await file.close()
        raise HTTPException(422, 'Hỗ trợ WAV, MP3, FLAC, OGG hoặc M4A.')
    try:
        data = await file.read(200 * 1024 * 1024 + 1)
    finally:
        await file.close()
    if not data or len(data) > 200 * 1024 * 1024:
        raise HTTPException(413, 'File phải nhỏ hơn hoặc bằng 200 MB và không được rỗng.')
    try:
        midi = await asyncio.to_thread(transcribe_audio, data, suffix)
    except HTTPException:
        raise
    except Exception as exc:
        logger.exception('Transcription failed')
        raise HTTPException(500, 'Không chuyển đổi được âm thanh. Kiểm tra log dịch vụ.') from exc
    return Response(midi, media_type='audio/midi', headers={'Content-Disposition': 'attachment; filename="transcription.mid"'})
