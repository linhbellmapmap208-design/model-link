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
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, Response
from transkun.Data import writeMidi

logger = logging.getLogger(__name__)
inference_lock = threading.Lock()


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
    app.state.model = model
    logger.info('TransKun %s loaded on CPU', version('transkun'))
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
    return {'status': 'ready', 'model': 'TransKun', 'version': version('transkun'), 'device': 'cpu'}


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


@app.post('/transcribe', responses={200: {'content': {'audio/midi': {}}}})
async def transcribe(file: UploadFile = File(...)):
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
