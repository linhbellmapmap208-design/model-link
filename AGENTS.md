# Development findings

- The imported repo contained only `requirements 2.txt`, not application code. `app/` is a minimal FastAPI wrapper created to serve TransKun on port 3000.
- Run `docker compose -f docker-compose.base44.yml up -d`. The plain Python runtime bind-mounts this checkout; Uvicorn reloads Python edits. HTML is read on every request, so refresh the preview after HTML edits.
- First startup installs ffmpeg/sox and Python dependencies, including the 50 MB TransKun package with its bundled checkpoint. The named virtualenv and pip-cache volumes speed subsequent startups. CPU-only PyTorch and NumPy 1.x are intentional; setuptools is pinned for legacy dependencies.
- `requirements.base44.lock.txt` constrains all installed dependencies. When changing dependencies, regenerate the constraints from the tested container with `docker compose -f docker-compose.base44.yml exec -T web /opt/venv/bin/pip freeze` and rerun startup. Quote `requirements 2.txt` in pip includes because its filename contains a space.
- Startup loads the bundled TransKun 2.0.1 checkpoint on CPU before accepting requests. `/health` reports ready only after load; it is the Compose healthcheck. No external credentials are required.
- Verify `curl -fsS http://localhost:3000/health` and `curl -F 'file=@/tmp/piano.wav' http://localhost:3000/transcribe -o /tmp/result.mid`; successful MIDI starts with `MThd`. Synthetic 1-second audio was tested successfully. Do not infer transcription accuracy from that synthetic test.
- The demo accepts at most 200 MB and 40 minutes per audio, with one inference at a time. Uploads use temporary disk files deleted on completion; there is no durable file storage. API is unauthenticated, intended for development only. Add authentication and production hosting before wider use.
- `BASE44_PUBLIC_HOST_SUFFIX` is passed into the service to construct the copyable public preview URL without baking in an environment-specific host. This preview is not reliable 24/7 hosting.
