"""Export the TransKun neural core to ONNX + freeze the log-mel front-end constants.

The Python pipeline is split in two:

  audio -> [front-end: framing + rfft + mel + log]  ->  [neural core: backbone + scorer] -> score

The front-end is pure DSP with **fixed** constants (Hann window, the 5 learned Gaussian
windows, the mel filterbank), so it moves to C. The neural core is exported to ONNX and
executed from C through the ONNX Runtime C API.

Every segment is padded to the same length before inference, so the frame axis is a fixed
constant (T) and the exported graph can keep a static shape.

Run inside the `web` container (needs torch + onnx + onnxruntime + onnxscript):

    docker compose -f docker-compose.base44.yml exec -T web \
        /opt/venv/bin/python tools/export_onnx_core.py

Writes into `c/artifacts/`:
    transkun_core.onnx        fp32 core (mel features in, score/score_skip/ctx out)
    frontend.bin              float32 constants: hann[4096], gauss[5*4096], mel[2049*229]
    core_meta.json            shapes / config needed by the C program
"""

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import moduleconf
from importlib import resources

WINDOW_SIZE = 4096
HOP_SIZE = 1024
FS = 44100
SEGMENT_SIZE = FS * 16  # one 16 s segment, the unit the app transcribes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ART = os.path.join(REPO, "c", "artifacts")


def load_model():
    pk = resources.files("transkun")
    cm = moduleconf.parseFromFile(str(pk / "pretrained/2.0.conf"))
    m = cm["Model"].module.TransKun(conf=cm["Model"].config).to("cpu").eval()
    ck = torch.load(str(pk / "pretrained/2.0.pt"), map_location="cpu", weights_only=False)
    m.load_state_dict(ck.get("best_state_dict", ck.get("state_dict")), strict=False)
    m.backbone.useGradientCheckpoint = False
    return m


class Core(torch.nn.Module):
    """backbone + interval scorer: mel features -> pairwise scores + per-frame context."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, feats):  # (B, T, n_mels, n_channels)
        ctx = self.m.backbone(
            feats, outputIndices=torch.tensor(self.m.targetMIDIPitch)
        )
        s, s_skip = self.m.scorer(ctx)
        return s.flatten(-2, -1), s_skip.flatten(-2, -1), ctx


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    m = load_model()

    t_frames = math.ceil(SEGMENT_SIZE / HOP_SIZE) + 1  # makeFrame(): ceil(len/hop)+1
    core = Core(m).eval()
    core_path = os.path.join(ART, "transkun_core.onnx")
    print(f"exporting core (T={t_frames}) ...")
    torch.onnx.export(
        core,
        (torch.randn(1, t_frames, 229, 6),),
        core_path,
        dynamo=True,
        input_names=["features"],
        output_names=["score", "score_skip", "ctx"],
    )
    print(f"  {core_path} ({os.path.getsize(core_path)/1e6:.1f} MB)")

    # --- front-end constants -------------------------------------------------
    spec = m.framewiseFeatureExtractor.spectrogramExtractor
    hann = spec.win.numpy().astype(np.float32)                              # (4096,)
    # winGen.get() is (windowSize, nExtraWins); Spectrum uses its transpose, so store
    # the (nExtraWins, windowSize) layout the C front-end reads.
    gauss = spec.winGen.get().detach().t().numpy().astype(np.float32)       # (5, 4096)
    # stored transposed (n_mels, n_freqBins) so the C inner loop walks memory in order
    mel = m.framewiseFeatureExtractor.freq2mels.numpy().astype(np.float32).T  # (229, 2049)
    front_path = os.path.join(ART, "frontend.bin")
    with open(front_path, "wb") as f:
        f.write(hann.tobytes())
        f.write(gauss.tobytes())
        f.write(mel.tobytes())
    print(f"  {front_path} ({os.path.getsize(front_path)/1e6:.1f} MB)")

    meta = {
        "nMels": 229,
        "nChannels": 6,
        "nFreqBins": WINDOW_SIZE // 2 + 1,
        "windowSize": WINDOW_SIZE,
        "hopSize": HOP_SIZE,
        "fs": FS,
        "segmentSamples": SEGMENT_SIZE,
        "tFrames": t_frames,
        "logEps": 1e-5,
    }
    with open(os.path.join(ART, "core_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    # --- check the exported graph reproduces PyTorch ------------------------
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.log_severity_level = 3
    sess = ort.InferenceSession(core_path, sess_options=so, providers=["CPUExecutionProvider"])
    print("  onnx io:", [(i.name, i.shape) for i in sess.get_inputs()],
          [(o.name, o.shape) for o in sess.get_outputs()])
    probe = torch.randn(1, t_frames, 229, 6)
    with torch.inference_mode():
        ref = core(probe)
    got = sess.run(None, {"features": probe.numpy()})
    for name, a, b in zip(["score", "score_skip", "ctx"], ref, got):
        d = np.abs(a.numpy() - b)
        print(f"  {name}: max_abs={d.max():.3e} rel={d.mean()/(np.abs(a.numpy()).mean()+1e-9):.2e}")


if __name__ == "__main__":
    sys.exit(main())
