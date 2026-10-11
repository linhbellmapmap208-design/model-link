/* TransKun core runner in C.
 *
 * Replaces the Python front-end (framing -> rfft -> mel -> log) with plain C and runs the
 * exported neural core through the ONNX Runtime C API.
 *
 * Usage: ./transkun_core <audio.wav> [artifacts_dir] [dump_prefix] [model.onnx]
 *   artifacts_dir   default "artifacts" (frontend.bin, core_meta.json, transkun_core.onnx)
 *   dump_prefix     when given, write <prefix>_features.bin and <prefix>_ctx.bin for
 *                   comparison against the Python reference
 *   model.onnx      alternative graph, e.g. transkun_core_int8.onnx
 *
 * Build with the Makefile in this directory.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "onnxruntime_c_api.h"

#define NFFT 4096
#define HOP 1024
#define NMELS 229
#define NCH 6
#define NFREQ 2049
#define NGAUSS 5
#define FS 44100
#define LOG_EPS 1e-5f
#define PAD_SECONDS 8  /* padTimeBegin = segmentSize - step = 16 - 8 */

static const OrtApi *g_ort = NULL;

#define CHECK(expr)                                                            \
    do {                                                                       \
        OrtStatus *st_ = (expr);                                                \
        if (st_) {                                                              \
            fprintf(stderr, "ORT error: %s\n", g_ort->GetErrorMessage(st_));    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

static double now_sec(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

/* ---------------------------------------------------------------- FFT ---- */

static int *g_bitrev = NULL;
static float *g_tw_re = NULL, *g_tw_im = NULL;

static void fft_init(int n) {
    int logn = 0;
    while ((1 << logn) < n) logn++;
    g_bitrev = malloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        int r = 0;
        for (int j = 0; j < logn; j++)
            if (i & (1 << j)) r |= 1 << (logn - 1 - j);
        g_bitrev[i] = r;
    }
    g_tw_re = malloc(sizeof(float) * (n / 2));
    g_tw_im = malloc(sizeof(float) * (n / 2));
    for (int k = 0; k < n / 2; k++) {
        double a = -2.0 * M_PI * k / n;
        g_tw_re[k] = (float)cos(a);
        g_tw_im[k] = (float)sin(a);
    }
}

static void fft(int n, float *re, float *im) {
    for (int i = 0; i < n; i++) {
        int j = g_bitrev[i];
        if (j > i) {
            float tr = re[i]; re[i] = re[j]; re[j] = tr;
            float ti = im[i]; im[i] = im[j]; im[j] = ti;
        }
    }
    for (int len = 2; len <= n; len <<= 1) {
        int half = len >> 1, step = n / len;
        for (int i = 0; i < n; i += len) {
            for (int k = 0; k < half; k++) {
                float wr = g_tw_re[k * step], wi = g_tw_im[k * step];
                float xr = re[i + k + half], xi = im[i + k + half];
                float tr = wr * xr - wi * xi;
                float ti = wr * xi + wi * xr;
                re[i + k + half] = re[i + k] - tr;
                im[i + k + half] = im[i + k] - ti;
                re[i + k] += tr;
                im[i + k] += ti;
            }
        }
    }
}

/* ---------------------------------------------------------------- WAV ---- */

typedef struct {
    float *samples;
    long n;
    int rate;
    int channels;
} Audio;

static int read_wav(const char *path, Audio *a) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); return -1; }
    char id[4];
    unsigned int sz;
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "RIFF", 4)) { fclose(f); return -2; }
    fseek(f, 4, SEEK_CUR);
    if (fread(id, 1, 4, f) != 4 || memcmp(id, "WAVE", 4)) { fclose(f); return -2; }

    unsigned short fmt = 0, channels = 0, bits = 0;
    unsigned int rate = 0;
    long data_off = -1;
    unsigned int data_len = 0;

    while (fread(id, 1, 4, f) == 4) {
        if (fread(&sz, 4, 1, f) != 1) break;
        if (!memcmp(id, "fmt ", 4)) {
            fread(&fmt, 2, 1, f);
            fread(&channels, 2, 1, f);
            fread(&rate, 4, 1, f);
            fseek(f, 6, SEEK_CUR);
            fread(&bits, 2, 1, f);
            if (sz > 16) fseek(f, (long)sz - 16, SEEK_CUR);
        } else if (!memcmp(id, "data", 4)) {
            data_off = ftell(f);
            data_len = sz;
            fseek(f, (long)sz, SEEK_CUR);
        } else {
            fseek(f, (long)sz + (sz & 1), SEEK_CUR);
        }
    }
    if (data_off < 0 || (fmt != 1 && fmt != 3)) {
        fprintf(stderr, "unsupported wav (fmt=%u, data_off=%ld)\n", fmt, data_off);
        fclose(f);
        return -3;
    }

    int bytes = bits / 8;
    long frames = data_len / (bytes * channels);
    a->samples = malloc(sizeof(float) * frames);
    a->channels = channels;
    a->rate = (int)rate;
    a->n = frames;

    fseek(f, data_off, SEEK_SET);
    unsigned char *raw = malloc(data_len);
    size_t got = fread(raw, 1, data_len, f);
    (void)got;
    for (long i = 0; i < frames; i++) {
        double acc = 0;
        for (int c = 0; c < channels; c++) {
            long off = (i * channels + c) * bytes;
            if (fmt == 3) {
                float v;
                memcpy(&v, raw + off, 4);
                acc += v;
            } else if (bits == 16) {
                short v;
                memcpy(&v, raw + off, 2);
                acc += v / 32768.0;
            } else if (bits == 32) {
                int v;
                memcpy(&v, raw + off, 4);
                acc += v / 2147483648.0;
            } else if (bits == 8) {
                acc += (raw[off] - 128) / 128.0;
            }
        }
        a->samples[i] = (float)(acc / channels);
    }
    free(raw);
    fclose(f);
    return 0;
}

/* --------------------------------------------------------------- main ---- */

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <audio.wav> [artifacts_dir] [dump_prefix] [model.onnx]\n", argv[0]);
        return 2;
    }
    const char *wav_path = argv[1];
    const char *art = argc > 2 ? argv[2] : "artifacts";
    const char *dump = argc > 3 ? argv[3] : NULL;
    const char *model_name = argc > 4 ? argv[4] : "transkun_core.onnx";

    char path[4096];

    /* window constants */
    float *hann = malloc(sizeof(float) * NFFT);
    float *gauss = malloc(sizeof(float) * NGAUSS * NFFT);
    float *melmat = malloc(sizeof(float) * NFREQ * NMELS);
    snprintf(path, sizeof(path), "%s/frontend.bin", art);
    FILE *fp = fopen(path, "rb");
    if (!fp) { fprintf(stderr, "cannot open %s\n", path); return 1; }
    if (fread(hann, sizeof(float), NFFT, fp) != NFFT ||
        fread(gauss, sizeof(float), NGAUSS * NFFT, fp) != NGAUSS * NFFT ||
        fread(melmat, sizeof(float), NFREQ * NMELS, fp) != (size_t)NFREQ * NMELS) {
        fprintf(stderr, "frontend.bin has unexpected size\n");
        return 1;
    }
    fclose(fp);

    float *wins = malloc(sizeof(float) * NCH * NFFT);
    memcpy(wins, hann, sizeof(float) * NFFT);
    memcpy(wins + NFFT, gauss, sizeof(float) * NGAUSS * NFFT);

    /* audio -> padded signal -> first segment */
    Audio a;
    if (read_wav(wav_path, &a)) return 1;
    if (a.rate != FS) fprintf(stderr, "warning: sample rate %d, model expects %d\n", a.rate, FS);

    long pad = (long)PAD_SECONDS * FS;
    long seg = 16L * FS;
    long slice_len = seg;
    float *slice = calloc(slice_len, sizeof(float));
    long from_audio = slice_len - pad;           /* left pad is silence */
    if (from_audio > a.n) from_audio = a.n;
    memcpy(slice + pad, a.samples, sizeof(float) * from_audio);

    int T = (int)ceil((double)slice_len / HOP) + 1;   /* makeFrame(): ceil(len/hop)+1 */
    long total = (long)(T - 1) * HOP + NFFT;
    long left = NFFT / 2;
    float *frames = calloc(total, sizeof(float));
    memcpy(frames + left, slice, sizeof(float) * slice_len);
    fprintf(stderr, "frames: T=%d (padded %ld samples)\n", T, total);

    double t0 = now_sec();

    /* gain normalization over the frame matrix, as processFramesBatch does */
    double sum = 0, sum2 = 0;
    const long cnt = (long)T * NFFT;
#pragma omp parallel for reduction(+ : sum, sum2) schedule(static)
    for (int t = 0; t < T; t++)
        for (int i = 0; i < NFFT; i++) {
            float v = frames[(long)t * HOP + i];
            sum += v;
            sum2 += (double)v * v;
        }
    double mean = sum / cnt;
    double var = (sum2 - cnt * mean * mean) / (cnt - 1);
    float std = (float)sqrt(var);
    for (long i = 0; i < total; i++) frames[i] = (frames[i] - (float)mean) / (std + 1e-8f);

    /* front-end: window -> rfft -> power -> mel -> log */
    float *feats = malloc(sizeof(float) * (size_t)T * NMELS * NCH);
    fft_init(NFFT);

#pragma omp parallel for schedule(static)
    for (int t = 0; t < T; t++) {
        float re[NFFT], im[NFFT], pow_[NFREQ];
        for (int c = 0; c < NCH; c++) {
            const float *w = wins + (size_t)c * NFFT;
            for (int i = 0; i < NFFT; i++) {
                re[i] = frames[(long)t * HOP + i] * w[i];
                im[i] = 0.0f;
            }
            fft(NFFT, re, im);
            for (int k = 0; k < NFREQ; k++)
                pow_[k] = (re[k] * re[k] + im[k] * im[k]) / (float)NFFT;  /* rfft norm="ortho" */
            for (int m = 0; m < NMELS; m++) {
                const float *row = melmat + (size_t)m * NFREQ;  /* melmat is (n_mels, n_freqBins) */
                float acc = 0;
                for (int k = 0; k < NFREQ; k++) acc += pow_[k] * row[k];
                float v = (logf(acc + LOG_EPS) - logf(LOG_EPS)) / (-logf(LOG_EPS));
                feats[((size_t)t * NMELS + m) * NCH + c] = v;
            }
        }
    }
    double t_front = now_sec() - t0;
    fprintf(stderr, "front-end (C): %.3f s\n", t_front);

    if (dump) {
        snprintf(path, sizeof(path), "%s_features.bin", dump);
        fp = fopen(path, "wb");
        fwrite(feats, sizeof(float), (size_t)T * NMELS * NCH, fp);
        fclose(fp);
    }

    /* ------------------------------------------------------- ONNX core --- */
    g_ort = OrtGetApiBase()->GetApi(ORT_API_VERSION);
    OrtEnv *env;
    OrtSessionOptions *so;
    OrtSession *sess;
    OrtMemoryInfo *mem;
    CHECK(g_ort->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "transkun_c", &env));
    CHECK(g_ort->CreateSessionOptions(&so));
    CHECK(g_ort->SetIntraOpNumThreads(so, 4));
    CHECK(g_ort->SetSessionGraphOptimizationLevel(so, ORT_ENABLE_ALL));
    snprintf(path, sizeof(path), "%s/%s", art, model_name);
    double t1 = now_sec();
    CHECK(g_ort->CreateSession(env, path, so, &sess));
    fprintf(stderr, "session load: %.3f s (%s)\n", now_sec() - t1, path);

    CHECK(g_ort->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &mem));
    const int64_t dims[4] = {1, T, NMELS, NCH};
    size_t n_in = (size_t)T * NMELS * NCH;
    OrtValue *in_val = NULL;
    CHECK(g_ort->CreateTensorWithDataAsOrtValue(
        mem, feats, sizeof(float) * n_in, dims, 4,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &in_val));

    const char *in_names[] = {"features"};
    const char *out_names[] = {"score", "score_skip", "ctx"};
    OrtValue *out_vals[3] = {NULL, NULL, NULL};

    double t2 = now_sec();
    const OrtValue *in_vals[1] = {in_val};
    CHECK(g_ort->Run(sess, NULL, in_names, in_vals, 1, out_names, 3, out_vals));
    double t_infer = now_sec() - t2;
    fprintf(stderr, "core inference (ONNX Runtime): %.3f s\n", t_infer);

    float *ctx = NULL;
    for (int k = 0; k < 3; k++) {
        OrtTensorTypeAndShapeInfo *info = NULL;
        CHECK(g_ort->GetTensorTypeAndShape(out_vals[k], &info));
        size_t nd = 0;
        CHECK(g_ort->GetDimensionsCount(info, &nd));
        int64_t d[8];
        CHECK(g_ort->GetDimensions(info, d, nd));
        fprintf(stderr, "  out %-10s shape", out_names[k]);
        for (size_t i = 0; i < nd; i++) fprintf(stderr, " %lld", (long long)d[i]);
        fprintf(stderr, "\n");
        g_ort->ReleaseTensorTypeAndShapeInfo(info);
    }
    CHECK(g_ort->GetTensorMutableData(out_vals[2], (void **)&ctx));

    if (dump) {
        snprintf(path, sizeof(path), "%s_ctx.bin", dump);
        fp = fopen(path, "wb");
        fwrite(ctx, sizeof(float), (size_t)T * 90 * 256, fp);
        fclose(fp);
    }

    fprintf(stderr, "total: %.3f s\n", now_sec() - t0);
    return 0;
}
