# Build the C runner for the TransKun core.
#
#   make            build ./transkun_core
#   make check      build and run against a wav, dumping outputs for comparison
#
# ONNX Runtime is taken from the Python package already installed in the container
# (`pip install onnxruntime`); point ORT_LIB at another build to override.

CC ?= gcc
CFLAGS ?= -O3 -march=native -funroll-loops -fopenmp -Ithird_party -Wall
ORT_VERSION ?= v1.31.0
ORT_LIB ?= /opt/venv/lib/python3.11/site-packages/onnxruntime/capi/libonnxruntime.so.1.31.0

all: transkun_core

ORT_HEADER_BASE = https://raw.githubusercontent.com/microsoft/onnxruntime/$(ORT_VERSION)/include/onnxruntime/core/session
ORT_HEADERS = onnxruntime_c_api.h onnxruntime_error_code.h onnxruntime_ep_c_api.h

third_party/onnxruntime_c_api.h:
	mkdir -p third_party
	@for h in $(ORT_HEADERS); do \
	   if command -v curl >/dev/null; then curl -fsSL -o third_party/$$h $(ORT_HEADER_BASE)/$$h; \
	   else python3 -c "import urllib.request,sys; urllib.request.urlretrieve(sys.argv[1], sys.argv[2])" $(ORT_HEADER_BASE)/$$h third_party/$$h; fi; \
	done

# The pip package ships libonnxruntime.so.1.31.0 with SONAME libonnxruntime.so.1, so the
# loader needs a matching name next to it; a local symlink keeps that out of the venv.
lib/libonnxruntime.so.1: $(ORT_LIB)
	mkdir -p lib
	ln -sf $(ORT_LIB) $@

transkun_core: transkun_core.c third_party/onnxruntime_c_api.h lib/libonnxruntime.so.1
	$(CC) $(CFLAGS) -o $@ transkun_core.c $(ORT_LIB) -fopenmp -lm -Wl,-rpath,$(CURDIR)/lib

clean:
	rm -f transkun_core

.PHONY: all clean
