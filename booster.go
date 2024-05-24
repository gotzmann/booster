package main

/*
#cgo linux    CFLAGS: -O3 -std=c17   -I.          -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin   CFLAGS: -O3 -std=c17   -I.          -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DHAVE_BUGGY_APPLE_LINKER
#cgo linux  CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -DGGML_USE_LLAMAFILE -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DHAVE_BUGGY_APPLE_LINKER
#cgo linux   LDFLAGS: cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o cpp/unicode.o cpp/unicode-data.o cpp/sgemm.o cpp/ggml-cuda.o cpp/ggml-cuda/acc.o cpp/ggml-cuda/arange.o cpp/ggml-cuda/argsort.o cpp/ggml-cuda/binbcast.o cpp/ggml-cuda/clamp.o cpp/ggml-cuda/concat.o cpp/ggml-cuda/convert.o cpp/ggml-cuda/cpy.o cpp/ggml-cuda/diagmask.o cpp/ggml-cuda/dmmv.o cpp/ggml-cuda/fattn-tile-f16.o cpp/ggml-cuda/fattn-tile-f32.o cpp/ggml-cuda/fattn-vec-f16.o cpp/ggml-cuda/fattn-vec-f32.o cpp/ggml-cuda/fattn.o cpp/ggml-cuda/getrows.o cpp/ggml-cuda/im2col.o cpp/ggml-cuda/mmq.o cpp/ggml-cuda/mmvq.o cpp/ggml-cuda/norm.o cpp/ggml-cuda/pad.o cpp/ggml-cuda/pool2d.o cpp/ggml-cuda/quantize.o cpp/ggml-cuda/rope.o cpp/ggml-cuda/scale.o cpp/ggml-cuda/softmax.o cpp/ggml-cuda/sumrows.o cpp/ggml-cuda/tsembd.o cpp/ggml-cuda/unary.o cpp/ggml-cuda/upscale.o                        -lstdc++ -lm -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib
#cgo darwin  LDFLAGS: cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o cpp/unicode.o cpp/unicode-data.o cpp/sgemm.o cpp/ggml-metal.o cpp/ggml-metal-embed.o -lstdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit
*/
import "C"
import "github.com/gotzmann/booster/pkg/booster"

func main() {
	booster.Run()
}
