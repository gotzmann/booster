package main

/*
#cgo linux  CFLAGS:   -O3 -std=c17   -I.          -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin CFLAGS:   -O3 -std=c17   -I.          -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DHAVE_BUGGY_APPLE_LINKER
#cgo linux  CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -march=native -mtune=native -DNDEBUG -D_XOPEN_SOURCE=600 -D_GNU_SOURCE      -DGGML_USE_CUDA  -DGGML_CUDA_USE_GRAPHS           -DLOG_DISABLE_LOGS  -I/usr/local/cuda/include -I/opt/cuda/include -I/usr/local/cuda/targets/x86_64-linux/include
#cgo darwin CXXFLAGS: -O3 -std=c++17 -I. -Icommon -fPIC -pthread -mcpu=native                -DNDEBUG -D_XOPEN_SOURCE=600 -D_DARWIN_C_SOURCE -DGGML_USE_METAL -DGGML_LLAMA_METAL_EMBED_LIBRARY -DGGML_METAL_NDEBUG -DGGML_USE_ACCELERATE -DACCELERATE_NEW_LAPACK -DACCELERATE_LAPACK_ILP64 -DHAVE_BUGGY_APPLE_LINKER
#cgo linux  LDFLAGS:  cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o cpp/unicode.o cpp/unicode-data.o cpp/sgemm.o cpp/ggml-cuda.o                         -lstdc++ -lm -lcuda -lcublas -lculibos -lcudart -lcublasLt -lpthread -ldl -lrt -L/usr/local/cuda/lib64 -L/opt/cuda/lib64 -L/usr/local/cuda/targets/x86_64-linux/lib
#cgo darwin LDFLAGS:  cpp/llama.o cpp/bridge.o cpp/janus.o cpp/ggml.o cpp/ggml-backend.o cpp/ggml-alloc.o cpp/ggml-quants.o cpp/unicode.o cpp/unicode-data.o cpp/sgemm.o cpp/ggml-metal.o cpp/ggml-metal-embed.o -lstdc++ -framework Accelerate -framework Foundation -framework Metal -framework MetalKit
*/
import "C"
import "github.com/gotzmann/booster/pkg/booster"

func main() {
	booster.Run()
}
