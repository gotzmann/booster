# llamazoo

LLaMAZoo - platform for serving LLaMA (and other) GPT models at production scale with OpenAI API compatibility layer. Use any GPU with CUDA/OpenCL or x64/Apple/ARM CPU. Original LLM models and their quantised versions are supported.

# LLaMAZoo

![](./logo.png?raw=true)

## Motivation

While developing first version of **[llama.go](https://github.com/gotzmann/llama.go)** I was impressed by how fast original **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** project "moded fast and broke things".

Since then I've decided to start a new project where the high performant C++ / CUDA core will be embedded into Golang powered API server ready for inference at any scale within real production environments.

That's how LLaMAZoo was born.

## V0 Roadmap - Fall'23

- [x] Draft implementation with CGO llama.cpp backend
- [x] Simple REST API to allow text generation
- [x] Inference with Apple Silicon GPU using Metal framework
- [x] Parallel inference both with CPU and GPU
- [x] Support both AMD64  and ARM64 platforms
- [x] CUDA support and fast inference with Nvidia cards

## V1 Roadmap - Winter'23

- [x] Retain dialog history by Session ID parameter
- [x] Support moderm GGUF V3 model format
- [x] Inference for most popular LLM architectures
- [x] Janus Sampling for better non-English text generation
- [ ] Perplexity computation [ useful for benchmarking ]
- [ ] Support LLaVA multi-modal models inference
- [ ] OpenAI API endpoint specifications compatibile

## V2 Roadmap - Spring'24

- [ ] Full Windows support
- [ ] Better test coverage
- [ ] Prebuilt binaries for all platforms

## How to Run?

You shold go through steps below:

1) Build the server from sources [ pure CPU inference as example ]

```shell
make clean && make
```

2) Download the model [ Vicuna 13B v1.3 quantized for Q4KM format as example ]

```shell
wget https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GGML/resolve/main/vicuna-13b-v1.3.0.ggmlv3.q4_K_M.bin
```

3) Create configuration file and place it to the same directory [ see config.sample.yaml ] 

```shell
id: "LLaMAZoo"
host: localhost
port: 8080
log: llamazoo.log

pods: 

  -
    threads: 6
    gpus: [ 0 ]
    model: default

models:

  -
    id: default
    name: Mistral
    path: Mistral-7B-Instruct-v0.1-GGUF
    locale: en_US
    preamble: "You are a virtual assistant. Please help user."
    prefix: "\nUSER: "
    suffix: "\nASSISTANT:"
    contextsize: 2048
    predict: 1024
    temperature: 0.1
    top_k: 8
    top_p: 0.96
    repetition_penalty: 1.1
```    

4) When all is done, start the server with debug enabled to be sure it working

```shell
./llama --server --debug
```

5) Now POST JSON with unique ID and your question to `localhost:8080/jobs`

```shell
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc6",
    "prompt": "Who are you?"
}
```

6) See instructions within `llamazoo.service` file on how to create daemond service out of LLaMAZoo server.