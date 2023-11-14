# llamazoo

LLaMAZoo - platform for serving LLaMA (and other) GPT models at production scale with OpenAI API compatibility layer. Use any GPU with CUDA/OpenCL or x64/Apple/ARM CPU. Original LLM models and their quantised versions are supported.

# LLaMAZoo

![](./logo.png?raw=true)

## Motivation

While developing first version of **[llama.go](https://github.com/gotzmann/llama.go)** I was impressed by how fast original **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** project "moded fast and broke things".

Since then I've decided to start a new project where the high performant C++ / CUDA core will be embedded into Golang powered API server ready for inference at any scale within real production environments.

That's how LLaMAZoo was born.

## V0 Roadmap

- [x] Draft implementation with llama.cpp emedded within Golang with CGO
- [x] Simple REST API to core llama.cpp inference
- [x] Inference by Apple Silicon GPU with Metal framework
- [x] Parallel inference both with CPU and GPU
- [x] Support of AMD and ARM platforms
- [x] CUDA support and fast inference with Nvidia cards
- [x] Let Go shine! Enable multi-threading and messaging to boost performance

## V1 Roadmap - Autumn'23

- [x] Dialog history with session ID parameter
- [x] GGUF model format support
- [x] Compatible with most popular LLM architectures
- [ ] Perplexity computation [ useful for models benchmark ]
- [ ] Support LLaVA multi-modal inference
- [ ] Janus Sampling for better non-English text generation
- [ ] OpenAI API specs compatibile

## V2 Roadmap - Winter'23

- [ ] Full Windows support
- [ ] Prebuilt executable binaries for Linux / Windows / MacOS

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

3) Set config file [ config.yaml as example ] 

```shell
id: "LLaMAZoo"
host: localhost
port: 8080
log: llamazoo.log
debug:

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
    temp: 0.1
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