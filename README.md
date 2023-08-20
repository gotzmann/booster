# llamazoo

LLaMAZoo - Platform for serving LLaMA GPT models at production scale.

# LLaMAZoo

![](./logo.png?raw=true)

## Motivation

While developing first version of **[llama.go](https://github.com/gotzmann/llama.go)** I was shocked on how fast original **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** project went.

So I've decided to start a new project where the high performance C++ / CUDA core will be embedded into Golang powered API server ready for inference at any scale within real production.

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

- [ ] GGUF model format support
- [ ] Full Windows support
- [ ] Server built and ready for download for all platforms 

## How to Run?

You shold go through three steps:

1) Build the server from sources

```shell
make clean && make
```

2) Download the model

```shell
wget https://huggingface.co/TheBloke/vicuna-13b-v1.3.0-GGML/resolve/main/vicuna-13b-v1.3.0.ggmlv3.q4_K_M.bin
```

3) Set config file

```shell
id: "server"
host: localhost
port: 8080
log: llamazoo.log
deadline: 180

pods: 

  -
    threads: 6
    gpus: [ 0 ]
    model: vicuna
    mode: 

models:

  -
    id: vicuna
    name: Vicuna v1.3
    path: /vicuna-13b-v1.3.0.ggmlv3.q4_K_M.bin
    gqa: 1
    preamble: "You are a virtual assistant. Please help user."
    prefix: "\nUSER: "
    suffix: "\nASSISTANT:"
    contextsize: 2048
    predict: 1024
    temp: 0.1
    mirostat: 2
    mirostattau: 0.1
    mirostateta: 0.1
```    

When all is done, go try it for yourself:

```shell
./llama --server --debug
```