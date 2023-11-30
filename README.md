![](./collider.jpg?raw=true)

Large Hadron Collider is the world's most powerful particle accelerator.

**Large Model Collider aims to be an simple and mighty LLM inference accelerator both for those who needs to scale GPTs within production environment or just experiment with models on its own.**

## Superpowers

- Built with performance and scaling in mind thanks Golang and C++
- No more problems installing myriads of Python dependencies
- Most of modern CPUs are supported: Intel/AMD x64, server and Mac ARM64
- GPUs supported as well: Nvidia CUDA, Apple Metal, OpenCL cards
- Expect fast inference on CPUs and hell fast on beefy GPUs
- Both regular FP16/FP32 models and quantised versions are supported (4-bit rocks!)
- Popular LLM architectures already there: LLaMA, Starcoder, Baichuan, Mistral, etc...

## Motivation

While developing first version of **[llama.go](https://github.com/gotzmann/llama.go)** I was impressed by how **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** project showed there are no limits of how fast talented people could bring mega-features and move fast.

Since then I've decided to start a new project where high performant C++ / CUDA core will be embedded into Golang powered networking server ready for inference at any scale within real production environments.

## V0 Roadmap - Fall'23

- [x] Draft implementation with CGO llama.cpp backend
- [x] Simple REST API to allow text generation
- [x] Inference with Apple Silicon GPU using Metal framework
- [x] Parallel inference both with CPU and GPU
- [x] Support both AMD64  and ARM64 platforms
- [x] CUDA support and fast inference with Nvidia cards
- [x] Retain dialog history by Session ID parameter
- [x] Support moderm GGUF V3 model format
- [x] Inference for most popular LLM architectures
- [x] Janus Sampling for better non-English text generation

## V1 Roadmap - Winter'23

- [ ] Freeze JSON / YAML config format
- [ ] Rebrand project: LLaMAZoo => Large Model Collider
- [ ] Release V1 after half a year honing and testing
- [ ] Release OpenAI API compatible endpoints
- [ ] Perplexity computation [ useful for benchmarking ]
- [ ] Support LLaVA multi-modal models inference

## V2 Roadmap - Spring'24

- [ ] Full Windows support
- [ ] Prebuilt binaries for all platforms
- [ ] Better test coverage

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