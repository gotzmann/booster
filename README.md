![](./logo.jpg?raw=true)

Large Hadron Collider is the world's most powerful particle accelerator.

**Large Model Collider aims to be an simple and mighty LLM inference accelerator both for those who needs to scale GPTs within production environment or just experiment with models on its own.**

## Superpowers

- Built with performance and scaling in mind **thanks Golang and C++**
- **No more problems with Python** dependencies and broken compatibility
- **Most of modern CPUs are supported**: any Intel/AMD x64 platofrms, server and Mac ARM64
- GPUs supported as well: **Nvidia CUDA, Apple Metal, OpenCL** cards
- Expect good performance on CPUs and **fast as hell inference on beefy GPUs**
- Both regular FP16/FP32 models and quantised versions are supported - **4-bit really rocks!**
- **Popular LLM architectures** already there: **LLaMA**, Starcoder, Baichuan, Mistral, etc...
- Special bonus: **Janus Sampling** well suited for non English languages

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

- [ ] Rebrand project: LLaMAZoo => Large Model Collider
- [ ] Is it 30th of November, first birthday of ChatGPT? Celebrate ...
- [ ] ... then release the very first Collider version after half a year of honing it
- [ ] Freeze JSON / YAML config format for Native API
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

2) Download the model [ like Mistral 7B quantized to GGUF Q4KM format as an example ]

```shell
wget https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf
```

3) Create configuration file and place it to the same directory [ see config.sample.yaml ] 

```shell
id: "collider"
host: localhost
port: 8080
log: collider.log
deadline: 180
debug: full
swap:

pods: 

  -
    model: default
    threads: 6
    gpus: [ 0 ]
    batchsize: 512

models:

  -
    id: default
    name: Mistral
    path: mistral-7b-instruct-v0.1.Q4_K_M.gguf
    locale: en_US
    preamble: "You are a virtual assistant. Please answer the question."
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
./collider --server --debug
```

5) Now POST JSON with unique ID and your question to `localhost:8080/jobs`

```shell
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc6",
    "prompt": "Who are you?"
}
```

6) See instructions within `collider.service` file on how to create daemond service out of this API server.