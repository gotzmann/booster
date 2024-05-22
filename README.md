![](./logo.jpg?raw=true)

**Booster**, according to Merriam-Webster dictionary:

- an auxiliary device for increasing force, power, pressure, or effectiveness
- the first stage of a multistage rocket providing thrust for the launching and the initial part of the flight

**Large Model Booster aims to be an simple and mighty LLM inference accelerator both for those who needs to scale GPTs within production environment or just experiment with models on its own.**

## Superpowers

- Built with performance and scaling in mind **thanks Golang and C++**
- **No more problems with Python** dependencies
- **All modern CPUs are supported**: any Intel or AMD x64 platofrms, server ARM64 and Apple Silicon
- GPUs supported as well: **Nvidia CUDA, Apple Metal, even OpenCL cards**
- Split really big models between a number of GPU (**warp LLaMA 70B with 2x RTX 3090**)
- Great performance on CPU only machines, **fast as hell inference on monsters with beefy GPUs**
- Both regular FP16/FP32 models and their quantised versions are supported - **4-bit really rocks!**
- **Popular LLM architectures** already there: **LLaMA**, Mistral, Gemma, etc...
- **Special bonus: SOTA Janus Sampling implementation** for code generation and non English languages

## Motivation

Within first month of **[llama.go](https://github.com/gotzmann/llama.go)** development I was literally shocked of how original **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** project made it very clear - there are no limits for talented people on bringing mind-blowing features and moving to AI future.

So I've decided to start a new project where best-in-class C++ / CUDA core will be embedded into mighty Golang server ready for robust and performant inference at large scale within real production environments.

## V3 Roadmap - Summer'24

- [x] Rebrand project again :) **Collider => Booster**
- [x] Complete LLaMA v3 support
- [x] OpenAI API Chat Completion compatible endpoints
- [ ] Ollama compatible endpoints
- [ ] Interactive mode for chatting from command line
- [ ] Broader integration with Ollama ecosystem
- [ ] Smarter context shrinking when reaching its limits chatting with model
- [ ] Embedded web UI with no external dependencies
- [ ] Allow native Windows support
- [ ] Prebuilt binaries for all platforms
- [ ] Support LLaVA multi-modal models inference
- [ ] Better code test coverage
- [ ] Perplexity computation useful for benchmarking

## V2 Roadmap - Spring'24

- [x] Full LLaMA v2 support
- [x] Freeze JSON / YAML config format for Native API

## V1 Roadmap - Winter'23

- [x] Rebrand project: LLaMAZoo => Large Model Collider
- [x] Is it 2023, 30th of November? First birthday of ChatGPT! **Celebrate ...**
- [x] **... then release Collider V1** after half a year of honing it :)

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

## How to build on Mac?

Booster was (and still) developed on Mac with Apple Silicon M1 processor, so it's really easy peasy:

```shell
make mac
```

## How to compile for CUDA on Ubuntu?

Follow step 1 and step 2, then just make!

Ubuntu Step 1: Install C++ and Golang compilers, as well some developer libraries

```
sudo apt update -y && sudo apt upgrade -y && \
apt install -y git git-lfs make build-essential && \
wget https://golang.org/dl/go1.21.5.linux-amd64.tar.gz && \
tar -xf go1.21.5.linux-amd64.tar.gz -C /usr/local && \
rm go1.21.5.linux-amd64.tar.gz && \
echo 'export PATH="${PATH}:/usr/local/go/bin"' >> ~/.bashrc && source ~/.bashrc
```

Ubuntu Step 2: Install Nvidia drivers and CUDA Toolkit 12.2 with NVCC

```
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin && \
sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/3bf863cc.pub && \
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/ /" && \
sudo apt update -y && \
sudo apt install -y cuda-toolkit-12-2
```

Now you are ready to rock!

```shell
make cuda
```

## How to Run?

You shold go through steps below:

1) Build the server from sources [ Mac inference as example ]

```shell
make clean && make mac
```

2) Download the model, like [ Hermes 2 Pro ] based on [ LLaMA-v3-8B ] quantized to GGUF Q4KM format:

```shell
wget https://huggingface.co/NousResearch/Hermes-2-Pro-Llama-3-8B-GGUF/resolve/main/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
```

3) Create configuration file and place it to the same directory [ see config.sample.yaml ] 

```shell
id: mac
host: localhost
port: 8080
log: booster.log
deadline: 180

pods:

  gpu:
    model: hermes
    prompt: chat
    sampling: janus
    threads: 1
    gpus: [ 100 ]
    batch: 512

models:

  hermes:
    name: Hermes2 Pro 8B
    path: ~/models/Hermes-2-Pro-Llama-3-8B-Q4_K_M.gguf
    context: 8192
    predict: 1024

prompts:

  chat:
    locale: en_US
    system: "<|im_start|>system\nToday is {DATE}. You are virtual assistant. Please answer the question.<|im_end|>"
    user: "\n<|im_start|>user\n{USER}<|im_end|>"
    assistant: "\n<|im_start|>assistant\n{ASSISTANT}<|im_end|>"

samplings:

  janus:
    janus: 1
    depth: 200
    scale: 0.97
    hi: 0.99
    lo: 0.96

  mirostat:
    mirostat: 0
    mirostatent: 3.0
    mirostatlr: 0.1

  basic:
    temperature: 0.8
    top_k: 8
    topp: 0.9
    typicalp: 1.0
    repetition_penalty: 1.1
    penaltylastn: 200
```    

4) When all is done, start the server with debug enabled to be sure it working

Launch Booster in interactive mode to just chatting with the model:

```shell
./booster
```

Launch Booster as server to handle all API endpoints and show debug info:

```shell
./booster --server --debug
```

5) Now use Booster with Ollama / OpenAI API or POST JSON to native Async API `http://localhost:8080/jobs`

```shell
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc6",
    "prompt": "Who are you?"
}
```


6) See results with native HTTP GET to native Async API `http://localhost:8080/jobs/5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc6`

```shell
{
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9f77",
    "output": "I'm a virtual assistant.",
    "prompt": "Who are you?",
    "status": "finished"
}
}
```

7) See instructions within `booster.service` file on how to create daemond service out of this API server.