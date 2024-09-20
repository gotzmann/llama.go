![](./assets/images/terminal.png?raw=true)

## FINALLY - GOOD NEWS!

I've started to work on reimplementation of the library here: **[FastTensors](https://github.com/gotzmann/fast)**

Please star it if you'd like to see GGML-compatible implementation in pure Go.

## Looking for LLM Debug and Inference with Golang?

Please check out my related project **[Booster](https://github.com/gotzmann/booster)**

## Motivation

We dream of a world where fellow ML hackers are grokking **REALLY BIG GPT** models in their homelabs without having GPU clusters consuming a shit tons of **$$$**.

The code of the project is based on the legendary **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** framework of Georgi Gerganov written in C++ with the same attitude to performance and elegance.

We hope using Golang instead of *soo-powerful* but *too-low-level* language will allow much greater adoption.

## V0 Roadmap

- [x] Tensor math in pure Golang
- [x] Implement LLaMA neural net architecture and model loading
- [x] Test with smaller LLaMA-7B model
- [x] Be sure Go inference works exactly same way as C++
- [x] Let Go shine! Enable multi-threading and messaging to boost performance

## V1 Roadmap - Spring'23

- [x] Cross-patform compatibility with Mac, Linux and Windows
- [x] Release first stable version for ML hackers - v1.0
- [x] Enable bigger LLaMA models: 13B, 30B, 65B - v1.1
- [x] ARM NEON support on Apple Silicon (modern Macs) and ARM servers - v1.2
- [x] Performance boost with x64 AVX2 support for Intel and AMD - v1.2
- [x] Better memory use and GC optimizations - v1.3
- [x] Introduce Server Mode (embedded REST API) for use in real projects - v1.4
- [x] Release converted models for free access over the Internet - v1.4

## V2 Roadmap - Winter'23

- [ ] Support LLaMA V2 7B / 13B models architecture
- [ ] Implement LLaMA V2 34B / 70B Qrouped Query Attention
- [ ] Support modern GGUF V3 model format
- [ ] INT8 quantization to allow x4 bigger models fit same memory
- [ ] Benchmark LLaMA.go against some mainstream Python / C++ frameworks
- [ ] Enable some popular models of LLaMA family: Vicuna, Alpaca, etc
- [ ] Speed-up AVX2 with memory aligned tensors
- [ ] Extensive logging for production monitoring
- [ ] Interactive mode for real-time chat with GPT
- [ ] Automatic CPU / GPU features detection
- [ ] Implement metrics for RAM and CPU usage
- [ ] Standalone GUI or web interface for better access to framework
- [ ] Support popular open models: Open Assistant, StableLM, BLOOM, Anthropic, etc.
- [ ] AVX512 support - yet another performance boost for AMD Epyc and Intel Sapphire Rapids
- [ ] Nvidia GPUs support (CUDA or Tensor Cores)

## V3 Roadmap - Spring'23

- [ ] Allow plugins and external APIs for complex projects
- [ ] Allow model training and fine-tuning
- [ ] Speed up execution on GPU cards and clusters
- [ ] FP16 and BF16 math if hardware support is there
- [ ] INT4 and GPTQ quantization 
- [ ] AMD Radeon GPUs support with OpenCL

## How to Run?

First, obtain and convert original LLaMA models on your own, or just download ready-to-rock ones:

**LLaMA-7B:** [llama-7b-fp32.bin](https://nogpu.com/llama-7b-fp32.bin)

**LLaMA-13B:** [llama-13b-fp32.bin](https://nogpu.com/llama-13b-fp32.bin)

Both models store FP32 weights, so you'll needs at least 32Gb of RAM (not VRAM or GPU RAM) for LLaMA-7B. Double to 64Gb for LLaMA-13B.

Next, build app binary from sources (see instructions below), or just download already built one:

**Windows:** [llama-go-v1.4.0.exe](./builds/llama-go-v1.4.0.exe)

**MacOS:** [llama-go-v1.4.0-macos](./builds/llama-go-v1.4.0-macos)

**Linux:** [llama-go-v1.4.0-linux](./builds/llama-go-v1.4.0-linux)

So now you have both executable and model, go try it for yourself:

```shell
llama-go-v1.4.0-macos \
    --model ~/models/llama-7b-fp32.bin \
    --prompt "Why Golang is so popular?" \
```

## Useful command line flags:

```shell
--prompt   Text prompt from user to feed the model input
--model    Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]
--server   Start in Server Mode acting as REST API endpoint
--host     Host to allow requests from in Server Mode [ localhost by default ]
--port     Port listen to in Server Mode [ 8080 by default ]
--pods     Maximum pods or units of parallel execution allowed in Server Mode [ 1 by default ]
--threads  Adjust to the number of CPU cores you want to use [ all cores by default ]
--context  Context size in tokens [ 1024 by default ]
--predict  Number of tokens to predict [ 512 by default ]
--temp     Model temperature hyper parameter [ 0.5 by default ]
--silent   Hide welcome logo and other output [ shown by default ]
--chat     Chat with user in interactive mode instead of compute over static prompt
--profile  Profe CPU performance while running and store results to cpu.pprof file
--avx      Enable x64 AVX2 optimizations for Intel and AMD machines
--neon     Enable ARM NEON optimizations for Apple Macs and ARM server
```

## Going Production

LLaMA.go embeds standalone HTTP server exposing REST API. To enable it, run app with special flags:

```shell
llama-go-v1.4.0-macos \
    --model ~/models/llama-7b-fp32.bin \
    --server \
    --host 127.0.0.1 \
    --port 8080 \
    --pods 4 \
    --threads 4
```

Depending on the model size, how many CPU cores available there, how many requests you want to process in parallel, how fast you'd like to get answers, choose **pods** and **threads** parameters wisely.

**Pods** is a number of inference instances that might run in parallel.

**Threads** parameter sets how many cores will be used for tensor math within a pod.

So for example if you have machine with 16 hardware cores capable running 32 hyper-threads in parallel, you might end up with something like that: 

```shell
--server --pods 4 --threads 8
```

When there is no free pod to handle arriving request, it will be placed into the waiting queue and started when some pod gets job finished.

# REST API examples

## Place new job

Send POST request (with Postman) to your server address with JSON containing unique UUID v4 and prompt:

```json
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc3",
    "prompt": "Why Golang is so popular?"
}
```

## Check job status

Send GET request (with Postman or browser) to URL like http://host:port/jobs/status/:id

```shell
GET http://localhost:8080/jobs/status/5fb8ebd0-e0c9-4759-8f7d-35590f6c9fcb
```

## Get the results

Send GET request (with Postman or browser) to  URL like http://host:port/jobs/:id

```shell
GET http://localhost:8080/jobs/5fb8ebd0-e0c9-4759-8f7d-35590f6c9fcb
```

# How to build

First, install **Golang** and **git** (you'll need to download installers in case of Windows). 

```shell
brew install git
brew install golang
```

Then clone the repo and enter the project folder:

```
git clone https://github.com/gotzmann/llama.go.git
cd llama.go
```

Some Go magic to install external dependencies:

```
go mod tidy
go mod vendor
```

Now we are ready to build the binary from the source code:

```shell
go build -o llama-go-v1.exe -ldflags "-s -w" main.go
```

## FAQ

**1) From where I might obtain original LLaMA models?**

Contact Meta directly or just look around for some torrent alternatives.

**2) How to convert original LLaMA files into supported format?** 

Place original PyTorch FP16 files into **models** directory, then convert with command:

```shell
python3 ./scripts/convert.py ~/models/LLaMA/7B/ 0
```
