# LLaMA.go

![](./assets/images/terminal.png?raw=true)

## Motivation

We dream of a world where fellow ML hackers are grokking with **REALLY BIG GPT** models in their homelabs without having GPU clusters consuming a shit tons of **$$$**.

The code of the project is based on the legendary **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** framework of Georgi Gerganov written in C++ with the same attitude to performance and elegance.

We hope Golang instead of *soo-powerful* but *too-low-level* language will allow much greater adoption.

## V0 Roadmap

- [x] Run tensor math in pure Golang - based on C++ source
- [x] Implement LLaMA neural net architecture and model loading
- [x] Run smaller LLaMA-7B model
- [x] Be sure Go inference works EXACT SAME way as C++
- [x] Let Go shine! Enable multi-threading and boost performance

## V1 Roadmap - Spring'23

- [x] Cross-patform compatibility with Mac, Linux and Windows
- [x] Release first stable version for ML hackers - v1.0
- [x] Support bigger LLaMA models: 13B, 30B, 65B - v1.1
- [x] ARM NEON support on Apple Silicon (modern Macs) and ARM servers - v1.2
- [x] Performance boost with x64 AVX2 support for Intel and AMD - v1.2
- [x] RAM and GC optimizations - v1.3
- [x] Server Mode (embedded REST API) for use in real projects - v1.4
- [x] Download model weights from the Internet - v1.4
- [ ] INT8 quantization to allow x4 bigger models fit the same memory
- [ ] Support for popular models of LLaMA family: Vicuna, Alpaca, etc
- [ ] Speed-up AVX2 with memory aligned tensors
- [ ] Extensive logging for production monitoring
- [ ] Interactive mode for real-time chat with GPT

## V2 Roadmap - Summer'23

- [ ] Automatic CPU / GPU features detection
- [ ] Implement metrics for RAM and CPU usage
- [ ] Support popular open models: BLOOM, Anthropic, etc.
- [ ] AVX512 support - yet another performance boost for AMD Epyc
- [ ] Limited Nvidia GPU support (CUDA or Tensor Cores)

## V3 Roadmap - Fall'23

- [ ] Allow plugins and external APIs for complex projects
- [ ] Training capabilities - not inference only
- [ ] Speed execution on GPU cluster
- [ ] FP16 and BF16 support when hardware support there
- [ ] INT4 and GPTQ quantization 

## How to Run?

First, you need to obtain original LLaMA models and convert them into GGML-format, or just download already baked one:

**LLaMA-7B:** [llama-7b-fp32.bin](https://nogpu.com/llama-7b-fp32.bin)

**LLaMA-13B:** [llama-7b-fp32.bin](https://nogpu.com/llama-7b-fp32.bin)

Both models store FP32 weights, so one needs at least 32Gb of regular RAM (not VRAM or GPU RAM) for LLaMA-7B or at least 64Gb for LLaMA-13B.

Next, you should build app binary from sources (see instructions below), or again, just download already built one:

**Windows:** [llama-go-v1.4.0.exe](./builds/llama-go-v1.4.0.exe)

**MacOS:** [llama-go-v1.4.0-macos](./builds/llama-go-v1.4.0-macos)

**Linux:** [llama-go-v1.4.0-linux](./builds/llama-go-v1.4.0-linux)

So now you have both executable binary and model, go try it for yourself!

```shell
llama-go-v1.4.0-macos \
    --model ~/models/llama-7b-fp32.bin \
    --prompt "Why Golang is so popular?" \
```

## Useful CLI parameters:

```shell
--prompt   Text prompt from user to feed the model input
--model    Path and file name of converted .bin LLaMA model [ llama-7b-fp32.bin, etc ]
--server   Start in Server Mode acting as REST API endpoint
--host     Host to allow requests from in Server Mode [ localhost by default ]
--port     Port listen to in Server Mode [ 8080 by default ]
--pods     Maximum pods or units of parallel execution allowed in Server Mode [ single by default ]
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

LLaMA.go embeds standalone HTTP server exposing REST API to place jobs and look for results. Start it with command like this:

```shell
llama-go-v1.4.0-macos \
    --model ~/models/llama-7b-fp32.bin \
    --server
    --host 127.0.0.1
    --port 8080
    --pods 4
    --threads 4
    --avx
```

Depending on how many CPU cores you have, how many requests you need to process in parallel, how fast you'd like to get jobs done, choose **pods** and **threads** parameters wisely.

**Pods** basically means how many parallel instances of inference you'd like to run in parallel?

And **threads** set how many cores will be used to do tensor math within one pod.

So for example, if you have machine with 16 hardware cores capable run 32 hyper-threading executions in parallel, you might end up with something like that: 

--pods 4 --threads 8

When there no capacity for arriving request, it will be placed into the queue and started when there will be free pod available.

## REST API examples

# Place new job

Send POST request to you server with JSON like this:

```json
{
    "id": "5fb8ebd0-e0c9-4759-8f7d-35590f6c9fc3",
    "prompt": "Why Golang is so popular?"
}
```

## How to build by myself?

First, install Golang and git (you'll need to download installers in case of Windows). 

```shell
brew install git
brew install golang
```

Then clone the repo and enter the project folder:

```
git clone https://github.com/gotzmann/llama.go.git
cd llama.go
```

Some magic with external packages:

```
go tidy
go vendor
```

And finally run binary from the source! Do not forget about command-line flags:

```shell
go run main.go \
    --model ~/models/llama-7b-fp32.bin \
    --prompt "Why Golang is so popular?"
```

## FAQ

**1) Where might I get original LLaMA model files?**

Contact Meta directly or look around for some torrent alternatives

**2) How to convert original LLaMA files into supported format?** 

Place original PyTorch FP16 files into **models** directory, then convert with command:

```shell
python3 ./scripts/convert.py ~/models/LLaMA/7B/ 0
```
