# LLaMA.go

![](./assets/images/terminal.png?raw=true)

## The Goal

We dream of a world where ML hackers are able to grok with **REALLY BIG GPT** models without having GPU clusters consuming a shit tons of **$$$** - using only machines in their own homelabs.

The code of the project is based on the legendary **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** framework of Georgi Gerganov written in C++

We hope using our beloved Golang instead of *soo-powerful* but *too-low-level* language will allow much greater adoption of the **NoGPU** ideas.

The V1 supports only FP32 math, so you'll need at least 32GB RAM to work even with the smallest **LLaMA-7B** model. As a preliminary step you should have binary files converted from original LLaMA model locally.

## V0 Roadmap

- [x] Run tensor math in pure Golang based on C++ source
- [x] Implement LLaMA neural net architecture and model loading
- [x] Run smaller LLaMA-7B model
- [x] Be sure Go inference works EXACT SAME way as C++
- [x] Let Go shine! Enable multi-threading and boost performance

## V1 Roadmap

- [x] Cross-patform compatibility with Mac, Linux and Windows
- [x] Release first stable version for ML hackers
- [x] Support bigger LLaMA models: 13B, 30B, 65B
- [x] ARM NEON support on Apple Silicon (modern Macs) and ARM servers
- [x] Performance boost with x64 AVX2 support for Intel and AMD
- [ ] Speed-up AVX2 with memory aligned tensors
- [ ] INT8 quantization to allow x4 bigger models fit the same memory
- [ ] Enable interactive mode for real-time chat with GPT
- [ ] Allow automatic download converted model weights from the Internet
- [ ] Implement metrics for RAM and CPU usage
- [ ] Server Mode for use in Clouds as part of Microservice Architecture

## V2 Roadmap

- [ ] Allow plugins and external APIs for complex projects
- [ ] AVX512 support - yet another performance boost for AMD Epyc
- [ ] FP16 and BF16 support when hardware support there
- [ ] Support INT4 and GPTQ quantization 

## How to Run

```shell
go run main.go \
    --model ~/models/7B/ggml-model-f32.bin \
    --temp 0.80 \
    --context 128 \
    --predict 128 \
    --prompt "Why Golang is so popular?"
```

Or build it with Makefile and then run binary.

## Useful CLI parameters:

```shell
--prompt   Text prompt from user to feed the model input
--model    Path and file name of converted .bin LLaMA model
--threads  Adjust to the number of CPU cores you want to use [ all cores by default ]
--context  Context size in tokens [ 1024 by default ]
--predict  Number of tokens to predict [ 512 by default ]
--temp     Model temperature hyper parameter [ 0.5 by default ]
--silent   Hide welcome logo and other output [ show by default ]
--chat     Chat with user in interactive mode instead of compute over static prompt
--profile  Profe CPU performance while running and store results to [cpu.pprof] file
--avx      Enable x64 AVX2 optimizations for Intel and AMD machines
--neon     Enable ARM NEON optimizations for Apple Macs and ARM server
```

## FAQ

**1] Where might I get original LLaMA model files?**

Contact Meta directly or look around for some torrent alternatives

**2] How to convert original LLaMA files into supported format?** 

Youl'll need original FP16 files placed into **models** directory, then convert with command:

```shell
python3 ./scripts/convert.py ~/models/LLaMA/7B/ 0
```
