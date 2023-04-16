# LLaMA.go

![](./assets/images/terminal.png?raw=true)

[![Coverage](https://img.shields.io/badge/Coverage-0-red)](https://github.com/gotzmann/llama.go/actions/workflows/coverage.yml)

## The Goal

We dream of a world where ML hackers are able to grok with **REALLY BIG GPT** models without having GPU clusters consuming a shit tons of **$$$** - using only machines in their own homelabs.

The code of the project is based on the legendary **[ggml.cpp](https://github.com/ggerganov/llama.cpp)** framework of Georgi Gerganov written in C++

We hope using our beloved Golang instead of *soo-powerful* but *too-low-level* language will allow much greater adoption of the **NoGPU** ideas.

**NB!** The V1 supports only FP32 math, so you'll need at least 32GB RAM to work even with the smallest **LLaMA-7B** model. As a preliminary step you should have binary files converted from original LLaMA model locally.

## V0 Roadmap

- [x] Move FP32 tensor math from C++ to pure Golang package GoML
- [x] Implement LLaMA neural net architecture and model loading in Golang
- [x] Support smaller LLaMA-7B model
- [x] Be sure Go inference works EXACT SAME way as C++ for static prompts
- [x] Let Go shine! Enable multi-threading and boost performance

## V1 Roadmap

- [x] Check cross-patform compatibility with Mac and Windows
- [x] Release first stable version for ML hackers
- [x] Support bigger LLaMA models: 13B, 30B, 65B
- [ ] Enable interactive mode for real-time chat with GPT
- [ ] Allow automatic download converted model weights from the Internet
- [ ] Implement metrics for RAM and CPU usage
- [ ] x8 performance boost with AVX2 support
- [ ] INT8 quantization to allow x4 bigger models fit the same memory
- [ ] Server Mode for use in clouds as part of microservice architecture

## V2 Roadmap

- [ ] x2 performance boost with AVX512 support
- [ ] ARM NEON support on Mac machines and ARM servers
- [ ] FP16 and BF16 support where possible
- [ ] Support INT4 and GPTQ quantization 

## How to Run

```shell
go run main.go --threads 8 --model ~/models/7B/ggml-model-f32.bin --temp 0.80 --context 128 --predict 128 --prompt "Why Golang is so popular?"
```

Or edit the Makefile and compile and run:

```shell
make
./llama --threads 8 --model ~/models/7B/ggml-model-f32.bin --temp 0.80 --context 128 --predict 128 --prompt "Why Golang is so popular?"
```

## FAQ

**1] Where might I get original LLaMA model files?**

Contact Meta directly or look around for some torrent alternatives

**2] How to convert original LLaMA files into supported format?** 

Youl'll need original FP16 files placed into **models** directory, then convert with command:

```shell
python3 ./scripts/convert.py ~/models/LLaMA/7B/ 0
```
