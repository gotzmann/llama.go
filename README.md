# llama.go
Meta's LLaMA large language model inference in pure Golang using only CPU. No GPU needed.

It will stress all CPU cores using FP32 math - so you'll need at least 32Gb RAM for 7B model. 

AVX2/AVX-512 and ARM NEON optimizations will come later. More details come a bit later too...

<p align="center">
  <img width="100%" src="terminal.png">
</p>
