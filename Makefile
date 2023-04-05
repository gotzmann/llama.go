convert16:
	python3 convert.py ./LLaMA/7B/ 1

convert32:
	python3 convert.py ./LLaMA/7B/ 0

quantize:
	./quantize ~/models/7B/ggml-model-f32.bin ~/models/7B/ggml-model-q4_0.bin 2	

int4:
	make -j && ./main -m ./models/7B/ggml-model-q4_0.bin -p "How to create conversational AI:" -n 512	

fp16:
	make -j && ./main -m ./models/7B/ggml-model-f16.bin -p "How to create conversational AI:" -n 512