package main

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"reflect"
	"strings"
	"unsafe"

	"github.com/x448/float16"

	"github.com/gotzmann/llama.go/ml"
)

/*
https://huggingface.co/docs/transformers/main/model_doc/llama

vocab_size (int, optional, defaults to 32000) — Vocabulary size of the LLaMA model. Defines the number of different tokens that can be represented by the inputs_ids passed when calling LlamaModel

hidden_size (int, optional, defaults to 4096) — Dimension of the hidden representations.

intermediate_size (int, optional, defaults to 11008) — Dimension of the MLP representations.

num_hidden_layers (int, optional, defaults to 32) — Number of hidden layers in the Transformer encoder.

num_attention_heads (int, optional, defaults to 32) — Number of attention heads for each attention layer in the Transformer encoder.

hidden_act (str or function, optional, defaults to "silu") — The non-linear activation function (function or string) in the decoder.

initializer_range (float, optional, defaults to 0.02) — The standard deviation of the truncated_normal_initializer for initializing all weight matrices.

rms_norm_eps (float, optional, defaults to 1e-12) — The epsilon used by the rms normalization layers.

use_cache (bool, optional, defaults to True) — Whether or not the model should return the last key/values attentions (not used by all models). Only relevant if config.is_decoder=True.

tie_word_embeddings(bool, optional, defaults to False) — Whether to tie weight embeddings Example —
*/

/*
#include "ggml.h"

#include "utils.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
#include <signal.h>
#include <unistd.h>
#elif defined (_WIN32)
#include <signal.h>
#endif

#define ANSI_COLOR_RED     "\x1b[31m"
#define ANSI_COLOR_GREEN   "\x1b[32m"
#define ANSI_COLOR_YELLOW  "\x1b[33m"
#define ANSI_COLOR_BLUE    "\x1b[34m"
#define ANSI_COLOR_MAGENTA "\x1b[35m"
#define ANSI_COLOR_CYAN    "\x1b[36m"
#define ANSI_COLOR_RESET   "\x1b[0m"
#define ANSI_BOLD          "\x1b[1m"
*/

var (
	// determine number of model parts based on the dimension
	llamaParts = map[uint32]uint32{
		4096: 1,
		5120: 2,
		6656: 4,
		8192: 8,
	}

	// default hparams (LLaMA 7B)
	hparamsVocabSize = uint32(32000)
	hparamsCtx       = uint32(512) // this is provided as user input?
	hparamsEmbd      = uint32(4096)
	hparamsMult      = uint32(256)
	hparamsHeads     = uint32(32)
	hparamsLayers    = uint32(32)
	hparamsRot       = uint32(64)
	hparamsF16       = uint32(1)
)

type llamaLayer struct {

	// normalization
	////struct ggml_tensor * attention_norm;
	attentionNorm *ml.Tensor

	// attention
	///struct ggml_tensor * wq;
	wq *ml.Tensor
	////struct ggml_tensor * wk;
	wk *ml.Tensor
	////struct ggml_tensor * wv;
	wv *ml.Tensor
	////struct ggml_tensor * wo;
	wo *ml.Tensor

	// normalization
	////struct ggml_tensor * ffn_norm;
	ffn_norm *ml.Tensor

	// ff
	w1 *ml.Tensor
	w2 *ml.Tensor
	w3 *ml.Tensor
	////struct ggml_tensor * w2;
	////struct ggml_tensor * w3;
}

type llamaModel struct {

	//hparams llama_hparams hparams;

	////struct ggml_tensor * tok_embeddings;
	tokEmbeddings *ml.Tensor

	////struct ggml_tensor * norm;
	norm *ml.Tensor
	////struct ggml_tensor * output;
	output *ml.Tensor

	////std::vector<llama_layer> layers;
	layers []llamaLayer

	// key + value memory
	////struct ggml_tensor * memory_k;
	memoryK *ml.Tensor
	////struct ggml_tensor * memory_v;
	memoryV *ml.Tensor

	ctx *ml.Context // ggml_context

	tensors map[string]*ml.Tensor //std::map<std::string, struct ggml_tensor *> tensors;
}

func NewModel() llamaModel {
	return llamaModel{
		layers:  make([]llamaLayer, 0),
		tensors: make(map[string]*ml.Tensor),
	}
}

// NB! INT = 32 bits
func readInt(reader *bufio.Reader) (uint32, error) {
	buf := make([]byte, 4)
	if count, err := io.ReadFull(reader, buf); err != nil || count != 4 {
		fmt.Print("\n[ERROR] Failed to read data from model")
		//os.Exit(1)
		return 0, err
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0]), nil
}

func readString(reader *bufio.Reader, len uint32) string {
	buf := make([]byte, len)
	if count, err := io.ReadFull(reader, buf); err != nil || count != int(len) {
		fmt.Print("\n[ERROR] Failed to read data from model")
		os.Exit(1)
	}
	return string(buf)
}

func readFP16ToFP32(reader *bufio.Reader) float32 {
	buf := make([]byte, 2)
	if count, err := io.ReadFull(reader, buf); err != nil || count != 2 {
		fmt.Print("\n[ERROR] Failed to read data from model")
		os.Exit(1)
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32()
}

func readFP32(reader *bufio.Reader) float32 {
	buf := make([]byte, 4)
	if count, err := io.ReadFull(reader, buf); err != nil || count != 4 {
		fmt.Print("\n[ERROR] Failed to read data from model")
		os.Exit(1)
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	//bits := uint32(buf[1])<<24 | uint32(buf[0])<<16 | uint32(buf[3])<<8 | uint32(buf[2])
	return math.Float32frombits(bits)
}

// load the model's weights from a file
func llamaModelLoad(fileName string, model *llamaModel, vocab *ml.GPTVocab, n_ctx uint32) error {
	fmt.Printf("\n[llamaModelLoad] Loading model from '%s' - please wait ...\n", fileName)

	data, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer data.Close()
	reader := bufio.NewReader(data)
	//reader := io.NewReader(data)

	//var magic []byte
	//magic := make([]byte, 4)
	//if _, err := reader.Read(magic); err != nil {
	//	return err
	//}

	//var magicInt int32
	//magicInt := int32(magic[3])<<24 | int32(magic[2])<<16 | int32(magic[1])<<8 | int32(magic[0])
	magic, _ := readInt(reader)
	if magic != 0x67676d6c {
		fmt.Printf("\n[llamaModelLoad] Invalid model file '%s' (bad magic)", fileName)
		return nil // FIXME ERR
	}

	//var buf []byte // std::vector<char> f_buf(1024*1024);
	//	buf := make([]byte, 1024*1024)
	//	if _, err := reader.Read(buf); err != nil {
	//		fmt.Printf("\n[llamaModelLoad] Error '%w'", err)
	//		return nil
	//	}

	////auto fin = std::ifstream(fname, std::ios::binary);
	////fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
	////if (!fin) {
	////    fmt.Printf("%s: failed to open '%s'\n", __func__, fname.c_str());
	////    return false;
	////}
	/*
	   // verify magic
	   {
	       uint32_t magic;
	       fin.read((char *) &magic, sizeof(magic));
	       if (magic != 0x67676d6c) {
	           fmt.Printf("%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
	           return false;
	       }
	   }
	*/

	var n_ff, n_parts uint32

	// load hparams
	{
		hparamsVocabSize, _ = readInt(reader) // vocab_size
		hparamsEmbd, _ = readInt(reader)      // dim
		hparamsMult, _ = readInt(reader)      // multiple_of
		hparamsHeads, _ = readInt(reader)     // n_heads
		hparamsLayers, _ = readInt(reader)    // n_layers
		hparamsRot, _ = readInt(reader)       // rot = dim // n_heads [obsolete]
		hparamsF16, _ = readInt(reader)       // ftype

		hparamsCtx = n_ctx

		//n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
		n_ff = ((2*(4*hparamsEmbd)/3 + hparamsMult - 1) / hparamsMult) * hparamsMult
		//n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
		n_parts = llamaParts[hparamsEmbd]

		fmt.Printf("\nvocab  = %d", hparamsVocabSize)
		//fmt.Printf("\nctx   = %d", hparamsCtx)
		fmt.Printf("\nembd   = %d", hparamsEmbd)
		fmt.Printf("\nmult   = %d", hparamsMult)
		fmt.Printf("\nheads  = %d", hparamsHeads)
		fmt.Printf("\nlayers = %d", hparamsLayers)
		//fmt.Printf("\nrot   = %d", hparamsRot)
		//fmt.Printf("\nf16     = %d", hparamsF16)
		//fmt.Printf("\nn_ff    = %d", n_ff)
		//fmt.Printf("\nn_parts = %d", n_parts)
	}

	// --- load vocab
	for i := uint32(0); i < hparamsVocabSize; i++ {
		len, _ := readInt(reader)
		//word := make([]byte, len)
		//if count, err := io.ReadFull(reader, word); err != nil || count != int(len) {
		//	fmt.Printf("\n[llamaModelLoad] Problem reading vocabulary from '%s'", fileName)
		//	return nil // FIXME ERR
		//}
		word := readString(reader, len)

		//if i%6 == 0 {
		//	fmt.Println()
		//}
		//fmt.Printf("| vocab[%d] = %s ] ", i, string(word))

		vocab.Token2ID[word] = i
		vocab.ID2Token[i] = word
	}

	//return nil

	// for the big tensors, we have the option to store the data in 16-bit floats or quantized
	// in order to save memory and also to speed up the computation
	//wtype := ml.TYPE_COUNT

	////switch (model.hparams.f16) {
	//// case 0: wtype = GGML_TYPE_F32;  break;

	////case 1: wtype = GGML_TYPE_F16;  break;

	wtype := ml.TYPE_F16 // FIXME dtype

	////case 2: wtype = GGML_TYPE_Q4_0; break;
	////case 3: wtype = GGML_TYPE_Q4_1; break;
	////default:
	////        {
	////            fmt.Printf("%s: invalid model file '%s' (bad f16 value %d)\n",
	////                    __func__, fname.c_str(), model.hparams.f16);
	////            return false;
	////        }
	////}

	//wtype2 := ml.TYPE_F32

	////auto & ctx = model.ctx;
	ctx := model.ctx

	// FIXME Context size calculations - do we need this ??
	//{
	//typeSize := ml.TypeSizeFloat(wtype)
	typeSize := ml.TYPE_SIZE[wtype]
	ctxSize := uint32(0)
	////const auto & hparams = model.hparams;
	embd := hparamsEmbd
	layers := hparamsLayers
	////const int n_ctx   = hparams.n_ctx;
	vocabSize := hparamsVocabSize

	ctxSize += embd * vocabSize * typeSize                              /* ggml_type_sizef(wtype) */         // tok_embeddings
	ctxSize += embd * 4                                                 /* ggml_type_sizef(GGML_TYPE_F32) */ // norm
	ctxSize += embd * vocabSize * typeSize                              /* ggml_type_sizef(wtype) */         // output
	ctxSize += layers * (embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */) // attention_norm

	ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wq
	ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wk
	ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wv
	ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wo

	ctxSize += layers * (embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */) // ffn_norm

	ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w1
	ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w2
	ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w3

	ctxSize += ctxSize * layers * embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */ // memory_k
	ctxSize += ctxSize * layers * embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */ // memory_v

	ctxSize += (5 + 10*layers) * 256 // object overhead

	////fmt.Printf("\nggml ctx size = %.2f MB", float32(ctxSize)/(1024*1024))
	//}

	// create the ggml context
	{
		params := ml.InitParams{
			MemSize:   uint64(ctxSize),
			MemBuffer: nil,
		}

		model.ctx = ml.Init(params)
		if model.ctx == nil {
			fmt.Printf("\nggml_init() failed")
			return nil // FIXME ERR
		}
	}

	// prepare memory for the weights
	{
		//const auto & hparams = model.hparams;

		embd := hparamsEmbd
		layers := hparamsLayers
		//ctxSize := hparamsCtx
		vocabSize := hparamsVocabSize

		////model.layers.resize(layers) // FIXME

		model.tokEmbeddings = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, vocabSize)

		model.norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embd)
		model.output = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, vocabSize)

		// map by name
		model.tensors["tok_embeddings.weight"] = model.tokEmbeddings

		model.tensors["norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]llamaLayer, layers)
		for i := uint32(0); i < layers; i++ {
			//auto & layer = model.layers[i];

			model.layers[i].attentionNorm = ml.NewTensor1D(ctx, ml.TYPE_F32, embd)

			model.layers[i].wq = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, embd)
			model.layers[i].wk = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, embd)
			model.layers[i].wv = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, embd)
			model.layers[i].wo = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, embd)

			model.layers[i].ffn_norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embd)

			model.layers[i].w1 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, n_ff)
			model.layers[i].w2 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, n_ff, embd)
			model.layers[i].w3 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embd, n_ff)

			// map by name
			prefix := fmt.Sprintf("layers.%d.", i)

			model.tensors[prefix+"attention_norm.weight"] = model.layers[i].attentionNorm

			model.tensors[prefix+"attention.wq.weight"] = model.layers[i].wq
			model.tensors[prefix+"attention.wk.weight"] = model.layers[i].wk
			model.tensors[prefix+"attention.wv.weight"] = model.layers[i].wv
			model.tensors[prefix+"attention.wo.weight"] = model.layers[i].wo

			model.tensors[prefix+"ffn_norm.weight"] = model.layers[i].ffn_norm

			model.tensors[prefix+"feed_forward.w1.weight"] = model.layers[i].w1
			model.tensors[prefix+"feed_forward.w2.weight"] = model.layers[i].w2
			model.tensors[prefix+"feed_forward.w3.weight"] = model.layers[i].w3
		}
	}

	// key + value memory
	{
		//const auto & hparams = model.hparams;

		embd := hparamsEmbd
		layers := hparamsLayers
		//ctxSize := hparamsCtx
		//mem := layers * ctxSize
		//elements := embd * mem
		elements := embd * layers // FIXME

		model.memoryK = ml.NewTensor1D(ctx, ml.TYPE_F32, elements)
		model.memoryV = ml.NewTensor1D(ctx, ml.TYPE_F32, elements)

		////memorySize = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

		////fmt.Printf("\nmemory_size = %8.2f MB, n_mem = %d\n", memorySize/1024.0/1024.0, mem);
	}

	////const size_t file_offset = fin.tellg();

	////fin.close();

	//std::vector<uint8_t> tmp;

	////tmp := []byte{}

	for i := uint32(0); i < n_parts; /*++i*/ i++ {

		part_id := i
		//commented const int part_id = n_parts - i - 1;

		fname_part := fileName
		if i > 0 {
			fname_part += "." + fmt.Sprintf("%d", i)
		}

		fmt.Printf("\n\n[llamaModelLoad] Loading model part %d / %d from '%s'\n", i+1, n_parts, fname_part)

		//fin = std::ifstream(fname_part, std::ios::binary);
		//fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
		//fin.seekg(file_offset);

		// load weights
		{
			n_tensors := uint32(0)

			////total_size := uint64(0)

			//fmt.Printf("%s: ", __func__);

			for {
				//var n_dims, length, ftype uint32
				//fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
				//fin.read(reinterpret_cast<char *>(&length), sizeof(length));
				//fin.read(reinterpret_cast<char *>(&ftype),  sizeof(ftype));

				dims, err := readInt(reader)

				// FIXME Check for EOF
				if err != nil || dims > 2 {
					fmt.Printf("\n[STOP] Model was read...")
					break
				}

				length, _ := readInt(reader)
				ftype, _ := readInt(reader)

				//fmt.Printf("\ndims = %d", dims)
				//fmt.Printf("\nlength = %d", length)
				//fmt.Printf("\nftype = %d", ftype)

				////if (fin.eof()) {
				////break;
				////}

				nelements := uint32(1)
				//int32_t ne[2] = { 1, 1 };
				ne := [2]uint32{1, 1} // FIXME Why only 2 ??
				for i = uint32(0); i < dims; i++ {
					////fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
					ne[i], _ = readInt(reader)
					////nelements *= ne[i]
					nelements *= ne[i]
				}

				////std::string name(length, 0);
				////fin.read(&name[0], length);
				name := readString(reader, length)
				//fmt.Printf("\nname = %s", name)

				typeStr := "FP32"
				if ftype == 1 {
					typeStr = "FP16"
				}
				nStr := fmt.Sprintf("%d", nelements)
				if nelements > 1000000 {
					nStr = fmt.Sprintf("%.1f M", float32(nelements)/1024/1024)
				}

				fmt.Printf("\n\n=== Tensor # %d === [ %s | %s | dims = %d | n = %s ] ===\n\n", n_tensors, typeStr, name, dims, nStr)

				if _, ok := model.tensors[name]; !ok {
					fmt.Printf("\n[ERROR] Unknown tensor '%s' in model file", name)
					os.Exit(1)
					//return false;
				}

				// splitType = 0: split by columns
				// splitType = 1: split by rows
				splitType := uint32(0)

				// splitType = 0:
				// regex:
				//   - tok_embeddings.*
				//   - layers.*.attention.wo.weight
				//   - layers.*.feed_forward.w2.weight

				// splitType = 1:
				// regex:
				//   - output.*
				//   - layers.*.attention.wq.weight
				//   - layers.*.attention.wk.weight
				//   - layers.*.attention.wv.weight
				//   - layers.*.feed_forward.w1.weight
				//   - layers.*.feed_forward.w3.weight

				if strings.Contains(name, "tok_embeddings") {
					splitType = 0
				} else if strings.Contains(name, "layers") {
					if strings.Contains(name, "attention.wo.weight") {
						splitType = 0
					} else if strings.Contains(name, "feed_forward.w2.weight") {
						splitType = 0
					} else {
						splitType = 1
					}
				} else if strings.Contains(name, "output") {
					splitType = 1
				}

				////auto tensor = model.tensors[name.data()];
				tensor := model.tensors[name]
				tensorSize := tensor.Nelements()

				if dims == 1 {
					if tensorSize != nelements {
						fmt.Printf("\n[ERROR] Tensor '%s' has wrong size in model file", name)
						os.Exit(1)
						//return false;
					}
				} else {
					if tensorSize/n_parts != nelements {
						fmt.Printf("\n[ERROR] Tensor '%s' has wrong size in model file", name)
						os.Exit(1)
						//return false;
					}
				}

				if dims == 1 {
					if tensor.NE[0] != ne[0] || tensor.NE[1] != ne[1] {
						fmt.Printf("\n[ERROR] Tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
							name, tensor.NE[0], tensor.NE[1], ne[0], ne[1])
						os.Exit(1)
						//return false;
					}
				} else {
					if splitType == 0 {
						if tensor.NE[0]/n_parts != ne[0] || tensor.NE[1] != ne[1] {
							fmt.Printf("\n[ERROR] Tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
								name, tensor.NE[0]/n_parts, tensor.NE[1], ne[0], ne[1])
							os.Exit(1)
							//return false;
						}
					} else {
						if tensor.NE[0] != ne[0] || tensor.NE[1]/n_parts != ne[1] {
							fmt.Printf("\n[ERROR] Tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
								name, tensor.NE[0], tensor.NE[1]/n_parts, ne[0], ne[1])
							os.Exit(1)
							//return false;
						}
					}
				}

				////if 0 {
				////static const char * ftype_str[] = { "f32", "f16", "q4_0", "q4_1", };
				////fmt.Printf("%24s - [%5d, %5d], type = %6s, split = %d\n", name.data(), ne[0], ne[1], ftype_str[ftype], splitType);
				////}
				/*
					bpe := uint32(0) // FIXME or 64

					switch ftype {
					case 0:
						bpe = ml.TYPE_SIZE[ml.TYPE_F32]
					case 1:
						bpe = ml.TYPE_SIZE[ml.TYPE_F16]
					case 2:
						bpe = ml.TYPE_SIZE[ml.TYPE_Q4_0] //; assert(ne[0] % 64 == 0); break;
					case 3:
						bpe = ml.TYPE_SIZE[ml.TYPE_Q4_1] //; assert(ne[0] % 64 == 0); break;
					default:
						fmt.Printf("\n[ERROR] unknown ftype %d in model file", ftype)
						os.Exit(1)
						//return false;
					}
				*/
				if dims == 1 || n_parts == 1 {
					////if (nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
					////fmt.Printf("\n[ERROR] tensor '%s' has wrong size in model file: got %zu, expected %zu",
					////    __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
					////os.Exit(1)
					//return false;
					////}

					if part_id == 0 {

						////fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
						// NB! ggml_nbytes == (ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type])/GGML_BLCK_SIZE[tensor->type];
						//fmt.Printf("\n\nReading %d Tensor elements...\n", tensor.Nelements())

						//dataHeader := (*reflect.SliceHeader) (unsafe.Pointer(&tensor.Data))
						//dataHeader.Data

						if ftype == 1 { // --- FP16

							for n := uint32(0); n < tensorSize; n++ {
								tensor.Data[n] = readFP16ToFP32(reader)
							}

						} else { // --- FP32

							var fake []byte

							fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
							dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))

							fakeHeader.Data = dataHeader.Data
							fakeHeader.Len = int(tensorSize * 4)
							fakeHeader.Cap = int(tensorSize * 4)

							//fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
							if count, err := io.ReadFull(reader, fake); err != nil || count != int(tensorSize*4) {
								fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
								fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
								os.Exit(1)
							}
							//os.Exit(0)

							//for n := uint32(0); n < tensorSize; n++ {
							//	tensor.Data[n] = readFP32(reader)
							//}
						}

						// DEBUG Print each 1,000th or 10,000,000th element
						//if tensorSize > 10000000 && n%10000000 == 0 {
						//	fmt.Printf("| %f |", tensor.Data[n])
						//} else {
						//	if tensorSize < 10000 && n%1000 == 0 {
						//		fmt.Printf("| %f |", tensor.Data[n])
						//	}
						//}
						//}

					} else {
						////fin.seekg(ggml_nbytes(tensor), std::ios::cur);
						fmt.Printf("\n[ERROR] The multi-part models are not supported yet")
						os.Exit(1)
					}

					//os.Exit(0)

					////total_size += ggml_nbytes(tensor)

				} else {

					fmt.Printf("\nNOT EXPECTED WAY")
					os.Exit(0)

					////if ((nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)/n_parts) {
					////fmt.Printf("\n[ERROR] tensor '%s' has wrong size in model file: got %zu, expected %zu",
					////    name.data(), ggml_nbytes(tensor)/n_parts, nelements*bpe);
					////os.Exit(1)
					//return false;
					////}

					if splitType == 0 {
						np0 := ne[0]

						////const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
						row_size := tensor.NE[0] * ml.TYPE_SIZE[tensor.Type] // FIXME Check twice
						////assert(row_size == tensor->nb[1]);

						for i1 := uint32(0); i1 < ne[1]; i1++ {
							//const size_t offset_row = i1*row_size;
							offset_row := i1 * row_size
							////offset = offset_row + ((part_id*np0)/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);

							offset := offset_row + part_id*np0*ml.TYPE_SIZE[tensor.Type]
							fmt.Print(offset)

							////fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
						}
					} else {
						np1 := ne[1]

						////const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
						row_size := tensor.NE[0] * ml.TYPE_SIZE[tensor.Type]

						for i1 := uint32(0); i1 < ne[1]; i1++ {
							////const size_t offset_row = (i1 + part_id*np1)*row_size;
							offset_row := (i1 + part_id*np1) * row_size
							////fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
							fmt.Print(offset_row)
						}
					}

					////total_size += ggml_nbytes(tensor)/n_parts;
				}

				//fmt.Printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
				n_tensors++
				if n_tensors%8 == 0 {
					fmt.Printf(".")
					////fflush(stderr);
				}

			}

			////fmt.Printf("\ndone")

			////fmt.Printf("\nmodel size = %.2f MB / num tensors = %d", total_size/1024.0/1024.0, n_tensors)
		}

		////fin.close();
	}

	return nil
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//

func llamaEval(model *llamaModel, n_threads, n_past uint32, embdInp *[]uint32, embdW *[]float32, memPerToken *uint32) error {

	N := len(*embdInp)

	// FIXME Load hyper parameters into model itself
	//const auto & hparams = model.hparams;

	////embd := hparamsEmbd
	layers := hparamsLayers
	////ctx := hparamsCtx
	////heads := hparamsHeads
	////vocab := hparamsVocab
	////rot := hparamsEmbd / hparamsHeads

	////dKey := embd / heads

	// TODO: check if this size scales with n_ctx linearly and remove constant. somehow I feel it wasn't the case
	// static size_t buf_size = hparams.n_ctx*1024*1024;
	////static size_t buf_size = 512u*1024*1024;
	////static void * buf = malloc(buf_size);

	////if (mem_per_token > 0 && mem_per_token*N > buf_size) {
	////    const size_t buf_size_new = 1.1*(mem_per_token*N); // add 10% to account for ggml object overhead
	//fmt.Printf("\n%s: reallocating buffer from %zu to %zu bytes\n", __func__, buf_size, buf_size_new);

	// reallocate
	////    buf_size = buf_size_new;
	////    buf = realloc(buf, buf_size);
	////    if (buf == nullptr) {
	////        fmt.Printf("%s: failed to allocate %zu bytes\n", __func__, buf_size);
	////        return false;
	////    }
	////}

	////struct ggml_init_params params = {
	////    / *.mem_size   =* / buf_size,
	////    / *.mem_buffer =* / buf,
	////};

	////struct ggml_context * ctx0 = ggml_init(params);
	ctx0 := ml.Init(ml.InitParams{})
	////ggml_cgraph gf = {};
	////gf := ml.Graph{}
	////gf.threads = n_threads

	embd := ml.NewTensor1D(ctx0, ml.TYPE_I32, N)
	////memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd));
	// ^^ memcpy(embd->data, embd_inp.data(), N*ggml_element_size(embd)) // FIXME

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layers; il++ {
		////inpSA := inpL

		var cur *ml.Tensor

		// norm
		{
			cur = ml.RMSNorm(ctx0, inpL)

			// cur = attention_norm*cur
			cur = ml.Mul(ctx0, ml.Repeat(ctx0, model.layers[il].attentionNorm, cur), cur)
		}

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			// store key and value to memory
			////if N >= 1 {
			////struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
			////struct ggml_tensor * v = ggml_view_1d(ctx0, model.memory_v, N*n_embd, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past));

			////ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
			////ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
			////}
			/*
			   // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
			   struct ggml_tensor * Q =
			       ggml_permute(ctx0,
			               ggml_rope(ctx0,
			                   ggml_cpy(ctx0,
			                       Qcur,
			                       ggml_new_tensor_3d(ctx0, GGML_TYPE_F32, n_embd/n_head, n_head, N)),
			                   n_past, n_rot, 0),
			               0, 2, 1, 3);
			*/
			/*
			   // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
			   struct ggml_tensor * K =
			       ggml_permute(ctx0,
			               ggml_rope(ctx0,
			                   ggml_reshape_3d(ctx0,
			                       ggml_view_1d(ctx0, model.memory_k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_k)*n_embd),
			                       n_embd/n_head, n_head, n_past + N),
			                   n_past, n_rot, 1),
			               0, 2, 1, 3);
			*/
			// K * Q
			////struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
			/*
			   // KQ_scaled = KQ / sqrt(n_embd/n_head)
			   struct ggml_tensor * KQ_scaled =
			       ggml_scale(ctx0,
			               KQ,
			               ggml_new_f32(ctx0, 1.0f/sqrt(float(n_embd)/n_head))
			               );
			*/
			// KQ_masked = mask_past(KQ_scaled)
			////struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

			// KQ = soft_max(KQ_masked)
			////struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
			/*
			   // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
			   struct ggml_tensor * V_trans =
			       ggml_permute(ctx0,
			               ggml_reshape_3d(ctx0,
			                   ggml_view_1d(ctx0, model.memory_v, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd),
			                   n_embd/n_head, n_head, n_past + N),
			               1, 2, 0, 3);
			*/
			// KQV = transpose(V) * KQ_soft_max
			////struct ggml_tensor * KQV = ggml_mul_mat(ctx0, V_trans, KQ_soft_max);

			// KQV_merged = KQV.permute(0, 2, 1, 3)
			////struct ggml_tensor * KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

			// cur = KQV_merged.contiguous().view(n_embd, N)
			////cur = ggml_cpy(ctx0,
			////        KQV_merged,
			////        ml.NewTensor2D(ctx0, GGML_TYPE_F32, n_embd, N));

			// projection (no bias)
			////cur = ggml_mul_mat(ctx0,
			////        model.layers[il].wo,
			////        cur);
		}

		////struct ggml_tensor * inpFF = ggml_add(ctx0, cur, inpSA);

		// feed-forward network
		{
			// norm
			{
				////cur = ggml_rms_norm(ctx0, inpFF);

				// cur = ffn_norm*cur
				////cur = ggml_mul(ctx0,
				////        ggml_repeat(ctx0, model.layers[il].ffn_norm, cur),
				////        cur);
			}

			////struct ggml_tensor * tmp = ggml_mul_mat(ctx0,
			////        model.layers[il].w3,
			////        cur);

			////cur = ggml_mul_mat(ctx0,
			////        model.layers[il].w1,
			////        cur);

			// SILU activation
			////cur = ggml_silu(ctx0, cur);

			////cur = ggml_mul(ctx0, cur, tmp);

			////cur = ggml_mul_mat(ctx0,
			////        model.layers[il].w2,
			////        cur);
		}

		////cur  = ggml_add(ctx0, cur, inpFF);

		// input for next layer
		inpL = cur
	}

	// norm
	{
		////inpL = ggml_rms_norm(ctx0, inpL);

		// inpL = norm*inpL
		////inpL = ggml_mul(ctx0,
		////            ggml_repeat(ctx0, model.norm, inpL),
		////            inpL);
	}

	// lm_head
	{
		////inpL = ggml_mul_mat(ctx0, model.output, inpL);
	}

	// logits -> probs
	//inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	////ggml_build_forward_expand(&gf, inpL);
	////ggml_graph_compute       (ctx0, &gf);

	//if (n_past%100 == 0) {
	//    ggml_graph_print   (&gf);
	//    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
	//}

	//embd_w.resize(n_vocab*N);
	//memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

	// return result for just the last token
	////embd_w.resize(n_vocab);
	////memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);

	////if (mem_per_token == 0) {
	////    mem_per_token = ggml_used_mem(ctx0)/N;
	////}
	//fmt.Printf("used_mem = %zu\n", ggml_used_mem(ctx0));

	////ggml_free(ctx0);

	return nil
}

/*

static bool is_interacting = false;

#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
void sigint_handler(int signo) {
    printf(ANSI_COLOR_RESET);
    if (signo == SIGINT) {
        if (!is_interacting) {
            is_interacting=true;
        } else {
            _exit(130);
        }
    }
}
#endif

const char * llama_print_system_info(void) {
    static std::string s;

    s  = "";
    s += "AVX = "       + std::to_string(ggml_cpu_has_avx())       + " | ";
    s += "AVX2 = "      + std::to_string(ggml_cpu_has_avx2())      + " | ";
    s += "AVX512 = "    + std::to_string(ggml_cpu_has_avx512())    + " | ";
    s += "FMA = "       + std::to_string(ggml_cpu_has_fma())       + " | ";
    s += "NEON = "      + std::to_string(ggml_cpu_has_neon())      + " | ";
    s += "ARM_FMA = "   + std::to_string(ggml_cpu_has_arm_fma())   + " | ";
    s += "F16C = "      + std::to_string(ggml_cpu_has_f16c())      + " | ";
    s += "FP16_VA = "   + std::to_string(ggml_cpu_has_fp16_va())   + " | ";
    s += "WASM_SIMD = " + std::to_string(ggml_cpu_has_wasm_simd()) + " | ";
    s += "BLAS = "      + std::to_string(ggml_cpu_has_blas())      + " | ";
    s += "SSE3 = "      + std::to_string(ggml_cpu_has_sse3())      + " | ";
    s += "VSX = "       + std::to_string(ggml_cpu_has_vsx())       + " | ";

    return s.c_str();
}

*/

func main() {
	//int main(int argc, char ** argv) {

	////ggml_time_init();
	////const int64_t t_main_start_us = ggml_time_us();

	//var params gptParams//gpt_params params;
	////params.model = "models/llama-7B/ggml-model.bin";

	////if (gpt_params_parse(argc, argv, params) == false) {
	////    return 1;
	////}

	////if (params.n_ctx > 2048) {
	////fmt.Printf("%s: warning: model does not support context sizes greater than 2048 tokens (%d specified);"
	////    "expect poor results\n", __func__, params.n_ctx);
	////}

	////if (params.seed < 0) {
	////params.seed = time(NULL);
	////}

	//fmt.Printf("\n[main] seed = %d", params.seed)
	/*
	       std::mt19937 rng(params.seed);
	       if (params.prompt.empty()) {
	           params.prompt = gpt_random_prompt(rng);
	       }

	   //    params.prompt = R"(// this function checks if the number n is prime
	   //bool is_prime(int n) {)";

	       int64_t t_load_us = 0;

	       gpt_vocab vocab;*/

	//modelName := "./LLaMA/7B/ggml-model-f16.bin"
	modelName := "./LLaMA/7B/ggml-model-fp32.bin"
	model := NewModel()
	vocab := ml.NewVocab()

	// load the model
	if err := llamaModelLoad(modelName, &model, vocab, hparamsCtx); err != nil {
		fmt.Printf("\n[main] Failed to load model from '%s'", modelName)
		return
	}

	// print system information
	////{
	//fmt.Printf("\nsystem_info: n_threads = %d / %d | %s\n",
	//    params.n_threads, std::thread::hardware_concurrency(), llama_print_system_info());
	////}

	////n_past := 0

	////int64_t t_sample_us  = 0;
	////int64_t t_predict_us = 0;

	////std::vector<float> logits;

	// tokenize the prompt
	prompt := "The best programming language to create general AI and profitable ML startup: "
	// Add a space in front of the first character to match OG llama tokenizer behavior
	prompt = " " + prompt
	////std::vector<gpt_vocab::id> embd_inp = ::llama_tokenize(vocab, params.prompt, true);
	embdInp := ml.Tokenize(vocab, prompt, true)
	fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", embdInp)

	////params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (int) embd_inp.size());

	// tokenize the reverse prompt
	////std::vector<gpt_vocab::id> antiprompt_inp = ::llama_tokenize(vocab, params.antiprompt, false);

	fmt.Printf("\nprompt: '%s'\n", prompt)
	fmt.Printf("\nnumber of tokens in prompt = %d\n", len(embdInp))
	for i := 0; i < len(embdInp); i++ {
		fmt.Printf("\n%d => '%s'", embdInp[i], vocab.ID2Token[embdInp[i]])
	}

	////if (params.interactive) {

	////#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
	////        struct sigaction sigint_action;
	////        sigint_action.sa_handler = sigint_handler;
	////        sigemptyset (&sigint_action.sa_mask);
	////        sigint_action.sa_flags = 0;
	////        sigaction(SIGINT, &sigint_action, NULL);
	////#elif defined (_WIN32)
	////       signal(SIGINT, sigint_handler);
	////#endif

	////fmt.Printf("%s: interactive mode on.\n", __func__);

	////if(antiprompt_inp.size()) {
	////    fmt.Printf("%s: reverse prompt: '%s'\n", __func__, params.antiprompt.c_str());
	////    fmt.Printf("%s: number of tokens in reverse prompt = %zu\n", __func__, antiprompt_inp.size());
	////    for (int i = 0; i < (int) antiprompt_inp.size(); i++) {
	////        fmt.Printf("%6d -> '%s'\n", antiprompt_inp[i], vocab.id_to_token.at(antiprompt_inp[i]).c_str());
	////    }
	////    fmt.Printf("\n");
	////}

	////}

	fmt.Printf("\n\nsampling parameters: temp = %f, top_k = %d, top_p = %f, repeat_last_n = %i, repeat_penalty = %f", params.temp, params.top_k, params.top_p, params.repeat_last_n, params.repeat_penalty)

	var embd []uint32

	// determine the required inference memory per token:
	memPerToken = uint64(0)
	ml.llamaEval(model, params.n_threads, 0, []uint32{0, 1, 2, 3}, logits, &memPerToken)

	////int last_n_size = params.repeat_last_n;
	////std::vector<gpt_vocab::id> last_n_tokens(last_n_size);
	///std::fill(last_n_tokens.begin(), last_n_tokens.end(), 0);

	////if (params.interactive) {
	////fmt.Printf("== Running in interactive mode. ==\n"
	////#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__)) || defined (_WIN32)
	////              " - Press Ctrl+C to interject at any time.\n"
	////#endif
	////              " - Press Return to return control to LLaMa.\n"
	////              " - If you want to submit another line, end your input in '\\'.\n");
	////   }

	////remainingTokens = params.n_predict
	remainingTokens := uint32(100) // FIXME
	inputConsumed := uint32(0)
	inputNoEcho := false

	// prompt user immediately after the starting prompt has been loaded
	////if (params.interactive_start) {
	////    is_interacting = true;
	////}

	// set the color for the prompt which will be output initially
	////if (params.use_color) {
	////    printf(ANSI_COLOR_YELLOW);
	////}

	for remaining_tokens > 0 {

		// predict
		if len(embd) > 0 {
			////const int64_t t_start_us = ggml_time_us();

			if !llamaEval(model, params.n_threads, n_past, embd, logits, mem_per_token) {
				fmt.Printf("\n[ERRRO] Failed to predict")
				os.Exit(1)
			}

			////t_predict_us += ggml_time_us() - t_start_us;
		}

		n_past += len(embd)
		embd = []uint32{} ////embd.clear();

		if len(embdInp) <= inputConsumed {

			// out of user input, sample next token
			var top_k float32 = params.top_k
			var top_p float32 = params.top_p
			var temp float32 = params.temp
			var repeat_penalty float32 = params.repeat_penalty

			vocabSize = model.hparams.n_vocab

			id := 0

			{
				////const int64_t t_start_sample_us = ggml_time_us();

				////id = llama_sample_top_p_top_k(vocab, logits.data() + (logits.size() - n_vocab), last_n_tokens, repeat_penalty, top_k, top_p, temp, rng);

				////last_n_tokens.erase(last_n_tokens.begin());
				////last_n_tokens.push_back(id);

				////t_sample_us += ggml_time_us() - t_start_sample_us;
			}

			// add it to the context
			////embd.push_back(id);

			// echo this to console
			inputNoEcho = false

			// decrement remaining sampling budget
			remainingTokens--

		} else {

			// some user input remains from prompt or interaction, forward it to processing
			////while (embd_inp.size() > input_consumed) {
			////embd.push_back(embd_inp[input_consumed]);
			////last_n_tokens.erase(last_n_tokens.begin());
			////last_n_tokens.push_back(embd_inp[input_consumed]);
			////++input_consumed;
			////if (embd.size() > params.n_batch) {
			////break;
			////}
			////}

			// reset color to default if we there is no pending user input
			////if (!input_noecho && params.use_color && embd_inp.size() == input_consumed) {
			////printf(ANSI_COLOR_RESET);
			////}
		}

		// display text
		if !inputNoEcho {
			//for (auto id : embd) {
			////for (auto id : embd) {
			for _, id := range embd { // FIXME Ordered / Unordered ??
				fmt.Printf("%s", vocab.ID2Token[id])
			}
			////fflush(stdout);
		}

		// in interactive mode, and not currently processing queued inputs;
		// check if we should prompt the user for more
		////if (params.interactive && embd_inp.size() <= input_consumed) {
		// check for reverse prompt
		////    if (antiprompt_inp.size() && std::equal(antiprompt_inp.rbegin(), antiprompt_inp.rend(), last_n_tokens.rbegin())) {
		// reverse prompt found
		////        is_interacting = true;
		////    }
		////    if (is_interacting) {
		// currently being interactive
		////        bool another_line=true;
		////        while (another_line) {
		////            fflush(stdout);
		////            char buf[256] = {0};
		////            int n_read;
		////            if(params.use_color) printf(ANSI_BOLD ANSI_COLOR_GREEN);
		////            if (scanf("%255[^\n]%n%*c", buf, &n_read) <= 0) {
		////                // presumable empty line, consume the newline
		////                std::ignore = scanf("%*c");
		////                n_read=0;
		////            }
		////            if(params.use_color) printf(ANSI_COLOR_RESET);

		////            if (n_read > 0 && buf[n_read-1]=='\\') {
		////                another_line = true;
		////                buf[n_read-1] = '\n';
		////                buf[n_read] = 0;
		////            } else {
		////                another_line = false;
		////                buf[n_read] = '\n';
		////                buf[n_read+1] = 0;
		////            }

		////            std::vector<gpt_vocab::id> line_inp = ::llama_tokenize(vocab, buf, false);
		////            embd_inp.insert(embd_inp.end(), line_inp.begin(), line_inp.end());

		////            remaining_tokens -= line_inp.size();

		////            input_noecho = true; // do not echo this again
		////        }

		////        is_interacting = false;
		////    }
		////}

		// end of text token
		////if (embd.back() == 2) {
		////fmt.Printf(" [ EOF ]\n");
		////break
		////}
	}

	////#if defined (_WIN32)
	////    signal(SIGINT, SIG_DFL);
	////#endif

	// report timing
	////{
	////    const int64_t t_main_end_us = ggml_time_us();

	////    fmt.Printf("\n\n");
	////    fmt.Printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
	////    fmt.Printf("%s:     load time = %8.2f ms\n", __func__, t_load_us/1000.0f);
	////    fmt.Printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us/1000.0f);
	////    fmt.Printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us/1000.0f, t_predict_us/1000.0f/n_past);
	////    fmt.Printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us)/1000.0f);
	////}

	////ggml_free(model.ctx);

	////if (params.use_color) {
	////    printf(ANSI_COLOR_RESET);
	////}

	return
}
