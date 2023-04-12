package llama

import (
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"sort"
	"strings"
	"time"
	"unsafe"

	colorable "github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/schollz/progressbar/v3"
	"github.com/x448/float16"
	"golang.org/x/exp/slices"

	"github.com/gotzmann/llama.go/ml"
)

const (
	LLAMA_FILE_VERSION           = 1
	LLAMA_FILE_MAGIC             = 0x67676a74 // 'ggjt' in hex
	LLAMA_FILE_MAGIC_OLD         = 0x67676d66 // 'ggmf' in hex
	LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c // 'ggml' pre-versioned files
)

var (
	// determine number of model parts based on the dimension
	LLAMA_N_PARTS = map[uint32]uint32{
		4096: 1,
		5120: 2,
		6656: 4,
		8192: 8,
	}
)

type pair struct {
	first  float32
	second uint32
}

type Context struct {

	////std::mt19937 rng;
	////int64_t t_load_us = 0;
	////int64_t t_start_us = 0;
	////int64_t t_sample_us = 0;
	////int64_t t_eval_us   = 0;
	////n_sample uint32 // number of tokens sampled
	////n_eval   uint32 // number of eval calls
	////size_t mem_per_token = 0;

	Model *Model
	Vocab *ml.Vocab

	// decode output (2-dimensional array: [n_tokens][n_vocab])
	Logits    []float32
	LogitsAll bool

	// input embedding (1-dimensional array: [n_embd])
	Embedding []float32
}

func NewContext() *Context {
	return &Context{
		Model:     NewModel(),
		Vocab:     ml.NewVocab(0),
		Logits:    make([]float32, 0, 0), // NewFloatSlice(0, 0),
		Embedding: make([]float32, 0, 0), // NewFloatSlice(0, 0),
	}
}

// struct llama_context_params {
type ContextParams struct {
	CtxSize    uint32 // text context
	PartsCount int    // -1 for default
	Seed       int    // RNG seed, 0 for random

	////f16_kv    bool // use fp16 for KV cache
	LogitsAll bool // the llama_eval() call computes all logits, not just the last one
	VocabOnly bool // only load the vocabulary, no weights
	UseLock   bool // force system to keep model in RAM
	Embedding bool // embedding mode only
}

type Layer struct {

	// normalization
	attentionNorm *ml.Tensor

	// attention
	wq *ml.Tensor
	wk *ml.Tensor
	wv *ml.Tensor
	wo *ml.Tensor

	// normalization
	ffn_norm *ml.Tensor

	// ff
	w1 *ml.Tensor
	w2 *ml.Tensor
	w3 *ml.Tensor
}

// default hparams (LLaMA 7B)
type HParams struct {
	ctxSize     uint32 // 512 // this is provided as user input?
	vocabSize   uint32 // 32000
	embdSize    uint32 // 4096
	multSize    uint32 // 256
	headsCount  uint32 // 32
	layersCount uint32 // 32
	rotCount    uint32 // 64
	f16         uint32 // 1
}

type ModelType uint8

// available llama models
const (
	MODEL_UNKNOWN ModelType = iota
	MODEL_7B
	MODEL_13B
	MODEL_30B
	MODEL_65B
)

type KVCache struct {
	K *ml.Tensor
	V *ml.Tensor

	N uint32 // number of tokens currently in the cache
}

type Model struct {
	Type    ModelType
	ctx     *ml.Context // ggml_context
	hparams HParams

	tokEmbeddings *ml.Tensor
	norm          *ml.Tensor
	output        *ml.Tensor

	layers []Layer

	// key + value cache for the self attention
	// TODO: move to llama_state
	kvSelf KVCache // llama_kv_cache

	// the model memory buffer
	////std::vector<uint8_t> buf;

	// tensors
	loadedCount uint32
	tensors     map[string]*ml.Tensor // std::unordered_map<std::string, struct ggml_tensor *> tensors;
}

func NewModel() *Model {
	return &Model{
		hparams: HParams{
			ctxSize:     512,
			vocabSize:   32000,
			embdSize:    4096,
			multSize:    256,
			headsCount:  32,
			layersCount: 32,
			rotCount:    64,
			f16:         1,
		},
		layers:  make([]Layer, 0),
		tensors: make(map[string]*ml.Tensor),
		kvSelf: KVCache{
			K: &ml.Tensor{},
			V: &ml.Tensor{},
		},
	}
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// FIXME Double Check
// Safe Resize() for using instead of C++ std::vector:resize()
// https://go.dev/play/p/VlQ7N75E5AD
func Resize(slice []float32, size int) []float32 {
	newSlice := make([]float32, size)
	for i := 0; i < min(size, len(slice)); i++ {
		newSlice[i] = slice[i]
	}
	return newSlice
}

// FIXME Double Check
// NB! This do not clear the underlying array when resizing
// https://go.dev/play/p/DbK4dFqwrZn
func ResizeInplace(slice *[]float32, size int) {
	if len(*slice) == size {
		return
	} else if size < len(*slice) {
		*slice = (*slice)[:size]
	} else {
		*slice = slices.Grow(*slice, size)
		*slice = (*slice)[:size]
	}
}

// evaluate the transformer
//
//   - lctx:      llama context
//   - tokens:    new batch of tokens to process
//   - n_past:    the context size so far
//   - n_threads: number of threads to use
//

func Eval(

	lctx *Context,
	tokens []uint32,
	tokensCount uint32,
	pastCount uint32,
	threadsCount int) error {

	N := tokensCount
	model := lctx.Model
	kvSelf := model.kvSelf

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	ctxSize := model.hparams.ctxSize
	headsCount := model.hparams.headsCount
	vocabSize := model.hparams.vocabSize
	rotCount := model.hparams.embdSize / model.hparams.headsCount

	ctx0 := &ml.Context{} //ctx0 := ml.Init(ml.InitParams{})

	// for big prompts, if BLAS is enabled, it is better to use only one thread
	// otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
	////ggml_cgraph gf = {};
	////gf.n_threads = N > 255 && ggml_cpu_has_blas() ? 1 : n_threads;
	graph := ml.Graph{ThreadsCount: threadsCount}

	embd := ml.NewTensor1D(ctx0, ml.TYPE_F32 /*ml.TYPE_I32*/, N) // FIXME Will be created as FP32 anyway
	////memcpy(embd->data, tokens, N*ggml_element_size(embd));
	// FIXME Refactore inline initialization
	for id := uint32(0); id < N; id++ {
		embd.Data[id] = float32(tokens[id]) // FIXME copy() for slices
	}

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layersCount; il++ {

		//if il > 0 {
		//	break // DEBUG
		//}

		inpSA := inpL
		cur := &ml.Tensor{}

		// norm
		cur = ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)

		cur = ml.Mul(ctx0, rep, cur)

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			// store key and value to memory
			if N >= 1 {

				////struct ggml_tensor * k = ggml_view_1d(ctx0, kv_self.k, N*n_embd, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past));
				////struct ggml_tensor * v = ggml_view_1d(ctx0, kv_self.v, N*n_embd, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past));

				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Kcur, k));
				////ggml_build_forward_expand(&graph, ggml_cpy(ctx0, Vcur, v));

				// NB! ggml_element_size(kv_self.k) = 2 for FP16
				k := ml.View1D(ctx0, kvSelf.K, N*embdSize, embdSize*(il*ctxSize+pastCount))
				v := ml.View1D(ctx0, kvSelf.V, N*embdSize, embdSize*(il*ctxSize+pastCount))

				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(&graph, ml.Copy(ctx0, Vcur, v))
			}

			// Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
			Q :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Copy(ctx0,
							Qcur,
							ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/headsCount, headsCount, N)),
						pastCount, rotCount, 0),
					0, 2, 1, 3)

			// K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
			K :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Reshape3D(ctx0,
							////ggml_view_1d(ctx0, kv_self.k, (n_past + N)*n_embd, il*n_ctx*ggml_element_size(kv_self.k)*n_embd),
							////n_embd/n_head, n_head, n_past + N),
							ml.View1D(ctx0, kvSelf.K, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						pastCount, rotCount, 1),
					0, 2, 1, 3)

			// K * Q
			////struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
			KQ := ml.MulMat(ctx0, K, Q)

			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewFP32(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(headsCount)))),
				)

			// KQ_masked = mask_past(KQ_scaled)
			////struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, pastCount)

			// KQ = soft_max(KQ_masked)
			////struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

			// V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
			VTrans :=
				ml.Copy(ctx0,
					ml.Permute(ctx0,
						ml.Reshape3D(ctx0,
							ml.View1D(ctx0, kvSelf.V, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						1, 2, 0, 3),
					ml.NewTensor3D(ctx0, ml.TYPE_F32 /* kv_self.v->type */, pastCount+N, embdSize/headsCount, headsCount))

			// KQV = transpose(V) * KQ_soft_max
			KQV := ml.MulMat(ctx0, VTrans, KQSoftMax)

			// KQV_merged = KQV.permute(0, 2, 1, 3)
			KQVMerged := ml.Permute(ctx0, KQV, 0, 2, 1, 3)

			// cur = KQV_merged.contiguous().view(n_embd, N)
			cur = ml.Copy(ctx0,
				KQVMerged,
				ml.NewTensor2D(ctx0, ml.TYPE_F32, embdSize, N))

			// projection (no bias)
			cur = ml.MulMat(ctx0,
				model.layers[il].wo,
				cur)
		}

		inpFF := ml.Add(ctx0, cur, inpSA)

		// feed-forward network
		{
			// norm
			{
				cur = ml.RMSNorm(ctx0, inpFF)

				// cur = ffn_norm*cur
				cur = ml.Mul(ctx0,
					ml.Repeat(ctx0, model.layers[il].ffn_norm, cur),
					cur)
			}

			tmp := ml.MulMat(ctx0,
				model.layers[il].w3,
				cur)

			cur = ml.MulMat(ctx0,
				model.layers[il].w1,
				cur)

			// SILU activation
			cur = ml.Silu(ctx0, cur)

			cur = ml.Mul(ctx0, cur, tmp)

			cur = ml.MulMat(ctx0,
				model.layers[il].w2,
				cur)
		}

		cur = ml.Add(ctx0, cur, inpFF)

		// input for next layer
		inpL = cur

	}

	// used at the end to optionally extract the embeddings
	////var embeddings *ml.Tensor

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	embeddings := inpL

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// logits -> probs
	// COMMentED inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	ml.BuildForwardExpand(&graph, inpL)

	ml.GraphCompute(ctx0, &graph)

	// --- extract logits

	//logitsOut := lctx.Logits // FIXME ASAP What we'll doing with this? Just lost in thin air?

	//fmt.Printf("\n\n=== INPL 09 === [%d,%d,%d,%d] ===\n", inpL.NE[0], inpL.NE[1], inpL.NE[2], inpL.NE[3]) // DEBUG
	//for ii := 0; ii < 12; ii++ {
	//	fmt.Printf("%.4f  ", inpL.Data[ii])
	//}

	if lctx.LogitsAll {

		fmt.Print("\n[HALT] Not Expected : lctx.LogitsAll == true")
		os.Exit(1)
		////logits_out.resize(n_vocab * N);
		///////////////////////////////////////////////////////////logitsOut = Resize(logitsOut, int(vocabSize*N)) // FIXME ASAP Why N multipy?
		////memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
		// FIXME Double Check !! Why multiply for N? Replace with copy() for slices
		for i := uint32(0); i < vocabSize*N; i++ {
			lctx.Logits[i] = inpL.Data[i] // FIXME ASAP Overflow ??
		}

	} else {

		// return result for just the last token
		////logits_out.resize(n_vocab);
		////////////////////////////////////////////////////////logitsOut = Resize(logitsOut, int(vocabSize))

		// FIXME ASAP
		////logitsOut = NewFloatSlice(vocabSize, vocabSize) // FIXME Duplicate rearrangment?

		////memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
		// FIXME Double Check !! Replace with copy() for slices

		// FIXME ASAP Logits LEN = 32,000 without *N | INPL LEN = 256,000
		//memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
		for i := uint32(0); i < vocabSize; i++ {
			//lctx.Logits[i] = inpL.Data[i]
			lctx.Logits[i] = inpL.Data[vocabSize*(N-1)+i]
		}
	}

	if ml.DEBUG {
		printTensor(inpL, "INPL")

		fmt.Printf("\n\n=== LOGITS === %d ===\n", len(lctx.Logits)) // DEBUG
		for ii := 0; ii < 13; ii++ {
			fmt.Printf("%.4f  ", lctx.Logits[ii])
		}
	}

	//os.Exit(0)

	// --- extract embeddings

	if len(lctx.Embedding) > 0 {
		////embeddingOut := lctx.Embedding

		////embedding_out.resize(n_embd);
		//////////////////////////////embeddingOut = Resize(embeddingOut, int(embdSize)) // FIXME ASAP ^^^ down
		///////////////////////////////////////embeddingOut = NewFloatSlice(embdSize, embdSize)
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
		// FIXME ASAP Replace with copy() for slices
		//////////////////////for i := uint32(0); i < embdSize; i++ {
		////////////////////////////	(*embeddingOut)[i] = (*lctx.Embedding)[i]
		//////////////////////////////}

		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);

		for i := uint32(0); i < embdSize; i++ {
			lctx.Embedding[i] = embeddings.Data[(embdSize*(N-1))+i] // FIXME ASAP
		}
	}

	////if (mem_per_token == 0) {
	////    mem_per_token = ggml_used_mem(ctx0)/N;
	////}
	//fmt.Printf("used_mem = %zu\n", ggml_used_mem(ctx0));

	////#if 0
	////    printf("\n%s: used_mem = %.3f MB, scratch -- %.3f MB %.3f MB\n", __func__,
	////            ggml_used_mem(ctx0)/1024.0/1024.0,
	////            lctx.get_buf_max_mem(0)/1024.0/1024.0,
	////            lctx.get_buf_max_mem(1)/1024.0/1024.0);
	////#endif

	////ggml_free(ctx0);

	// measure the performance only for the single-token evals
	////if (N == 1) {
	////    lctx.t_eval_us += ggml_time_us() - t_start_us;
	////    lctx.n_eval++;
	////}
	////else if (N > 1) {
	////    lctx.t_p_eval_us += ggml_time_us() - t_start_us;
	////    lctx.n_p_eval += N;
	////}

	return nil
}

func printTensor(tensor *ml.Tensor, name string) {
	var dt string
	if tensor.Type == ml.TYPE_F16 {
		dt = "FP16"
	}
	if tensor.Type == ml.TYPE_F32 {
		dt = "FP32"
	}
	if tensor.Type == ml.TYPE_Q4_0 {
		dt = "INT4"
	}

	fmt.Printf("\n\n=== [ %s | %s | %d:%d:%d ] ===\n",
		name, dt, tensor.NE[0], tensor.NE[1], tensor.NE[2])

	for nn := 0; nn < min(12, int(tensor.NE[1])); nn++ {
		fmt.Printf("\n %d x %d ...\t", nn, tensor.NE[0])
		for ii := 0; ii < min(12, int(tensor.NE[0])); ii++ {
			fmt.Printf("%.3f\t", tensor.Data[nn*int(tensor.NE[0])+ii])
		}
	}
}

func sampleTopK(logitsID []pair, topK uint32) []pair {
	// find the top K tokens

	// std::partial_sort
	// Rearranges elements such that the range [first, middle) contains
	// the sorted middle − first smallest elements in the range [first, last).
	// The order of equal elements is not guaranteed to be preserved.
	// The order of the remaining elements in the range [middle, last) is unspecified.

	/*std::partial_sort(
	        logits_id.begin(),
	        logits_id.begin() + top_k, logits_id.end(),
	        [](const std::pair<double, gpt_vocab::id> & a, const std::pair<double, gpt_vocab::id> & b) {
	    return a.first > b.first;
	});*/

	//keys := make([]double, 0, len(logitsID))
	//for k := range logitsID {
	//	keys = append(keys, k)
	//}
	//sort.Float64s(keys)

	sort.Slice(
		logitsID[:topK],
		func(i, j int) bool {
			return logitsID[i].first < logitsID[j].first // FIXME ASAP We need bigger elements first
		})

	// logits_id.resize(top_k);
	//for i := uint32(0); i < len(keys)-topK; i++ {
	//delete(logitsID, keys[i])
	//}

	ret := make([]pair, 0, topK)
	copy(ret, logitsID)

	return ret
}

// llama_sample_top_p_top_k
// sample next token given probabilities for each embedding
//
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
//

// std::mt19937 = A Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.
func SampleTopPTopK(
	lctx *Context,
	lastNTokens []uint32,
	lastNTokensSize uint32, // FIXME Remove
	topK uint32,
	topP float32,
	temp float32,
	repeatPenalty float32,
) uint32 {

	////auto & rng = lctx.rng;
	////logitsCount := uint32(len(vocab.ID2Token))
	logitsCount := lctx.Model.hparams.vocabSize
	logits := lctx.Logits

	if ml.DEBUG {
		fmt.Printf("\n\n>>> SampleTopPTopK <<<\n")
		fmt.Printf("\n=== LOGITS | %d ===\n", len(logits))
		for i := 0; i < 8; i++ {
			fmt.Printf("%.4f ", logits[i])
		}
		fmt.Printf(" ... ")
		for i := int(len(logits)) - 1; i >= int(len(logits))-8; i-- {
			fmt.Printf("%.4f ", logits[i])
		}
		fmt.Printf("\n=== LAST N TOKENS | %d ===\n", len(lastNTokens))
		for i := 0; i < int(lastNTokensSize); i++ {
			fmt.Printf("%d ", lastNTokens[i])
		}
	}

	////if (temp <= 0) {
	////    // select the token with the highest logit directly
	////    float max_logit = plogits[0];
	////    llama_vocab::id max_id = 0;
	////
	////    for (int i = 1; i < n_logits; ++i) {
	////        if (plogits[i] > max_logit) {
	////            max_logit = plogits[i];
	////            max_id = i;
	////        }
	////    }
	////    return max_id;
	////}

	////const auto * plogits = logits.data() + logits.size() - n_logits;
	//plogits := logits[len(logits)-int(logitsCount):] // FIXME ASAP
	plogits := logits[:]

	////std::vector<std::pair<double, llama_vocab::id>> logits_id;
	////logits_id.reserve(n_logits);
	logitsID := make([]pair, 0, logitsCount) // FIXME LEN vs CAP

	{
		scale := float32(1.0 / temp)
		for i := uint32(0); i < logitsCount; i++ {

			// repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
			// credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

			// if lastNTokens already contains i-th token, append it with repeat penatly
			////if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
			if slices.IndexFunc(lastNTokens, func(el uint32) bool { return el == i }) != -1 {
				// if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
				if plogits[i] < 0.0 {
					////logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
					logitsID = append(logitsID, pair{plogits[i] * scale * repeatPenalty, i})
				} else {
					////logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
					logitsID = append(logitsID, pair{plogits[i] * scale / repeatPenalty, i})
				}
				// else append pair to logitsID	scaling probability
			} else {
				logitsID = append(logitsID, pair{plogits[i] * scale, i})
			}
		}
	}

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID AFTER | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// sort logitsID slice and return only top K elements
	//// sampleTopK(logitsID, topK)

	// NB! Inline logic for [sampleTopK] right here

	//// std::partial_sort(
	////	logits_id.begin(),
	////	logits_id.begin() + top_k, logits_id.end(),
	////	[](const std::pair<float, llama_vocab::id> & a, const std::pair<float, llama_vocab::id> & b) {
	//// return a.first > b.first;
	//// });
	//// logits_id.resize(top_k);

	sort.Slice(
		logitsID, // logitsID[:topK],
		func(a, b int) bool {
			return logitsID[a].first > logitsID[b].first
		})

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID SORTED | TOP K = %d ===\n", topK)
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	logitsID = logitsID[:topK]

	if ml.DEBUG {
		fmt.Printf("\n=== LOGITS ID RESIZED | %d ===\n", len(logitsID))
		for i := 0; i < min(6, len(logitsID)); i++ {
			fmt.Printf("{ %.3f | %d }", logitsID[i].first, logitsID[i].second)
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(logitsID)-1; i++ {
			fmt.Printf("{ %.3f | %d } ", logitsID[i].first, logitsID[i].second)
		}
	}

	// FIXME Why loop? We've already SORTED logitsID and the MAX is just the FIRST element
	////double maxl = -INFINITY;
	maxl := float32(math.Inf(-1))
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		//// maxl = std::max(maxl, kv.first);
		maxl = max(maxl, kv.first)
	}

	//fmt.Printf("\nmaxl = %.3f", maxl)

	// compute probs for the top k tokens
	////probs.reserve(logits_id.size());
	probs := make([]float32, 0, len(logitsID)) // FIXME LEN vs CAP

	sum := float64(0.0)
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		// double p = exp(kv.first - maxl);
		p := math.Exp(float64(kv.first - maxl))
		probs = append(probs, float32(p))
		sum += p
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	// normalize the probs
	for i := range probs {
		probs[i] /= float32(sum)
	}

	if ml.DEBUG {
		fmt.Printf("\n=== PROBS NORM | %d ===\n", len(probs))
		for i := 0; i < min(6, len(probs)); i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
		fmt.Printf(" ... ")
		for i := len(logitsID) - 6; i < len(probs)-1; i++ {
			fmt.Printf("%.3f  ", probs[i])
		}
	}

	if topP < 1.0 {

		cumsum := float32(0.0) // TODO float64 for better math?
		for i := uint32(0); i < uint32(len(probs)); i++ {
			cumsum += probs[i]
			if cumsum >= topP {
				////probs.resize(i + 1) // FIXME ASAP
				probs = probs[:i+1]
				////logits_id.resize(i + 1) // FIXME ASAP
				logitsID = logitsID[:i+1]
				break
			}
		}

		cumsum = 1.0 / cumsum
		for i := uint32(0); i < uint32(len(probs)); i++ {
			probs[i] *= cumsum
		}
	}

	if ml.DEBUG {
		if len(probs) > 6 {
			fmt.Printf("\n=== PROBS POST | %d ===\n", len(probs))
			for i := 0; i < min(6, len(probs)); i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
			fmt.Printf(" ... ")
			for i := len(logitsID) - 6; i < len(probs)-1; i++ {
				fmt.Printf("%.3f  ", probs[i])
			}
		}
	}

	////std::discrete_distribution<> dist(probs.begin(), probs.end());
	////int idx = dist(rng);
	////return logits_id[idx].second;

	// --- discrete distribution
	//     TODO Do we need something better than Serge Gotsuliak's hand-crafted formula here?

	seed := time.Now().UnixNano()
	source := rand.NewSource(seed)

	for i := 0; i < len(probs); i++ {
		f := float32(source.Int63()) / (1 << 63)
		probs[i] = probs[i] * probs[i] * f * f
	}

	idx := 0
	maxProb := probs[0]
	for i := 1; i < len(probs); i++ {
		if probs[i] > maxProb {
			idx = i
			maxProb = probs[i]
		}
	}

	if ml.DEBUG {
		fmt.Printf("\nidx = %d", idx)
		fmt.Printf("\nlogitsID = %d | weight = %f", logitsID[idx].second, logitsID[idx].first)
	}

	return logitsID[idx].second
}

// llama_model_load
// load the model's weights from a file
// see convert-pth-to-ggml.py for details on format

func LoadModel(
	fileName string,
	//partsCount int,
	silent bool,
) (*Context, error) {

	lctx := NewContext()

	file, err := os.Open(fileName)
	if err != nil {
		return nil, err
	}
	defer file.Close()

	// --- check header magic and format version

	magic := readInt(file)

	if magic == LLAMA_FILE_MAGIC_UNVERSIONED || magic == LLAMA_FILE_MAGIC_OLD {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Too old, regenerate!", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	if magic != LLAMA_FILE_MAGIC {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Wrong MAGIC in header", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	version := readInt(file)

	if version != LLAMA_FILE_VERSION {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Unsupported version", fileName)
		return nil, fmt.Errorf("invalid model file")
	}

	// --- load hparams

	vocabSize := readInt(file)   // vocab_size
	embdSize := readInt(file)    // dim
	multSize := readInt(file)    // multiple_of
	headsCount := readInt(file)  // n_heads
	layersCount := readInt(file) // n_layers
	rotCount := readInt(file)    // rot = dim // n_heads [obsolete]
	f16 := readInt(file)         // ftype

	model := lctx.Model

	model.hparams.vocabSize = vocabSize
	model.hparams.embdSize = embdSize
	model.hparams.multSize = multSize
	model.hparams.headsCount = headsCount
	model.hparams.layersCount = layersCount
	model.hparams.rotCount = rotCount
	model.hparams.f16 = f16

	// --- init cache
	//KVCacheInit(&lctx.Model.hparams, &lctx.Model.kvSelf, ml.TYPE_F32)
	dt := ml.TYPE_F32
	size := embdSize * layersCount * 512 /*ctxSize*/ // FIXME ctxSize
	lctx.Model.kvSelf.K = ml.NewTensor1D(nil, dt, size)
	lctx.Model.kvSelf.V = ml.NewTensor1D(nil, dt, size)

	// NB! Do not try to resize / relocate secondary pointers
	lctx.Vocab = ml.NewVocab(vocabSize)
	vocab := lctx.Vocab

	//ctx.LogitsAll = params.LogitsAll
	//if params.LogitsAll {
	//ctx.Logits = make([]float32, ctx.Model.hparams.ctxSize*ctx.Model.hparams.vocabSize) // .reserve(hparams.n_ctx*hparams.n_vocab);
	//} else {
	// FIXME 32K -> 512 ?? Already reserved, skip
	//ctx.Logits = make([]float32, ctx.Model.hparams.ctxSize) // .reserve(hparams.n_ctx);
	//}

	// FIXME Reserve extra space for tokensCount (N) = 8 (as with LogitsAll == true)
	//lctx.Logits = make([]float32, vocabSize*8, vocabSize*8) // NewFloatSlice(vocabSize, vocabSize) // FIXME ASAP
	lctx.Logits = make([]float32, vocabSize, vocabSize) // use just vocab size as CPP version does by default

	//hparamsCtx = n_ctx

	//n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
	//n_ff := ((2*(4*hparamsEmbd)/3 + hparamsMult - 1) / hparamsMult) * hparamsMult

	//if partsCount < 1 {
	partsCount := int(LLAMA_N_PARTS[embdSize]) // FIXME ASAP
	//}

	// temp warning to tell the user to use "--n_parts"
	////if (hparams.f16 == 4 && n_parts != 1) {
	////fprintf(stderr, "%s: GPTQ model detected - are you sure n_parts should be %d? we normally expect it to be 1\n", __func__, n_parts);
	////fprintf(stderr, "%s: use '--n_parts 1' if necessary\n", __func__);
	////}

	////if (hparams.n_layer == 32) {
	////model.type = e_model::MODEL_7B;
	////}

	////if (hparams.n_layer == 40) {
	////model.type = e_model::MODEL_13B;
	////}

	////if (hparams.n_layer == 60) {
	////model.type = e_model::MODEL_30B;
	////}

	////if (hparams.n_layer == 80) {
	////model.type = e_model::MODEL_65B;
	////}

	if ml.DEBUG {
		fmt.Printf("\nvocab  = %d", vocabSize)
		fmt.Printf("\nembd   = %d", embdSize)
		fmt.Printf("\nmult   = %d", multSize)
		fmt.Printf("\nheads  = %d", headsCount)
		fmt.Printf("\nlayers = %d", layersCount)
		fmt.Printf("\nrot    = %d", rotCount)
		fmt.Printf("\nf16    = %d", f16)
	}

	//fmt.Printf("\nctx   = %d", hparamsCtx)
	//fmt.Printf("\nn_ff    = %d", n_ff)
	//fmt.Printf("\nn_parts = %d", n_parts)

	n_ff := ((2*(4*embdSize)/3 + multSize - 1) / multSize) * multSize

	// --- load vocab

	if !silent && runtime.GOOS == "windows" {
		Colorize("\n\n[magenta][ INIT ][white] Loading vocab...")
	}

	vocabBar := progressbar.NewOptions(
		int(vocabSize),
		progressbar.OptionFullWidth(),
		//progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model vocab...  [light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	for i := uint32(0); i < vocabSize; i++ {

		if !silent && runtime.GOOS != "windows" && i%100 == 0 {
			vocabBar.Set(int(i))
		}

		len := readInt(file)
		token := readString(file, len)
		score := readFP32(file)

		vocab.Token2ID[token] = i
		vocab.ID2Token[i] = ml.TokenScore{Token: token, Score: score}
	}

	if !silent && runtime.GOOS != "windows" {
		vocabBar.Finish()
		fmt.Printf("\n")
	}

	// for the big tensors, we have the option to store the data in 16-bit floats or quantized
	// in order to save memory and also to speed up the computation
	//wtype := ml.TYPE_COUNT

	////switch (model.hparams.f16) {
	//// case 0: wtype = GGML_TYPE_F32;  break;
	////case 1: wtype = GGML_TYPE_F16;  break;
	////wtype := ml.TYPE_F16 // FIXME dtype
	////case 2: wtype = GGML_TYPE_Q4_0; break;
	////case 3: wtype = GGML_TYPE_Q4_1; break;
	////default:
	////        {
	////            fmt.Printf("%s: invalid model file '%s' (bad f16 value %d)\n",
	////                    __func__, fname.c_str(), model.hparams.f16);
	////            return false;
	////        }
	////}

	ctx := model.ctx

	// --- prepare memory for the weights
	{
		model.tokEmbeddings = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize)

		model.norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)
		model.output = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize)

		// map by name
		model.tensors["tok_embeddings.weight"] = model.tokEmbeddings

		model.tensors["norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]Layer, layersCount)
		for i := uint32(0); i < layersCount; i++ {
			//auto & layer = model.layers[i];

			model.layers[i].attentionNorm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)

			model.layers[i].wq = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wk = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wv = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)
			model.layers[i].wo = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, embdSize)

			model.layers[i].ffn_norm = ml.NewTensor1D(ctx, ml.TYPE_F32, embdSize)

			model.layers[i].w1 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, n_ff)
			model.layers[i].w2 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, n_ff, embdSize)
			model.layers[i].w3 = ml.NewTensor2D(ctx, ml.TYPE_F32 /*wtype*/, embdSize, n_ff)

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

	if !silent && runtime.GOOS == "windows" {
		Colorize("\n[magenta][ INIT ][white] Loading model - please wait ...\n")
	}

	// https://pkg.go.dev/github.com/schollz/progressbar/v3#Option
	bar := progressbar.NewOptions(int(layersCount*9),
		progressbar.OptionFullWidth(),
		//progressbar.OptionSetWidth(40),
		progressbar.OptionEnableColorCodes(true),
		progressbar.OptionSetPredictTime(false),
		progressbar.OptionSetElapsedTime(false),
		progressbar.OptionSetDescription("[light_magenta][ INIT ][light_blue] Loading model weights...[light_cyan]"),
		progressbar.OptionSetTheme(progressbar.Theme{
			Saucer:        "[light_magenta]▒[reset]",
			SaucerHead:    "[white]▒[reset]",
			SaucerPadding: "[dark_gray]▒[reset]",
			BarStart:      "[dark_gray]║[reset]",
			BarEnd:        "[dark_gray]║[reset]",
		}))

	for i := 0; i < int(partsCount); i++ {

		part_id := i
		//commented const int part_id = n_parts - i - 1;

		fname_part := fileName
		if i > 0 {
			fname_part += "." + fmt.Sprintf("%d", i)
		}

		//fmt.Printf("\n\n[llamaModelLoad] Loading model part %d / %d from '%s'\n", i+1, partsCount, fname_part)

		// --- Python ---
		// fout.write(struct.pack("iii", len(data.shape), len(sname), ftype_cur))
		// for dim in reversed(data.shape):
		//     fout.write(struct.pack("i", dim))
		// fout.write(sname)

		//fin = std::ifstream(fname_part, std::ios::binary);
		//fin.rdbuf()->pubsetbuf(f_buf.data(), f_buf.size());
		//fin.seekg(file_offset);

		// load weights
		{
			tensorsCount := uint32(0)

			for {
				dims := readInt(file)

				// FIXME Check for EOF
				//_, err := file.Seek(0, io.SeekCurrent)
				//if err == io.EOF {
				//if err != nil || dims > 2 {
				if dims == 0 || dims > 2 {
					//fmt.Printf("\n[STOP] Model was read...")
					break
				}

				length := readInt(file)
				ftype := readInt(file)
				nelements := uint32(1)
				//int32_t ne[2] = { 1, 1 };
				ne := [2]uint32{1, 1} // FIXME Why only 2 ??
				for i := uint32(0); i < dims; i++ {
					////fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
					ne[i] = readInt(file)
					////nelements *= ne[i]
					nelements *= ne[i]
				}

				name := readString(file, length)

				if ml.DEBUG {
					typeStr := "FP32"
					if ftype == 1 {
						typeStr = "FP16"
					}
					memStr := fmt.Sprintf("%dM", nelements*4/1024/1024) // FIXME element size
					fmt.Printf("\n=== LAYER #%d === %s | %s | %s ===", tensorsCount, typeStr, name, memStr)
				}

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
					if tensorSize/uint32(partsCount) != nelements {
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
						if tensor.NE[0]/uint32(partsCount) != ne[0] || tensor.NE[1] != ne[1] {
							fmt.Printf("\n[ERROR] Tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
								name, tensor.NE[0]/uint32(partsCount), tensor.NE[1], ne[0], ne[1])
							os.Exit(1)
							//return false;
						}
					} else {
						if tensor.NE[0] != ne[0] || tensor.NE[1]/uint32(partsCount) != ne[1] {
							fmt.Printf("\n[ERROR] Tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]",
								name, tensor.NE[0], tensor.NE[1]/uint32(partsCount), ne[0], ne[1])
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
				if dims == 1 || partsCount == 1 {
					////if (nelements*bpe)/ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
					////fmt.Printf("\n[ERROR] tensor '%s' has wrong size in model file: got %zu, expected %zu",
					////    __func__, name.data(), ggml_nbytes(tensor), nelements*bpe);
					////os.Exit(1)
					//return false;
					////}

					if part_id == 0 {

						if ftype == 1 { // --- FP16

							for n := uint32(0); n < tensorSize; n++ {
								tensor.Data[n] = readFP16ToFP32(file)
							}

						} else { // --- FP32

							var fake []byte

							fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
							// NB! unsafe.Pointer(tensor.Data) for *Data VS unsafe.Pointer(&tensor.Data) for Data
							dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))

							fakeHeader.Data = dataHeader.Data
							fakeHeader.Len = int(tensorSize * 4)
							fakeHeader.Cap = int(tensorSize * 4)

							// --- all tensors in file are aligned for 32 bytes

							alignment := int64(32)
							offset, _ := file.Seek(0, io.SeekCurrent)
							for ; offset%alignment != 0; offset++ {
							}
							file.Seek(offset, io.SeekStart)

							//fmt.Printf("\n== FAKE []BYTE LEN = %d", len(fake))
							if count, err := io.ReadFull(file, fake); err != nil || count != int(tensorSize*4) {
								fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
								fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
								os.Exit(1)
							}
						}

					} else {
						////fin.seekg(ggml_nbytes(tensor), std::ios::cur);
						fmt.Printf("\n[ERROR] The multi-part models are not supported yet")
						os.Exit(1)
					}

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

							offset := offset_row + uint32(part_id)*np0*ml.TYPE_SIZE[tensor.Type]
							fmt.Print(offset)

							////fin.read(reinterpret_cast<char *>(tensor->data) + offset, row_size/n_parts);
						}
					} else {
						np1 := ne[1]

						////const size_t row_size = (tensor->ne[0]/ggml_blck_size(tensor->type))*ggml_type_size(tensor->type);
						row_size := tensor.NE[0] * ml.TYPE_SIZE[tensor.Type]

						for i1 := uint32(0); i1 < ne[1]; i1++ {
							////const size_t offset_row = (i1 + part_id*np1)*row_size;
							offset_row := (i1 + uint32(part_id)*np1) * row_size
							////fin.read(reinterpret_cast<char *>(tensor->data) + offset_row, row_size);
							fmt.Print(offset_row)
						}
					}

					////total_size += ggml_nbytes(tensor)/n_parts;
				}

				//fmt.Printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ftype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);

				tensorsCount++
				model.loadedCount++
				if !silent && runtime.GOOS != "windows" {
					bar.Add(1)
				}
			}
		}

		if !silent && runtime.GOOS != "windows" {
			bar.Finish()
		}
	}

	return lctx, nil
}

func max(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

// NB! INT = 32 bits
func readInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

func readString(file *os.File, len uint32) string {
	buf := make([]byte, len)
	if count, err := file.Read(buf); err != nil || count != int(len) {
		return ""
	}
	return string(buf)
}

func readFP16ToFP32(file *os.File) float32 {
	buf := make([]byte, 2)
	if count, err := file.Read(buf); err != nil || count != 2 {
		return 0.0
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32()
}

func readFP32(file *os.File) float32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0.0
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	return math.Float32frombits(bits)
}

func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}
