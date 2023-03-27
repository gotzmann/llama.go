package llama

import (
	"bufio"
	"fmt"
	"io"
	"log"
	"math"
	"os"
	"reflect"
	"sort"
	"strings"
	"unsafe"

	"github.com/x448/float16"
	"golang.org/x/exp/slices"

	"github.com/gotzmann/llama.go/ml"
)

const (
	LLAMA_FILE_VERSION           = 1
	LLAMA_FILE_MAGIC             = 0x67676d66 // 'ggmf' in hex
	LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c // pre-versioned files
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
	first  float64
	second uint32
}

type Context struct {
	////std::mt19937 rng;

	////int64_t t_load_us = 0;
	////int64_t t_start_us = 0;

	////int64_t t_sample_us = 0;
	////int64_t t_eval_us   = 0;

	n_sample uint32 // number of tokens sampled
	n_eval   uint32 // number of eval calls

	Model *Model
	Vocab *ml.Vocab

	////size_t mem_per_token = 0;

	// decode output (2-dimensional array: [n_tokens][n_vocab])
	Logits    *[]float32
	LogitsAll bool

	// input embedding (1-dimensional array: [n_embd])
	Embedding *[]float32
}

func NewFloatSlice(len, cap uint32) *[]float32 {
	slice := make([]float32, len, cap)
	return &slice
}

func NewContext() *Context {
	return &Context{
		Model:     NewModel(),
		Vocab:     ml.NewVocab(0),
		Logits:    NewFloatSlice(0, 0),
		Embedding: NewFloatSlice(0, 0),
	}
}

// struct llama_context_params {
type ContextParams struct {
	////n_ctx   int // text context
	partsCount int // -1 for default
	seed       int // RNG seed, 0 for random

	////f16_kv    bool // use fp16 for KV cache
	logitsAll bool // the llama_eval() call computes all logits, not just the last one
	vocabOnly bool // only load the vocabulary, no weights
	use_mlock bool // force system to keep model in RAM
	embedding bool // embedding mode only

	// called with a progress value between 0 and 1, pass NULL to disable
	////llama_progress_callback progress_callback;
	// context pointer passed to the progress callback
	////void * progress_callback_user_data;
}

func NewContextParams() ContextParams {
	return ContextParams{
		partsCount: -1,
	}
}

func KVCacheInit(hparams *HParams, cache *KVCache, dt ml.DType /*, int n_ctx*/) error {
	////const int n_embd  = hparams.n_embd;
	////const int n_layer = hparams.n_layer;

	////const int n_mem      = n_layer*n_ctx;
	////const int n_elements = n_embd*n_mem;

	////cache.buf.resize(2u*n_elements*ggml_type_size(wtype) + 2u*MB);

	////struct ggml_init_params params;
	////params.mem_size   = cache.buf.size();
	////params.mem_buffer = cache.buf.data();

	////cache.ctx = ggml_init(params);

	////if (!cache.ctx) {
	////	fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
	////	return false;
	////}

	////cache.k = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);
	////cache.v = ggml_new_tensor_1d(cache.ctx, wtype, n_elements);

	count := hparams.embdSize * hparams.layersCount /* *n_ctx */
	cache.K = ml.NewTensor1D(nil, dt, count)
	cache.V = ml.NewTensor1D(nil, dt, count)
	// FIXME Should we alter cache.K ??

	return nil
}

// //struct llama_context * llama_init_from_file(
func InitFromFile(fileName string, params *ContextParams) (*Context, error) {
	////ggml_time_init();

	// FIXME Calculate model parts number from defaults ??
	ctx := NewContext()

	////if (params.seed <= 0) {
	////params.seed = time(NULL);
	////}

	////ctx->rng = std::mt19937(params.seed);
	ctx.LogitsAll = params.logitsAll

	////ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

	err := LoadModel(fileName, ctx /*params.n_ctx,*/, params.partsCount, /*memory_type,*/
		params.vocabOnly, /*params.progress_callback,
		params.progress_callback_user_data*/)

	if err != nil {
		fmt.Printf("\n[ERROR] Failed to load LLaMMA model!")
		////llama_free(ctx);
		return nil, err
	}

	////if (params.use_mlock) {
	////char *err;
	////if (!ggml_mlock(ctx->model.ctx, &err)) {
	////fprintf(stderr, "%s\n", err);
	////free(err);
	////llama_free(ctx);
	////return nullptr;
	////}
	////}

	// --- reserve memory for context buffers

	////{
	////if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
	////fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
	////llama_free(ctx);
	////return nullptr;
	////}

	// kv_cache_init
	KVCacheInit(&ctx.Model.hparams, &ctx.Model.kvSelf, ml.TYPE_F32 /*, ctx.Model.hparams.n_ctx*/)

	////{
	////const size_t memory_size = ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
	////fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
	////}

	////const auto & hparams = ctx->model.hparams;
	////if (params.logits_all) {
	////ctx->logits.reserve(hparams.n_ctx*hparams.n_vocab);
	////} else {
	////ctx->logits.reserve(hparams.n_ctx);
	////}

	////if (params.embedding){
	///ctx->embedding.reserve(hparams.n_embd);
	////}

	////ctx->buf_compute.resize(MEM_REQ_EVAL.at(ctx->model.type));

	////ctx->buf_scratch[0].resize(MEM_REQ_SCRATCH0.at(ctx->model.type));
	////ctx->buf_scratch[1].resize(MEM_REQ_SCRATCH1.at(ctx->model.type));
	////}

	return ctx, nil
}

type Layer struct {

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

// default hparams (LLaMA 7B)
type HParams struct {
	////int32_t n_ctx   = 512;   // this is provided as user input?
	vocabSize   uint32 // = 32000;
	embdSize    uint32 //  = 4096;
	multSize    uint32 //  = 256;
	headsCount  uint32 //  = 32;
	layersCount uint32 // = 32;
	rotCount    uint32 //   = 64;
	f16         uint32 //    = 1;
}

func NewHparams() HParams {
	return HParams{
		vocabSize:   32000,
		embdSize:    4096,
		multSize:    256,
		headsCount:  32,
		layersCount: 32,
		rotCount:    64,
		f16:         1,
	}
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

	////ctx *ml.Context
	////std::vector<uint8_t> buf;

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

	/*
		// key + value memory
		////struct ggml_tensor * memory_k;
		memoryK *ml.Tensor
		////struct ggml_tensor * memory_v;
		memoryV *ml.Tensor
	*/
}

func NewModel() *Model {
	return &Model{
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

// Safe Resize() for using instead of C++ std::vector:resize()
// https://go.dev/play/p/VlQ7N75E5AD
func Resize(slice []float32, size int) []float32 {
	newSlice := make([]float32, size)
	for i := 0; i < min(size, len(slice)); i++ {
		newSlice[i] = slice[i]
	}
	return newSlice
}

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
	threadsCount uint32) error {

	N := tokensCount
	model := lctx.Model
	kvSelf := model.kvSelf

	fmt.Printf("\n=== N = %d", N)
	//// LLAMA_ASSERT(!!kv_self.ctx);

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	////ctx := hparamsCtx
	headsCount := model.hparams.headsCount
	vocabSize := model.hparams.vocabSize
	rotCount := model.hparams.embdSize / model.hparams.headsCount

	////auto & mem_per_token = lctx.mem_per_token;
	////auto & buf_compute   = lctx.buf_compute;

	////struct ggml_init_params params = {
	////    /*.mem_size   =*/ buf_compute.size(),
	////    /*.mem_buffer =*/ buf_compute.data(),
	////};

	////struct ggml_context * ctx0 = ggml_init(params);
	ctx0 := ml.Init(ml.InitParams{})

	// for big prompts, if BLAS is enabled, it is better to use only one thread
	// otherwise, the threads are spin-lock waiting for the BLAS calls and are degrading the performance
	////ggml_cgraph gf = {};
	////gf.n_threads = N > 255 && ggml_cpu_has_blas() ? 1 : n_threads;
	gf := ml.Graph{ThreadsCount: threadsCount}

	embd := ml.NewTensor1D(ctx0, ml.TYPE_F32 /*ml.TYPE_I32*/, N) // FIXME Will be created as FP32 anyway
	////memcpy(embd->data, tokens, N*ggml_element_size(embd));
	// FIXME Refactore inline initialization
	for id := uint32(0); id < N; id++ {
		(*embd.Data)[id] = float32(tokens[id]) // FIXME copy() for slices
	}

	fmt.Printf("\n\n=== EMBD === LEN = %d * %d\n", embd.NE[0], embd.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| EMBD[%d] = %f |", ii, (*embd.Data)[ii])
	}

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	fmt.Printf("\n\n=== INPL 01 === LEN = %d * %d\n", inpL.NE[0], inpL.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| INPL[%d] = %f |", ii, (*inpL.Data)[ii])
	}

	////fmt.Printf("\n\nmodel.tokEmbeddings = %+v", model.tokEmbeddings) // DEBUG

	for il := uint32(0); il < layersCount; il++ {

		inpSA := inpL
		cur := &ml.Tensor{}

		// norm
		cur = ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)
		cur = ml.Mul(ctx0, rep, cur)

		////////////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] Self-attention #%d...", il)

		// self-attention
		{
			Qcur := ml.MulMat(ctx0, model.layers[il].wq, cur)
			Kcur := ml.MulMat(ctx0, model.layers[il].wk, cur)
			Vcur := ml.MulMat(ctx0, model.layers[il].wv, cur)

			//fmt.Printf("\n\nOK\n %+v %+v %+v", Qcur, Kcur, Vcur)
			//os.Exit(0)

			// store key and value to memory
			if N >= 1 {

				// !!! FIXME !!!
				/*
					////struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
					k := ml.View1D(ctx0, model.memoryK, N*embdSize / *(ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past)* /)
					v := ml.View1D(ctx0, model.memoryV, N*embdSize / *, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past)* /)
					////ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
					ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Kcur, k)) // K
					ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Vcur, v)) // V
				*/
				k := ml.View1D(ctx0, kvSelf.K, N*embdSize /*, (ggml_element_size(kv_self.k)*n_embd)*(il*n_ctx + n_past)*/)
				v := ml.View1D(ctx0, kvSelf.V, N*embdSize /*, (ggml_element_size(kv_self.v)*n_embd)*(il*n_ctx + n_past)*/)

				ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Vcur, v))
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
							ml.View1D(ctx0, kvSelf.K, (pastCount+N)*embdSize /*, il*n_ctx*ggml_element_size(model.memory_k)*n_embd*/),
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
							ml.View1D(ctx0, kvSelf.V, (pastCount+N)*embdSize), /* (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd)*/
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

		/////////////////////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] Feed-forward network #%d...", il)

		////lctx.use_buf(ctx0, 1);

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

	////lctx.use_buf(ctx0, 0);

	/////////////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] RMS Norm...")

	// used at the end to optionally extract the embeddings
	////var embeddings *ml.Tensor

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	//fmt.Printf("\n\n=== INPL 05 === LEN = %d\n", len(inpL.Data)) // DEBUG
	fmt.Printf("\n\n=== INPL 05 === LEN = %d * %d\n", inpL.NE[0], inpL.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| INPL[%d] = %f |", ii, (*inpL.Data)[ii])
	}

	////embeddings := inpL

	///////////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] LM Head...")

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	//fmt.Printf("\n\n=== INPL 06 === LEN = %d\n", len(inpL.Data)) // DEBUG
	fmt.Printf("\n\n=== INPL 06 === LEN = %d * %d\n", inpL.NE[0], inpL.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| INPL[%d] = %f |", ii, (*inpL.Data)[ii])
	}

	////lctx.use_buf(ctx0, -1);

	// logits -> probs
	// COMMentED inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	////////////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] BuildForwardExpand...")
	ml.BuildForwardExpand(&gf, inpL)

	///////////////////////////////////////////////////////////////////fmt.Printf("\n[EVAL] GraphCompute...")
	ml.GraphCompute(ctx0, &gf)

	//fmt.Printf("\n\n=== INPL 08 === LEN = %d\n", len(inpL.Data)) // DEBUG
	fmt.Printf("\n\n=== INPL 08 === LEN = %d * %d\n", inpL.NE[0], inpL.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| INPL[%d] = %f |", ii, (*inpL.Data)[ii])
	}

	// COMMenteD  if (n_past%100 == 0) {
	// COMMenteD    ggml_graph_print   (&gf);
	// COMMenteD    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
	// COMMenteD }

	// COMMenteD embd_w.resize(n_vocab*N);
	// COMMenteD  memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

	// --- extract logits

	logitsOut := *lctx.Logits // FIXME ASAP What we'll doing with this? Just lost in thin air?

	fmt.Printf("\n\n=== INPL 09 === LEN = %d * %d\n", inpL.NE[0], inpL.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| INPL[%d] = %f |", ii, (*inpL.Data)[ii])
	}

	fmt.Printf("\n\n=== BEFORE === len(logitsOut) = %d\n", len(logitsOut)) // DEBUG
	for ii := 0; ii < 7; ii++ {
		fmt.Printf("| logitsOut[%d] = %f |", ii, logitsOut[ii])
	}

	if lctx.LogitsAll {

		fmt.Print("\n[HALT] Not Expected : lctx.LogitsAll == true")
		os.Exit(1)
		////logits_out.resize(n_vocab * N);
		///////////////////////////////////////////////////////////logitsOut = Resize(logitsOut, int(vocabSize*N)) // FIXME ASAP Why N multipy?
		////memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N);
		// FIXME Double Check !! Replace with copy() for slices
		for i := uint32(0); i < vocabSize*N; i++ {
			logitsOut[i] = (*inpL.Data)[i] // FIXME ASAP Overflow ??
		}

	} else {

		// return result for just the last token
		////logits_out.resize(n_vocab);
		////////////////////////////////////////////////////////logitsOut = Resize(logitsOut, int(vocabSize))

		// FIXME ASAP
		////logitsOut = NewFloatSlice(vocabSize, vocabSize) // FIXME Duplicate rearrangment?

		////memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab);
		// FIXME Double Check !! Replace with copy() for slices

		// FIXME ASAP Logits LEN = 32,000 | INPL LEN = 256,000
		for i := uint32(0); i < vocabSize; i++ {
			logitsOut[i] = (*inpL.Data)[i]
		}
	}

	fmt.Printf("\n\n=== AFTER === len(logitsOut) = %d\n", len(logitsOut)) // DEBUG
	for ii := 0; ii < 7; ii++ {
		fmt.Printf("| logitsOut[%d] = %f |", ii, logitsOut[ii])
	}

	os.Exit(0) // DEBUG

	// --- extract embeddings

	if len(*lctx.Embedding) > 0 {
		embeddingOut := lctx.Embedding

		////embedding_out.resize(n_embd);
		//////////////////////////////embeddingOut = Resize(embeddingOut, int(embdSize)) // FIXME ASAP ^^^ down
		embeddingOut = NewFloatSlice(embdSize, embdSize)
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
		// FIXME ASAP Replace with copy() for slices
		for i := uint32(0); i < embdSize; i++ {
			(*embeddingOut)[i] = (*lctx.Embedding)[i]
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
	topK uint32,
	topP float64,
	temp float64,
	repeatPenalty float64,
) uint32 {

	////auto & rng = lctx.rng;

	////////////////////////////////logitsCount := uint32(len(vocab.ID2Token))
	logitsCount := lctx.Model.hparams.vocabSize
	logits := *lctx.Logits

	fmt.Printf("\nlogitsCount = %d", logitsCount)   // DEBUG
	fmt.Printf("\nlen(logits) = %d\n", len(logits)) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| logits[%d] = %f |", ii, (logits)[ii])
	}

	////const auto * plogits = logits.data() + logits.size() - n_logits;
	//plogits := logits[len(logits)-int(logitsCount):] // FIXME ASAP
	plogits := logits[:]

	////std::vector<std::pair<double, llama_vocab::id>> logits_id;
	////logits_id.reserve(n_logits);
	logitsID := make([]pair, 0, logitsCount) // FIXME LEN vs CAP

	{
		scale := 1.0 / temp
		for i := uint32(0); i < logitsCount; i++ {

			// repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
			// credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

			// if lastNTokens already contains i-th token, append it with repeat penatly
			////if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
			if slices.IndexFunc(lastNTokens, func(el uint32) bool { return el == i }) != -1 {
				// if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
				if plogits[i] < 0.0 {
					////logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
					logitsID = append(logitsID, pair{float64(plogits[i]) * scale * repeatPenalty, i})
				} else {
					////logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
					logitsID = append(logitsID, pair{float64(plogits[i]) * scale / repeatPenalty, i})
				}
				// else append pair to logitsID	scaling probability
			} else {
				logitsID = append(logitsID, pair{float64(plogits[i]) * scale, i})
			}
		}
	}

	// sort logitsID slice and return only top K elements
	sampleTopK(logitsID, topK)

	////double maxl = -INFINITY;
	maxl := math.Inf(-1)
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		//// maxl = std::max(maxl, kv.first);
		maxl = max(maxl, kv.first)
	}

	// compute probs for the top k tokens
	////probs.reserve(logits_id.size());
	probs := make([]float64, 0, len(logitsID)) // FIXME LEN vs CAP

	sum := 0.0
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		// double p = exp(kv.first - maxl);
		p := math.Exp(kv.first - maxl)
		probs = append(probs, p)
		sum += p
	}

	// normalize the probs
	for i := range probs {
		probs[i] = probs[i] / sum
	}

	if topP < 1.0 {
		cumsum := 0.0
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

	// COMMENTED printf("\n");
	// COMMENTED for (int i = 0; i < (int) 10; i++) {
	// COMMENTED    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
	// COMMENTED }
	// COMMENTED printf("\n\n");
	// COMMENTED exit(0);

	////std::discrete_distribution<> dist(probs.begin(), probs.end());
	////int idx = dist(rng);

	////return logits_id[idx].second;

	fmt.Printf("\nSampleTopPTopK = %d", logitsID[0].second) // DEBUG

	return logitsID[0].second // FIXME ASAP
}

// llama_model_load
// load the model's weights from a file
// WAS func LoadModel(fileName string, model *Model, vocab *ml.Vocab) error {

func LoadModel(
	fileName string, //const std::string & fname,
	lctx *Context,
	////n_ctx uint32,
	partsCount int,
	////ggml_type memory_type,
	vocabOnly bool,
	////llama_progress_callback progress_callback,
	////void *progress_callback_user_data
) error {

	fmt.Printf("\n[LoadModel] Loading model from '%s' - please wait ...\n", fileName)

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

	if magic == LLAMA_FILE_MAGIC_UNVERSIONED {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Too old, regenerate!", fileName)
		return fmt.Errorf("invalid model file")
	}

	if magic != LLAMA_FILE_MAGIC {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Wrong MAGIC in header", fileName)
		return fmt.Errorf("invalid model file")
	}

	version, _ := readInt(reader)

	if version != LLAMA_FILE_VERSION {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Unsupported version", fileName)
		return fmt.Errorf("invalid model file")
	}

	/*
		if magic != 0x67676d6c {
			fmt.Printf("\n[llamaModelLoad] Invalid model file '%s' (bad magic)", fileName)
			return nil // FIXME ERR
		} */

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

	///////////////////////////////////////////////////////////////////////var n_ff, n_parts uint32

	// --- load hparams

	vocabSize, _ := readInt(reader)   // vocab_size
	embdSize, _ := readInt(reader)    // dim
	multSize, _ := readInt(reader)    // multiple_of
	headsCount, _ := readInt(reader)  // n_heads
	layersCount, _ := readInt(reader) // n_layers
	rotCount, _ := readInt(reader)    // rot = dim // n_heads [obsolete]
	f16, _ := readInt(reader)         // ftype

	model := lctx.Model

	model.hparams.vocabSize = vocabSize
	model.hparams.embdSize = embdSize
	model.hparams.multSize = multSize
	model.hparams.headsCount = headsCount
	model.hparams.layersCount = layersCount
	model.hparams.rotCount = rotCount
	model.hparams.f16 = f16

	// NB! Do not try to resize / relocate secondary pointers
	lctx.Vocab = ml.NewVocab(vocabSize)
	vocab := lctx.Vocab

	lctx.Logits = NewFloatSlice(vocabSize, vocabSize) // FIXME ASAP

	//hparamsCtx = n_ctx

	//n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
	//n_ff := ((2*(4*hparamsEmbd)/3 + hparamsMult - 1) / hparamsMult) * hparamsMult

	if partsCount < 1 {
		partsCount = int(LLAMA_N_PARTS[embdSize])
	}

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

	fmt.Printf("\nvocab  = %d", vocabSize)
	fmt.Printf("\nembd   = %d", embdSize)
	fmt.Printf("\nmult   = %d", multSize)
	fmt.Printf("\nheads  = %d", headsCount)
	fmt.Printf("\nlayers = %d", layersCount)
	fmt.Printf("\nrot    = %d", rotCount)
	fmt.Printf("\nf16    = %d", f16)

	//fmt.Printf("\nctx   = %d", hparamsCtx)
	//fmt.Printf("\nn_ff    = %d", n_ff)
	//fmt.Printf("\nn_parts = %d", n_parts)

	n_ff := ((2*(4*embdSize)/3 + multSize - 1) / multSize) * multSize

	// --- load vocab

	fmt.Printf("\n\n[LoadModel] Loading vocab...\n")

	// --- Python ---
	// fout.write(struct.pack("i", len(text)))
	// fout.write(text)
	// fout.write(struct.pack("f", tokenizer.get_score(i)))

	// Allocate memory and increase len / cap for the whole space
	// FIXME Was already done with NewVocab() call
	/////////////////////////////////////////////////////////vocab.ID2Token = slices.Grow(vocab.ID2Token, int(vocabSize))
	///////////////////////////////////////////////////////////////////vocab.ID2Token = vocab.ID2Token[0:vocabSize:vocabSize]

	for i := uint32(0); i < vocabSize; i++ {

		len, _ := readInt(reader)
		//word := make([]byte, len)
		//if count, err := io.ReadFull(reader, word); err != nil || count != int(len) {
		//	fmt.Printf("\n[llamaModelLoad] Problem reading vocabulary from '%s'", fileName)
		//	return nil // FIXME ERR
		//}
		token := readString(reader, len)
		score := readFP32(reader)

		//if i%6 == 0 {
		//	fmt.Println()
		//}
		//fmt.Printf("| vocab[%d] = %s ] ", i, string(word))

		vocab.Token2ID[token] = i
		vocab.ID2Token[i] = ml.TokenScore{Token: token, Score: score}

		// DEBUG
		if i%1000 == 0 {
			fmt.Printf("| %+v ", ml.TokenScore{Token: token, Score: score}) // DEBUG
		}
	}

	//return nil

	// for the big tensors, we have the option to store the data in 16-bit floats or quantized
	// in order to save memory and also to speed up the computation
	//wtype := ml.TYPE_COUNT

	////switch (model.hparams.f16) {
	//// case 0: wtype = GGML_TYPE_F32;  break;

	////case 1: wtype = GGML_TYPE_F16;  break;

	/////////////////////////////////////////////////////////////////////wtype := ml.TYPE_F16 // FIXME dtype

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
	//////////////////////////////////////////////////////////////////typeSize := ml.TYPE_SIZE[wtype]
	////ctxSize := uint32(0)
	////const auto & hparams = model.hparams;
	/////////////////////////////////////////////////////////////////embd := hparamsEmbd
	////////////////////////////////////////////////////////////////layers := hparamsLayers
	////const int n_ctx   = hparams.n_ctx;
	///////////////////////////////////////////////////////////////////vocabSize := hparamsVocabSize

	////ctxSize += embd * vocabSize * typeSize                              /* ggml_type_sizef(wtype) */         // tok_embeddings
	////ctxSize += embd * 4                                                 /* ggml_type_sizef(GGML_TYPE_F32) */ // norm
	////ctxSize += embd * vocabSize * typeSize                              /* ggml_type_sizef(wtype) */         // output
	////ctxSize += layers * (embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */) // attention_norm

	////ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wq
	////ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wk
	////ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wv
	////ctxSize += layers * (embd * embd * typeSize /* ggml_type_sizef(wtype) */) // wo

	/////ctxSize += layers * (embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */) // ffn_norm

	////ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w1
	////ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w2
	////ctxSize += layers * (n_ff * embd * typeSize /* ggml_type_sizef(wtype) */) // w3

	////ctxSize += ctxSize * layers * embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */ // memory_k
	/////ctxSize += ctxSize * layers * embd * 4 /* ggml_type_sizef(GGML_TYPE_F32) */ // memory_v

	////ctxSize += (5 + 10*layers) * 256 // object overhead

	////fmt.Printf("\nggml ctx size = %.2f MB", float32(ctxSize)/(1024*1024))
	//}

	// --- create the ggml context
	////{
	//// lctx.model.buf.resize(ctx_size);

	////params := ml.InitParams{
	////MemSize:   uint64(ctxSize),
	////MemBuffer: nil,
	////}

	///////////////////////////////////////////////////////////////model.ctx = ml.Init(ml.InitParams{})
	////if model.ctx == nil {
	////fmt.Printf("\nggml_init() failed")
	////return nil // FIXME ERR
	////}
	////}

	// prepare memory for the weights
	{
		//const auto & hparams = model.hparams;

		////embd := EmbdSize
		////layers := hparamsLayers
		//ctxSize := hparamsCtx
		////vocabSize := hparamsVocabSize

		////model.layers.resize(layers) // FIXME ASAP

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

	////if (progress_callback) {
	////progress_callback(0.0, progress_callback_user_data);
	////}

	/* REMOVED FROM v2 ?

	// key + value memory
	{
		//const auto & hparams = model.hparams;

		////embd := hparamsEmbd
		////layers := hparamsLayers
		//ctxSize := hparamsCtx
		//mem := layers * ctxSize
		//elements := embd * mem
		elements := embdSize * layersCount // FIXME

		model.memoryK = ml.NewTensor1D(ctx, ml.TYPE_F32, elements)
		model.memoryV = ml.NewTensor1D(ctx, ml.TYPE_F32, elements)

		////memorySize = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

		////fmt.Printf("\nmemory_size = %8.2f MB, n_mem = %d\n", memorySize/1024.0/1024.0, mem);
	}
	*/
	////const size_t file_offset = fin.tellg();

	////fin.close();

	//std::vector<uint8_t> tmp;

	////tmp := []byte{}

	////if (progress_callback) {
	////progress_callback(0.0, progress_callback_user_data);
	////}

	for i := 0; i < partsCount; /*++i*/ i++ {

		part_id := i
		//commented const int part_id = n_parts - i - 1;

		fname_part := fileName
		if i > 0 {
			fname_part += "." + fmt.Sprintf("%d", i)
		}

		fmt.Printf("\n\n[llamaModelLoad] Loading model part %d / %d from '%s'\n", i+1, partsCount, fname_part)

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
				for i := uint32(0); i < dims; i++ {
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

				// DEBUG
				//fmt.Printf("\n\n=== Tensor # %d === [ %s | %s | dims = %d | n = %s ] ===\n\n", n_tensors, typeStr, name, dims, nStr)
				fmt.Printf("\n[ # %d | %s | %s | dims = %d | n = %s ]", n_tensors, typeStr, name, dims, nStr)
				//if n_tensors%3 == 0 {
				//	fmt.Printf("\n")
				//}

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

						////fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));
						// NB! ggml_nbytes == (ggml_nelements(tensor)*GGML_TYPE_SIZE[tensor->type])/GGML_BLCK_SIZE[tensor->type];
						//fmt.Printf("\n\nReading %d Tensor elements...\n", tensor.Nelements())

						//dataHeader := (*reflect.SliceHeader) (unsafe.Pointer(&tensor.Data))
						//dataHeader.Data

						if ftype == 1 { // --- FP16

							for n := uint32(0); n < tensorSize; n++ {
								(*tensor.Data)[n] = readFP16ToFP32(reader)
							}

						} else { // --- FP32

							var fake []byte

							fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
							// FIXME unsafe.Pointer(tensor.Data) VS unsafe.Pointer(&tensor.Data)
							// FIXME It's REALLY depends on how Data defined in Tensor struct (pointer VS simple slice)
							///////////////////////////dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))
							dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(tensor.Data))

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

				n_tensors++
				model.loadedCount++

				// progress
				////if (progress_callback) {
				////double current_file_progress = double(size_t(fin.tellg()) - file_offset) / double(file_size - file_offset);
				////double current_progress = (double(i) + current_file_progress) / double(n_parts);
				////progress_callback(current_progress, progress_callback_user_data);
				////}

				////if n_tensors%8 == 0 {
				////fmt.Printf(".")
				////fflush(stderr);
				////}

			}

			////fmt.Printf("\ndone")

			////fprintf(stderr, "%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size/1024.0/1024.0, model.n_loaded);
			////if (model.n_loaded == 0) {
			////fprintf(stderr, "%s: WARN no tensors loaded from model file - assuming empty model for testing\n", __func__);
			////} else if (model.n_loaded != (int) model.tensors.size()) {
			////fprintf(stderr, "%s: ERROR not all tensors loaded from model file - expected %zu, got %d\n", __func__, model.tensors.size(), model.n_loaded);
			////return false;
			////}
		}

		////fin.close();
	}

	////lctx.t_load_us = ggml_time_us() - t_start_us;

	////if (progress_callback) {
	////progress_callback(1.0, progress_callback_user_data);
	////}

	return nil
}

func max(a, b float64) float64 {
	if a >= b {
		return a
	}
	return b
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

/*
func readFloat(reader *bufio.Reader) (float32, error) {
	buf := make([]byte, 4)
	if count, err := io.ReadFull(reader, buf); err != nil || count != 4 {
		fmt.Print("\n[ERROR] Failed to read data from model")
		//os.Exit(1)
		return 0, err
	}
	return // uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0]), nil
} */

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
