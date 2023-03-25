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
	Logits    []float32
	LogitsAll bool

	// input embedding (1-dimensional array: [n_embd])
	Embedding []float32
}

func NewContext() *Context {
	return &Context{
		Model:     NewModel(),
		Vocab:     ml.NewVocab(),
		Logits:    make([]float32, 0), // TODO Cap?
		Embedding: make([]float32, 0), // TODO Cap?
	}
}

// struct llama_context_params {
type ContextParams struct {
	n_ctx   int // text context
	n_parts int // -1 for default
	seed    int // RNG seed, 0 for random

	f16_kv    bool // use fp16 for KV cache
	logitsAll bool // the llama_eval() call computes all logits, not just the last one
	vocabOnly bool // only load the vocabulary, no weights
	use_mlock bool // force system to keep model in RAM
	embedding bool // embedding mode only

	// called with a progress value between 0 and 1, pass NULL to disable
	////llama_progress_callback progress_callback;
	// context pointer passed to the progress callback
	////void * progress_callback_user_data;
}

//
// interface implementation
//

// //struct llama_context * llama_init_from_file(
func InitFromFile(fileName string, params *ContextParams) (*Context, error) {
	////ggml_time_init();

	ctx := NewContext()

	////if (params.seed <= 0) {
	////params.seed = time(NULL);
	////}

	////ctx->rng = std::mt19937(params.seed);
	ctx.LogitsAll = params.logitsAll

	////ggml_type memory_type = params.f16_kv ? GGML_TYPE_F16 : GGML_TYPE_F32;

	err := LoadModel(fileName, ctx /*params.n_ctx,*/, uint32(params.n_parts), /*memory_type,*/
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

	// reserve memory for context buffers
	{
		////if (!kv_cache_init(ctx->model.hparams, ctx->model.kv_self, memory_type, ctx->model.hparams.n_ctx)) {
		////fprintf(stderr, "%s: kv_cache_init() failed for self-attention cache\n", __func__);
		////llama_free(ctx);
		////return nullptr;
		////}

		{
			////const size_t memory_size = ggml_nbytes(ctx->model.kv_self.k) + ggml_nbytes(ctx->model.kv_self.v);
			////fprintf(stderr, "%s: kv self size  = %7.2f MB\n", __func__, memory_size / 1024.0 / 1024.0);
		}

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
	}

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

type Model struct {

	//hparams llama_hparams hparams;

	////struct ggml_tensor * tok_embeddings;
	tokEmbeddings *ml.Tensor

	////struct ggml_tensor * norm;
	norm *ml.Tensor
	////struct ggml_tensor * output;
	output *ml.Tensor

	////std::vector<llama_layer> layers;
	layers []Layer

	// key + value memory
	////struct ggml_tensor * memory_k;
	memoryK *ml.Tensor
	////struct ggml_tensor * memory_v;
	memoryV *ml.Tensor

	ctx *ml.Context // ggml_context

	tensors map[string]*ml.Tensor //std::map<std::string, struct ggml_tensor *> tensors;
}

func NewModel() *Model {
	return &Model{
		layers:  make([]Layer, 0),
		tensors: make(map[string]*ml.Tensor),
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
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//

//bool llama_eval(
//    const llama_model & model,
//    const int n_threads,
//    const int n_past,
//    const std::vector<gpt_vocab::id> & embd_inp,
//          std::vector<float>         & embd_w,
//          size_t                     & mem_per_token) {

func Eval(
	lctx *Context,
	tokens []uint32,
	n_tokens uint32,
	n_past uint32,
	threadsCount uint32,
	/*
		model *Model,
		threads,
		n_past uint32,
		embdInp []uint32,
		embdW []float32,
		memPerToken *uint32*/) error {

	//N := uint32(len(embdInp))
	N := n_tokens

	// FIXME Load hyper parameters into model itself
	//const auto & hparams = model.hparams;

	model := lctx.Model
	////hparams := model.hparams

	embdSize := hparamsEmbd
	layers := hparamsLayers
	////ctx := hparamsCtx
	n_head := hparamsHeads
	vocabSize := 32000 // hparamsVocab
	rot := hparamsEmbd / hparamsHeads

	////auto & mem_per_token = lctx.mem_per_token;

	// TODO: fix this hardcoded size
	////static size_t buf_size = 2048u*1024*1024; // TMP !!!
	////static void * buf = malloc(buf_size);

	////if (mem_per_token > 0 && mem_per_token*N > buf_size) {
	////    const size_t buf_size_new = 1.3*(mem_per_token*N); // add 30% to account for ggml object overhead

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
	gf := ml.Graph{Threads: threadsCount}

	embd := ml.NewTensor1D(ctx0, ml.TYPE_I32, N) // FIXME Will be created as FP32 anyway

	////memcpy(embd->data, tokens, N*ggml_element_size(embd));
	// FIXME Refactore inline initialization
	embd.Type = ml.TYPE_F32
	for em := uint32(0); em < N; em++ {
		embd.Data[em] = float32(tokens[em])
	}

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layers; il++ {
		inpSA := inpL

		//var cur *ml.Tensor

		// norm
		cur := ml.RMSNorm(ctx0, inpL)

		// cur = attention_norm*cur
		rep := ml.Repeat(ctx0, model.layers[il].attentionNorm, cur)
		cur = ml.Mul(ctx0, rep, cur)

		fmt.Printf("\n[EVAL] Self-attention #%d...", il)

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

				////struct ggml_tensor * k = ggml_view_1d(ctx0, model.memory_k, N*n_embd, (ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past));
				k := ml.View1D(ctx0, model.memoryK, N*embdSize /*(ggml_element_size(model.memory_k)*n_embd)*(il*n_ctx + n_past)*/)

				v := ml.View1D(ctx0, model.memoryV, N*embdSize /*, (ggml_element_size(model.memory_v)*n_embd)*(il*n_ctx + n_past)*/)

				////ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
				ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Kcur, k)) // K

				ml.BuildForwardExpand(&gf, ml.Copy(ctx0, Vcur, v)) // V
			}

			// Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
			Q := ml.Permute(ctx0,
				ml.Rope(ctx0,
					ml.Copy(ctx0,
						Qcur,
						ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/n_head, n_head, N)),
					n_past, rot, 0),
				0, 2, 1, 3)

			// K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
			K := ml.Permute(ctx0,
				ml.Rope(ctx0,
					ml.Reshape3D(ctx0,
						ml.View1D(ctx0, model.memoryK, (n_past+N)*embdSize /*, il*n_ctx*ggml_element_size(model.memory_k)*n_embd*/),
						embdSize/n_head, n_head, n_past+N),
					n_past, rot, 1),
				0, 2, 1, 3)

			// K * Q
			////struct ggml_tensor * KQ = ggml_mul_mat(ctx0, K, Q);
			KQ := ml.MulMat(ctx0, K, Q)

			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewFP32(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(n_head)))),
				)

				// KQ_masked = mask_past(KQ_scaled)
				////struct ggml_tensor * KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, n_past)

			// KQ = soft_max(KQ_masked)
			////struct ggml_tensor * KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

			// V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
			VTrans :=
				ml.Copy(ctx0,
					ml.Permute(ctx0,
						ml.Reshape3D(ctx0, // FIXME down ^^^
							ml.View1D(ctx0, model.memoryV, (n_past+N)*embdSize), /* (n_past + N)*n_embd, il*n_ctx*ggml_element_size(model.memory_v)*n_embd)*/
							embdSize/n_head, n_head, n_past+N),
						1, 2, 0, 3),
					ml.NewTensor3D(ctx0, ml.TYPE_F32, n_past+N, embdSize/n_head, n_head))

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

		fmt.Printf("\n[EVAL] Feed-forward network #%d...", il)

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

	fmt.Printf("\n[EVAL] RMS Norm...")

	// used at the end to optionally extract the embeddings
	////var embeddings *ml.Tensor

	// --- norm
	{
		inpL = ml.RMSNorm(ctx0, inpL)

		// inpL = norm*inpL
		inpL = ml.Mul(ctx0,
			ml.Repeat(ctx0, model.norm, inpL),
			inpL)

		////embeddings = inpL
	}

	fmt.Printf("\n[EVAL] LM Head...")

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// logits -> probs
	//inpL = ggml_soft_max(ctx0, inpL);

	// run the computation
	fmt.Printf("\n[EVAL] BuildForwardExpand...")
	ml.BuildForwardExpand(&gf, inpL)

	fmt.Printf("\n[EVAL] GraphCompute...")
	ml.GraphCompute(ctx0, &gf)

	//if (n_past%100 == 0) {
	//    ggml_graph_print   (&gf);
	//    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
	//}

	//embd_w.resize(n_vocab*N);
	//memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

	// extract logits
	{
		logitsOut := lctx.Logits

		if lctx.LogitsAll {
			////logits_out.resize(n_vocab * N);
			logitsOut = Resize(logitsOut, vocabSize*int(N))
			////memcpy(logits_out.data(), (float *) ggml_get_data(inpL), sizeof(float)*n_vocab*N); // FIXME ASAP
		} else {

			// return result for just the last token
			////logits_out.resize(n_vocab);
			logitsOut = Resize(logitsOut, vocabSize)
			////memcpy(logits_out.data(), (float *) ggml_get_data(inpL) + (n_vocab*(N-1)), sizeof(float)*n_vocab); // FIXME ASAP
		}
	}

	// extract embeddings
	if len(lctx.Embedding) > 0 {
		embeddingOut := lctx.Embedding

		////embedding_out.resize(n_embd);
		embeddingOut = Resize(embeddingOut, int(embdSize))
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd); // FIXME ASAP
	}

	////if (mem_per_token == 0) {
	////    mem_per_token = ggml_used_mem(ctx0)/N;
	////}
	//fmt.Printf("used_mem = %zu\n", ggml_used_mem(ctx0));

	////ggml_free(ctx0);

	// measure the performance only for the single-token evals
	if N == 1 {
		////lctx.t_eval_us += ggml_time_us() - t_start_us;
		////lctx.n_eval++;
	}

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

	sort.Slice(logitsID[:topK], func(i, j int) bool {
		return logitsID[i].first < logitsID[j].first
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
	lctx Context,
	lastNTokens []uint32,
	topK uint32,
	topP float64,
	temp float64,
	repeatPenalty float64,
) uint32 {

	//-//auto & rng = lctx.rng;
	//-//const auto & vocab = lctx.vocab;
	vocab := lctx.Vocab
	//-//const auto & logits = lctx.logits;
	logits := lctx.Logits

	n_logits := uint32(len(vocab.ID2Token))

	////std::vector<std::pair<double, gpt_vocab::id>> logits_id;
	////logits_id.reserve(n_logits);
	////logitsID := make(map[float64]uint32, n_logits)
	logitsID := make([]pair, n_logits)

	{
		scale := float64(1.0) / temp
		for i := uint32(0); i < n_logits; i++ {
			// repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
			// credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main
			////if (std::find(last_n_tokens.begin(), last_n_tokens.end(), i) != last_n_tokens.end()) {
			if slices.IndexFunc(lastNTokens, func(el uint32) bool { return el == i }) != -1 {
				// if score < 0 then repetition penalty has to multiplied to reduce the previous token probability
				if logits[i] < 0.0 {
					////logits_id.push_back(std::make_pair(logits[i]*scale*repeat_penalty, i));
					logitsID = append(logitsID, pair{float64(logits[i]) * scale * repeatPenalty, i})
				} else {
					////logits_id.push_back(std::make_pair(logits[i]*scale/repeat_penalty, i));
					logitsID = append(logitsID, pair{float64(logits[i]) * scale / repeatPenalty, i})
				}
			} else {
				logitsID = append(logitsID, pair{float64(logits[i]) * scale, i})
			}
		}
	}

	sampleTopK(logitsID, topK)

	////double maxl = -INFINITY;
	maxl := math.Inf(-1)
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		//// maxl = std::max(maxl, kv.first);
		maxl = max(maxl, kv.first)
	}

	// compute probs for the top k tokens
	probs := make([]float64, 0, uint32(len(logitsID)))
	////probs.reserve(logits_id.size());

	sum := float64(0.0)
	////for (const auto & kv : logits_id) {
	for _, kv := range logitsID {
		// double p = exp(kv.first - maxl);
		p := math.Exp(kv.first - maxl)
		probs = append(probs, p)
		sum += p
	}

	// normalize the probs
	for i, _ := range probs {
		probs[i] = probs[i] / sum
	}

	if topP < 1.0 {
		cumsum := float64(0.0)
		for i := uint32(0); i < uint32(len(probs)); i++ {
			cumsum += probs[i]
			if cumsum >= topP {
				////probs.resize(i + 1)
				////logits_id.resize(i + 1)
				break
			}
		}

		cumsum = 1.0 / cumsum
		for i := uint32(0); i < uint32(len(probs)); i++ {
			probs[i] *= cumsum
		}
	}

	//printf("\n");
	//for (int i = 0; i < (int) 10; i++) {
	//    printf("%d: '%s' %f\n", i, vocab.id_to_token.at(logits_id[i].second).c_str(), probs[i]);
	//}
	//printf("\n\n");
	//exit(0);

	////std::discrete_distribution<> dist(probs.begin(), probs.end());

	// std::mt19937(since C++11) class is a very efficient pseudo-random number generator
	// and is defined in a random header file. It produces 32-bit pseudo-random numbers
	////idx := dist(rng)

	////v, _ := logitsID[idx]
	////return logitsID[idx].second;
	return logitsID[0].second // FIXME ASAP
}

// llama_model_load
// load the model's weights from a file
// WAS func LoadModel(fileName string, model *Model, vocab *ml.Vocab) error {

func LoadModel(
	fileName string, //const std::string & fname,
	lctx *Context,
	////n_ctx uint32,
	n_parts uint32,
	////ggml_type memory_type,
	vocabOnly bool,
	////llama_progress_callback progress_callback,
	////void *progress_callback_user_data
) error {

	fmt.Printf("\n[LoadModel] Loading model from '%s' - please wait ...\n", fileName)

	model := lctx.Model
	vocab := lctx.Vocab

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

	// load hparams
	{
		hparamsVocabSize, _ = readInt(reader) // vocab_size
		hparamsEmbd, _ = readInt(reader)      // dim
		hparamsMult, _ = readInt(reader)      // multiple_of
		hparamsHeads, _ = readInt(reader)     // n_heads
		hparamsLayers, _ = readInt(reader)    // n_layers
		hparamsRot, _ = readInt(reader)       // rot = dim // n_heads [obsolete]
		hparamsF16, _ = readInt(reader)       // ftype

		//hparamsCtx = n_ctx

		//n_ff = ((2*(4*hparams.n_embd)/3 + hparams.n_mult - 1)/hparams.n_mult)*hparams.n_mult;
		//n_ff := ((2*(4*hparamsEmbd)/3 + hparamsMult - 1) / hparamsMult) * hparamsMult

		//n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
		//////////////////////////////////////////////////n_parts = llamaParts[hparamsEmbd]

		////if (n_parts < 1) {
		////n_parts = LLAMA_N_PARTS.at(hparams.n_embd);
		////}

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

	n_ff := ((2*(4*hparamsEmbd)/3 + hparamsMult - 1) / hparamsMult) * hparamsMult

	// --- load vocab

	fmt.Printf("\n\n[LoadModel] Loading vocab...")

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
		vocab.ID2Token[i] = ml.TokenScore{Token: word}
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
	ctxSize := uint32(0)
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

		model.layers = make([]Layer, layers)
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

	for i := 0; i < int(n_parts); /*++i*/ i++ {

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
				////if n_tensors%8 == 0 {
				////fmt.Printf(".")
				////fflush(stderr);
				////}

			}

			////fmt.Printf("\ndone")

			////fmt.Printf("\nmodel size = %.2f MB / num tensors = %d", total_size/1024.0/1024.0, n_tensors)
		}

		////fin.close();
	}

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
