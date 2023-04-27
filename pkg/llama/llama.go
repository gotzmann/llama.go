package llama

import (
	"container/ring"
	"fmt"
	"io"
	"math"
	"math/rand"
	"os"
	"reflect"
	"runtime"
	"sort"
	"time"
	"unsafe"

	//progressbar "github.com/schollz/progressbar/v3"
	"github.com/mattn/go-colorable"
	"github.com/mitchellh/colorstring"
	"github.com/x448/float16"
	"golang.org/x/exp/slices"

	"github.com/gotzmann/llama.go/pkg/ml"
)

const (
	LLAMA_FILE_VERSION           = 1
	LLAMA_FILE_MAGIC             = 0x67676a74 // 'ggjt' in hex
	LLAMA_FILE_MAGIC_OLD         = 0x67676d66 // 'ggmf' in hex
	LLAMA_FILE_MAGIC_UNVERSIONED = 0x67676d6c // 'ggml' pre-versioned files
)

type ModelParams struct {
	Model  string // model path
	Prompt string

	MaxThreads int

	UseAVX  bool
	UseNEON bool

	Seed         int
	PredictCount uint32 // new tokens to predict
	RepeatLastN  uint32 // last n tokens to penalize
	PartsCount   int    // amount of model parts (-1 = determine from model dimensions)
	CtxSize      uint32 // context size
	BatchSize    uint32 // batch size for prompt processing
	KeepCount    uint32

	// --- sampling parameters

	TopK          uint32  // 40
	TopP          float32 // 0.95
	Temp          float32 // 0.80
	RepeatPenalty float32 // 1.10

	InputPrefix string   // string to prefix user inputs with
	Antiprompt  []string // string upon seeing which more user input is prompted

	MemoryFP16   bool // use f16 instead of f32 for memory kv
	RandomPrompt bool // do not randomize prompt if none provided
	UseColor     bool // use color to distinguish generations and inputs
	Interactive  bool // interactive mode

	Embedding        bool // get only sentence embedding
	InteractiveStart bool // wait for user input immediately

	Instruct   bool // instruction mode (used for Alpaca models)
	IgnoreEOS  bool // do not stop generating after eos
	Perplexity bool // compute perplexity over the prompt
	UseMLock   bool // use mlock to keep model in memory
	MemTest    bool // compute maximum memory usage

	VerbosePrompt bool
}

// pair is a C++ inspired struct
type pair struct {
	first  float32
	second uint32
}

// Context is the context of the model.
type Context struct {
	kvSelf    KVCache   // key-value store for the self attention
	Logits    []float32 // decode output 2D array [tokensCount][vocabSize]
	Embedding []float32 // input embedding 1D array [embdSize]
	MLContext *ml.Context
}

// NewContext creates a new context.
func NewContext(model *Model, params *ModelParams) *Context {
	dt := ml.TYPE_F32
	size := model.hparams.embdSize * model.hparams.layersCount * params.CtxSize
	return &Context{
		kvSelf: KVCache{
			K: ml.NewTensor1D(nil, dt, size), // Fixed OK
			V: ml.NewTensor1D(nil, dt, size), // Fixed OK
		},
		Logits:    make([]float32, model.hparams.vocabSize, model.hparams.vocabSize),
		Embedding: make([]float32, 0, 0), // FIXME: vocab.Size ?
		MLContext: ml.NewContext(params.MaxThreads, params.UseAVX, params.UseNEON),
	}
}

func (ctx *Context) ReleaseContext() {
	// not sure if it makes sense to nil explicitly
	ctx.kvSelf.K = nil
	ctx.kvSelf.V = nil
	ctx.Logits = nil
	ctx.Embedding = nil
	// close sync channel and stop compute workers
	ctx.MLContext.ReleaseContext()
}

// ContextParams are the parameters for the context.
// struct llama_context_params {
type ContextParams struct {
	CtxSize    uint32 // text context
	PartsCount int    // -1 for default
	Seed       int    // RNG seed, 0 for random
	LogitsAll  bool   // the llama_eval() call computes all logits, not just the last one
	VocabOnly  bool   // only load the vocabulary, no weights
	UseLock    bool   // force system to keep model in RAM
	Embedding  bool   // embedding mode only
}

// Layer is a single layer of the model.
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

// HParams are the hyperparameters of the model (LLaMA-7B commented as example).
type HParams struct {
	ctxSize     uint32
	vocabSize   uint32 // 32000
	embdSize    uint32 // 4096
	multSize    uint32 // 256
	headsCount  uint32 // 32
	layersCount uint32 // 32
	rotCount    uint32 // 64
	f16         uint32
}

// ModelType is the type of the model.
type ModelType uint8

// available llama models
const (
	MODEL_UNKNOWN ModelType = iota
	MODEL_7B
	MODEL_13B
	MODEL_30B
	MODEL_65B
)

// KVCache is a key-value cache for the self attention.
type KVCache struct {
	K *ml.Tensor
	V *ml.Tensor

	N uint32 // number of tokens currently in the cache
}

// Model is the representation of any NN model (and LLaMA too).
type Model struct {
	Type    ModelType
	ctx     *ml.Context
	hparams *HParams

	tokEmbeddings *ml.Tensor
	norm          *ml.Tensor
	output        *ml.Tensor

	layers []Layer

	tensors map[string]*ml.Tensor
}

// NewModel creates a new model with default hyperparameters.
func NewModel(params *ModelParams) *Model {
	return &Model{
		hparams: &HParams{
			ctxSize: params.CtxSize,
		},
		layers:  make([]Layer, 0),
		tensors: make(map[string]*ml.Tensor),
	}
}

// Eval runs one inference iteration over the LLaMA model
// lctx = model context with all LLaMA data
// tokens = new batch of tokens to process
// pastCount = the context size so far
// params = all other parameters like max threads allowed, etc
func Eval(
	lctx *Context,
	vocab *ml.Vocab,
	model *Model,
	tokens []uint32,
	pastCount uint32,
	params *ModelParams,
) error {

	N := uint32(len(tokens))
	kvSelf := lctx.kvSelf

	embdSize := model.hparams.embdSize
	layersCount := model.hparams.layersCount
	ctxSize := model.hparams.ctxSize
	headsCount := model.hparams.headsCount
	vocabSize := model.hparams.vocabSize
	rotCount := model.hparams.embdSize / model.hparams.headsCount

	ctx0 := lctx.MLContext

	graph := &ml.Graph{
		//MaxThreads: params.MaxThreads,
		//UseNEON:    params.UseNEON,
		//UseAVX:     params.UseAVX,
	}

	// Initialize the embd tensor with the tokensFloat32 data
	embd := ml.NewTensor1D(ctx0, ml.TYPE_F32, uint32(len(tokens))) // Reusable OK
	for i, token := range tokens {
		embd.Data[i] = float32(token)
	}

	inpL := ml.GetRows(ctx0, model.tokEmbeddings, embd)

	for il := uint32(0); il < layersCount; il++ {

		//if il > 0 {
		//	break // DEBUG
		//}

		inpSA := inpL

		// norm
		cur := ml.RMSNorm(ctx0, inpL)

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

				// NB! ggml_element_size(kv_self.k) = 2 for FP16
				k := ml.View1D(ctx0, kvSelf.K, N*embdSize, embdSize*(il*ctxSize+pastCount))
				v := ml.View1D(ctx0, kvSelf.V, N*embdSize, embdSize*(il*ctxSize+pastCount))

				ml.BuildForwardExpand(graph, ml.Copy(ctx0, Kcur, k))
				ml.BuildForwardExpand(graph, ml.Copy(ctx0, Vcur, v))
			}

			Q :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Copy(ctx0,
							Qcur,
							ml.NewTensor3D(ctx0, ml.TYPE_F32, embdSize/headsCount, headsCount, N)), // Reusable OK
						pastCount, rotCount, 0),
					0, 2, 1, 3)

			K :=
				ml.Permute(ctx0,
					ml.Rope(ctx0,
						ml.Reshape3D(ctx0,
							ml.View1D(ctx0, kvSelf.K, (pastCount+N)*embdSize, il*ctxSize*embdSize),
							embdSize/headsCount, headsCount, pastCount+N),
						pastCount, rotCount, 1),
					0, 2, 1, 3)

			// K * Q
			KQ := ml.MulMat(ctx0, K, Q)

			// KQ_scaled = KQ / sqrt(n_embd/n_head)
			KQScaled :=
				ml.Scale(ctx0,
					KQ,
					ml.NewFP32(ctx0, float32(1.0/math.Sqrt(float64(embdSize)/float64(headsCount)))),
				)

			// KQ_masked = mask_past(KQ_scaled)
			KQMasked := ml.DiagMaskInf(ctx0, KQScaled, pastCount)

			// KQ = soft_max(KQ_masked)
			KQSoftMax := ml.SoftMax(ctx0, KQMasked)

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
				ml.NewTensor2D(ctx0, ml.TYPE_F32, embdSize, N)) // Reusable OK

			// projection (no bias)
			cur = ml.MulMat(ctx0, model.layers[il].wo, cur)

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

			tmp := ml.MulMat(ctx0, model.layers[il].w3, cur)

			cur = ml.MulMat(ctx0, model.layers[il].w1, cur)

			// SILU activation
			cur = ml.Silu(ctx0, cur)

			cur = ml.Mul(ctx0, cur, tmp)

			cur = ml.MulMat(ctx0, model.layers[il].w2, cur)
		}

		cur = ml.Add(ctx0, cur, inpFF)

		// input for next layer
		inpL = cur
	}

	// --- norm

	inpL = ml.RMSNorm(ctx0, inpL)

	// inpL = norm*inpL
	inpL = ml.Mul(ctx0,
		ml.Repeat(ctx0, model.norm, inpL),
		inpL)

	embeddings := inpL

	// lm_head
	inpL = ml.MulMat(ctx0, model.output, inpL)

	// run the computation
	ml.BuildForwardExpand(graph, inpL)

	ml.GraphCompute(ctx0, graph)

	// --- extract logits

	// Copy only the relevant part of inpL.Data to lctx.Logits
	for i := uint32(0); i < vocabSize; i++ {
		srcIndex := vocabSize*(N-1) + i
		if i >= uint32(len(lctx.Logits)) || srcIndex >= uint32(len(inpL.Data)) {
			fmt.Println("Error: Index out of bounds during Logits copy")
			os.Exit(1)
		}
		lctx.Logits[i] = inpL.Data[srcIndex]
	}

	if ml.DEBUG {
		printTensor(inpL, "INPL")

		fmt.Printf("\n\n=== LOGITS === %d ===\n", len(lctx.Logits)) // DEBUG
		for ii := 0; ii < 13; ii++ {
			fmt.Printf("%.4f  ", lctx.Logits[ii])
		}
	}

	// --- extract embeddings

	if len(lctx.Embedding) > 0 {
		////memcpy(embedding_out.data(), (float *) ggml_get_data(embeddings) + (n_embd*(N - 1)), sizeof(float)*n_embd);
		for i := uint32(0); i < embdSize; i++ {
			lctx.Embedding[i] = embeddings.Data[(embdSize*(N-1))+i]
		}
	}

	// It really helps to eliminate degradation of performance when
	// the garbage collector do it job more often
	runtime.GC()

	return nil
}

// printTensor prints a tensor
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

// SampleTopPTopK samples next token given probabilities for each embedding:
//   - consider only the top K tokens
//   - from them, consider only the top tokens with cumulative probability > P
func SampleTopPTopK(
	logits []float32,
	lastNTokens *ring.Ring, // TODO: Use custom performant container
	lastNTokensSize uint32, // TODO: Remove
	topK uint32,
	topP float32,
	temp float32,
	repeatPenalty float32,
) uint32 {

	logitsCount := uint32(len(logits))

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
		extractedTokens := ExtractTokens(lastNTokens.Move(-int(lastNTokensSize)), int(lastNTokensSize))
		fmt.Printf("\n=== LAST N TOKENS | %d ===\n", len(extractedTokens))
		for i := 0; i < int(lastNTokensSize); i++ {
			fmt.Printf("%d ", extractedTokens[i])
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

	logitsID := make([]pair, 0, logitsCount)

	scale := float32(1.0 / temp)
	for i := uint32(0); i < logitsCount; i++ {

		// Repetition penalty from ctrl paper (https://arxiv.org/abs/1909.05858)
		// Credit https://github.com/facebookresearch/llama/compare/main...shawwn:llama:main

		// Check if the i-th token is present in the last_n_tokens ring buffer
		tokenExists := false
		// TODO: Ompimize [ 32,000 * 1024 ~ 100 ms ] loop with better data structure for lastNTokens
		lastNTokens.Do(func(p interface{}) {
			if p.(uint32) == i {
				tokenExists = true
			}
		})

		// If lastNTokens already contains i-th token, append it with repeat penalty
		if tokenExists {
			// If score < 0, then repetition penalty has to be multiplied to reduce the previous token probability
			if logits[i] < 0.0 {
				logitsID = append(logitsID, pair{logits[i] * scale * repeatPenalty, i})
			} else {
				logitsID = append(logitsID, pair{logits[i] * scale / repeatPenalty, i})
			}
			// Else append pair to logitsID, scaling probability
		} else {
			logitsID = append(logitsID, pair{logits[i] * scale, i})
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

	// --- sort logitsID slice and return only top K elements

	// std::partial_sort
	// Rearranges elements such that the range [first, middle) contains
	// the sorted middle − first smallest elements in the range [first, last).
	// The order of equal elements is not guaranteed to be preserved.
	// The order of the remaining elements in the range [middle, last) is unspecified.

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

	// Since logitsID is already sorted, the max value is the first element
	maxl := logitsID[0].first

	// Compute probabilities for the top k tokens
	probs := make([]float32, len(logitsID))

	sum := 0.0
	for i, kv := range logitsID {
		p := math.Exp(float64(kv.first - maxl))
		probs[i] = float32(p)
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
				probs = probs[:i+1]
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

	// --- Hand-crafted Discrete Distribution math - do we need something better?

	// Original C++ version with rng = std::mt19937
	// Mersenne Twister pseudo-random generator of 32-bit numbers with a state size of 19937 bits.

	// std::discrete_distribution<> dist(probs.begin(), probs.end());
	// int idx = dist(rng);
	// return logits_id[idx].second;

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
		fmt.Printf("\n=== PROVED === ")
		for i := 0; i < min(8, len(probs)); i++ {
			fmt.Printf("%.3f | ", probs[i])
		}
		fmt.Printf(" === idx = %d | logitsID = %d | weight = %.3f | ", idx, logitsID[idx].second, logitsID[idx].first)
	}

	/*
		// --- experimental approach seems doesn't work right yet

		rng := rand.New(source)

		cumulative := make([]float32, len(probs))
		cumulative[0] = probs[0]
		for i := 1; i < len(probs); i++ {
			cumulative[i] = cumulative[i-1] + probs[i]
		}

		target := rng.Float32() * cumulative[len(cumulative)-1]
		idx := sort.Search(len(cumulative), func(i int) bool { return cumulative[i] >= target })

		if ml.DEBUG {
			fmt.Printf("\n=== EXPERIMENTAL === ")
			for i := 0; i < min(8, len(probs)); i++ {
				fmt.Printf("%.3f | ", probs[i])
			}
			fmt.Printf(" === idx = %d | logitsID = %d | weight = %.3f | ", idx, logitsID[idx].second, logitsID[idx].first)
		}
	*/

	return logitsID[idx].second
}

// LoadModel loads a model's weights from a file
// See convert-pth-to-ggml.py for details on format
// func LoadModel(fileName string, params ModelParams, silent bool) (*Context, error) {
func LoadModel(fileName string, params *ModelParams, silent bool) (*ml.Vocab, *Model, error) {

	file, err := os.Open(fileName)
	if err != nil {
		return nil, nil, err
	}
	defer file.Close()

	// --- check header magic and format version

	magic := readInt(file)

	if magic == LLAMA_FILE_MAGIC_UNVERSIONED || magic == LLAMA_FILE_MAGIC_OLD {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Too old, regenerate!", fileName)
		return nil, nil, fmt.Errorf("invalid model file")
	}

	if magic != LLAMA_FILE_MAGIC {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Wrong MAGIC in header", fileName)
		return nil, nil, fmt.Errorf("invalid model file")
	}

	version := readInt(file)

	if version != LLAMA_FILE_VERSION {
		fmt.Printf("\n[ERROR] Invalid model file '%s'! Unsupported version", fileName)
		return nil, nil, fmt.Errorf("invalid model file")
	}

	// --- load hparams

	vocabSize := readInt(file)   // vocab_size
	embdSize := readInt(file)    // dim
	multSize := readInt(file)    // multiple_of
	headsCount := readInt(file)  // n_heads
	layersCount := readInt(file) // n_layers
	rotCount := readInt(file)    // [obsolete] rot = dim // n_heads
	f16 := readInt(file)         // ftype

	model := NewModel(params)

	model.hparams.vocabSize = vocabSize
	model.hparams.embdSize = embdSize
	model.hparams.multSize = multSize
	model.hparams.headsCount = headsCount
	model.hparams.layersCount = layersCount
	model.hparams.rotCount = rotCount
	model.hparams.f16 = f16

	ffSize := ((2*(4*embdSize)/3 + multSize - 1) / multSize) * multSize

	vocab := ml.NewVocab(vocabSize)

	if ml.DEBUG {
		fmt.Printf("\nvocab  = %d", vocabSize)
		fmt.Printf("\nembd   = %d", embdSize)
		fmt.Printf("\nmult   = %d", multSize)
		fmt.Printf("\nheads  = %d", headsCount)
		fmt.Printf("\nlayers = %d", layersCount)
		fmt.Printf("\nff     = %d", ffSize)
		fmt.Printf("\nrot    = %d", rotCount)
		fmt.Printf("\nf16    = %d", f16)
	}

	// --- load vocab

	if !silent && runtime.GOOS == "windows" {
		Colorize("[magenta][ INIT ][white] Loading vocab...")
	}
	/*
	       // https://pkg.go.dev/github.com/schollz/progressbar/v3#Option
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
	*/
	for i := uint32(0); i < vocabSize; i++ {

		//if !silent && runtime.GOOS != "windows" && i%100 == 0 {
		//	vocabBar.Set(int(i))
		//}

		length := readInt(file)
		token := readString(file, length)
		score := readFP32(file)

		vocab.Token2ID[token] = i
		vocab.ID2Token[i] = ml.TokenScore{Token: token, Score: score}
	}

	//if !silent && runtime.GOOS != "windows" {
	//	vocabBar.Finish()
	//	fmt.Printf("\n")
	//}

	// --- prepare memory for the weights
	{
		model.tokEmbeddings = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize) // Fixed OK

		model.norm = ml.NewTensor1D(nil, ml.TYPE_F32, embdSize)                        // Fixed OK
		model.output = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, vocabSize) // Fixed OK

		// map by name
		model.tensors["tok_embeddings.weight"] = model.tokEmbeddings

		model.tensors["norm.weight"] = model.norm
		model.tensors["output.weight"] = model.output

		model.layers = make([]Layer, layersCount)
		for i := uint32(0); i < layersCount; i++ {

			model.layers[i].attentionNorm = ml.NewTensor1D(nil, ml.TYPE_F32, embdSize) // Fixed OK

			model.layers[i].wq = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, embdSize) // Fixed OK
			model.layers[i].wk = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, embdSize) // Fixed OK
			model.layers[i].wv = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, embdSize) // Fixed OK
			model.layers[i].wo = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, embdSize) // Fixed OK

			model.layers[i].ffn_norm = ml.NewTensor1D(nil, ml.TYPE_F32, embdSize)

			model.layers[i].w1 = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, ffSize) // Fixed OK
			model.layers[i].w2 = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, ffSize, embdSize) // Fixed OK
			model.layers[i].w3 = ml.NewTensor2D(nil, ml.TYPE_F32 /*wtype*/, embdSize, ffSize) // Fixed OK

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

	if !silent /* && runtime.GOOS == "windows" */ {
		//Colorize("[magenta][ INIT ][white] Loading model - please wait ...")
		Colorize("[light_magenta][ INIT ][light_blue] Loading model, please wait ")
	}
	/*
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
	*/

	// --- load weights
	var tensorsCount uint32
	for {
		dims := readInt(file)
		if dims < 1 || dims > 2 { // TODO Check for EOF
			break
		}

		nameLength := readInt(file)
		shardType := ml.DType(readInt(file))

		nelements := 1
		ne := [2]uint32{1, 1}
		for i := 0; i < int(dims); i++ {
			ne[i] = readInt(file)
			nelements *= int(ne[i])
		}

		name := readString(file, nameLength)
		tensor, ok := model.tensors[name]
		if !ok {
			fmt.Printf("\n[ERROR] Unknown tensor '%s' in model file", name)
			os.Exit(1)
		}

		if ml.DEBUG {
			typeStr := "FP32"
			if shardType == ml.TYPE_F16 {
				typeStr = "FP16"
			}
			memStr := fmt.Sprintf("%dM", nelements*4/1024/1024)
			fmt.Printf("\n=== LAYER #%d === %s | %s | %s ===", tensorsCount, typeStr, name, memStr)
		}

		tensorSize := tensor.Nelements()

		// --- All tensors in file are aligned for 32 bytes

		// TODO: Align with one modulo operation
		alignment := int64(32)
		offset, _ := file.Seek(0, io.SeekCurrent)
		for ; offset%alignment != 0; offset++ {
		}
		_, err = file.Seek(offset, io.SeekStart)
		if err != nil {
			return nil, nil, err
		}

		// --- Read tensor into memory

		switch shardType {
		case ml.TYPE_F16:
			for n := uint32(0); n < tensorSize; n++ {
				tensor.Data[n] = readFP16ToFP32(file)
			}
		case ml.TYPE_F32:
			var fake []byte
			fakeHeader := (*reflect.SliceHeader)(unsafe.Pointer(&fake))
			dataHeader := (*reflect.SliceHeader)(unsafe.Pointer(&tensor.Data))

			fakeHeader.Data = dataHeader.Data
			fakeHeader.Len = int(tensorSize * 4)
			fakeHeader.Cap = int(tensorSize * 4)

			if count, err := io.ReadFull(file, fake); err != nil || count != int(tensorSize*4) {
				fmt.Printf("\n[ERROR] Failed to read BIG FP32 chunk from model!")
				fmt.Printf("\n[ERROR] COUNT = %d | ERR = %s", count, err.Error())
				os.Exit(1)
			}
		default:
			fmt.Printf("\n[ERROR] Tensor data type is not supported yet!")
			os.Exit(0)
		}

		// TODO: Implement just simple dots increasing count for Windows
		tensorsCount++
		if !silent && tensorsCount%10 == 0 {
			Colorize("[light_blue].")
		}
		// if !silent && runtime.GOOS != "windows" {
		// bar.Add(1)
		// }
	}

	// if !silent && runtime.GOOS != "windows" {
	// bar.Finish()
	// }

	return vocab, model, nil
}

// max returns the maximum of two float32 values
func max(a, b float32) float32 {
	if a >= b {
		return a
	}
	return b
}

// readInt reads 32-bit integer from the file
func readInt(file *os.File) uint32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0
	}
	return uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
}

// readString reads a string from the file
func readString(file *os.File, len uint32) string {
	buf := make([]byte, len)
	if count, err := file.Read(buf); err != nil || count != int(len) {
		return ""
	}
	return string(buf)
}

// readFP16ToFP32 reads a 16-bit float from the file and converts it to 32-bit
func readFP16ToFP32(file *os.File) float32 {
	buf := make([]byte, 2)
	if count, err := file.Read(buf); err != nil || count != 2 {
		return 0.0
	}
	bits := uint16(buf[1])<<8 | uint16(buf[0])
	f16 := float16.Frombits(bits)
	return f16.Float32()
}

// readFP32 reads a 32-bit float from the file
func readFP32(file *os.File) float32 {
	buf := make([]byte, 4)
	if count, err := file.Read(buf); err != nil || count != 4 {
		return 0.0
	}
	bits := uint32(buf[3])<<24 | uint32(buf[2])<<16 | uint32(buf[1])<<8 | uint32(buf[0])
	return math.Float32frombits(bits)
}

// ExtractTokens is a function to extract a slice of tokens from the ring buffer
func ExtractTokens(r *ring.Ring, count int) []uint32 {
	tokens := make([]uint32, count)
	for i := 0; i < count; i++ {
		tokens[i] = r.Value.(uint32)
		r = r.Next()
	}
	return tokens
}

// Colorize is a function to print colored text to the console
func Colorize(format string, opts ...interface{}) (n int, err error) {
	var DefaultOutput = colorable.NewColorableStdout()
	return fmt.Fprintf(DefaultOutput, colorstring.Color(format), opts...)
}

// min returns the minimum of a and b.
func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

// Resize() (safe) for using instead of C++ std::vector:resize()
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
