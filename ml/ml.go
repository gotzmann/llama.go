package ml

import (
	"fmt"
	"os"
	// "atomic"
	// "fmt"
)

const (
	MAX_DIMS     = 4
	MAX_NODES    = 4096
	MAX_PARAMS   = 16
	MAX_CONTEXTS = 64
	MAX_OPT      = 4

	QK = 32 // quantization
)

type dtype uint8

// TODO FP8, BFLOAT16
const (
	TYPE_Q4_0 dtype = iota
	TYPE_Q4_1
	TYPE_I8
	TYPE_I16
	TYPE_I32
	TYPE_F16   // TODO FP16
	TYPE_F32   // TODO FP32
	TYPE_COUNT // NB! COUNT should be the last
)

var BLCK_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{QK, QK, 1, 1, 1, 1, 1}

var TYPE_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{ /* 4 + QK/2 */ 1 /* 4*2 + QK/2 */, 1, 1, 2, 4, 2, 4} // FIXME

func TypeSizeFloat(dt dtype) float32 {
	return float32(TYPE_SIZE[dt]) / float32(BLCK_SIZE[dt]) // FIXME
}

// available tensor operations
type optype uint8

const (
	OP_NONE optype = iota
	OP_DUP
	OP_ADD
	OP_SUB
	OP_MUL
	OP_DIV
	OP_SQR
	OP_SQRT
	OP_SUM
	OP_MEAN
	OP_REPEAT
	OP_ABS
	OP_SGN
	OP_NEG
	OP_STEP
	OP_RELU
	OP_GELU
	OP_SILU
	OP_NORM // normalize
	OP_RMS_NORM

	OP_MUL_MAT

	OP_SCALE
	OP_CPY
	OP_RESHAPE
	OP_VIEW
	OP_PERMUTE
	OP_TRANSPOSE
	OP_GET_ROWS
	OP_DIAG_MASK_INF
	OP_SOFT_MAX
	OP_ROPE
	OP_CONV_1D_1S
	OP_CONV_1D_2S

	OP_FLASH_ATTN
	OP_FLASH_FF

	OP_COUNT
)

// n-dimensional tensor
type Tensor struct {
	Type dtype

	Dims uint32
	NE   [MAX_DIMS]uint32 // number of elements
	NB   [MAX_DIMS]uint32 // stride in bytes:
	// nb[0] = sizeof(type)
	// nb[1] = nb[0]   * ne[0] + padding
	// nb[i] = nb[i-1] * ne[i-1]

	// compute data
	op optype

	isParam bool

	grad *Tensor
	src0 *Tensor
	src1 *Tensor
	opt  [MAX_OPT]*Tensor

	// thread scheduling
	n_tasks uint32

	// performance
	perfRuns   uint32
	perfCycles uint32
	perfTime   uint64

	Data    []float32
	padding [8]byte
}

func AreSameShape(t0, t1 *Tensor) bool {
	////static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

	return (t0.NE[0] == t1.NE[0]) && (t0.NE[1] == t1.NE[1]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3])
}

func (t *Tensor) Nelements() uint32 {
	////static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
	return t.NE[0] * t.NE[1] * t.NE[2] * t.NE[3]
}

// struct ggml_tensor * ggml_view_tensor(
func ViewTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], src.Data)
}

// ggml.c : ggml_dup_tensor
func DupTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], nil)
}

// struct ggml_tensor * ggml_mul(
func Mul(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, false)
}

// struct ggml_tensor * ggml_mul_inplace(
func MulInplace(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, true)
}

// struct ggml_tensor * ggml_mul_impl(
func MulImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////GGML_ASSERT(ggml_are_same_shape(a, b));

	isNode := false

	if inplace && (a.grad != nil || b.grad != nil) {
		isNode = true
	}

	if inplace {
		////GGML_ASSERT(is_node == false);
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_MUL

	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result = nil
	}

	return result
}

// static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
func CanMulMat(t0, t1 *Tensor) bool {

	////static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");

	return (t0.NE[0] == t1.NE[0]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3]) // FIXME Where NE[1] ??
}

// ggml_mul_mat

// struct ggml_tensor * ggml_mul_mat(
func MulMat(ctx *Context, a, b *Tensor) *Tensor {
	////GGML_ASSERT(ggml_can_mul_mat(a, b));

	isNode := false

	if a.grad != nil || b.grad != nil {
		isNode = true
	}

	////const int ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
	result := NewTensor(ctx, TYPE_F32, min(a.Dims, b.Dims), a.NE[1], b.NE[1], a.NE[2], b.NE[3], nil) // Check for indexes

	result.op = OP_MUL_MAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_repeat

// struct ggml_tensor * ggml_repeat(
func Repeat(ctx *Context, a, b *Tensor) *Tensor {
	////GGML_ASSERT(ggml_can_repeat(a, b));

	isNode := false

	if a.grad != nil {
		isNode = true
	}

	if AreSameShape(a, b) && !isNode {
		return a
	}

	//struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);
	result := NewTensor(ctx, a.Type, b.Dims, b.NE[0], 1, 1, 1, nil)

	result.op = OP_REPEAT
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_get_rows

func GetRows(ctx *Context, a, b *Tensor) *Tensor {
	////GGML_ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == GGML_TYPE_I32);

	////isNode := false

	if a.grad != nil || b.grad != nil {
		////GGML_ASSERT(false); // TODO: implement backward
		////isNode = true
		fmt.Printf("\n[STOP] ml.GetRows") // FIXME ??
		os.Exit(1)                        // FIXME ??
	}

	// TODO: implement non F32 return
	//struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
	result := NewTensor2D(ctx, TYPE_F32, a.NE[0], b.NE[0])

	result.op = OP_GET_ROWS
	////if isNode {
	////    result.grad = DupTensor(ctx, result)
	////} else {
	////    result.grad = nil
	////}

	result.src0 = a
	result.src1 = b

	return result
}

func RMSNorm(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, false)
}

func RMSNormInplace(ctx *Context, a *Tensor) *Tensor {
	return RMSNormImpl(ctx, a, true)
}

// //struct ggml_tensor * ggml_rms_norm_impl(
func RMSNormImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		////GGML_ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] ml.GetRows") // FIXME ??
		os.Exit(1)                        // FIXME ??
	}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_RMS_NORM

	result.src0 = a
	result.src1 = nil // TODO: maybe store epsilon here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// computation graph
type Graph struct {
	nodesCount uint32 // FIXME Do not need
	leafCount  uint32 // FIXME Do not need
	threads    uint32

	workSize uint64
	work     *Tensor

	nodes [MAX_NODES]*Tensor
	grads [MAX_NODES]*Tensor
	leafs [MAX_NODES]*Tensor

	// performance
	perfRuns   uint64
	perfCycles uint64
	////int64_t perf_time_us;
}

type State struct {
	Contexts [MAX_CONTEXTS]ContextContainer
}

type ContextContainer struct {
	Used bool
	Ctx  Context
}

// global state
var gState State
var gStateBarrier int // FIXME atomic_int

type InitParams struct {
	// memory pool
	MemSize   uint64 // bytes
	MemBuffer []byte // if NULL, memory will be allocated internally
}

// scratch buffer
type Scratch struct {
	Offs uint64
	Size uint64
	Data []byte
}

type Object struct {
	Offs uint64
	Size uint64

	Next *Object

	Padding [8]byte
}

// ml/ggml.c:2248
type Context struct {
	MemSize        uint64
	MemBuffer      []byte
	MemBufferOwned bool

	//Objects uint64
	Objects []Object // FIXME Speedup with *Object?

	ObjectsBegin *Object
	ObjectsEnd   *Object

	Scratch     Scratch
	ScratchSave Scratch
}

/*
// ggml_new_tensor
func NewTensor(ctx *Context, dt dtype, dims, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensorImpl(ctx, dt, dims, ne0, ne1, ne2, ne3, nil)
}
*/
// ggml_new_tensor_1d
func NewTensor1D(ctx *Context, dt dtype, ne uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne, 1, 1, 1, nil)
}

// ggml_new_tensor_2d
func NewTensor2D(ctx *Context, dt dtype, ne0, ne1 uint32) *Tensor {
	//ne := []uint32{ne0, ne1}
	//return NewTensor(ctx, typ, 2, ne)
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil) // FIXME
}

func NewTensor3D(ctx *Context, dt dtype, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil) // FIXME
}

func NewTensor4D(ctx *Context, dt dtype, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensor(ctx, dt, 4, ne0, ne1, ne2, ne3, nil) // FIXME
}

// TODO ne2 for 3D tensors?
// ggml_new_tensor_impl
// func NewTensorImpl(ctx *Context, dt dtype, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {
func NewTensor(ctx *Context, dt dtype, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	if dt != TYPE_F32 {
		fmt.Printf("\n[ERROR] NewTensorImpl got not TYPE_F32!")
		os.Exit(1)
	}

	// always insert objects at the end of the context's memory pool
	////struct ggml_object * obj_cur = ctx->objects_end;

	////const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
	////const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
	////const size_t cur_end  = cur_offs + cur_size;

	//sizeNeeded := uint64(0)

	//if data == nil {
	////size_needed += TYPE_SIZE[type]*(ne[0]/BLCK_SIZE[type]);
	////for (int i = 1; i < n_dims; i++) {
	////    size_needed *= ne[i];
	////}
	// align to MEM_ALIGN
	////size_needed = ((size_needed + MEM_ALIGN - 1)/MEM_ALIGN)*MEM_ALIGN;
	//}

	////char * const mem_buffer = ctx->mem_buffer;
	////struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

	//if ctx.Scratch.Data == nil || data != nil {
	////size_needed += sizeof(struct ggml_tensor);

	////if (cur_end + size_needed + OBJECT_SIZE > ctx->mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + size_needed + OBJECT_SIZE, ctx->mem_size);
	////assert(false);
	////return NULL;
	////}

	////objNew := &Object{
	//Offs: cur_end + OBJECT_SIZE,
	////Size: 0, // FIXME size_needed,
	////Next: nil,
	////}

	//} else {

	//	if ctx.Scratch.Offs+sizeNeeded > ctx.Scratch.Size {
	//PRINT("%s: not enough space in the scratch memory\n", __func__);
	//assert(false);
	//		return nil
	//	}
	//}

	////if (cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE > ctx->mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE, ctx->mem_size);
	////assert(false);
	////return NULL;
	////}

	////data = (char * const) ctx->scratch.data + ctx->scratch.offs;

	////*obj_new = (struct ggml_object) {
	////.offs = cur_end + OBJECT_SIZE,
	////.size = sizeof(struct ggml_tensor),
	////.next = NULL,
	////};

	//printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

	////ctx->scratch.offs += size_needed;
	////}

	//if objCur != nil {
	//	objCur.Next = objNew
	//} else {
	// this is the first object in this context
	//	ctx.ObjectsBegin = objNew
	//}

	//ctx.ObjectsEnd = objNew

	//printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

	////struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new->offs);

	////ggml_assert_aligned(result);

	var retData []float32
	if data == nil {
		retData = make([]float32, ne0*ne1*ne2*ne3)
	} else {
		retData = data
	}

	return &Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		NB:   [4]uint32{0, 0, 0, 0},
		op:   OP_NONE,
		opt:  [4]*Tensor{nil, nil, nil, nil},
		Data: retData,
	}
}

// uitils.h
type GPTVocab struct {
	Token2ID map[string]uint32
	ID2Token map[uint32]string
}

func NewVocab() *GPTVocab {
	return &GPTVocab{
		Token2ID: make(map[string]uint32),
		ID2Token: make(map[uint32]string),
	}
}

func min(a, b uint32) uint32 {
	if a <= b {
		return a
	}
	return b
}

// FIXME Would it work with UTF-8? Rewrite for runes
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
// std::vector<gpt_vocab::id> llamaTokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
func Tokenize(vocab *GPTVocab, text string, bos bool) []uint32 {

	// TODO: Calculate this constant from the vocabulary
	MAX_TOKEN_LEN := uint32(18)
	length := uint32(len(text))

	////std::vector<gpt_vocab::id> res;
	res := make([]uint32, 0)
	////std::vector<int> score;
	//var score []uint32
	////std::vector<gpt_vocab::id> prev;
	//var prev []uint32
	////int len = text.length();

	////score.resize(len + 1);
	score := make([]uint32, length+1)
	////prev.resize(len + 1);
	prev := make([]uint32, length+1)

	// Forward pass
	for i := uint32(0); i < length; i++ {
		maxLen := min(length-i, MAX_TOKEN_LEN)
		for subLen := uint32(1); subLen <= maxLen; subLen++ {
			////auto sub = text.substr(i, sub_len);
			sub := text[i : i+subLen]
			////auto token = vocab.token_to_id.find(sub);
			token, ok := vocab.Token2ID[sub] // FIXME if not found?
			//if token != vocab.token2id.end() {
			if ok {
				tokenScore := uint32(len(sub) * len(sub))
				localScore := score[i] + tokenScore
				next := i + subLen
				if score[next] < localScore {
					score[next] = localScore
					////prev[next] = (*token).second
					prev[next] = token
				}
			}
		}
	}

	// Backward pass
	i := len(text)
	for i > 0 {
		////gpt_vocab::id token_id = prev[i];
		tokenID := prev[i]
		if tokenID == 0 {
			// TODO: Return error or something more meaningful
			fmt.Printf("\n[ERROR] Failed to tokenize string!")
			break
		}
		////res.push_back(token_id);
		res = append(res, tokenID)
		////auto token = (*vocab.id_to_token.find(token_id)).second;
		token, _ := vocab.ID2Token[tokenID]
		i -= len(token)
	}

	if bos {
		////res.push_back(1); // TODO: replace with vocab.bos
		res = append(res, 1) // TODO: replace with vocab.bos
	}

	// Pieces are in reverse order so correct that
	////std::reverse(res.begin(), res.end());
	//sort.Reverse(sort.IntSlice(res))

	//fmt.Printf("\n\n=== PREV ===\n\n%+v", prev)
	//fmt.Printf("\n\n=== RES ===\n\n%+v", res)

	reversed := make([]uint32, 0, len(res))
	for n := len(res); n > 0; n-- {
		reversed = append(reversed, res[n-1])
	}

	return reversed
}

func Init(params InitParams) *Context {
	// make this function thread safe
	////ggml_critical_section_start();

	isFirstCall := true // FIXME static ??

	if isFirstCall {
		// initialize GELU, SILU and EXP F32 tables
		////{
		////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

		////ggml_fp16_t ii;
		////for (int i = 0; i < (1 << 16); ++i) {
		////uint16_t ui = i;
		////memcpy(&ii, &ui, sizeof(ii));
		////const float f = table_f32_f16[i] = COMPUTE_FP16_TO_FP32(ii);
		////table_gelu_f16[i] = FP32_TO_FP16(ggml_gelu_f32(f));
		////table_silu_f16[i] = FP32_TO_FP16(ggml_silu_f32(f));
		////table_exp_f16[i]  = FP32_TO_FP16(exp(f));
		////}

		////const uint64_t t_end = ggml_time_us(); UNUSED(t_end);

		////PRINT_DEBUG("%s: GELU, SILU and EXP tables initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
		////}

		// initialize g_state
		{
			////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

			gState = State{
				Contexts: [MAX_CONTEXTS]ContextContainer{},
			}

			for i := uint32(0); i < MAX_CONTEXTS; i++ {
				gState.Contexts[i].Used = false
			}

			////const uint64_t t_end = ggml_time_us(); UNUSED(t_end);
			//var end uint64 = ggml_time_us(); UNUSED(t_end)

			////PRINT_DEBUG("%s: g_state initialized in %f ms\n", __func__, (t_end - t_start)/1000.0f);
		}

		isFirstCall = false
	}

	// find non-used context in g_state
	var ctx *Context

	for i := uint32(0); i < MAX_CONTEXTS; i++ {
		if !gState.Contexts[i].Used {
			gState.Contexts[i].Used = true
			ctx = &gState.Contexts[i].Ctx

			////PRINT_DEBUG("%s: found unused context %d\n", __func__, i)
			break
		}
	}

	if ctx == nil {
		////PRINT_DEBUG("%s: no unused context found\n", __func__);
		////ggml_critical_section_end();
		return nil
	}

	var buf []byte
	if params.MemBuffer == nil {
		buf = make([]byte, params.MemSize)

	} else {
		buf = params.MemBuffer
	}

	ctx = &Context{
		MemSize:        params.MemSize,
		MemBuffer:      buf,
		MemBufferOwned: params.MemBuffer != nil,
		//Objects:        0,
		Objects:      make([]Object, 0),
		ObjectsBegin: nil,
		ObjectsEnd:   nil,
		Scratch:      Scratch{0, 0, nil},
		ScratchSave:  Scratch{0, 0, nil},
	}

	////ggml_assert_aligned(ctx->mem_buffer);

	////PRINT_DEBUG("%s: context initialized\n", __func__);

	////ggml_critical_section_end();

	return ctx
}

/*
void ggml_free(struct ggml_context * ctx) {
    // make this function thread safe
    ggml_critical_section_start();

    bool found = false;

    for (int i = 0; i < MAX_CONTEXTS; i++) {
        if (&g_state.contexts[i].context == ctx) {
            g_state.contexts[i].used = false;

            PRINT_DEBUG("%s: context %d with %d objects has been freed. memory used = %zu\n",
                    __func__, i, ctx->n_objects, ctx->objects_end->offs + ctx->objects_end->size);

            if (ctx->mem_buffer_owned) {
                free(ctx->mem_buffer);
            }

            found = true;
            break;
        }
    }

    if (!found) {
        PRINT_DEBUG("%s: context not found\n", __func__);
    }

    ggml_critical_section_end();
}

size_t ggml_used_mem(const struct ggml_context * ctx) {
    return ctx->objects_end->offs + ctx->objects_end->size;
}

size_t ggml_set_scratch(struct ggml_context * ctx, struct ggml_scratch scratch) {
    const size_t result = ctx->scratch.data ? ctx->scratch.offs : 0;

    ctx->scratch = scratch;

    return result;
}
*/
////////////////////////////////////////////////////////////////////////////////
/*
struct ggml_tensor * ggml_new_tensor_impl(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int* ne,
        void*  data) {
    // always insert objects at the end of the context's memory pool
    struct ggml_object * obj_cur = ctx->objects_end;

    const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur->offs;
    const size_t cur_size = obj_cur == NULL ? 0 : obj_cur->size;
    const size_t cur_end  = cur_offs + cur_size;

    size_t size_needed = 0;

    if (data == NULL) {
        size_needed += TYPE_SIZE[type]*(ne[0]/BLCK_SIZE[type]);
        for (int i = 1; i < n_dims; i++) {
            size_needed *= ne[i];
        }
        // align to MEM_ALIGN
        size_needed = ((size_needed + MEM_ALIGN - 1)/MEM_ALIGN)*MEM_ALIGN;
    }

    char * const mem_buffer = ctx->mem_buffer;
    struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

    if (ctx->scratch.data == NULL || data != NULL) {
        size_needed += sizeof(struct ggml_tensor);

        if (cur_end + size_needed + OBJECT_SIZE > ctx->mem_size) {
            PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + size_needed + OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        *obj_new = (struct ggml_object) {
            .offs = cur_end + OBJECT_SIZE,
            .size = size_needed,
            .next = NULL,
        };
    } else {
        if (ctx->scratch.offs + size_needed > ctx->scratch.size) {
            PRINT("%s: not enough space in the scratch memory\n", __func__);
            assert(false);
            return NULL;
        }

        if (cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE > ctx->mem_size) {
            PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
                    __func__, cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE, ctx->mem_size);
            assert(false);
            return NULL;
        }

        data = (char * const) ctx->scratch.data + ctx->scratch.offs;

        *obj_new = (struct ggml_object) {
            .offs = cur_end + OBJECT_SIZE,
            .size = sizeof(struct ggml_tensor),
            .next = NULL,
        };

        //printf("scratch offs = %zu, size_needed = %zu\n", ctx->scratch.offs, size_needed);

        ctx->scratch.offs += size_needed;
    }

    if (obj_cur != NULL) {
        obj_cur->next = obj_new;
    } else {
        // this is the first object in this context
        ctx->objects_begin = obj_new;
    }

    ctx->objects_end = obj_new;

    //printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new->size);

    struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new->offs);

    ggml_assert_aligned(result);

    *result = (struct ggml_tensor) {
        / *.type         =* / type,
        / *.n_dims       =* / n_dims,
        / *.ne           =* / { 1, 1, 1, 1 },
        / *.nb           =* / { 0, 0, 0, 0 },
        / *.op           =* / OP_NONE,
        / *.is_param     =* / false,
        / *.grad         =* / NULL,
        / *.src0         =* / NULL,
        / *.src1         =* / NULL,
        / *.opt          =* / { NULL },
        / *.n_tasks      =* / 0,
        / *.perf_runs    =* / 0,
        / *.perf_cycles  =* / 0,
        / *.perf_time_us =* / 0,
        / *.data         =* / data == NULL ? (void *)(result + 1) : data,
        / *.pad          =* / { 0 },
    };

    ggml_assert_aligned(result->data);

    for (int i = 0; i < n_dims; i++) {
        result->ne[i] = ne[i];
    }

    result->nb[0] = TYPE_SIZE[type];
    result->nb[1] = result->nb[0]*(result->ne[0]/BLCK_SIZE[type]);
    for (int i = 2; i < MAX_DIMS; i++) {
        result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
    }

    ctx->n_objects++;

    return result;
}

struct ggml_tensor * ggml_new_tensor(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    n_dims,
        const int * ne) {
    return ggml_new_tensor_impl(ctx, type, n_dims, ne, NULL);
}

struct ggml_tensor * ggml_new_tensor_1d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0) {
    return ggml_new_tensor(ctx, type, 1, &ne0);
}

struct ggml_tensor * ggml_new_tensor_2d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1) {
    const int ne[2] = { ne0, ne1 };
    return ggml_new_tensor(ctx, type, 2, ne);
}

struct ggml_tensor * ggml_new_tensor_3d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2) {
    const int ne[3] = { ne0, ne1, ne2 };
    return ggml_new_tensor(ctx, type, 3, ne);
}

struct ggml_tensor * ggml_new_tensor_4d(
        struct ggml_context * ctx,
        enum   ggml_type type,
        int    ne0,
        int    ne1,
        int    ne2,
        int    ne3) {
    const int ne[4] = { ne0, ne1, ne2, ne3 };
    return ggml_new_tensor(ctx, type, 4, ne);
}

struct ggml_tensor * ggml_new_i32(struct ggml_context * ctx, int32_t value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, TYPE_I32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_i32(result, value);

    return result;
}

struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
    ctx->scratch_save = ctx->scratch;
    ctx->scratch.data = NULL;

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, TYPE_F32, 1);

    ctx->scratch = ctx->scratch_save;

    ggml_set_f32(result, value);

    return result;
}

struct ggml_tensor * ggml_dup_tensor(struct ggml_context * ctx, const struct ggml_tensor * src) {
    return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, NULL);
}

struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor) {
    memset(tensor->data, 0, ggml_nbytes(tensor));
    return tensor;
}

struct ggml_tensor * ggml_set_i32 (struct ggml_tensor * tensor, int32_t value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    return tensor;
}
*/
/*
struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
    const int n     = ggml_nrows(tensor);
    const int nc    = tensor->ne[0];
    const size_t n1 = tensor->nb[1];

    char * const data = tensor->data;

    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                assert(tensor->nb[0] == sizeof(int8_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_I16:
            {
                assert(tensor->nb[0] == sizeof(int16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_I32:
            {
                assert(tensor->nb[0] == sizeof(int32_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_F16:
            {
                assert(tensor->nb[0] == sizeof(ggml_fp16_t));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
                }
            } break;
        case TYPE_F32:
            {
                assert(tensor->nb[0] == sizeof(float));
                for (int i = 0; i < n; i++) {
                    ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
                }
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    return tensor;
}

int32_t ggml_get_i32_1d(const struct ggml_tensor * tensor, int i) {
    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case TYPE_I16:
            {
                ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case TYPE_I32:
            {
                ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case TYPE_F16:
            {
                ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            } break;
        case TYPE_F32:
            {
                ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    return 0.0f;
}

void ggml_set_i32_1d(const struct ggml_tensor * tensor, int i, int32_t value) {
    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_I16:
            {
                ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_I32:
            {
                ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_F16:
            {
                ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = FP32_TO_FP16(value);
            } break;
        case TYPE_F32:
            {
                ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}
*/
/*
float ggml_get_f32_1d(const struct ggml_tensor * tensor, int i) {
    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                ASSERT(tensor->nb[0] == sizeof(int8_t));
                return ((int8_t *)(tensor->data))[i];
            } break;
        case TYPE_I16:
            {
                ASSERT(tensor->nb[0] == sizeof(int16_t));
                return ((int16_t *)(tensor->data))[i];
            } break;
        case TYPE_I32:
            {
                ASSERT(tensor->nb[0] == sizeof(int32_t));
                return ((int32_t *)(tensor->data))[i];
            } break;
        case TYPE_F16:
            {
                ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                return FP16_TO_FP32(((ggml_fp16_t *)(tensor->data))[i]);
            } break;
        case TYPE_F32:
            {
                ASSERT(tensor->nb[0] == sizeof(float));
                return ((float *)(tensor->data))[i];
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    return 0.0f;
}

void ggml_set_f32_1d(const struct ggml_tensor * tensor, int i, float value) {
    switch (tensor->type) {
        case TYPE_Q4_0:
            {
                ASSERT(false);
            } break;
        case TYPE_Q4_1:
            {
                ASSERT(false);
            } break;
        case TYPE_I8:
            {
                ASSERT(tensor->nb[0] == sizeof(int8_t));
                ((int8_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_I16:
            {
                ASSERT(tensor->nb[0] == sizeof(int16_t));
                ((int16_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_I32:
            {
                ASSERT(tensor->nb[0] == sizeof(int32_t));
                ((int32_t *)(tensor->data))[i] = value;
            } break;
        case TYPE_F16:
            {
                ASSERT(tensor->nb[0] == sizeof(ggml_fp16_t));
                ((ggml_fp16_t *)(tensor->data))[i] = FP32_TO_FP16(value);
            } break;
        case TYPE_F32:
            {
                ASSERT(tensor->nb[0] == sizeof(float));
                ((float *)(tensor->data))[i] = value;
            } break;
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

void * ggml_get_data(const struct ggml_tensor * tensor) {
    return tensor->data;
}

float * ggml_get_data_f32(const struct ggml_tensor * tensor) {
    assert(tensor->type == TYPE_F32);
    return (float *)(tensor->data);
}

struct ggml_tensor * ggml_view_tensor(
        struct ggml_context * ctx,
        const struct ggml_tensor * src) {
    return ggml_new_tensor_impl(ctx, src->type, src->n_dims, src->ne, src->data);
}
*/
/*
////////////////////////////////////////////////////////////////////////////////

// ggml_dup

struct ggml_tensor * ggml_dup_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_DUP;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_dup(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_dup_impl(ctx, a, false);
}

struct ggml_tensor * ggml_dup_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    return ggml_dup_impl(ctx, a, true);
}

// ggml_add

struct ggml_tensor * ggml_add_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_ADD;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_add(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_add_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_add_impl(ctx, a, b, true);
}

// ggml_sub

struct ggml_tensor * ggml_sub_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_SUB;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_sub(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_sub_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_sub_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_sub_impl(ctx, a, b, true);
}

// ggml_mul

struct ggml_tensor * ggml_mul_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        ASSERT(is_node == false);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_MUL;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_mul(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_mul_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_mul_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_mul_impl(ctx, a, b, true);
}

// ggml_div

struct ggml_tensor * ggml_div_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b,
        bool inplace) {
    ASSERT(ggml_are_same_shape(a, b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        is_node = true;
    }

    if (inplace) {
        ASSERT(is_node == false);
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_DIV;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_div(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_div_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_div_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    return ggml_div_impl(ctx, a, b, true);
}

// ggml_sqr

struct ggml_tensor * ggml_sqr_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_SQR;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sqr(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqr_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sqr_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqr_impl(ctx, a, true);
}

// ggml_sqrt

struct ggml_tensor * ggml_sqrt_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_SQRT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sqrt(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqrt_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sqrt_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sqrt_impl(ctx, a, true);
}

// ggml_sum

struct ggml_tensor * ggml_sum(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_1d(ctx, a->type, 1);

    result->op   = OP_SUM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_mean

struct ggml_tensor * ggml_mean(
        struct ggml_context * ctx,
        struct ggml_tensor * a) {
    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement
        is_node = true;
    }

    int ne[MAX_DIMS] = { 1, a->ne[1], a->ne[2], a->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, a->n_dims, ne);

    result->op   = OP_MEAN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_repeat

struct ggml_tensor * ggml_repeat(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    ASSERT(ggml_can_repeat(a, b));

    bool is_node = false;

    if (a->grad) {
        is_node = true;
    }

    if (ggml_are_same_shape(a, b) && !is_node) {
        return a;
    }

    struct ggml_tensor * result = ggml_new_tensor(ctx, a->type, b->n_dims, b->ne);

    result->op   = OP_REPEAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_abs

struct ggml_tensor * ggml_abs_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_ABS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_abs(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_abs_impl(ctx, a, false);
}

struct ggml_tensor * ggml_abs_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_abs_impl(ctx, a, true);
}

*/
/*
// ggml_sgn

struct ggml_tensor * ggml_sgn_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_SGN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_sgn(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sgn_impl(ctx, a, false);
}

struct ggml_tensor * ggml_sgn_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_sgn_impl(ctx, a, true);
}

// ggml_neg

struct ggml_tensor * ggml_neg_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_NEG;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_neg(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_neg_impl(ctx, a, false);
}

struct ggml_tensor * ggml_neg_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_neg_impl(ctx, a, true);
}

// ggml_step

struct ggml_tensor * ggml_step_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_STEP;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_step(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_step_impl(ctx, a, false);
}

struct ggml_tensor * ggml_step_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_step_impl(ctx, a, true);
}

// ggml_relu

struct ggml_tensor * ggml_relu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_RELU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_relu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_relu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_relu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_relu_impl(ctx, a, true);
}

// ggml_gelu

struct ggml_tensor * ggml_gelu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_GELU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_gelu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_gelu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_gelu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_gelu_impl(ctx, a, true);
}

// ggml_silu

struct ggml_tensor * ggml_silu_impl(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_SILU;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_silu(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, false);
}

struct ggml_tensor * ggml_silu_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_silu_impl(ctx, a, true);
}

// ggml_norm

struct ggml_tensor * ggml_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct ggml_tensor * ggml_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_norm_impl(ctx, a, false);
}

struct ggml_tensor * ggml_norm_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_norm_impl(ctx, a, true);
}

struct ggml_tensor * ggml_rms_norm_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        bool inplace) {
    bool is_node = false;

    if (!inplace && (a->grad)) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);

    result->op   = OP_RMS_NORM;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store epsilon here?

    return result;
}

struct ggml_tensor * ggml_rms_norm(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_rms_norm_impl(ctx, a, false);
}

struct ggml_tensor * ggml_rms_norm_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    return ggml_rms_norm_impl(ctx, a, true);
}

// ggml_mul_mat

struct ggml_tensor * ggml_mul_mat(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    ASSERT(ggml_can_mul_mat(a, b));

    bool is_node = false;

    if (a->grad || b->grad) {
        is_node = true;
    }

    const int ne[4] = { a->ne[1], b->ne[1], a->ne[2], b->ne[3] };
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, MIN(a->n_dims, b->n_dims), ne);

    result->op   = OP_MUL_MAT;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_scale

struct ggml_tensor * ggml_scale_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {
    ASSERT(ggml_is_scalar(b));
    ASSERT(ggml_is_padded_1d(a));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = OP_SCALE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_scale(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_scale_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_scale_impl(ctx, a, b, true);
}

// ggml_cpy

struct ggml_tensor * ggml_cpy_impl(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b,
        bool inplace) {
    ASSERT(ggml_nelements(a) == ggml_nelements(b));

    bool is_node = false;

    if (!inplace && (a->grad || b->grad)) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // make a view of the destination
    struct ggml_tensor * result = ggml_view_tensor(ctx, b);

    result->op   = OP_CPY;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

struct ggml_tensor * ggml_cpy(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_cpy_impl(ctx, a, b, false);
}

struct ggml_tensor * ggml_cpy_inplace(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    return ggml_cpy_impl(ctx, a, b, true);
}

// ggml_reshape

struct ggml_tensor * ggml_reshape(
        struct ggml_context * ctx,
        struct ggml_tensor * a,
        struct ggml_tensor * b) {
    ASSERT(ggml_is_contiguous(a));
    ASSERT(ggml_is_contiguous(b));
    ASSERT(ggml_nelements(a) == ggml_nelements(b));

    bool is_node = false;

    if (a->grad || b->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, b->n_dims, b->ne, a->data);

    result->op   = OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1) {
    ASSERT(ggml_is_contiguous(a));
    ASSERT(ggml_nelements(a) == ne0*ne1);

    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[2] = { ne0, ne1 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, a->data);

    result->op   = OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

struct ggml_tensor * ggml_reshape_3d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        int                   ne2) {
    ASSERT(ggml_is_contiguous(a));
    ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[3] = { ne0, ne1, ne2 };
    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 3, ne, a->data);

    result->op   = OP_RESHAPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_view_1d

struct ggml_tensor * ggml_view_1d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        size_t                offset) {
    if (a->grad) {
        ASSERT(false); // gradient propagation is not supported
    }

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 1, &ne0, (char *) a->data + offset);

    result->op   = OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// ggml_view_2d

struct ggml_tensor * ggml_view_2d(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   ne0,
        int                   ne1,
        size_t                nb1,
        size_t                offset) {
    if (a->grad) {
        ASSERT(false); // gradient propagation is not supported
    }

    const int ne[MAX_DIMS] = { ne0, ne1, 1, 1 };

    struct ggml_tensor * result = ggml_new_tensor_impl(ctx, a->type, 2, ne, (char *) a->data + offset);

    result->nb[1] = nb1;
    result->nb[2] = result->nb[1]*ne1;
    result->nb[3] = result->nb[2];

    result->op   = OP_VIEW;
    result->grad = NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the offset here?

    return result;
}

// ggml_permute

struct ggml_tensor * ggml_permute(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   axis0,
        int                   axis1,
        int                   axis2,
        int                   axis3) {
    ASSERT(axis0 >= 0 && axis0 < MAX_DIMS);
    ASSERT(axis1 >= 0 && axis1 < MAX_DIMS);
    ASSERT(axis2 >= 0 && axis2 < MAX_DIMS);
    ASSERT(axis3 >= 0 && axis3 < MAX_DIMS);

    ASSERT(axis0 != axis1);
    ASSERT(axis0 != axis2);
    ASSERT(axis0 != axis3);
    ASSERT(axis1 != axis2);
    ASSERT(axis1 != axis3);
    ASSERT(axis2 != axis3);

    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    int ne[MAX_DIMS];
    int nb[MAX_DIMS];

    ne[axis0] = a->ne[0];
    ne[axis1] = a->ne[1];
    ne[axis2] = a->ne[2];
    ne[axis3] = a->ne[3];

    nb[axis0] = a->nb[0];
    nb[axis1] = a->nb[1];
    nb[axis2] = a->nb[2];
    nb[axis3] = a->nb[3];

    result->ne[0] = ne[0];
    result->ne[1] = ne[1];
    result->ne[2] = ne[2];
    result->ne[3] = ne[3];

    result->nb[0] = nb[0];
    result->nb[1] = nb[1];
    result->nb[2] = nb[2];
    result->nb[3] = nb[3];

    result->op   = OP_PERMUTE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL; // TODO: maybe store the permutation here?

    return result;
}

// ggml_transpose

struct ggml_tensor * ggml_transpose(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->ne[0] = a->ne[1];
    result->ne[1] = a->ne[0];

    result->nb[0] = a->nb[1];
    result->nb[1] = a->nb[0];

    result->op   = OP_TRANSPOSE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_get_rows

struct ggml_tensor * ggml_get_rows(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b->type == TYPE_I32);

    bool is_node = false;

    if (a->grad || b->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: implement non F32 return
    //struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a->type, a->ne[0], b->ne[0]);
    struct ggml_tensor * result = ggml_new_tensor_2d(ctx, TYPE_F32, a->ne[0], b->ne[0]);

    result->op   = OP_GET_ROWS;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_diag_mask_inf

struct ggml_tensor * ggml_diag_mask_inf(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past) {
    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);
    struct ggml_tensor * b = ggml_new_i32(ctx, n_past);

    result->op   = OP_DIAG_MASK_INF;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_soft_max

struct ggml_tensor * ggml_soft_max(
        struct ggml_context * ctx,
        struct ggml_tensor  * a) {
    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    result->op   = OP_SOFT_MAX;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = NULL;

    return result;
}

// ggml_rope

struct ggml_tensor * ggml_rope(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        int                   n_past,
        int                   n_dims,
        int                   mode) {
    ASSERT(n_past >= 0);
    bool is_node = false;

    if (a->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    // TODO: when implement backward, fix this:
    //struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_view_tensor(ctx, a);

    struct ggml_tensor * b = ggml_new_tensor_1d(ctx, TYPE_I32, 3);
    ((int32_t *) b->data)[0] = n_past;
    ((int32_t *) b->data)[1] = n_dims;
    ((int32_t *) b->data)[2] = mode;

    result->op   = OP_ROPE;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_conv_1d_1s

struct ggml_tensor * ggml_conv_1d_1s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    ASSERT(ggml_is_matrix(b));
    ASSERT(a->ne[1] == b->ne[1]);
    ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[4] = { b->ne[0], a->ne[2], 1, 1, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, 2, ne);

    result->op   = OP_CONV_1D_1S;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_conv_1d_2s

struct ggml_tensor * ggml_conv_1d_2s(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b) {
    ASSERT(ggml_is_matrix(b));
    ASSERT(a->ne[1] == b->ne[1]);
    ASSERT(a->ne[3] == 1);
    bool is_node = false;

    if (a->grad || b->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    const int ne[4] = { b->ne[0]/2, a->ne[2], 1, 1, };
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, 2, ne);

    result->op   = OP_CONV_1D_2S;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b;

    return result;
}

// ggml_flash_attn
*/
/*
struct ggml_tensor * ggml_flash_attn(
        struct ggml_context * ctx,
        struct ggml_tensor  * q,
        struct ggml_tensor  * k,
        struct ggml_tensor  * v,
        bool                  masked) {
    ASSERT(ggml_can_mul_mat(k, q));
    // TODO: check if vT can be multiplied by (k*qT)

    bool is_node = false;

    if (q->grad || k->grad || v->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct ggml_tensor * result = ggml_dup_tensor(ctx, q);
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, 4, q->ne);

    result->op   = OP_FLASH_ATTN;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = q;
    result->src1 = k;
    result->opt[0] = v;
    result->opt[1] = ggml_new_i32(ctx, masked ? 1 : 0);

    return result;
}

// ggml_flash_ff

struct ggml_tensor * ggml_flash_ff(
        struct ggml_context * ctx,
        struct ggml_tensor  * a,
        struct ggml_tensor  * b0,
        struct ggml_tensor  * b1,
        struct ggml_tensor  * c0,
        struct ggml_tensor  * c1) {
    ASSERT(ggml_can_mul_mat(b0, a));
    // TODO: more checks

    bool is_node = false;

    if (a->grad || b0->grad || b1->grad || c0->grad || c1->grad) {
        ASSERT(false); // TODO: implement backward
        is_node = true;
    }

    //struct ggml_tensor * result = ggml_dup_tensor(ctx, a);
    struct ggml_tensor * result = ggml_new_tensor(ctx, TYPE_F32, 4, a->ne);

    result->op   = OP_FLASH_FF;
    result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
    result->src0 = a;
    result->src1 = b0;
    result->opt[0] = b1;
    result->opt[1] = c0;
    result->opt[2] = c1;

    return result;
}

////////////////////////////////////////////////////////////////////////////////

void ggml_set_param(
        struct ggml_context * ctx,
        struct ggml_tensor * tensor) {
    tensor->is_param = true;

    ASSERT(tensor->grad == NULL);
    tensor->grad = ggml_dup_tensor(ctx, tensor);
}

// ggml_compute_forward_dup

static void ggml_compute_forward_dup_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(params->ith == 0);
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    if (ggml_is_contiguous(src0) && src0->type == dst->type) {
        memcpy(dst->data, src0->data, ggml_nelements(dst) * TYPE_SIZE[src0->type]);
        return;
    }

    if (src0->nb[0] == sizeof(ggml_fp16_t)) {
        if (dst->type == TYPE_F16) {
            int id = 0;
            const size_t rs = ne00*nb00;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        char * dst_ptr = (char *) dst->data + id*rs;

                        memcpy(dst_ptr, src0_ptr, rs);

                        id++;
                    }
                }
            }
        } else if (dst->type == TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = FP16_TO_FP32(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            ASSERT(false); // TODO: implement
        }
    } else {
        //printf("%s: this is not optimal - fix me\n", __func__);

        if (dst->type == TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = FP16_TO_FP32(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else if (dst->type == TYPE_F16) {
            int id = 0;
            ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const ggml_fp16_t * src0_ptr = (ggml_fp16_t *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = *src0_ptr;
                            id++;
                        }
                    }
                }
            }
        } else {
            ASSERT(false); // TODO: implement
        }
    }
}

static void ggml_compute_forward_dup_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(params->ith == 0);
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    if (ggml_is_contiguous(src0) && src0->type == dst->type) {
        memcpy(dst->data, src0->data, ggml_nelements(dst) * TYPE_SIZE[src0->type]);
        return;
    }

    if (src0->nb[0] == sizeof(float)) {
        if (dst->type == TYPE_F32) {
            int id = 0;
            const size_t rs = ne00*nb00;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
                        char * dst_ptr = (char *) dst->data + id*rs;

                        memcpy(dst_ptr, src0_ptr, rs);

                        id++;
                    }
                }
            }
        } else if (dst->type == TYPE_F16) {
            int id = 0;
            ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = FP32_TO_FP16(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            ASSERT(false); // TODO: implement
        }
    } else {
        //printf("%s: this is not optimal - fix me\n", __func__);

        if (dst->type == TYPE_F32) {
            int id = 0;
            float * dst_ptr = (float *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = *src0_ptr;
                            id++;
                        }
                    }
                }
            }
        } else if (dst->type == TYPE_F16) {
            int id = 0;
            ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

            for (int i03 = 0; i03 < ne03; i03++) {
                for (int i02 = 0; i02 < ne02; i02++) {
                    for (int i01 = 0; i01 < ne01; i01++) {
                        for (int i00 = 0; i00 < ne00; i00++) {
                            const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

                            dst_ptr[id] = FP32_TO_FP16(*src0_ptr);
                            id++;
                        }
                    }
                }
            }
        } else {
            ASSERT(false); // TODO: implement
        }
    }
}

static void ggml_compute_forward_dup(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_dup_f16(params, src0, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_dup_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_add

static void ggml_compute_forward_add_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    const size_t nb00 = src0->nb[0];
    const size_t nb01 = src0->nb[1];

    const size_t nb10 = src1->nb[0];
    const size_t nb11 = src1->nb[1];

    const size_t nb0 = dst->nb[0];
    const size_t nb1 = dst->nb[1];

    ASSERT( nb0 == sizeof(float));
    ASSERT(nb00 == sizeof(float));

    if (nb10 == sizeof(float)) {
        const int j0 = (n/nth)*ith;
        const int j1 = ith == nth - 1 ? n : (n/nth)*(ith + 1);

        for (int j = j0; j < j1; j++) {
            ggml_vec_add_f32(nc,
                    (float *) ((char *) dst->data  + j*nb1),
                    (float *) ((char *) src0->data + j*nb01),
                    (float *) ((char *) src1->data + j*nb11));
        }
    } else {
        // src1 is not contiguous
        for (int j = ith; j < n; j += nth) {
            float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
            float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
            for (int i = 0; i < nc; i++) {
                float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);

                dst_ptr[i] = src0_ptr[i] + *src1_ptr;
            }
        }
    }
}

static void ggml_compute_forward_add(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_add_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sub

static void ggml_compute_forward_sub_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sub_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_sub(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_sub_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_mul

static void ggml_compute_forward_mul_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_mul_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_mul(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_mul_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_div
*/
/*
static void ggml_compute_forward_div_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));
    assert(src1->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_div_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])),
                (float *) ((char *) src1->data + i*(src1->nb[1])));
    }
}

static void ggml_compute_forward_div(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_div_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sqr

static void ggml_compute_forward_sqr_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n     = ggml_nrows(src0);
    const int nc    = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sqr_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sqr(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_sqr_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sqrt

static void ggml_compute_forward_sqrt_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sqrt_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sqrt(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_sqrt_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sum

static void ggml_compute_forward_sum_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_is_scalar(dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    assert(ggml_is_scalar(dst));
    assert(src0->nb[0] == sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (float *) (dst->data),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));
            }
        }
    }
}

static void ggml_compute_forward_sum(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_sum_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_mean

static void ggml_compute_forward_mean_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    assert(src0->nb[0] == sizeof(float));

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    const int ne3 = dst->ne[3];

    assert(ne0 == 1);
    assert(ne1 == ne01);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    UNUSED(ne0);
    UNUSED(ne1);
    UNUSED(ne2);
    UNUSED(ne3);

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = 0; i01 < ne01; i01++) {
                ggml_vec_sum_f32(ne00,
                        (float *) ((char *)  dst->data + i01*nb1  + i02*nb2  + i03*nb3),
                        (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03));

                *(float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3) /= (float) ne00;
            }
        }
    }
}

static void ggml_compute_forward_mean(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_mean_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_repeat

static void ggml_compute_forward_repeat_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_can_repeat(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    // TODO: implement support for rank > 2 tensors
    assert(src0->ne[2] == 1);
    assert(src0->ne[3] == 1);
    assert( dst->ne[2] == 1);
    assert( dst->ne[3] == 1);

    const int nc  = dst->ne[0];
    const int nr  = dst->ne[1];
    const int nc0 = src0->ne[0];
    const int nr0 = src0->ne[1];
    const int ncr = nc/nc0; // guaranteed to be an integer due to the check in ggml_can_repeat
    const int nrr = nr/nr0; // guaranteed to be an integer due to the check in ggml_can_repeat

    // TODO: support for transposed / permuted tensors
    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    // TODO: maybe this is not optimal?
    for (int i = 0; i < nrr; i++) {
        for (int j = 0; j < ncr; j++) {
            for (int k = 0; k < nr0; k++) {
                ggml_vec_cpy_f32(nc0,
                        (float *) ((char *)  dst->data + (i*nr0 + k)*( dst->nb[1]) + j*nc0*( dst->nb[0])),
                        (float *) ((char *) src0->data + (        k)*(src0->nb[1])));
            }
        }
    }
}

static void ggml_compute_forward_repeat(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_repeat_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_abs

static void ggml_compute_forward_abs_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_abs_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_abs(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_abs_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_sgn

static void ggml_compute_forward_sgn_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_sgn_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_sgn(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_sgn_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_neg

static void ggml_compute_forward_neg_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_neg_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_neg(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_neg_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_step

static void ggml_compute_forward_step_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_step_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_step(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_step_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_relu

static void ggml_compute_forward_relu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];

    assert(dst->nb[0]  == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < n; i++) {
        ggml_vec_relu_f32(nc,
                (float *) ((char *) dst->data  + i*( dst->nb[1])),
                (float *) ((char *) src0->data + i*(src0->nb[1])));
    }
}

static void ggml_compute_forward_relu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_relu_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_gelu

static void ggml_compute_forward_gelu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(ggml_is_contiguous(src0));
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_gelu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}

static void ggml_compute_forward_gelu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_gelu_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    //printf("XXXXXXXX gelu\n");
}

// ggml_compute_forward_silu

static void ggml_compute_forward_silu_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(ggml_is_contiguous(src0));
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_silu_f32(nc,
                (float *) ((char *) dst->data  + i1*( dst->nb[1])),
                (float *) ((char *) src0->data + i1*(src0->nb[1])));

#ifndef NDEBUG
        for (int k = 0; k < nc; k++) {
            const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
            UNUSED(x);
            assert(!isnan(x));
            assert(!isinf(x));
        }
#endif
    }
}
*/
/*
static void ggml_compute_forward_silu(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_silu_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}


// ggml_compute_forward_norm

static void ggml_compute_forward_norm_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const ggml_float eps = 1e-5f; // TODO: make this a parameter

    // TODO: optimize
    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float mean = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    mean += x[i00];
                }

                mean /= ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                ggml_float sum2 = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    ggml_float v = x[i00] - mean;
                    y[i00] = v;
                    sum2 += v*v;
                }

                const float scale = 1.0/sqrt(sum2/ne00 + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void ggml_compute_forward_norm(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_norm_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

static void ggml_compute_forward_rms_norm_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    ASSERT(src0->nb[0] == sizeof(float));

    const int ith = params->ith;
    const int nth = params->nth;

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const size_t nb01 = src0->nb[1];
    const size_t nb02 = src0->nb[2];
    const size_t nb03 = src0->nb[3];

    const size_t nb1 = dst->nb[1];
    const size_t nb2 = dst->nb[2];
    const size_t nb3 = dst->nb[3];

    const ggml_float eps = 1e-5f; // TODO: make this a parameter

    // TODO: optimize
    for (int i03 = 0; i03 < ne03; i03++) {
        for (int i02 = 0; i02 < ne02; i02++) {
            for (int i01 = ith; i01 < ne01; i01 += nth) {
                const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);

                ggml_float mean = 0.0;
                for (int i00 = 0; i00 < ne00; i00++) {
                    mean += x[i00] * x[i00];
                }

                mean /= ne00;

                float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);

                memcpy(y, x, ne00 * sizeof(float));
                // for (int i00 = 0; i00 < ne00; i00++) {
                //     y[i00] = x[i00];
                // }

                const float scale = 1.0/sqrt(mean + eps);

                ggml_vec_scale_f32(ne00, y, scale);
            }
        }
    }
}

static void ggml_compute_forward_rms_norm(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_rms_norm_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}


// ggml_compute_forward_mul_mat

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
// helper function to determine if it is better to use BLAS or not
// for large matrices, BLAS is faster
static bool ggml_compute_forward_mul_mat_use_blas(
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    UNUSED(src0);

    const int ne10 = src1->ne[0];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];

    // TODO: find the optimal values for these
    if (ggml_is_contiguous(src0) &&
        ggml_is_contiguous(src1) && ((ne0 >= 32 && ne1 >= 32 && ne10 >= 32))) {
        //printf("BLAS: %d %d %d\n", ne0, ne1, ne10);
        return true;
    }

    return false;
}
#endif

static void ggml_compute_forward_mul_mat_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    assert(ne02 == ne12);
    assert(ne03 == ne13);
    assert(ne2  == ne12);
    assert(ne3  == ne13);

    // TODO: we don't support permuted src0
    assert(nb00 == sizeof(float) || nb01 == sizeof(float));

    // dst cannot be transposed or permuted
    assert(nb0 == sizeof(float));
    assert(nb0 <= nb1);
    assert(nb1 <= nb2);
    assert(nb2 <= nb3);

    assert(ne0 == ne01);
    assert(ne1 == ne11);
    assert(ne2 == ne02);
    assert(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == TASK_INIT) {
            return;
        }

        if (params->type == TASK_FINALIZE) {
            return;
        }

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                const float * x = (float *) (src0->data);
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // zT = y * xT
                {
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne10,
                                     x, ne10,
                            0.0f,    d, ne01);
                }
            }
        }

        //printf("CBLAS F32 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == TASK_INIT) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        //assert(params->wsize == (ggml_nbytes(dst) + CACHE_LINE_SIZE)*nth);

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        ggml_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            ggml_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1
        assert(nb10 == sizeof(float));

        // parallelize by src0 rows using ggml_vec_dot_f32

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            for (int ic = 0; ic < ne11; ++ic) {
                // src1 indices
                const int i13 = i03;
                const int i12 = i02;
                const int i11 = ic;

                // dst indices
                const int i0 = i01;
                const int i1 = i11;
                const int i2 = i02;
                const int i3 = i03;

                ggml_vec_dot_f32(ne00,
                        (float *) ((char *)  dst->data + (i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3)),
                        (float *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03)),
                        (float *) ((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13)));
            }
        }
    } else {
        // parallelize by src1 columns using ggml_vec_mad_f32
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        // dst indices
                        const int i1 = i11;
                        const int i2 = i12;
                        const int i3 = i13;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        ggml_vec_mad_f32(ne01,
                                (float *) (wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0),
                                (float *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03)),
                               *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13)));
                    }
                }
            }
        }
    }

    //int64_t t1 = ggml_perf_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
    //    printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}
*/
/*
static void ggml_compute_forward_mul_mat_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    ASSERT(ne02 == ne12);
    ASSERT(ne03 == ne13);
    ASSERT(ne2  == ne12);
    ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    ASSERT(nb00 == sizeof(ggml_fp16_t) || nb01 == sizeof(ggml_fp16_t));

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    ASSERT(ne0 == ne01);
    ASSERT(ne1 == ne11);
    ASSERT(ne2 == ne02);
    ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == TASK_INIT) {
            return;
        }

        if (params->type == TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        for (int i00 = 0; i00 < ne00; ++i00) {
                            wdata[id++] = FP16_TO_FP32(*(ggml_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        }
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        / *printf("CBLAS F16 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);* /

        return;
    }
#endif

    if (params->type == TASK_INIT) {
        if (nb01 >= nb00) {
            ggml_fp16_t * const wdata = params->wdata;

            int id = 0;
            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        for (int i10 = 0; i10 < ne10; ++i10) {
                            wdata[id++] = FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        }
                    }
                }
            }

            ASSERT(id*sizeof(ggml_fp16_t) <= params->wsize);

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        //assert(params->wsize == (ggml_nbytes(dst) + CACHE_LINE_SIZE)*nth);

        ggml_fp16_t * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        for (int i = ic0; i < ic1; ++i) {
            ((float *) dst->data)[i] = FP16_TO_FP32(wdata[i]);
        }

        for (int k = 1; k < nth; k++) {
            for (int i = ic0; i < ic1; ++i) {
                ((float *) dst->data)[i] += FP16_TO_FP32(wdata[(ne + CACHE_LINE_SIZE_F32)*k + i]);
            }
        }

        return;
    }

    if (nb01 >= nb00) {
        // fp16 -> half the size, so divide by 2
        // TODO: do not support transposed src1
        assert(nb10/2 == sizeof(ggml_fp16_t));

        // parallelize by src0 rows using ggml_vec_dot_f16

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        ggml_fp16_t * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            ggml_fp16_t * src0_row = (ggml_fp16_t *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            ggml_fp16_t * src1_col =                                wdata + (       0 + i12*ne11 + i13*ne12*ne11)*ne00;

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                ggml_vec_dot_f16(ne00, &dst_col[ic*ne0], src0_row, src1_col + ic*ne00);
            }
        }
    } else {
        // parallelize by src1 columns using ggml_vec_mad_f16
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        ggml_fp16_t * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    ggml_fp16_t * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(ggml_fp16_t)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        ggml_fp16_t * src0_col =  (ggml_fp16_t *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float         src1_val = *      (float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        ggml_vec_mad_f16(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}
*/
/*
static void ggml_compute_forward_mul_mat_q4_0_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    ASSERT(ne02 == ne12);
    ASSERT(ne03 == ne13);
    ASSERT(ne2  == ne12);
    ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    ASSERT(nb00 == (int) TYPE_SIZE[TYPE_Q4_0] || nb01 == (int) TYPE_SIZE[TYPE_Q4_0]);

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    ASSERT(ne0 == ne01);
    ASSERT(ne1 == ne11);
    ASSERT(ne2 == ne02);
    ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == TASK_INIT) {
            return;
        }

        if (params->type == TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        //for (int i00 = 0; i00 < ne00; ++i00) {
                        //    wdata[id++] = FP16_TO_FP32(*(ggml_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        //}
                        dequantize_row_q4_0((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        / *printf("CBLAS Q4_0 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);* /

        return;
    }
#endif

    if (params->type == TASK_INIT) {
        //printf("HHHHHHHHH ith = %d, nth = %d\n", ith, nth);
        if (nb01 >= nb00) {
            char * wdata = params->wdata;

            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        //for (int i10 = 0; i10 < ne10; ++i10) {
                        //    wdata[id++] = FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        //}
                        quantize_row_q4_0((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                        wdata += (ne10*TYPE_SIZE[TYPE_Q4_0])/BLCK_SIZE[TYPE_Q4_0];
                    }
                }
            }

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        ggml_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            ggml_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1

        // parallelize by src0 rows using ggml_vec_dot_q4_0

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        void * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*TYPE_SIZE[TYPE_Q4_0])/BLCK_SIZE[TYPE_Q4_0]);

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                ggml_vec_dot_q4_0(ne00, &dst_col[ic*ne0], src0_row, ((void *) (src1_col + (ic*ne00*TYPE_SIZE[TYPE_Q4_0])/BLCK_SIZE[TYPE_Q4_0])));
            }
        }
    } else {
        //printf("AAAAA ith = %d, nth = %d\n", ith, nth);
        // parallelize by src1 columns using ggml_vec_mad_q4_0
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    float * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        void * src0_col =   (void *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float  src1_val = *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        ggml_vec_mad_q4_0(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}
*/
/*

static void ggml_compute_forward_mul_mat_q4_1_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    const int ne12 = src1->ne[2];
    const int ne13 = src1->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    const int ne2  = dst->ne[2];
    const int ne3  = dst->ne[3];
    const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    const int nb12 = src1->nb[2];
    const int nb13 = src1->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    ASSERT(ne02 == ne12);
    ASSERT(ne03 == ne13);
    ASSERT(ne2  == ne12);
    ASSERT(ne3  == ne13);

    // TODO: we don't support permuted src0
    ASSERT(nb00 == (int) TYPE_SIZE[TYPE_Q4_1] || nb01 == (int) TYPE_SIZE[TYPE_Q4_1]);

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    ASSERT(ne0 == ne01);
    ASSERT(ne1 == ne11);
    ASSERT(ne2 == ne02);
    ASSERT(ne3 == ne03);

    // nb01 >= nb00 - src0 is not transposed
    //   compute by src0 rows
    //
    // nb00 <  nb01 - src0 is transposed
    //   compute by src0 columns

#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
        ASSERT(nb10 == sizeof(float));

        if (params->ith != 0) {
            return;
        }

        if (params->type == TASK_INIT) {
            return;
        }

        if (params->type == TASK_FINALIZE) {
            return;
        }

        float * const wdata = params->wdata;

        for (int i03 = 0; i03 < ne03; i03++) {
            for (int i02 = 0; i02 < ne02; i02++) {
                {
                    int id = 0;
                    for (int i01 = 0; i01 < ne01; ++i01) {
                        //for (int i00 = 0; i00 < ne00; ++i00) {
                        //    wdata[id++] = FP16_TO_FP32(*(ggml_fp16_t *) ((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01 + i00*nb00));
                        //}
                        dequantize_row_q4_1((char *) src0->data + i03*nb03 + i02*nb02 + i01*nb01, wdata + id, ne00);
                        id += ne00;
                    }
                }

                const float * x = wdata;
                const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

                //      float * z =                          wdata + ne00*ne01;

                // z = x * yT
                //{
                //    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                //            ne01, ne11, ne00,
                //            1.0f, x, ne00,
                //                  y, ne00,
                //            0.0f, z, ne11);
                //}

                float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

                // transpose z
                //for (int j = 0; j < ne11; ++j) {
                //    for (int i = 0; i < ne01; ++i) {
                //        d[j*ne01 + i] = z[i*ne11 + j];
                //    }
                //}

                {
#if 1
                    // zT = y * xT
                    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                            ne11, ne01, ne10,
                            1.0f,    y, ne00,
                                     x, ne00,
                            0.0f,    d, ne01);
#else
                    // zT = (xT * y)T
                    cblas_sgemm(CblasColMajor, CblasTrans, CblasNoTrans,
                            ne01, ne11, ne10,
                            1.0f,    x, ne00,
                                     y, ne00,
                            0.0f,    d, ne01);
#endif
                }
            }
        }

        //printf("CBLAS = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

        return;
    }
#endif

    if (params->type == TASK_INIT) {
        //printf("HHHHHHHHH ith = %d, nth = %d\n", ith, nth);
        if (nb01 >= nb00) {
            char * wdata = params->wdata;

            for (int i13 = 0; i13 < ne13; ++i13) {
                for (int i12 = 0; i12 < ne12; ++i12) {
                    for (int i11 = 0; i11 < ne11; ++i11) {
                        //for (int i10 = 0; i10 < ne10; ++i10) {
                        //    wdata[id++] = FP32_TO_FP16(*(float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11 + i10*nb10));
                        //}
                        quantize_row_q4_1((float *)((char *) src1->data + i13*nb13 + i12*nb12 + i11*nb11), (void *) wdata, ne10);
                        wdata += (ne10*TYPE_SIZE[TYPE_Q4_1])/BLCK_SIZE[TYPE_Q4_1];
                    }
                }
            }

            return;
        }

        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);
        return;
    }

    if (params->type == TASK_FINALIZE) {
        if (nb01 >= nb00) {
            return;
        }

        float * const wdata = params->wdata;

        // cols per thread
        const int dc = (ne + nth - 1)/nth;

        // col range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, ne);

        ggml_vec_cpy_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + ic0);

        for (int k = 1; k < nth; k++) {
            ggml_vec_acc_f32(ic1 - ic0, (float *) dst->data + ic0, wdata + (ne + CACHE_LINE_SIZE_F32)*k + ic0);
        }

        return;
    }

    if (nb01 >= nb00) {
        // TODO: do not support transposed src1

        // parallelize by src0 rows using ggml_vec_dot_q4_1

        // total rows in src0
        const int nr = ne01*ne02*ne03;

        // rows per thread
        const int dr = (nr + nth - 1)/nth;

        // row range for this thread
        const int ir0 = dr*ith;
        const int ir1 = MIN(ir0 + dr, nr);

        void * wdata = params->wdata;

        for (int ir = ir0; ir < ir1; ++ir) {
            // src0 indices
            const int i03 = ir/(ne02*ne01);
            const int i02 = (ir - i03*ne02*ne01)/ne01;
            const int i01 = (ir - i03*ne02*ne01 - i02*ne01);

            const int i13 = i03;
            const int i12 = i02;

            const int i0 = i01;
            const int i2 = i02;
            const int i3 = i03;

            void * src0_row = (void *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03));
            char * src1_col =          ((char *)      wdata + (      (0 + i12*ne11 + i13*ne12*ne11)*ne00*TYPE_SIZE[TYPE_Q4_1])/BLCK_SIZE[TYPE_Q4_1]);

            float * dst_col = (float *) ((char *) dst->data + (i0*nb0 + 0*nb1 + i2*nb2 + i3*nb3));

            assert(ne00 % 32 == 0);

            for (int ic = 0; ic < ne11; ++ic) {
                ggml_vec_dot_q4_1(ne00, &dst_col[ic*ne0], src0_row, ((void *) (src1_col + (ic*ne00*TYPE_SIZE[TYPE_Q4_1])/BLCK_SIZE[TYPE_Q4_1])));
            }
        }
    } else {
        //printf("AAAAA ith = %d, nth = %d\n", ith, nth);
        // parallelize by src1 columns using ggml_vec_mad_q4_1
        // each thread has its own work data
        // during FINALIZE we accumulate all work data into dst

        // total columns in src1
        const int nc = ne10;

        // columns per thread
        const int dc = (nc + nth - 1)/nth;

        // column range for this thread
        const int ic0 = dc*ith;
        const int ic1 = MIN(ic0 + dc, nc);

        // work data for thread
        const int wo = (ne + CACHE_LINE_SIZE_F32)*ith;
        float * const wdata = params->wdata;

        for (int i13 = 0; i13 < ne13; ++i13) {
            for (int i12 = 0; i12 < ne12; ++i12) {
                for (int i11 = 0; i11 < ne11; ++i11) {
                    // dst indices
                    const int i1 = i11;
                    const int i2 = i12;
                    const int i3 = i13;

                    float * dst_row = wdata + wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0;

                    for (int ic = ic0; ic < ic1; ++ic) {
                        // src1 indices
                        const int i10 = ic;

                        // src0 indices
                        const int i03 = i13;
                        const int i02 = i12;
                        const int i00 = ic;

                        assert(sizeof(float)*(wo + i3*ne2*ne1*ne0 + i2*ne1*ne0 + i1*ne0 + ne01) <= params->wsize);

                        void * src0_col =   (void *) ((char *) src0->data + (i00*nb00 + i02*nb02 + i03*nb03));
                        float  src1_val = *(float *) ((char *) src1->data + (i10*nb10 + i11*nb11 + i12*nb12 + i13*nb13));

                        ggml_vec_mad_q4_1(ne01, dst_row, src0_col, src1_val);
                    }
                }
            }
        }
    }

    //int64_t t1 = ggml_time_us();
    //static int64_t acc = 0;
    //acc += t1 - t0;
    //if (t1 - t0 > 10) {
    //    printf("\n");
    //    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
    //    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
    //    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);

    //    printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
    //}
}

static void ggml_compute_forward_mul_mat(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_Q4_0:
            {
                ggml_compute_forward_mul_mat_q4_0_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_1:
            {
                ggml_compute_forward_mul_mat_q4_1_f32(params, src0, src1, dst);
            } break;
        case TYPE_F16:
            {
                ggml_compute_forward_mul_mat_f16_f32(params, src0, src1, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_mul_mat_f32(params, src0, src1, dst);
            } break;
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

#if 0
    if (src0->type == TYPE_F16 || src0->type == TYPE_Q4_1) {
        static int first = 8;
        printf("src0: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src0->ne[0], src0->ne[1], src0->ne[2]);
        printf("src1: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src1->ne[0], src1->ne[1], src1->ne[2]);
        printf("dst:  ne0 = %5d, ne1 = %5d, ne2 = %5d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
        if (first) {
            --first;
        } else {
            for (int k = 0; k < dst->ne[1]; ++k) {
                for (int j = 0; j < dst->ne[0]/16; ++j) {
                    for (int i = 0; i < 16; ++i) {
                        printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
                    }
                    printf("\n");
                }
                printf("\n");
            }
            printf("\n");
            exit(0);
        }
    } else {
        printf("aaaa src0: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src0->ne[0], src0->ne[1], src0->ne[2]);
        printf("aaaa src1: ne0 = %5d, ne1 = %5d, ne2 = %5d\n", src1->ne[0], src1->ne[1], src1->ne[2]);
        printf("aaaa dst:  ne0 = %5d, ne1 = %5d, ne2 = %5d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    }
#endif
}

// ggml_compute_forward_scale

static void ggml_compute_forward_scale_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    ASSERT(ggml_is_contiguous(src0));
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_are_same_shape(src0, dst));
    ASSERT(ggml_is_scalar(src1));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    // scale factor
    const float v = *(float *) src1->data;

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), v);
    }
}

static void ggml_compute_forward_scale(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_scale_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_cpy

static void ggml_compute_forward_cpy(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ggml_compute_forward_dup(params, src0, dst);
}

// ggml_compute_forward_reshape

static void ggml_compute_forward_reshape(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
    UNUSED(dst);
}
*/
/*
// ggml_compute_forward_view

static void ggml_compute_forward_view(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_permute

static void ggml_compute_forward_permute(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_transpose

static void ggml_compute_forward_transpose(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0) {
    // NOP
    UNUSED(params);
    UNUSED(src0);
}

// ggml_compute_forward_get_rows

static void ggml_compute_forward_get_rows_q4_0(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == TYPE_SIZE[TYPE_Q4_0]);

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dequantize_row_q4_0(
                (const void *) ((char *) src0->data + r*src0->nb[1]),
                     (float *) ((char *)  dst->data + i*dst->nb[1]), nc);
    }
}

static void ggml_compute_forward_get_rows_q4_1(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == TYPE_SIZE[TYPE_Q4_1]);

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        dequantize_row_q4_1(
                (const void *) ((char *) src0->data + r*src0->nb[1]),
                     (float *) ((char *)  dst->data + i*dst->nb[1]), nc);
    }
}

static void ggml_compute_forward_get_rows_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(ggml_fp16_t));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        for (int j = 0; j < nc; ++j) {
            ggml_fp16_t v = ((ggml_fp16_t *) ((char *) src0->data + r*src0->nb[1]))[j];
            ((float *) ((char *)  dst->data + i*dst->nb[1]))[j] = FP16_TO_FP32(v);
        }
    }
}

static void ggml_compute_forward_get_rows_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    assert(params->ith == 0);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int nc = src0->ne[0];
    const int nr = ggml_nelements(src1);

    assert( dst->ne[0] == nc);
    assert( dst->ne[1] == nr);
    assert(src0->nb[0] == sizeof(float));

    for (int i = 0; i < nr; ++i) {
        const int r = ((int32_t *) src1->data)[i];

        ggml_vec_cpy_f32(nc,
                (float *) ((char *)  dst->data + i*dst->nb[1]),
                (float *) ((char *) src0->data + r*src0->nb[1]));
    }
}

static void ggml_compute_forward_get_rows(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_Q4_0:
            {
                ggml_compute_forward_get_rows_q4_0(params, src0, src1, dst);
            } break;
        case TYPE_Q4_1:
            {
                ggml_compute_forward_get_rows_q4_1(params, src0, src1, dst);
            } break;
        case TYPE_F16:
            {
                ggml_compute_forward_get_rows_f16(params, src0, src1, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_get_rows_f32(params, src0, src1, dst);
            } break;
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }

    //static bool first = true;
    //printf("ne0 = %d, ne1 = %d, ne2 = %d\n", dst->ne[0], dst->ne[1], dst->ne[2]);
    //if (first) {
    //    first = false;
    //} else {
    //    for (int k = 0; k < dst->ne[1]; ++k) {
    //        for (int j = 0; j < dst->ne[0]/16; ++j) {
    //            for (int i = 0; i < 16; ++i) {
    //                printf("%8.4f ", ((float *) dst->data)[k*dst->ne[0] + j*16 + i]);
    //            }
    //            printf("\n");
    //        }
    //        printf("\n");
    //    }
    //    printf("\n");
    //    exit(0);
    //}
}

// ggml_compute_forward_diag_mask_inf

static void ggml_compute_forward_diag_mask_inf_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == TYPE_I32);
    assert(ggml_nelements(src1) == 1);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];

    // TODO: handle transposed/permuted matrices

    const int n  = ggml_nrows(src0);
    const int nc = src0->ne[0];
    const int nr = src0->ne[1];
    const int nz = n/nr;

    assert( dst->nb[0] == sizeof(float));
    assert(src0->nb[0] == sizeof(float));

    for (int k = 0; k < nz; k++) {
        for (int j = 0; j < nr; j++) {
            for (int i = n_past; i < nc; i++) {
                if (i > n_past + j) {
                    *(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = -INFINITY;
                }
            }
        }
    }
}

static void ggml_compute_forward_diag_mask_inf(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_diag_mask_inf_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_soft_max

static void ggml_compute_forward_soft_max_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    ASSERT(ggml_is_contiguous(src0));
    ASSERT(ggml_is_contiguous(dst));
    ASSERT(ggml_are_same_shape(src0, dst));

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    // TODO: handle transposed/permuted matrices

    const int ith = params->ith;
    const int nth = params->nth;

    const int nc = src0->ne[0];
    const int nr = ggml_nrows(src0);

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float *p = (float *)((char *) dst->data + i1*dst->nb[1]);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            //printf("p[%d] = %f\n", i, p[i]);
            assert(!isnan(p[i]));
        }
#endif

        float max = -INFINITY;
        ggml_vec_max_f32(nc, &max, p);

        ggml_float sum = 0.0;

        uint16_t scvt;
        for (int i = 0; i < nc; i++) {
            if (p[i] == -INFINITY) {
                p[i] = 0.0f;
            } else {
                //const float val = (p[i] == -INFINITY) ? 0.0 : exp(p[i] - max);
                ggml_fp16_t s = FP32_TO_FP16(p[i] - max);
                memcpy(&scvt, &s, sizeof(scvt));
                const float val = FP16_TO_FP32(table_exp_f16[scvt]);
                sum += val;
                p[i] = val;
            }
        }

        assert(sum > 0.0f);

        sum = 1.0/sum;
        ggml_vec_scale_f32(nc, p, sum);

#ifndef NDEBUG
        for (int i = 0; i < nc; ++i) {
            assert(!isnan(p[i]));
            assert(!isinf(p[i]));
        }
#endif
    }
}

static void ggml_compute_forward_soft_max(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F32:
            {
                ggml_compute_forward_soft_max_f32(params, src0, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_F16:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_rope

static void ggml_compute_forward_rope_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == TYPE_I32);
    assert(ggml_nelements(src1) == 3);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int ne0 = src0->ne[0];
    const int ne1 = src0->ne[1];
    const int ne2 = src0->ne[2];
    const int ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(float));

    // TODO: optimize
    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = (mode == 0 ? n_past + i2 : i2);
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const double theta = pow(10000.0, ((double)-i0)/n_dims);

                    const double cos_theta = cos(p*theta);
                    const double sin_theta = sin(p*theta);

                    const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                          float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                    double x0 = src[0];
                    double x1 = src[1];

                    dst_data[0] = x0*cos_theta - x1*sin_theta;
                    dst_data[1] = x0*sin_theta + x1*cos_theta;
                }
            }
        }
    }
}

static void ggml_compute_forward_rope_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    assert(params->ith == 0);
    assert(src1->type == TYPE_I32);
    assert(ggml_nelements(src1) == 3);

    if (params->type == TASK_INIT || params->type == TASK_FINALIZE) {
        return;
    }

    const int n_past = ((int32_t *) src1->data)[0];
    const int n_dims = ((int32_t *) src1->data)[1];
    const int mode   = ((int32_t *) src1->data)[2];

    //const int ne0 = src0->ne[0];
    const int ne1 = src0->ne[1];
    const int ne2 = src0->ne[2];
    const int ne3 = src0->ne[3];

    const int nb0 = src0->nb[0];
    const int nb1 = src0->nb[1];
    const int nb2 = src0->nb[2];
    const int nb3 = src0->nb[3];

    //printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
    //printf("n_past = %d, ne2 = %d\n", n_past, ne2);

    assert(nb0 == sizeof(ggml_fp16_t));

    for (int i3 = 0; i3 < ne3; i3++) {
        for (int i2 = (mode == 0 ? 0 : n_past); i2 < ne2; i2++) {
            const int p = (mode == 0 ? n_past + i2 : i2);
            for (int i1 = 0; i1 < ne1; i1++) {
                for (int i0 = 0; i0 < n_dims; i0 += 2) {
                    const double theta = pow(10000.0, ((double)-i0)/n_dims);

                    const double cos_theta = cos(p*theta);
                    const double sin_theta = sin(p*theta);

                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
                          ggml_fp16_t * dst_data  = (ggml_fp16_t *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);

                    double x0 = ggml_fp16_to_fp32(src[0]);
                    double x1 = ggml_fp16_to_fp32(src[1]);

                    dst_data[0] = ggml_fp32_to_fp16(x0*cos_theta - x1*sin_theta);
                    dst_data[1] = ggml_fp32_to_fp16(x0*sin_theta + x1*cos_theta);
                }
            }
        }
    }
}

static void ggml_compute_forward_rope(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_rope_f16(params, src0, src1, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_rope_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_conv_1d_1s

static void ggml_compute_forward_conv_1d_1s_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    ASSERT(src0->type == TYPE_F16);
    ASSERT(src1->type == TYPE_F32);
    ASSERT( dst->type == TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    ASSERT(nb00 == sizeof(ggml_fp16_t));
    ASSERT(nb10 == sizeof(float));

    if (params->type == TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                ggml_fp16_t * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f16(ew0, &v,
                        (ggml_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (ggml_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_1s_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    ASSERT(src0->type == TYPE_F32);
    ASSERT(src1->type == TYPE_F32);
    ASSERT( dst->type == TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    ASSERT(nb00 == sizeof(float));
    ASSERT(nb10 == sizeof(float));

    if (params->type == TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; ++i0) {
            dst_data[i0] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0] += v;
            }
        }
    }
}
*/
/*
static void ggml_compute_forward_conv_1d_1s(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_conv_1d_1s_f16_f32(params, src0, src1, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_conv_1d_1s_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_conv_1d_2s

static void ggml_compute_forward_conv_1d_2s_f16_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    ASSERT(src0->type == TYPE_F16);
    ASSERT(src1->type == TYPE_F32);
    ASSERT( dst->type == TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    ASSERT(nb00 == sizeof(ggml_fp16_t));
    ASSERT(nb10 == sizeof(float));

    if (params->type == TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const ggml_fp16_t * const src = (ggml_fp16_t *)((char *) src0->data + i02*nb02 + i01*nb01);
                    ggml_fp16_t * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            ggml_fp16_t * const wdata = (ggml_fp16_t *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                ggml_fp16_t * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = FP32_TO_FP16(src[i10]);
                }
            }
        }

        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f16(ew0, &v,
                        (ggml_fp16_t *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (ggml_fp16_t *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}

static void ggml_compute_forward_conv_1d_2s_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
              struct ggml_tensor * dst) {
    ASSERT(src0->type == TYPE_F32);
    ASSERT(src1->type == TYPE_F32);
    ASSERT( dst->type == TYPE_F32);

    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int ne00 = src0->ne[0];
    const int ne01 = src0->ne[1];
    const int ne02 = src0->ne[2];
    //const int ne03 = src0->ne[3];

    const int ne10 = src1->ne[0];
    const int ne11 = src1->ne[1];
    //const int ne12 = src1->ne[2];
    //const int ne13 = src1->ne[3];

    //const int ne0  = dst->ne[0];
    //const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];
    //const int ne   = ne0*ne1*ne2*ne3;

    const int nb00 = src0->nb[0];
    const int nb01 = src0->nb[1];
    const int nb02 = src0->nb[2];
    //const int nb03 = src0->nb[3];

    const int nb10 = src1->nb[0];
    const int nb11 = src1->nb[1];
    //const int nb12 = src1->nb[2];
    //const int nb13 = src1->nb[3];

    //const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    //const int nb2  = dst->nb[2];
    //const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int nk = ne00;
    const int nh = nk/2;

    const int ew0 = ggml_up32(ne01);

    ASSERT(ne00 % 2 == 1); // TODO: support even kernel sizes
    ASSERT(nb00 == sizeof(float));
    ASSERT(nb10 == sizeof(float));

    if (params->type == TASK_INIT) {
        // TODO: fix this memset (wsize is overestimated)
        memset(params->wdata, 0, params->wsize);

        // prepare kernel data (src0)
        {
            float * const wdata = (float *) params->wdata + 0;

            for (int i02 = 0; i02 < ne02; i02++) {
                for (int i01 = 0; i01 < ne01; i01++) {
                    const float * const src = (float *)((char *) src0->data + i02*nb02 + i01*nb01);
                    float * dst_data = wdata + i02*ew0*ne00;
                    for (int i00 = 0; i00 < ne00; i00++) {
                        dst_data[i00*ew0 + i01] = src[i00];
                    }
                }
            }
        }

        // prepare source data (src1)
        {
            float * const wdata = (float *) params->wdata + ne02*ew0*ne00;

            for (int i11 = 0; i11 < ne11; i11++) {
                const float * const src = (float *)((char *) src1->data + i11*nb11);
                float * dst_data = wdata;
                for (int i10 = 0; i10 < ne10; i10++) {
                    dst_data[(i10 + nh)*ew0 + i11] = src[i10];
                }
            }
        }

        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // total rows in dst
    const int nr = ne02;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int i1 = ir0; i1 < ir1; i1++) {
        float * dst_data = (float *)((char *) dst->data + i1*nb1);
        for (int i0 = 0; i0 < ne10; i0 += 2) {
            dst_data[i0/2] = 0;
            for (int k = -nh; k <= nh; k++) {
                float v = 0.0f;
                ggml_vec_dot_f32(ew0, &v,
                        (float *) params->wdata +   i1*ew0*ne00 +      (nh + k)*ew0,
                        (float *) params->wdata + ne02*ew0*ne00 + (i0 + nh + k)*ew0);

                dst_data[i0/2] += v;
            }
        }
    }
}
*/
/*
static void ggml_compute_forward_conv_1d_2s(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * src0,
        const struct ggml_tensor * src1,
        struct ggml_tensor * dst) {
    switch (src0->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_conv_1d_2s_f16_f32(params, src0, src1, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_conv_1d_2s_f32(params, src0, src1, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_flash_attn

static void ggml_compute_forward_flash_attn_f32(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
             struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int neq0 = q->ne[0];
    const int neq1 = q->ne[1];
    const int neq2 = q->ne[2];
    const int neq3 = q->ne[3];

    const int nek0 = k->ne[0];
    const int nek1 = k->ne[1];
    //const int nek2 = k->ne[2];
    //const int nek3 = k->ne[3];

    //const int nev0 = v->ne[0];
    const int nev1 = v->ne[1];
    //const int nev2 = v->ne[2];
    //const int nev3 = v->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = neq0;
    const int N = neq1;
    const int P = nek1 - N;
    const int M = P + N;

    const int Mup = ggml_up(M, SOFT_MAX_UNROLL);

    ASSERT(ne0 == D);
    ASSERT(ne1 == N);
    ASSERT(P >= 0);

    ASSERT(nbq0 == sizeof(float));
    ASSERT(nbk0 == sizeof(float));
    ASSERT(nbv0 == sizeof(float));

    ASSERT(neq0 == D);
    ASSERT(nek0 == D);
    ASSERT(nev1 == D);

    ASSERT(neq1 == N);
    ASSERT(nek1 == N + P);
    ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    if (params->type == TASK_INIT) {
        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0/sqrt((double) D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        for (int ic = 0; ic < nek1; ++ic) {
            // k indices
            const int ik3 = iq3;
            const int ik2 = iq2;
            const int ik1 = ic;

            // S indices
            const int i1 = ik1;

            ggml_vec_dot_f32(neq0,
                    S + i1,
                    (float *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                    (float *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
        }

        // scale
        ggml_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            ggml_vec_max_f32(M, &max, S);

            float sum = 0.0f;
            {
#ifdef SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                ggml_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[SOFT_MAX_UNROLL];
                ggml_float sump[SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            ggml_fp16_t s = FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0f);

            sum = 1.0/sum;
            ggml_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        for (int ic = 0; ic < nev1; ++ic) {
            // dst indices
            const int i1 = iq1;
            const int i2 = iq2;
            const int i3 = iq3;

            ggml_vec_dot_f32(nek1,
                    (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                    (float *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                    S);
        }
    }
}

static void ggml_compute_forward_flash_attn_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
             struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int neq0 = q->ne[0];
    const int neq1 = q->ne[1];
    const int neq2 = q->ne[2];
    const int neq3 = q->ne[3];

    const int nek0 = k->ne[0];
    const int nek1 = k->ne[1];
    //const int nek2 = k->ne[2];
    //const int nek3 = k->ne[3];

    //const int nev0 = v->ne[0];
    const int nev1 = v->ne[1];
    //const int nev2 = v->ne[2];
    //const int nev3 = v->ne[3];

    const int ne0  = dst->ne[0];
    const int ne1  = dst->ne[1];
    //const int ne2  = dst->ne[2];
    //const int ne3  = dst->ne[3];

    const int nbk0 = k->nb[0];
    const int nbk1 = k->nb[1];
    const int nbk2 = k->nb[2];
    const int nbk3 = k->nb[3];

    const int nbq0 = q->nb[0];
    const int nbq1 = q->nb[1];
    const int nbq2 = q->nb[2];
    const int nbq3 = q->nb[3];

    const int nbv0 = v->nb[0];
    const int nbv1 = v->nb[1];
    const int nbv2 = v->nb[2];
    const int nbv3 = v->nb[3];

    const int nb0  = dst->nb[0];
    const int nb1  = dst->nb[1];
    const int nb2  = dst->nb[2];
    const int nb3  = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = neq0;
    const int N = neq1;
    const int P = nek1 - N;
    const int M = P + N;

    const int Mup = ggml_up(M, SOFT_MAX_UNROLL);

    ASSERT(ne0 == D);
    ASSERT(ne1 == N);
    ASSERT(P >= 0);

    ASSERT(nbq0 == sizeof(ggml_fp16_t));
    ASSERT(nbk0 == sizeof(ggml_fp16_t));
    ASSERT(nbv0 == sizeof(ggml_fp16_t));

    ASSERT(neq0 == D);
    ASSERT(nek0 == D);
    ASSERT(nev1 == D);

    ASSERT(neq1 == N);
    ASSERT(nek1 == N + P);
    ASSERT(nev1 == D);

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    if (params->type == TASK_INIT) {
        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // parallelize by q rows using ggml_vec_dot_f32

    // total rows in q
    const int nr = neq1*neq2*neq3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    const float scale = 1.0/sqrt((double) D);

    //printf("P=%d N=%d D=%d ir0=%d ir1=%d scale = %f\n", P, N, D, ir0, ir1, scale);

    for (int ir = ir0; ir < ir1; ++ir) {
        // q indices
        const int iq3 = ir/(neq2*neq1);
        const int iq2 = (ir - iq3*neq2*neq1)/neq1;
        const int iq1 = (ir - iq3*neq2*neq1 - iq2*neq1);

        float * S = (float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32);

        for (int i = M; i < Mup; ++i) {
            S[i] = -INFINITY;
        }

        if (VEC_DOT_UNROLL > 2 || nek1 % VEC_DOT_UNROLL != 0) {
            for (int ic = 0; ic < nek1; ++ic) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                ggml_vec_dot_f16(neq0,
                        S + i1,
                        (ggml_fp16_t *) ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (ggml_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        } else {
            for (int ic = 0; ic < nek1; ic += VEC_DOT_UNROLL) {
                // k indices
                const int ik3 = iq3;
                const int ik2 = iq2;
                const int ik1 = ic;

                // S indices
                const int i1 = ik1;

                ggml_vec_dot_f16_unroll(neq0, nbk1,
                        S + i1,
                        ((char *) k->data + (ik1*nbk1 + ik2*nbk2 + ik3*nbk3)),
                        (ggml_fp16_t *) ((char *) q->data + (iq1*nbq1 + iq2*nbq2 + iq3*nbq3)));
            }
        }

        // scale
        ggml_vec_scale_f32(nek1, S, scale);

        if (masked) {
            for (int i = P; i < M; i++) {
                if (i > P + iq1) {
                    S[i] = -INFINITY;
                }
            }
        }

        // softmax
        {
            float max = -INFINITY;
            ggml_vec_max_f32(M, &max, S);

            float sum = 0.0f;
            {
#ifdef SOFT_MAX_ACCELERATE
                max = -max;
                vDSP_vsadd(S, 1, &max, S, 1, Mup);
                vvexpf(S, S, &Mup);
                ggml_vec_sum_f32(Mup, &sum, S);
#else
                uint16_t   scvt[SOFT_MAX_UNROLL];
                ggml_float sump[SOFT_MAX_UNROLL] = { 0.0 };

                for (int i = 0; i < Mup; i += SOFT_MAX_UNROLL) {
                    float * SS = S + i;

                    for (int j = 0; j < SOFT_MAX_UNROLL; ++j) {
                        if (SS[j] == -INFINITY) {
                            SS[j] = 0.0f;
                        } else {
                            ggml_fp16_t s = FP32_TO_FP16(SS[j] - max);
                            memcpy(&scvt[j], &s, sizeof(uint16_t));
                            const float val = FP16_TO_FP32(table_exp_f16[scvt[j]]);
                            sump[j] += val;
                            SS[j] = val;
                        }
                    }
                }

                for (int i = 0; i < SOFT_MAX_UNROLL; i++) {
                    sum += sump[i];
                }
#endif
            }

            assert(sum > 0.0f);

            sum = 1.0/sum;
            ggml_vec_scale_f32(M, S, sum);

#ifndef NDEBUG
            for (int i = 0; i < M; ++i) {
                assert(!isnan(S[i]));
                assert(!isinf(S[i]));
            }
#endif
        }

        ggml_fp16_t * S16 = (ggml_fp16_t *) ((float *) params->wdata + ith*(2*Mup + CACHE_LINE_SIZE_F32) + Mup);

        for (int i = 0; i < M; i++) {
            S16[i] = FP32_TO_FP16(S[i]);
        }

        if (VEC_DOT_UNROLL == 1 || (nev1 % VEC_DOT_UNROLL != 0)) {
            for (int ic = 0; ic < nev1; ++ic) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                ggml_vec_dot_f16(nek1,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        (ggml_fp16_t *) ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        } else {
            for (int ic = 0; ic < nev1; ic += VEC_DOT_UNROLL) {
                // dst indices
                const int i1 = iq1;
                const int i2 = iq2;
                const int i3 = iq3;

                ggml_vec_dot_f16_unroll(nek1, nbv1,
                        (float *) ((char *) dst->data + (ic*nb0 + i1*nb1  + i2*nb2  + i3*nb3)),
                        ((char *) v->data   + (         ic*nbv1 + i2*nbv2 + i3*nbv3)),
                        S16);
            }
        }
    }
}
*/
/*
static void ggml_compute_forward_flash_attn(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * q,
        const struct ggml_tensor * k,
        const struct ggml_tensor * v,
        const bool masked,
        struct ggml_tensor * dst) {
    switch (q->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_flash_attn_f16(params, q, k, v, masked, dst);
            } break;
        case TYPE_F32:
            {
                ggml_compute_forward_flash_attn_f32(params, q, k, v, masked, dst);
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

// ggml_compute_forward_flash_ff

static void ggml_compute_forward_flash_ff_f16(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * a,  // F16
        const struct ggml_tensor * b0, // F16 fc_w
        const struct ggml_tensor * b1, // F32 fc_b
        const struct ggml_tensor * c0, // F16 proj_w
        const struct ggml_tensor * c1, // F32 proj_b
        struct ggml_tensor * dst) {
    int64_t t0 = ggml_perf_time_us();
    UNUSED(t0);

    const int nea0 = a->ne[0];
    const int nea1 = a->ne[1];
    const int nea2 = a->ne[2];
    const int nea3 = a->ne[3];

    const int neb00 = b0->ne[0];
    const int neb01 = b0->ne[1];
    //const int neb02 = b0->ne[2];
    //const int neb03 = b0->ne[3];

    const int neb10 = b1->ne[0];
    const int neb11 = b1->ne[1];
    //const int neb12 = b1->ne[2];
    //const int neb13 = b1->ne[3];

    const int nec00 = c0->ne[0];
    const int nec01 = c0->ne[1];
    //const int nec02 = c0->ne[2];
    //const int nec03 = c0->ne[3];

    const int nec10 = c1->ne[0];
    const int nec11 = c1->ne[1];
    //const int nec12 = c1->ne[2];
    //const int nec13 = c1->ne[3];

    const int ne0 = dst->ne[0];
    const int ne1 = dst->ne[1];
    const int ne2 = dst->ne[2];
    //const int ne3 = dst->ne[3];

    const int nba0 = a->nb[0];
    const int nba1 = a->nb[1];
    const int nba2 = a->nb[2];
    const int nba3 = a->nb[3];

    const int nbb00 = b0->nb[0];
    const int nbb01 = b0->nb[1];
    const int nbb02 = b0->nb[2];
    const int nbb03 = b0->nb[3];

    const int nbb10 = b1->nb[0];
    //const int nbb11 = b1->nb[1];
    //const int nbb12 = b1->nb[2];
    //const int nbb13 = b1->nb[3];

    const int nbc00 = c0->nb[0];
    const int nbc01 = c0->nb[1];
    const int nbc02 = c0->nb[2];
    const int nbc03 = c0->nb[3];

    const int nbc10 = c1->nb[0];
    //const int nbc11 = c1->nb[1];
    //const int nbc12 = c1->nb[2];
    //const int nbc13 = c1->nb[3];

    const int nb0 = dst->nb[0];
    const int nb1 = dst->nb[1];
    const int nb2 = dst->nb[2];
    const int nb3 = dst->nb[3];

    const int ith = params->ith;
    const int nth = params->nth;

    const int D = nea0;
    //const int N = nea1;
    const int M = neb01;

    ASSERT(ne0 == nea0);
    ASSERT(ne1 == nea1);
    ASSERT(ne2 == nea2);

    ASSERT(nba0  == sizeof(ggml_fp16_t));
    ASSERT(nbb00 == sizeof(ggml_fp16_t));
    ASSERT(nbb10 == sizeof(float));
    ASSERT(nbc00 == sizeof(ggml_fp16_t));
    ASSERT(nbc10 == sizeof(float));

    ASSERT(neb00 == D);
    ASSERT(neb01 == M);
    ASSERT(neb10 == M);
    ASSERT(neb11 == 1);

    ASSERT(nec00 == M);
    ASSERT(nec01 == D);
    ASSERT(nec10 == D);
    ASSERT(nec11 == 1);

    // dst cannot be transposed or permuted
    ASSERT(nb0 == sizeof(float));
    ASSERT(nb0 <= nb1);
    ASSERT(nb1 <= nb2);
    ASSERT(nb2 <= nb3);

    if (params->type == TASK_INIT) {
        return;
    }

    if (params->type == TASK_FINALIZE) {
        return;
    }

    // parallelize by a rows using ggml_vec_dot_f32

    // total rows in a
    const int nr = nea1*nea2*nea3;

    // rows per thread
    const int dr = (nr + nth - 1)/nth;

    // row range for this thread
    const int ir0 = dr*ith;
    const int ir1 = MIN(ir0 + dr, nr);

    for (int ir = ir0; ir < ir1; ++ir) {
        // a indices
        const int ia3 = ir/(nea2*nea1);
        const int ia2 = (ir - ia3*nea2*nea1)/nea1;
        const int ia1 = (ir - ia3*nea2*nea1 - ia2*nea1);

        float * S = (float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32);

        for (int ic = 0; ic < neb01; ++ic) {
            // b0 indices
            const int ib03 = ia3;
            const int ib02 = ia2;
            const int ib01 = ic;

            // S indices
            const int i1 = ib01;

            ggml_vec_dot_f16(nea0,
                    S + i1,
                    (ggml_fp16_t *) ((char *) b0->data + (ib01*nbb01 + ib02*nbb02 + ib03*nbb03)),
                    (ggml_fp16_t *) ((char *)  a->data + ( ia1*nba1  +  ia2*nba2  +  ia3*nba3)));
        }

        ggml_vec_add_f32(neb01, S, S, (float *) b1->data);
        //ggml_vec_gelu_f32(neb01, S, S);

        ggml_fp16_t * S16 = (ggml_fp16_t *) ((float *) params->wdata + ith*(2*M + CACHE_LINE_SIZE_F32) + M);

        for (int i = 0; i < M; i++) {
            S16[i] = FP32_TO_FP16(S[i]);
        }

        ggml_vec_gelu_f16(neb01, S16, S16);

        {
            // dst indices
            const int i1 = ia1;
            const int i2 = ia2;
            const int i3 = ia3;

            for (int ic = 0; ic < nec01; ++ic) {

                ggml_vec_dot_f16(neb01,
                        (float *)       ((char *) dst->data + (ic*nb0 + i1*nb1   + i2*nb2   + i3*nb3)),
                        (ggml_fp16_t *) ((char *) c0->data  + (         ic*nbc01 + i2*nbc02 + i3*nbc03)),
                        S16);
            }

            ggml_vec_add_f32(nec01,
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) ((char *) dst->data + (i1*nb1 + i2*nb2 + i3*nb3)),
                    (float *) c1->data);
        }
    }
}

static void ggml_compute_forward_flash_ff(
        const struct ggml_compute_params * params,
        const struct ggml_tensor * a,
        const struct ggml_tensor * b0,
        const struct ggml_tensor * b1,
        const struct ggml_tensor * c0,
        const struct ggml_tensor * c1,
        struct ggml_tensor * dst) {
    switch (b0->type) {
        case TYPE_F16:
            {
                ggml_compute_forward_flash_ff_f16(params, a, b0, b1, c0, c1, dst);
            } break;
        case TYPE_F32:
            {
                ASSERT(false); // TODO
            } break;
        case TYPE_Q4_0:
        case TYPE_Q4_1:
        case TYPE_I8:
        case TYPE_I16:
        case TYPE_I32:
        case TYPE_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

/////////////////////////////////

static void ggml_compute_forward(struct ggml_compute_params * params, struct ggml_tensor * tensor) {
    ASSERT(params);

    switch (tensor->op) {
        case OP_DUP:
            {
                ggml_compute_forward_dup(params, tensor->src0, tensor);
            } break;
        case OP_ADD:
            {
                ggml_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_SUB:
            {
                ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_MUL:
            {
                ggml_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_DIV:
            {
                ggml_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_SQR:
            {
                ggml_compute_forward_sqr(params, tensor->src0, tensor);
            } break;
        case OP_SQRT:
            {
                ggml_compute_forward_sqrt(params, tensor->src0, tensor);
            } break;
        case OP_SUM:
            {
                ggml_compute_forward_sum(params, tensor->src0, tensor);
            } break;
        case OP_MEAN:
            {
                ggml_compute_forward_mean(params, tensor->src0, tensor);
            } break;
        case OP_REPEAT:
            {
                ggml_compute_forward_repeat(params, tensor->src0, tensor);
            } break;
        case OP_ABS:
            {
                ggml_compute_forward_abs(params, tensor->src0, tensor);
            } break;
        case OP_SGN:
            {
                ggml_compute_forward_sgn(params, tensor->src0, tensor);
            } break;
        case OP_NEG:
            {
                ggml_compute_forward_neg(params, tensor->src0, tensor);
            } break;
        case OP_STEP:
            {
                ggml_compute_forward_step(params, tensor->src0, tensor);
            } break;
        case OP_RELU:
            {
                ggml_compute_forward_relu(params, tensor->src0, tensor);
            } break;
        case OP_GELU:
            {
                ggml_compute_forward_gelu(params, tensor->src0, tensor);
            } break;
        case OP_SILU:
            {
                ggml_compute_forward_silu(params, tensor->src0, tensor);
            } break;
        case OP_NORM:
            {
                ggml_compute_forward_norm(params, tensor->src0, tensor);
            } break;
        case OP_RMS_NORM:
            {
                ggml_compute_forward_rms_norm(params, tensor->src0, tensor);
            } break;
        case OP_MUL_MAT:
            {
                ggml_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_SCALE:
            {
                ggml_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_CPY:
            {
                ggml_compute_forward_cpy(params, tensor->src0, tensor);
            } break;
        case OP_RESHAPE:
            {
                ggml_compute_forward_reshape(params, tensor->src0, tensor);
            } break;
        case OP_VIEW:
            {
                ggml_compute_forward_view(params, tensor->src0);
            } break;
        case OP_PERMUTE:
            {
                ggml_compute_forward_permute(params, tensor->src0);
            } break;
        case OP_TRANSPOSE:
            {
                ggml_compute_forward_transpose(params, tensor->src0);
            } break;
        case OP_GET_ROWS:
            {
                ggml_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_DIAG_MASK_INF:
            {
                ggml_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_SOFT_MAX:
            {
                ggml_compute_forward_soft_max(params, tensor->src0, tensor);
            } break;
        case OP_ROPE:
            {
                ggml_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_CONV_1D_1S:
            {
                ggml_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_CONV_1D_2S:
            {
                ggml_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
            } break;
        case OP_FLASH_ATTN:
            {
                int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
                ASSERT(t == 0 || t == 1);
                bool masked = t != 0;
                ggml_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
            } break;
        case OP_FLASH_FF:
            {
                ggml_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
            } break;
        case OP_NONE:
            {
                // nop
            } break;
        case OP_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

////////////////////////////////////////////////////////////////////////////////
*/
/*
static void ggml_compute_backward(struct ggml_context * ctx, struct ggml_tensor * tensor, bool inplace) {
    struct ggml_tensor * src0 = tensor->src0;
    struct ggml_tensor * src1 = tensor->src1;

    switch (tensor->op) {
        case OP_DUP:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case OP_ADD:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = ggml_add_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case OP_SUB:
            {
                if (src0->grad) {
                    src0->grad = ggml_add_impl(ctx, src0->grad, tensor->grad, inplace);
                }
                if (src1->grad) {
                    src1->grad = ggml_sub_impl(ctx, src1->grad, tensor->grad, inplace);
                }
            } break;
        case OP_MUL:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx, src1, tensor->grad),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_impl(ctx,
                                src1->grad,
                                ggml_mul(ctx, src0, tensor->grad),
                                inplace);
                }
            } break;
        case OP_DIV:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_div(ctx, tensor->grad, src1),
                                inplace);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_sub_impl(ctx,
                                src1->grad,
                                ggml_mul(ctx,
                                    tensor->grad,
                                    ggml_div(ctx, tensor, src1)),
                                inplace);
                }
            } break;
        case OP_SQR:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_mul(ctx, src0, tensor->grad),
                                    ggml_repeat(ctx, ggml_new_f32(ctx, 2.0f), src0)),
                                inplace);
                }
            } break;
        case OP_SQRT:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_div(ctx,
                                    ggml_repeat(ctx, ggml_new_f32(ctx, 0.5f), tensor),
                                    tensor),
                                inplace);
                }
            } break;
        case OP_SUM:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_repeat(ctx, tensor->grad, src0->grad),
                                inplace);
                }
            } break;
        case OP_MEAN:
            {
                ASSERT(false); // TODO: implement
            } break;
        case OP_REPEAT:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_sum(ctx, tensor->grad),
                                inplace);
                }
            } break;
        case OP_ABS:
            {
                if (src0->grad) {
                    src0->grad =
                        ggml_add_impl(ctx,
                                src0->grad,
                                ggml_mul(ctx,
                                    ggml_sgn(ctx, src0),
                                    tensor->grad),
                                inplace);
                }
            } break;
        case OP_SGN:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case OP_NEG:
            {
                if (src0->grad) {
                    src0->grad = ggml_sub_impl(ctx, src0->grad, tensor->grad, inplace);
                }
            } break;
        case OP_STEP:
            {
                if (src0->grad) {
                    // noop
                }
            } break;
        case OP_RELU:
            {
                if (src0->grad) {
                    src0->grad = ggml_sub_impl(ctx,
                            src0->grad,
                            ggml_mul(ctx,
                                ggml_step(ctx, src0),
                                tensor->grad),
                            inplace);
                }
            } break;
        case OP_GELU:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_SILU:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_NORM:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_RMS_NORM:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_MUL_MAT:
            {
                if (src0->grad) {
                    // TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor->grad);
                    ASSERT(false);
                }
                if (src1->grad) {
                    src1->grad =
                        ggml_add_impl(ctx,
                                src1->grad,
                                // TODO: fix transpose, the node will break the graph connections
                                ggml_mul_mat(ctx, ggml_transpose(ctx, src0), tensor->grad),
                                inplace);
                }
            } break;
        case OP_SCALE:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_CPY:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_RESHAPE:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_VIEW:
            {
                ASSERT(false); // not supported
            } break;
        case OP_PERMUTE:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_TRANSPOSE:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_GET_ROWS:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_DIAG_MASK_INF:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_SOFT_MAX:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_ROPE:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_CONV_1D_1S:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_CONV_1D_2S:
            {
                ASSERT(false); // TODO: not implemented
            } break;
        case OP_FLASH_ATTN:
            {
                ASSERT(false); // not supported
            } break;
        case OP_FLASH_FF:
            {
                ASSERT(false); // not supported
            } break;
        case OP_NONE:
            {
                // nop
            } break;
        case OP_COUNT:
            {
                ASSERT(false);
            } break;
    }
}

static void ggml_visit_parents(struct ggml_cgraph * cgraph, struct ggml_tensor * node) {
    if (node->grad == NULL) {
        // this usually happens when we generate intermediate nodes from constants in the backward pass
        // it can also happen during forward pass, if the user performs computations with constants
        if (node->op != OP_NONE) {
            //PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node->op);
        }
    }

    // check if already visited
    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return;
        }
    }

    for (int i = 0; i < cgraph->n_leafs; i++) {
        if (cgraph->leafs[i] == node) {
            return;
        }
    }

    if (node->src0) {
        ggml_visit_parents(cgraph, node->src0);
    }

    if (node->src1) {
        ggml_visit_parents(cgraph, node->src1);
    }

    for (int i = 0; i < MAX_OPT; ++i) {
        if (node->opt[i]) {
            ggml_visit_parents(cgraph, node->opt[i]);
        }
    }

    if (node->op == OP_NONE && node->grad == NULL) {
        // reached a leaf node, not part of the gradient graph (e.g. a constant)
        ASSERT(cgraph->n_leafs < MAX_NODES);

        cgraph->leafs[cgraph->n_leafs] = node;
        cgraph->n_leafs++;
    } else {
        ASSERT(cgraph->n_nodes < MAX_NODES);

        cgraph->nodes[cgraph->n_nodes] = node;
        cgraph->grads[cgraph->n_nodes] = node->grad;
        cgraph->n_nodes++;
    }
}

static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
    if (!expand) {
        cgraph->n_nodes = 0;
        cgraph->n_leafs = 0;
    }

    const int n0 = cgraph->n_nodes;
    UNUSED(n0);

    ggml_visit_parents(cgraph, tensor);

    const int n_new = cgraph->n_nodes - n0;
    PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

    if (n_new > 0) {
        // the last added node should always be starting point
        ASSERT(cgraph->nodes[cgraph->n_nodes - 1] == tensor);
    }
}

void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
    ggml_build_forward_impl(cgraph, tensor, true);
}

struct ggml_cgraph ggml_build_forward(struct ggml_tensor * tensor) {
    struct ggml_cgraph result = {
        / *.n_nodes      =* / 0,
        / *.n_leafs      =* / 0,
        / *.n_threads    =* / 0,
        / *.work_size    =* / 0,
        / *.work         =* / NULL,
        / *.nodes        =* / { NULL },
        / *.grads        =* / { NULL },
        / *.leafs        =* / { NULL },
        / *.perf_runs    =* / 0,
        / *.perf_cycles  =* / 0,
        / *.perf_time_us =* / 0,
    };

    ggml_build_forward_impl(&result, tensor, false);

    return result;
}
*/
/*
struct ggml_cgraph ggml_build_backward(struct ggml_context * ctx, struct ggml_cgraph * gf, bool keep) {
    struct ggml_cgraph result = *gf;

    ASSERT(gf->n_nodes > 0);

    // if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
    if (keep) {
        for (int i = 0; i < gf->n_nodes; i++) {
            struct ggml_tensor * node = gf->nodes[i];

            if (node->grad) {
                node->grad = ggml_dup_tensor(ctx, node);
                gf->grads[i] = node->grad;
            }
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        // because we detached the grad nodes from the original graph, we can afford inplace operations
        if (node->grad) {
            ggml_compute_backward(ctx, node, keep);
        }
    }

    for (int i = gf->n_nodes - 1; i >= 0; i--) {
        struct ggml_tensor * node = gf->nodes[i];

        if (node->is_param) {
            PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
            ggml_build_forward_impl(&result, node->grad, true);
        }
    }

    return result;
}
*/
//
// thread data
//
// synchronization is done via busy loops
// I tried using spin locks, but not sure how to use them correctly - the things I tried were slower than busy loops
//
/*
#ifdef __APPLE__

//#include <os/lock.h>
//
//typedef os_unfair_lock ggml_lock_t;
//
//#define ggml_lock_init(x)    UNUSED(x)
//#define ggml_lock_destroy(x) UNUSED(x)
//#define ggml_lock_lock       os_unfair_lock_lock
//#define ggml_lock_unlock     os_unfair_lock_unlock
//
//#define LOCK_INITIALIZER OS_UNFAIR_LOCK_INIT

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define LOCK_INITIALIZER 0

typedef pthread_t ggml_thread_t;

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#else

//typedef pthread_spinlock_t ggml_lock_t;

//#define ggml_lock_init(x) pthread_spin_init(x, PTHREAD_PROCESS_PRIVATE)
//#define ggml_lock_destroy pthread_spin_destroy
//#define ggml_lock_lock    pthread_spin_lock
//#define ggml_lock_unlock  pthread_spin_unlock

typedef int ggml_lock_t;

#define ggml_lock_init(x)    UNUSED(x)
#define ggml_lock_destroy(x) UNUSED(x)
#define ggml_lock_lock(x)    UNUSED(x)
#define ggml_lock_unlock(x)  UNUSED(x)

#define LOCK_INITIALIZER 0

typedef pthread_t ggml_thread_t;

#define ggml_thread_create pthread_create
#define ggml_thread_join   pthread_join

#endif

struct ggml_compute_state_shared {
    ggml_lock_t spin;

    int n_threads;

    // synchronization primitives
    atomic_int  n_ready;
    atomic_bool has_work;
    atomic_bool stop; // stop all threads
};

struct ggml_compute_state {
    ggml_thread_t thrd;

    struct ggml_compute_params params;
    struct ggml_tensor * node;

    struct ggml_compute_state_shared * shared;
};

static thread_ret_t ggml_graph_compute_thread(void * data) {
    struct ggml_compute_state * state = (struct ggml_compute_state *) data;

    const int n_threads = state->shared->n_threads;

    while (true) {
        if (atomic_fetch_add(&state->shared->n_ready, 1) == n_threads - 1) {
            atomic_store(&state->shared->has_work, false);
        } else {
            while (atomic_load(&state->shared->has_work)) {
                if (atomic_load(&state->shared->stop)) {
                    return 0;
                }
                ggml_lock_lock  (&state->shared->spin);
                ggml_lock_unlock(&state->shared->spin);
            }
        }

        atomic_fetch_sub(&state->shared->n_ready, 1);

        // wait for work
        while (!atomic_load(&state->shared->has_work)) {
            if (atomic_load(&state->shared->stop)) {
                return 0;
            }
            ggml_lock_lock  (&state->shared->spin);
            ggml_lock_unlock(&state->shared->spin);
        }

        // check if we should stop
        if (atomic_load(&state->shared->stop)) {
            break;
        }

        if (state->node) {
            if (state->params.ith < state->params.nth) {
                ggml_compute_forward(&state->params, state->node);
            }

            state->node = NULL;
        } else {
            break;
        }
    }

    return 0;
}

void ggml_graph_compute(struct ggml_context * ctx, struct ggml_cgraph * cgraph) {
    const int n_threads = cgraph->n_threads;

    struct ggml_compute_state_shared state_shared = {
        / *.spin      =* / LOCK_INITIALIZER,
        / *.n_threads =* / n_threads,
        / *.n_ready   =* / 0,
        / *.has_work  =* / false,
        / *.stop      =* / false,
    };
    struct ggml_compute_state * workers = n_threads > 1 ? alloca(sizeof(struct ggml_compute_state)*(n_threads - 1)) : NULL;

    // create thread pool
    if (n_threads > 1) {
        ggml_lock_init(&state_shared.spin);

        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            workers[j] = (struct ggml_compute_state) {
                .thrd   = 0,
                .params = {
                    .type  = TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = n_threads,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                },
                .node   = NULL,
                .shared = &state_shared,
            };

            int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
            ASSERT(rc == 0);
            UNUSED(rc);
        }
    }

    // initialize tasks + work buffer
    {
        size_t work_size = 0;

        // thread scheduling for the different operations
        for (int i = 0; i < cgraph->n_nodes; i++) {
            struct ggml_tensor * node = cgraph->nodes[i];

            switch (node->op) {
                case OP_DUP:
                    {
                        node->n_tasks = 1;
                    } break;
                case OP_ADD:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_SUB:
                case OP_MUL:
                case OP_DIV:
                case OP_SQR:
                case OP_SQRT:
                case OP_SUM:
                case OP_MEAN:
                case OP_REPEAT:
                case OP_ABS:
                case OP_SGN:
                case OP_NEG:
                case OP_STEP:
                case OP_RELU:
                    {
                        node->n_tasks = 1;
                    } break;
                case OP_GELU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_SILU:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_NORM:
                case OP_RMS_NORM:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_MUL_MAT:
                    {
                        node->n_tasks = n_threads;

                        // TODO: use different scheduling for different matrix sizes
                        //const int nr0 = ggml_nrows(node->src0);
                        //const int nr1 = ggml_nrows(node->src1);

                        //node->n_tasks = MIN(n_threads, MAX(1, nr0/128));
                        //printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);

                        size_t cur = 0;

                        // TODO: better way to determine if the matrix is transposed
                        if (node->src0->nb[1] < node->src0->nb[0]) {
                            cur = ggml_nbytes(node)*node->n_tasks; // TODO: this can become (n_tasks-1)
                                                                   // TODO: overestimated by factor of x2 for FP16
                        } else {
                            if (node->src0->type == TYPE_F16 &&
                                node->src1->type == TYPE_F32) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
                                if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1; // TODO: this actually is doing nothing
                                                       //       the threads are still spinning
                                    cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                    //printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
                                    //printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
                                    //printf("cur = %zu\n", cur);
                                } else {
                                    cur = TYPE_SIZE[TYPE_F16]*ggml_nelements(node->src1);
                                }
#else
                                cur = TYPE_SIZE[TYPE_F16]*ggml_nelements(node->src1);
#endif
                            } else if (node->src0->type == TYPE_F32 &&
                                       node->src1->type == TYPE_F32) {
                                cur = 0;
                            } else if (node->src0->type == TYPE_Q4_0 &&
                                       node->src1->type == TYPE_F32) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
                                if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1;
                                    cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                } else {
                                    cur = (TYPE_SIZE[TYPE_Q4_0]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_0];
                                }
#else
                                cur = (TYPE_SIZE[TYPE_Q4_0]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_0];
#endif
                            } else if (node->src0->type == TYPE_Q4_1 &&
                                       node->src1->type == TYPE_F32) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
                                if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
                                    node->n_tasks = 1;
                                    cur = TYPE_SIZE[TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
                                } else {
                                    cur = (TYPE_SIZE[TYPE_Q4_1]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_1];
                                }
#else
                                cur = (TYPE_SIZE[TYPE_Q4_1]*ggml_nelements(node->src1))/BLCK_SIZE[TYPE_Q4_1];
#endif
                            } else {
                                ASSERT(false);
                            }
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case OP_SCALE:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_CPY:
                case OP_RESHAPE:
                case OP_VIEW:
                case OP_PERMUTE:
                case OP_TRANSPOSE:
                case OP_GET_ROWS:
                case OP_DIAG_MASK_INF:
                    {
                        node->n_tasks = 1;
                    } break;
                case OP_SOFT_MAX:
                    {
                        node->n_tasks = n_threads;
                    } break;
                case OP_ROPE:
                    {
                        node->n_tasks = 1;
                    } break;
                case OP_CONV_1D_1S:
                case OP_CONV_1D_2S:
                    {
                        node->n_tasks = n_threads;

                        ASSERT(node->src0->ne[3] == 1);
                        ASSERT(node->src1->ne[2] == 1);
                        ASSERT(node->src1->ne[3] == 1);

                        size_t cur = 0;
                        const int nk = node->src0->ne[0];

                        if (node->src0->type == TYPE_F16 &&
                            node->src1->type == TYPE_F32) {
                            cur = sizeof(ggml_fp16_t)*(
                                    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else if (node->src0->type == TYPE_F32 &&
                                   node->src1->type == TYPE_F32) {
                            cur = sizeof(float)*(
                                    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
                                    ( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
                                    );
                        } else {
                            ASSERT(false);
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case OP_FLASH_ATTN:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        const int ne11 = ggml_up(node->src1->ne[1], SOFT_MAX_UNROLL);

                        if (node->src1->type == TYPE_F32) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == TYPE_F16) {
                            cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case OP_FLASH_FF:
                    {
                        node->n_tasks = n_threads;

                        size_t cur = 0;

                        if (node->src1->type == TYPE_F32) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        if (node->src1->type == TYPE_F16) {
                            cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
                            cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
                        }

                        work_size = MAX(work_size, cur);
                    } break;
                case OP_NONE:
                    {
                        node->n_tasks = 1;
                    } break;
                case OP_COUNT:
                    {
                        ASSERT(false);
                    } break;
            }
        }

        if (cgraph->work != NULL && work_size > cgraph->work_size) {
            ASSERT(false); // TODO: better handling
        }

        if (work_size > 0 && cgraph->work == NULL) {
            cgraph->work_size = work_size + CACHE_LINE_SIZE*(n_threads - 1);

            PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
            cgraph->work = ggml_new_tensor_1d(ctx, TYPE_I8, cgraph->work_size);
        }
    }

    const int64_t perf_start_cycles  = ggml_perf_cycles();
    const int64_t perf_start_time_us = ggml_perf_time_us();

    for (int i = 0; i < cgraph->n_nodes; i++) {
        PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

        struct ggml_tensor * node = cgraph->nodes[i];

        // TODO: this could be used to avoid unnecessary computations, but it needs to be improved
        //if (node->grad == NULL && node->perf_runs > 0) {
        //    continue;
        //}

        const int64_t perf_node_start_cycles  = ggml_perf_cycles();
        const int64_t perf_node_start_time_us = ggml_perf_time_us();

        // INIT
        struct ggml_compute_params params = {
            / *.type  =* / TASK_INIT,
            / *.ith   =* / 0,
            / *.nth   =* / node->n_tasks,
            / *.wsize =* / cgraph->work ? ggml_nbytes(cgraph->work) : 0,
            / *.wdata =* / cgraph->work ? cgraph->work->data : NULL,
        };

        ggml_compute_forward(&params, node);

        // COMPUTE
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct ggml_compute_params) {
                    .type  = TASK_COMPUTE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_store(&state_shared.has_work, true);
        }

        params.type = TASK_COMPUTE;
        ggml_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }
        }

        // FINALIZE
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            // launch thread pool
            for (int j = 0; j < n_threads - 1; j++) {
                workers[j].params = (struct ggml_compute_params) {
                    .type  = TASK_FINALIZE,
                    .ith   = j + 1,
                    .nth   = node->n_tasks,
                    .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
                    .wdata = cgraph->work ? cgraph->work->data : NULL,
                };
                workers[j].node = node;
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) > 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_store(&state_shared.has_work, true);
        }

        params.type = TASK_FINALIZE;
        ggml_compute_forward(&params, node);

        // wait for thread pool
        if (node->n_tasks > 1) {
            if (atomic_fetch_add(&state_shared.n_ready, 1) == n_threads - 1) {
                atomic_store(&state_shared.has_work, false);
            }

            while (atomic_load(&state_shared.has_work)) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }

            atomic_fetch_sub(&state_shared.n_ready, 1);

            while (atomic_load(&state_shared.n_ready) != 0) {
                ggml_lock_lock  (&state_shared.spin);
                ggml_lock_unlock(&state_shared.spin);
            }
        }

        // performance stats (node)
        {
            int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_node_start_cycles;
            int64_t perf_time_us_cur = ggml_perf_time_us() - perf_node_start_time_us;

            node->perf_runs++;
            node->perf_cycles  += perf_cycles_cur;
            node->perf_time_us += perf_time_us_cur;
        }
    }

    // join thread pool
    if (n_threads > 1) {
        atomic_store(&state_shared.stop, true);
        atomic_store(&state_shared.has_work, true);

        for (int j = 0; j < n_threads - 1; j++) {
            int rc = ggml_thread_join(workers[j].thrd, NULL);
            ASSERT(rc == 0);
            UNUSED(rc);
        }

        ggml_lock_destroy(&state_shared.spin);
    }

    // performance stats (graph)
    {
        int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
        int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

        cgraph->perf_runs++;
        cgraph->perf_cycles  += perf_cycles_cur;
        cgraph->perf_time_us += perf_time_us_cur;

        PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
                __func__, cgraph->perf_runs,
                (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
                (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
                (double) perf_time_us_cur     / 1000.0,
                (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
    }
}
*/
/*
void ggml_graph_reset(struct ggml_cgraph * cgraph) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * grad = cgraph->grads[i];

        if (grad) {
            ggml_set_zero(grad);
        }
    }
}

void ggml_graph_print(const struct ggml_cgraph * cgraph) {
    int64_t perf_total_per_op_us[OP_COUNT] = {0};

    PRINT("=== GRAPH ===\n");

    PRINT_DEBUG("n_threads       = %d\n",       cgraph->n_threads);
    PRINT_DEBUG("total work size = %zu bytes\n",cgraph->work_size);

    PRINT("n_nodes = %d\n", cgraph->n_nodes);
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * node = cgraph->nodes[i];

        perf_total_per_op_us[node->op] += node->perf_time_us;

        PRINT(" - %3d: [ %6d, %6d, %6d] %16s %s (%3d) cpu = %7.3f / %7.3f ms, wall = %7.3f / %7.3f ms\n",
                i,
                node->ne[0], node->ne[1], node->ne[2],
                OP_LABEL[node->op], node->is_param ? "x" : node->grad ? "g" : " ", node->perf_runs,
                (double) node->perf_cycles  / (double) ggml_cycles_per_ms(),
                (double) node->perf_cycles  / (double) ggml_cycles_per_ms() / (double) node->perf_runs,
                (double) node->perf_time_us / 1000.0,
                (double) node->perf_time_us / 1000.0 / node->perf_runs);
    }

    PRINT("n_leafs = %d\n", cgraph->n_leafs);
    for (int i = 0; i < cgraph->n_leafs; i++) {
        struct ggml_tensor * node = cgraph->leafs[i];

        PRINT(" - %3d: [ %6d, %6d] %8s\n",
                i,
                node->ne[0], node->ne[1],
                OP_LABEL[node->op]);
    }

    for (int i = 0; i < OP_COUNT; i++) {
        PRINT("perf_total_per_op_us[%16s] = %7.3f ms\n", OP_LABEL[i], (double) perf_total_per_op_us[i] / 1000.0);
    }

    PRINT("========================================\n");
}

// check if node is part of the graph
static bool ggml_graph_find(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    if (cgraph == NULL) {
        return true;
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        if (cgraph->nodes[i] == node) {
            return true;
        }
    }

    return false;
}

static struct ggml_tensor * ggml_graph_get_parent(const struct ggml_cgraph * cgraph, const struct ggml_tensor * node) {
    for (int i = 0; i < cgraph->n_nodes; i++) {
        struct ggml_tensor * parent = cgraph->nodes[i];

        if (parent->grad == node) {
            return parent;
        }
    }

    return NULL;
}

void ggml_graph_dump_dot(const struct ggml_cgraph * gb, const struct ggml_cgraph * gf, const char * filename) {
    char color[16];

    FILE * fp = fopen(filename, "w");
    ASSERT(fp);

    fprintf(fp, "digraph G {\n");
    fprintf(fp, "  newrank = true;\n");
    fprintf(fp, "  rankdir = LR;\n");

    for (int i = 0; i < gb->n_nodes; i++) {
        struct ggml_tensor * node = gb->nodes[i];

        if (ggml_graph_get_parent(gb, node) != NULL) {
            continue;
        }

        if (node->is_param) {
            snprintf(color, sizeof(color), "yellow");
        } else if (node->grad) {
            if (ggml_graph_find(gf, node)) {
                snprintf(color, sizeof(color), "green");
            } else {
                snprintf(color, sizeof(color), "lightblue");
            }
        } else {
            snprintf(color, sizeof(color), "white");
        }

        fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"%d [%d, %d] | <x>%s",
                (void *) node, color,
                i, node->ne[0], node->ne[1],
                OP_SYMBOL[node->op]);

        if (node->grad) {
            fprintf(fp, " | <g>%s\"; ]\n", OP_SYMBOL[node->grad->op]);
        } else {
            fprintf(fp, "\"; ]\n");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct ggml_tensor * node = gb->leafs[i];

        snprintf(color, sizeof(color), "pink");

        if (ggml_nelements(node) == 1) {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>%.1e\"; ]\n",
                    (void *) node, color, ggml_get_f32_1d(node, 0));
        } else {
            fprintf(fp, "  \"%p\" [ \
style = filled; fillcolor = %s; shape = record; \
label=\"<x>CONST %d [%d, %d]\"; ]\n",
                    (void *) node, color,
                    i, node->ne[0], node->ne[1]);
        }
    }

    for (int i = 0; i < gb->n_nodes; i++) {
        struct ggml_tensor * node = gb->nodes[i];

        struct ggml_tensor * parent = ggml_graph_get_parent(gb, node);

        if (node->src0) {
            struct ggml_tensor * parent0 = ggml_graph_get_parent(gb, node->src0);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"x\"; ]\n",
                    parent0 ? (void *) parent0 : (void *) node->src0,
                    parent0 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }

        if (node->src1) {
            struct ggml_tensor * parent1 = ggml_graph_get_parent(gb, node->src1);

            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ arrowhead = %s; style = %s; label = \"y\"; ]\n",
                    parent1 ? (void *) parent1 : (void *) node->src1,
                    parent1 ? "g" : "x",
                    parent ? (void *) parent : (void *) node,
                    parent ? "g" : "x",
                    parent ? "empty" : "vee",
                    parent ? "dashed" : "solid");
        }
    }

    for (int i = 0; i < gb->n_leafs; i++) {
        struct ggml_tensor * node = gb->leafs[i];

        if (node->src0) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"x\"; ]\n",
                    (void *) node->src0, "x",
                    (void *) node, "x");
        }

        if (node->src1) {
            fprintf(fp, "  \"%p\":%s -> \"%p\":%s [ label = \"y\"; ]\n",
                    (void *) node->src1, "x",
                    (void *) node, "x");
        }
    }

    fprintf(fp, "}\n");

    fclose(fp);

    PRINT("%s: dot -Tpng %s -o %s.png && open %s.png\n", __func__, filename, filename, filename);
}

////////////////////////////////////////////////////////////////////////////////

static void OPt_set_params(int np, struct ggml_tensor * const ps[], const float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = ggml_nelements(ps[p]) ;
        // TODO: add function to set tensor from array
        for (int j = 0; j < ne; ++j) {
            ggml_set_f32_1d(ps[p], j, x[i++]);
        }
    }
}

static void OPt_get_params(int np, struct ggml_tensor * const ps[], float * x) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int j = 0; j < ne; ++j) {
            x[i++] = ggml_get_f32_1d(ps[p], j);
        }
    }
}

static void OPt_get_grad(int np, struct ggml_tensor * const ps[], float * g) {
    int i = 0;
    for (int p = 0; p < np; ++p) {
        const int ne = ggml_nelements(ps[p]) ;
        // TODO: add function to get all elements at once
        for (int j = 0; j < ne; ++j) {
            g[i++] = ggml_get_f32_1d(ps[p]->grad, j);
        }
    }
}

//
// ADAM
//
//   ref: https://arxiv.org/pdf/1412.6980.pdf
//

static enum OPt_result OPt_adam(
        struct ggml_context * ctx,
        struct OPt_params params,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb) {
    ASSERT(ggml_is_scalar(f));

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    // these will store the parameters we want to optimize
    struct ggml_tensor * ps[MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            ASSERT(np < MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += ggml_nelements(gf->nodes[i]);
        }
    }

    // constants
    const float alpha = params.adam.alpha;
    const float beta1 = params.adam.beta1;
    const float beta2 = params.adam.beta2;
    const float eps   = params.adam.eps;

    float * x  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // view of the parameters
    float * g1 = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // gradient
    float * g2 = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // gradient squared
    float * m  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // first moment
    float * v  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // second moment
    float * mh = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // first moment hat
    float * vh = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // second moment hat

    float * pf = params.past > 0 ? ggml_new_tensor_1d(ctx, TYPE_F32, params.past)->data : NULL; // past function values

    // initialize
    ggml_vec_set_f32(nx, m, 0.0f);
    ggml_vec_set_f32(nx, v, 0.0f);

    // update view
    OPt_get_params(np, ps, x);

    // compute the function value
    ggml_graph_reset  (gf);
    ggml_set_f32      (f->grad, 1.0f);
    ggml_graph_compute(ctx, gb);

    float fx_prev = ggml_get_f32_1d(f, 0);
    if (pf) {
        pf[0] = fx_prev;
    }

    int n_no_improvement = 0;
    float fx_best = fx_prev;

    // run the optimizer
    for (int t = 0; t < params.adam.n_iter; ++t) {
        PRINT_DEBUG  ("=== iter %d ===\n", t);

        PRINT_DEBUG  ("f      = %10.6f\n", ggml_get_f32_1d(f, 0));
        PRINT_DEBUG_5("df/dx0 = %10.6f\n", ggml_get_f32_1d(ps[0]->grad, 0));
        PRINT_DEBUG_5("df/dx1 = %10.6f\n", ggml_get_f32_1d(ps[1]->grad, 0));

        for (int i = 0; i < np; ++i) {
            PRINT_DEBUG("param %d: %10.6f, g = %10.6f\n", i,
                    ggml_get_f32_1d(ps[i], 0), ggml_get_f32_1d(ps[i]->grad, 0));
        }

        const int64_t t_start_wall = ggml_time_us();
        const int64_t t_start_cpu = ggml_cycles();
        UNUSED(t_start_wall);
        UNUSED(t_start_cpu);

        {
            // update the gradient
            OPt_get_grad(np, ps, g1);

            // m_t = beta1*m_t-1 + (1 - beta1)*g_t
            ggml_vec_scale_f32(nx, m, beta1);
            ggml_vec_mad_f32  (nx, m, g1, 1.0f - beta1);

            // g2 = g1^2
            ggml_vec_sqr_f32  (nx, g2, g1);

            // v_t = beta2*v_t-1 + (1 - beta2)*g_t^2
            ggml_vec_scale_f32(nx, v, beta2);
            ggml_vec_mad_f32  (nx, v, g2, 1.0f - beta2);

            // m^hat = m_t / (1 - beta1^t)
            // v^hat = v_t / (1 - beta2^t)
            // x_t = x_t-1 - alpha*m^hat/(sqrt(v^hat) + eps)
            ggml_vec_cpy_f32  (nx, mh, m);
            ggml_vec_cpy_f32  (nx, vh, v);

            ggml_vec_scale_f32(nx, mh, alpha/(1.0f - powf(beta1, t + 1)));
            ggml_vec_scale_f32(nx, vh,  1.0f/(1.0f - powf(beta2, t + 1)));

            ggml_vec_sqrt_f32 (nx, vh, vh);
            ggml_vec_acc1_f32 (nx, vh, eps);

            ggml_vec_div_f32  (nx, mh, mh, vh);
            ggml_vec_sub_f32  (nx, x,  x,  mh);

            // update the parameters
            OPt_set_params(np, ps, x);
        }

        ggml_graph_reset  (gf);
        ggml_set_f32      (f->grad, 1.0f);
        ggml_graph_compute(ctx, gb);

        const float fx = ggml_get_f32_1d(f, 0);

        // check convergence
        if (fabsf(fx - fx_prev)/fx < params.adam.eps_f) {
            PRINT_DEBUG("converged\n");

            return OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= t) {
                const float rate = (pf[t%params.past] - fx)/fx;

                if (fabs(rate) < params.delta) {
                    return OPT_OK;
                }
            }

            pf[t%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx_best > fx) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                ++n_no_improvement;

                if (n_no_improvement >= params.max_no_improvement) {
                    return OPT_OK;
                }
            }
        }

        fx_prev = fx;

        {
            const int64_t t_end_cpu = ggml_cycles();
            PRINT_DEBUG("time iter:      %5.3f s\n", ((float)(t_end_cpu - t_start_cpu))/CLOCKS_PER_SEC);
            UNUSED(t_end_cpu);

            const int64_t t_end_wall = ggml_time_us();
            PRINT_DEBUG("wall time iter: %5.3f s\n", (t_end_wall - t_start_wall)/1e6);
            UNUSED(t_end_wall);
        }
    }

    return OPT_DID_NOT_CONVERGE;
}
*/
/*
//
// L-BFGS
//
// the L-BFGS implementation below is based on the following implementation:
//
//   https://github.com/chokkan/liblbfgs
//

struct ggml_lbfgs_iteration_data {
    float alpha;
    float ys;
    float * s;
    float * y;
};

static enum OPt_result linesearch_backtracking(
        struct ggml_context * ctx,
        const struct OPt_params * params,
        int nx,
        float * x,
        float * fx,
        float * g,
        float * d,
        float * step,
        const float * xp,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb,
        const int np,
        struct ggml_tensor * ps[]) {
    int count = 0;

    float width  = 0.0f;
    float dg     = 0.0f;
    float finit  = 0.0f;
    float dginit = 0.0f;
    float dgtest = 0.0f;

    const float dec = 0.5f;
    const float inc = 2.1f;

    if (*step <= 0.) {
        return LINESEARCH_INVALID_PARAMETERS;
    }

    // compute the initial gradient in the search direction
    ggml_vec_dot_f32(nx, &dginit, g, d);

    // make sure that d points to a descent direction
    if (0 < dginit) {
        return LINESEARCH_FAIL;
    }

    // initialize local variables
    finit = *fx;
    dgtest = params->lbfgs.ftol*dginit;

    while (true) {
        ggml_vec_cpy_f32(nx, x, xp);
        ggml_vec_mad_f32(nx, x, d, *step);

        // evaluate the function and gradient values
        {
            OPt_set_params(np, ps, x);

            ggml_graph_reset  (gf);
            ggml_set_f32      (f->grad, 1.0f);
            ggml_graph_compute(ctx, gb);

            OPt_get_grad(np, ps, g);

            *fx = ggml_get_f32_1d(f, 0);
        }

        ++count;

        if (*fx > finit + (*step)*dgtest) {
            width = dec;
        } else {
            // Armijo condition is satisfied
            if (params->lbfgs.linesearch == LINESEARCH_BACKTRACKING_ARMIJO) {
                return count;
            }

            ggml_vec_dot_f32(nx, &dg, g, d);

            // check the Wolfe condition
            if (dg < params->lbfgs.wolfe * dginit) {
                width = inc;
            } else {
                if(params->lbfgs.linesearch == LINESEARCH_BACKTRACKING_WOLFE) {
                    // regular Wolfe conditions
                    return count;
                }

                if(dg > -params->lbfgs.wolfe*dginit) {
                    width = dec;
                } else {
                    // strong Wolfe condition (LINESEARCH_BACKTRACKING_STRONG_WOLFE)
                    return count;
                }
                return count;
            }
        }

        if (*step < params->lbfgs.min_step) {
            return LINESEARCH_MINIMUM_STEP;
        }
        if (*step > params->lbfgs.max_step) {
            return LINESEARCH_MAXIMUM_STEP;
        }
        if (params->lbfgs.max_linesearch <= count) {
            return LINESEARCH_MAXIMUM_ITERATIONS;
        }

        (*step) *= width;
    }

    return LINESEARCH_FAIL;
}

static enum OPt_result OPt_lbfgs(
        struct ggml_context * ctx,
        struct OPt_params params,
        struct ggml_tensor * f,
        struct ggml_cgraph * gf,
        struct ggml_cgraph * gb) {
    if (params.lbfgs.linesearch == LINESEARCH_BACKTRACKING_WOLFE ||
        params.lbfgs.linesearch == LINESEARCH_BACKTRACKING_STRONG_WOLFE) {
        if (params.lbfgs.wolfe <= params.lbfgs.ftol || 1. <= params.lbfgs.wolfe) {
            return OPT_INVALID_WOLFE;
        }
    }

    gf->n_threads = params.n_threads;
    gb->n_threads = params.n_threads;

    const int m = params.lbfgs.m;

    // these will store the parameters we want to optimize
    struct ggml_tensor * ps[MAX_PARAMS];

    int np = 0;
    int nx = 0;
    for (int i = 0; i < gf->n_nodes; ++i) {
        if (gf->nodes[i]->is_param) {
            PRINT_DEBUG("found param %d: grad->op = %d\n", np, gf->nodes[i]->grad->op);

            ASSERT(np < MAX_PARAMS);

            ps[np++] = gf->nodes[i];
            nx += ggml_nelements(gf->nodes[i]);
        }
    }

    float * x  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // current parameters
    float * xp = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // previous parameters
    float * g  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // current gradient
    float * gp = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // previous gradient
    float * d  = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data; // search direction

    float * pf = params.past > 0 ? ggml_new_tensor_1d(ctx, TYPE_F32, params.past)->data : NULL; // past function values

    float fx    = 0.0f; // cost function value
    float xnorm = 0.0f; // ||x||
    float gnorm = 0.0f; // ||g||
    float step  = 0.0f;

    // initialize x from the graph nodes
    OPt_get_params(np, ps, x);

    // the L-BFGS memory
    struct ggml_lbfgs_iteration_data * lm = alloca(sizeof(struct ggml_lbfgs_iteration_data)*m);

    for (int i = 0; i < m; ++i) {
        lm[i].alpha = 0.0f;
        lm[i].ys    = 0.0f;
        lm[i].s     = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data;
        lm[i].y     = ggml_new_tensor_1d(ctx, TYPE_F32, nx)->data;
    }

    // evaluate the function value and its gradient
    {
        OPt_set_params(np, ps, x);

        ggml_graph_reset  (gf);
        ggml_set_f32      (f->grad, 1.0f);
        ggml_graph_compute(ctx, gb);

        OPt_get_grad(np, ps, g);

        fx = ggml_get_f32_1d(f, 0);
    }

    if (pf) {
        pf[0] = fx;
    }

    float fx_best = fx;

    // search direction = -gradient
    ggml_vec_neg_f32(nx, d, g);

    // ||x||, ||g||
    ggml_vec_norm_f32(nx, &xnorm, x);
    ggml_vec_norm_f32(nx, &gnorm, g);

    if (xnorm < 1.0f) {
        xnorm = 1.0f;
    }

    // already optimized
    if (gnorm/xnorm <= params.lbfgs.eps) {
        return OPT_OK;
    }

    // initial step
    ggml_vec_norm_inv_f32(nx, &step, d);

    int j                = 0;
    int k                = 1;
    int ls               = 0;
    int end              = 0;
    int bound            = 0;
    int n_no_improvement = 0;

    float ys   = 0.0f;
    float yy   = 0.0f;
    float beta = 0.0f;

    while (true) {
        // store the current position and gradient vectors
        ggml_vec_cpy_f32(nx, xp, x);
        ggml_vec_cpy_f32(nx, gp, g);

        ls = linesearch_backtracking(ctx, &params, nx, x, &fx, g, d, &step, xp, f, gf, gb, np, ps);

        if (ls < 0) {
            // linesearch failed - go back to the previous point and return
            ggml_vec_cpy_f32(nx, x, xp);
            ggml_vec_cpy_f32(nx, g, gp);

            return ls;
        }

        ggml_vec_norm_f32(nx, &xnorm, x);
        ggml_vec_norm_f32(nx, &gnorm, g);

        PRINT_DEBUG("f = %10.6f\n", ggml_get_f32_1d(f, 0));

        if (xnorm < 1.0) {
            xnorm = 1.0;
        }
        if (gnorm/xnorm <= params.lbfgs.eps) {
            // converged
            return OPT_OK;
        }

        // delta-based convergence test
        if (pf != NULL) {
            // need at least params.past iterations to start checking for convergence
            if (params.past <= k) {
                const float rate = (pf[k%params.past] - fx)/fx;

                if (fabs(rate) < params.delta) {
                    return OPT_OK;
                }
            }

            pf[k%params.past] = fx;
        }

        // check for improvement
        if (params.max_no_improvement > 0) {
            if (fx < fx_best) {
                fx_best = fx;
                n_no_improvement = 0;
            } else {
                n_no_improvement++;

                if (n_no_improvement >= params.max_no_improvement) {
                    return OPT_OK;
                }
            }
        }

        if (params.lbfgs.n_iter != 0 && params.lbfgs.n_iter < k + 1) {
            // reached the maximum number of iterations
            return OPT_DID_NOT_CONVERGE;
        }

        // update vectors s and y:
        //   s_{k+1} = x_{k+1} - x_{k} = \step * d_{k}.
        //   y_{k+1} = g_{k+1} - g_{k}.
        //
        ggml_vec_sub_f32(nx, lm[end].s, x, xp);
        ggml_vec_sub_f32(nx, lm[end].y, g, gp);

        // compute scalars ys and yy:
        //     ys = y^t \cdot s    -> 1 / \rho.
        //     yy = y^t \cdot y.
        //
        ggml_vec_dot_f32(nx, &ys, lm[end].y, lm[end].s);
        ggml_vec_dot_f32(nx, &yy, lm[end].y, lm[end].y);

        lm[end].ys = ys;

        // find new search direction
        //   ref: https://en.wikipedia.org/wiki/Limited-memory_BFGS

        bound = (m <= k) ? m : k;
        k++;
        end = (end + 1)%m;

        // initialize search direction with -g
        ggml_vec_neg_f32(nx, d, g);

        j = end;
        for (int i = 0; i < bound; ++i) {
            j = (j + m - 1) % m;
            // \alpha_{j} = \rho_{j} s^{t}_{j} \cdot q_{k+1}
            ggml_vec_dot_f32(nx, &lm[j].alpha, lm[j].s, d);
            lm[j].alpha /= lm[j].ys;
            // q_{i} = q_{i+1} - \alpha_{i} y_{i}
            ggml_vec_mad_f32(nx, d, lm[j].y, -lm[j].alpha);
        }

        ggml_vec_scale_f32(nx, d, ys/yy);

        for (int i = 0; i < bound; ++i) {
            // \beta_{j} = \rho_{j} y^t_{j} \cdot \gamma_{i}
            ggml_vec_dot_f32(nx, &beta, lm[j].y, d);
            beta /= lm[j].ys;
            // \gamma_{i+1} = \gamma_{i} + (\alpha_{j} - \beta_{j}) s_{j}
            ggml_vec_mad_f32(nx, d, lm[j].s, lm[j].alpha - beta);
            j = (j + 1)%m;
        }

        step = 1.0;
    }

    return OPT_DID_NOT_CONVERGE;
}
*/
/*
struct OPt_params OPt_default_params(enum OPt_type type) {
    struct OPt_params result;

    switch (type) {
        case OPT_ADAM:
            {
                result = (struct OPt_params) {
                    .type      = OPT_ADAM,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 100,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .adam = {
                        .n_iter = 10000,
                        .alpha  = 0.001f,
                        .beta1  = 0.9f,
                        .beta2  = 0.999f,
                        .eps    = 1e-8f,
                        .eps_f  = 1e-5f,
                        .eps_g  = 1e-3f,
                    },
                };
            } break;
        case OPT_LBFGS:
            {
                result = (struct OPt_params) {
                    .type      = OPT_LBFGS,
                    .n_threads = 1,
                    .past      = 0,
                    .delta     = 1e-5f,

                    .max_no_improvement = 0,

                    .print_forward_graph  = true,
                    .print_backward_graph = true,

                    .lbfgs = {
                        .m              = 6,
                        .n_iter         = 100,
                        .max_linesearch = 20,

                        .eps      = 1e-5f,
                        .ftol     = 1e-4f,
                        .wolfe    = 0.9f,
                        .min_step = 1e-20f,
                        .max_step = 1e+20f,

                        .linesearch = LINESEARCH_DEFAULT,
                    },
                };
            } break;
    }

    return result;
}

enum OPt_result OPt(
        struct ggml_context * ctx,
        struct OPt_params params,
        struct ggml_tensor * f) {
    bool free_ctx = false;
    if (ctx == NULL) {
        struct ggml_init_params params_ctx = {
            .mem_size   = 16*1024*1024,
            .mem_buffer = NULL,
        };

        ctx = ggml_init(params_ctx);
        if (ctx == NULL) {
            return OPT_NO_CONTEXT;
        }

        free_ctx = true;
    }

    enum OPt_result result = OPT_OK;

    // build forward + backward compute graphs
    struct ggml_cgraph gf = ggml_build_forward (f);
    struct ggml_cgraph gb = ggml_build_backward(ctx, &gf, false);

    switch (params.type) {
        case OPT_ADAM:
            {
                result = OPt_adam(ctx, params, f, &gf, &gb);
            } break;
        case OPT_LBFGS:
            {
                result = OPt_lbfgs(ctx, params, f, &gf, &gb);
            } break;
    }

    if (params.print_forward_graph) {
        ggml_graph_print   (&gf);
        ggml_graph_dump_dot(&gf, NULL, "opt-forward.dot");
    }

    if (params.print_backward_graph) {
        ggml_graph_print   (&gb);
        ggml_graph_dump_dot(&gb, &gf, "opt-backward.dot");
    }

    if (free_ctx) {
        ggml_free(ctx);
    }

    return result;
}

////////////////////////////////////////////////////////////////////////////////
*/
/*
int ggml_cpu_has_avx(void) {
#if defined(__AVX__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx2(void) {
#if defined(__AVX2__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_avx512(void) {
#if defined(__AVX512F__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fma(void) {
#if defined(__FMA__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_neon(void) {
#if defined(__ARM_NEON)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_arm_fma(void) {
#if defined(__ARM_FEATURE_FMA)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_f16c(void) {
#if defined(__F16C__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_fp16_va(void) {
#if defined(__ARM_FEATURE_FP16_VECTOR_ARITHMETIC)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_wasm_simd(void) {
#if defined(__wasm_simd128__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_blas(void) {
#if defined(USE_ACCELERATE) || defined(USE_OPENBLAS)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_sse3(void) {
#if defined(__SSE3__)
    return 1;
#else
    return 0;
#endif
}

int ggml_cpu_has_vsx(void) {
#if defined(__POWER9_VECTOR__)
    return 1;
#else
    return 0;
#endif
}

////////////////////////////////////////////////////////////////////////////////
*/
