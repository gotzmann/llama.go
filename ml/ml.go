package ml

import (
	"fmt"
	"math"
	"os"

	//"github.com/x448/float16"
	"github.com/x448/float16"
)

const (
	MAX_DIMS     = 4
	MAX_NODES    = 4096
	MAX_PARAMS   = 16
	MAX_CONTEXTS = 64
	MAX_OPT      = 4

	QK = 32 // quantization

	TOKEN_BOS = 1
	TOKEN_EOS = 2
)

type DType uint8

// TODO FP8, BFLOAT16
const (
	TYPE_NONE  DType = 0
	TYPE_Q4_0  DType = 1
	TYPE_Q4_1  DType = 2
	TYPE_I8    DType = 3
	TYPE_I16   DType = 4
	TYPE_I32   DType = 5
	TYPE_F16   DType = 6 // TODO FP16
	TYPE_F32   DType = 7 // TODO FP32
	TYPE_COUNT DType = 8 // NB! COUNT should be the last
)

// precomputed exp table for f16 (128 KB)
// static ggml_fp16_t table_exp_f16[1 << 16];
var TableExpFP16 [1 << 16]float16.Float16

var BLCK_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{0, QK, QK, 1, 1, 1, 1, 1}

var TYPE_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{0, 4 + QK/2, 4*2 + QK/2, 1, 2, 4, 2, 4} // FIXME

func TypeSizeFloat(dt DType) float32 {
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
	Type DType

	Dims uint32
	NE   [MAX_DIMS]uint32 // number of elements
	NB   [MAX_DIMS]uint32 // stride in bytes: // FIXME ASAP

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
	TasksCount uint32

	// performance
	//perfRuns   uint32
	//perfCycles uint32
	//perfTime   uint64

	Data []float32 // FIXME Was simple slice before!
	//padding [8]byte
}

// static inline bool ggml_is_contiguous(const struct ggml_tensor * tensor) {
func (tensor *Tensor) IsContiguous() bool {
	//    static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
	//
	return tensor.NB[0] == TYPE_SIZE[tensor.Type] &&
		tensor.NB[1] == tensor.NB[0]*tensor.NE[0]/BLCK_SIZE[tensor.Type] &&
		tensor.NB[2] == tensor.NB[1]*tensor.NE[1] &&
		tensor.NB[3] == tensor.NB[2]*tensor.NE[2]
}

func AreSameShape(a, b *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return (a.NE[0] == b.NE[0]) && (a.NE[1] == b.NE[1]) && (a.NE[2] == b.NE[2]) && (a.NE[3] == b.NE[3])
}

func (t *Tensor) Nelements() uint32 {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return t.NE[0] * t.NE[1] * t.NE[2] * t.NE[3]
}

func (t *Tensor) Nrows() uint32 {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return t.NE[1] * t.NE[2] * t.NE[3]
}

// size_t ggml_nbytes(const struct ggml_tensor * tensor) {
func (t *Tensor) Nbytes() uint32 {
	////static_assert(GGML_MAX_DIMS == 4, "GGML_MAX_DIMS is not 4 - update this function");
	return (t.Nelements() * TYPE_SIZE[t.Type]) / BLCK_SIZE[t.Type]
}

// struct ggml_tensor * ggml_view_tensor(
func ViewTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], src.Data)
}

// ggml.c : ggml_dup_tensor
func DupTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], nil)
}

// struct ggml_tensor * Mul(
func Mul(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, false)
}

// struct ggml_tensor * Mul_inplace(
func MulInplace(ctx *Context, a, b *Tensor) *Tensor {
	return MulImpl(ctx, a, b, true)
}

// struct ggml_tensor * Mul_impl(
func MulImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	if !AreSameShape(a, b) {
		fmt.Printf("\n[STOP] MulImpl - tensors of different shapes!")
		os.Exit(1)
	}

	isNode := false

	if inplace && (a.grad != nil || b.grad != nil) {
		isNode = true
	}

	if inplace {
		////ASSERT(is_node == false);
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
		result.grad = nil
	}

	return result
}

// static inline bool ggml_can_mul_mat(const struct ggml_tensor * t0, const struct ggml_tensor * t1) {
func CanMulMat(t0, t1 *Tensor) bool {

	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");

	return (t0.NE[0] == t1.NE[0]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3]) // FIXME Where NE[1] ??
}

// struct ggml_tensor * ggml_mul_mat(
func MulMat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_mul_mat(a, b));
	////GGML_ASSERT(!ggml_is_transposed(a));

	isNode := false

	if a.grad != nil || b.grad != nil {
		isNode = true
	}

	////const int ne[4] = { a.ne[1], b.ne[1], a.ne[2], b.ne[3] };
	result := NewTensor(ctx, TYPE_F32, min32(a.Dims, b.Dims), a.NE[1], b.NE[1], a.NE[2], b.NE[3], nil) // Check for indexes

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

// ggml_add

func AddImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	//bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	////struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_ADD
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Add(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, false)
}

func AddInplace(ctx *Context, a, b *Tensor) *Tensor {
	return AddImpl(ctx, a, b, true)
}

// ggml_sum

func Sum(ctx *Context, a *Tensor) *Tensor {
	isNode := false

	if a.grad != nil {
		isNode = true
	}

	result := NewTensor1D(ctx, a.Type, 1)

	result.op = OP_SUM
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_sub

func SubImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a.grad || b.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SUB
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Sub(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, false)
}

func SubInplace(ctx *Context, a, b *Tensor) *Tensor {
	return SubImpl(ctx, a, b, true)
}

// ggml_div

func DivImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_are_same_shape(a, b));

	////bool is_node = false;

	////if (!inplace && (a->grad || b->grad)) {
	////is_node = true;
	////}

	////if (inplace) {
	////ASSERT(is_node == false);
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_DIV
	////result->grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Div(ctx *Context, a, b *Tensor) *Tensor {
	return DivImpl(ctx, a, b, false)
}

func DivInplace(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	return DivImpl(ctx, a, b, true)
}

// ggml_sgn

func SgnImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SGN
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Sgn(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, false)
}

func SgnInplace(ctx *Context, a *Tensor) *Tensor {
	return SgnImpl(ctx, a, true)
}

// Repeat

// struct ggml_tensor * Repeat(
func Repeat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_repeat(a, b));

	isNode := false

	if a.grad != nil {
		isNode = true
	}

	if AreSameShape(a, b) && !isNode {
		return a
	}

	//struct ggml_tensor * result = ggml_new_tensor(ctx, a.type, b.n_dims, b.ne);
	result := NewTensor(ctx, a.Type, b.Dims, b.NE[0], b.NE[1], b.NE[2], b.NE[3], nil)

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

func IsScalar(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[0] == 1 && tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsVector(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsMatrix(tensor *Tensor) bool {
	////static_assert(MAX_DIMS == 4, "MAX_DIMS is not 4 - update this function");
	return tensor.NE[2] == 1 && tensor.NE[3] == 1
}

// ggml_get_rows

func GetRows(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b.type == TYPE_I32);
	if !IsMatrix(a) || !IsVector(b) /* FIXME || b.Type != TYPE_I32 */ {
		fmt.Printf("\n[ERROR] GetRows fail basic assertions")
		os.Exit(1)
	}

	isNode := false

	if a.grad != nil || b.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows") // FIXME ??
		os.Exit(1)                        // FIXME ??
	}

	// TODO: implement non F32 return
	//struct ggml_tensor * result = ggml_new_tensor_2d(ctx, a.type, a.ne[0], b.ne[0]);
	result := NewTensor2D(ctx, TYPE_F32, a.NE[0], b.NE[0])

	result.op = OP_GET_ROWS
	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

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
		////ASSERT(false); // TODO: implement backward
		isNode = true
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

// ggml_view_1d
// NB! Originally offset in bytes, but here in floats (4-bytes)
func View1D(ctx *Context, a *Tensor, ne0 uint32, offset uint32) *Tensor {
	////if a.grad != nil {
	////	////ASSERT(false); // gradient propagation is not supported
	////	fmt.Printf("\n[STOP] View1D : gradient propagation is not supported")
	////	os.Exit(1)
	////}

	slice := a.Data[offset:]
	result := NewTensor(ctx, a.Type, 1, ne0, 1, 1, 1, slice) // FIXME

	result.op = OP_VIEW
	result.grad = nil
	result.src0 = a
	result.src1 = nil // TODO: maybe store the offset here?

	return result
}

// static void ggml_build_forward_impl(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor, bool expand) {
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	////UNUSED(n0); // FIXED

	VisitParents(graph, tensor)

	n_new := graph.NodesCount - n0
	////PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

	if n_new > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
		if !(graph.Nodes[graph.NodesCount-1] == tensor) {
			fmt.Printf("\n[STOP] BuildForwardImpl : the last added node should always be starting point!")
			os.Exit(1)
		}
	}
}

// void ggml_build_forward_expand(struct ggml_cgraph * cgraph, struct ggml_tensor * tensor) {
func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}

// static void ggml_visit_parents(struct ggml_cgraph * cgraph, struct ggml_tensor * node) {
func VisitParents(graph *Graph, node *Tensor) {

	if node.grad == nil {
		// this usually happens when we generate intermediate nodes from constants in the backward pass
		// it can also happen during forward pass, if the user performs computations with constants
		if node.op != OP_NONE {
			//PRINT_DEBUG("%s: warning: node %p has no grad, but op %d\n", __func__, (void *) node, node.op);
		}
	}

	// check if already visited
	for i := uint32(0); i < graph.NodesCount; i++ {
		if graph.Nodes[i] == node {
			return
		}
	}

	for i := uint32(0); i < graph.LeafsCount; i++ {
		if graph.Leafs[i] == node {
			return
		}
	}

	if node.src0 != nil {
		VisitParents(graph, node.src0)
	}

	if node.src1 != nil {
		VisitParents(graph, node.src1)
	}

	for i := 0; i < MAX_OPT; i++ {
		if node.opt[i] != nil {
			VisitParents(graph, node.opt[i])
		}
	}

	if node.op == OP_NONE && node.grad == nil {
		// reached a leaf node, not part of the gradient graph (e.g. a constant)
		////ASSERT(cgraph.n_leafs < MAX_NODES);

		graph.Leafs[graph.LeafsCount] = node
		graph.LeafsCount++
	} else {
		////ASSERT(cgraph.n_nodes < MAX_NODES);

		graph.Nodes[graph.NodesCount] = node
		graph.Grads[graph.NodesCount] = node.grad
		graph.NodesCount++
	}
}

// ggml_cpy

func CopyImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {

	////ASSERT(ggml_nelements(a) == ggml_nelements(b));
	if a.Nelements() != b.Nelements() {
		fmt.Printf("\n[HALT] Copy tensors of different dimensions!")
		os.Exit(1)
	}

	isNode := false

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] cpyImpl")
		os.Exit(1)
	}

	// make a view of the destination
	result := ViewTensor(ctx, b)

	result.op = OP_CPY
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Copy(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, false)
}

func CopyInplace(ctx *Context, a, b *Tensor) *Tensor {
	return CopyImpl(ctx, a, b, true)
}

// computation graph
type Graph struct {
	NodesCount   uint32 // FIXME Do not need, having len() ??
	LeafsCount   uint32 // FIXME Do not need, having len() ??
	ThreadsCount uint32

	WorkSize uint32
	Work     *Tensor

	Nodes [MAX_NODES]*Tensor
	Grads [MAX_NODES]*Tensor
	Leafs [MAX_NODES]*Tensor

	// performance
	//perfRuns   uint64
	//perfCycles uint64
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

	//Padding [8]byte
}

// ml/ggml.c:2248
type Context struct {
	//MemSize        uint64
	//MemBuffer      []byte
	//MemBufferOwned bool

	//Objects uint64
	//Objects []Object // FIXME Speedup with *Object?

	//ObjectsBegin *Object
	//ObjectsEnd   *Object

	//Scratch     Scratch
	//ScratchSave Scratch
}

// ggml_new_tensor_1d
func NewTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne0, 1, 1, 1, nil)
}

// ggml_new_tensor_2d
func NewTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil) // FIXME
}

func NewTensor3D(ctx *Context, dt DType, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil) // FIXME
}

func NewTensor4D(ctx *Context, dt DType, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensor(ctx, dt, 4, ne0, ne1, ne2, ne3, nil) // FIXME
}

// ggml_new_tensor_impl
func NewTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	if dt != TYPE_F32 && dt != TYPE_I32 {
		fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
		os.Exit(1)
	}

	// always insert objects at the end of the context's memory pool
	////struct ggml_object * obj_cur = ctx.objects_end;

	////const size_t cur_offs = obj_cur == NULL ? 0 : obj_cur.offs;
	////const size_t cur_size = obj_cur == NULL ? 0 : obj_cur.size;
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

	////char * const mem_buffer = ctx.mem_buffer;
	////struct ggml_object * const obj_new = (struct ggml_object *)(mem_buffer + cur_end);

	//if ctx.Scratch.Data == nil || data != nil {
	////size_needed += sizeof(struct ggml_tensor);

	////if (cur_end + size_needed + OBJECT_SIZE > ctx.mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + size_needed + OBJECT_SIZE, ctx.mem_size);
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

	////if (cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE > ctx.mem_size) {
	////PRINT("%s: not enough space in the context's memory pool (needed %zu, available %zu)\n",
	////    __func__, cur_end + sizeof(struct ggml_tensor) + OBJECT_SIZE, ctx.mem_size);
	////assert(false);
	////return NULL;
	////}

	////data = (char * const) ctx.scratch.data + ctx.scratch.offs;

	////*obj_new = (struct ggml_object) {
	////.offs = cur_end + OBJECT_SIZE,
	////.size = sizeof(struct ggml_tensor),
	////.next = NULL,
	////};

	//printf("scratch offs = %zu, size_needed = %zu\n", ctx.scratch.offs, size_needed);

	////ctx.scratch.offs += size_needed;
	////}

	//if objCur != nil {
	//	objCur.Next = objNew
	//} else {
	// this is the first object in this context
	//	ctx.ObjectsBegin = objNew
	//}

	//ctx.ObjectsEnd = objNew

	//printf("%s: inserted new object at %zu, size = %zu\n", __func__, cur_end, obj_new.size);

	////struct ggml_tensor * const result = (struct ggml_tensor *)(mem_buffer + obj_new.offs);

	////ggml_assert_aligned(result);

	result := Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		op:   OP_NONE,
		//opt:  [4]*Tensor{nil, nil, nil, nil},
	}

	////result->nb[0] = GGML_TYPE_SIZE[type];
	////result->nb[1] = result->nb[0]*(result->ne[0]/GGML_BLCK_SIZE[type]);
	////for (int i = 2; i < GGML_MAX_DIMS; i++) {
	////    result->nb[i] = result->nb[i - 1]*result->ne[i - 1];
	////}

	result.NB[0] = TYPE_SIZE[dt]
	result.NB[1] = TYPE_SIZE[dt] * (result.NE[0] / BLCK_SIZE[dt])
	result.NB[2] = result.NB[1] * result.NE[1]
	result.NB[3] = result.NB[2] * result.NE[2]

	total := ne0 * ne1 * ne2 * ne3

	if data == nil {
		//newData := make([]float32, total, total) // FIXME ASAP use CAP ??
		result.Data = make([]float32, total, total) // &newData
	} else {
		result.Data = data
	}

	return &result
}

// ggml_permute

func Permute(ctx *Context, a *Tensor, axis0, axis1, axis2, axis3 uint32) *Tensor {

	////ASSERT(axis0 >= 0 && axis0 < MAX_DIMS);
	////ASSERT(axis1 >= 0 && axis1 < MAX_DIMS);
	////ASSERT(axis2 >= 0 && axis2 < MAX_DIMS);
	////ASSERT(axis3 >= 0 && axis3 < MAX_DIMS);

	////ASSERT(axis0 != axis1);
	////ASSERT(axis0 != axis2);
	////ASSERT(axis0 != axis3);
	////ASSERT(axis1 != axis2);
	////ASSERT(axis1 != axis3);
	////ASSERT(axis2 != axis3);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Permute error")
		os.Exit(1)
	}

	result := ViewTensor(ctx, a)

	var ne [MAX_DIMS]uint32
	var nb [MAX_DIMS]uint32

	ne[axis0] = a.NE[0]
	ne[axis1] = a.NE[1]
	ne[axis2] = a.NE[2]
	ne[axis3] = a.NE[3]

	nb[axis0] = a.NB[0]
	nb[axis1] = a.NB[1]
	nb[axis2] = a.NB[2]
	nb[axis3] = a.NB[3]

	result.NE[0] = ne[0]
	result.NE[1] = ne[1]
	result.NE[2] = ne[2]
	result.NE[3] = ne[3]

	result.NB[0] = nb[0]
	result.NB[1] = nb[1]
	result.NB[2] = nb[2]
	result.NB[3] = nb[3]

	result.op = OP_PERMUTE
	result.src0 = a
	result.src1 = nil // TODO: maybe store the permutation here?

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

// ggml_rope

func Rope(ctx *Context, a *Tensor, past, dims, mode uint32) *Tensor {
	////ASSERT(n_past >= 0);

	isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] Rope error")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	b := NewTensor1D(ctx, TYPE_I32, 3)
	////((int32_t *) b.data)[0] = past
	b.Data[0] = float32(past)
	////((int32_t *) b.data)[1] = dims
	b.Data[1] = float32(dims)
	////((int32_t *) b.data)[2] = mode
	b.Data[2] = float32(mode)

	result.op = OP_ROPE
	result.src0 = a
	result.src1 = b

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Reshape3D(ctx *Context, a *Tensor, ne0, ne1, ne2 uint32) *Tensor {
	////ASSERT(ggml_is_contiguous(a));
	////ASSERT(ggml_nelements(a) == ne0*ne1*ne2);

	if !a.IsContiguous() {
		fmt.Printf("\n[STOP] Reshape3D : tensor is NOT contiguous!")
		os.Exit(1)
	}

	if a.Nelements() != ne0*ne1*ne2 {
		fmt.Printf("\n[STOP] Reshape3D : different elements number!")
		os.Exit(1)
	}

	////bool is_node = false;

	////if (a.grad) {
	////   //// ASSERT(false); // TODO: implement backward
	////    is_node = true;
	////}

	//ne := [3]uint32{ ne0, ne1, ne2 }
	result := NewTensor(ctx, a.Type, 3, ne0, ne1, ne2, 1, a.Data)

	result.op = OP_RESHAPE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// struct ggml_tensor * ggml_new_f32(struct ggml_context * ctx, float value) {
func NewFP32(ctx *Context, value float32) *Tensor {

	////ctx.scratch_save = ctx.scratch;
	////ctx.scratch.data = NULL;

	result := NewTensor1D(ctx, TYPE_F32, 1)

	////ctx.scratch = ctx.scratch_save;

	SetFP32(result, value)

	return result
}

// struct ggml_tensor * ggml_set_f32(struct ggml_tensor * tensor, float value) {
func SetFP32(tensor *Tensor, value float32) *Tensor {

	////n := tensor.Nrows()
	////nc := tensor.NE[0]
	////n1 := tensor.nb[1];

	////data := tensor.Data

	////switch (tensor.type) {
	////case TYPE_Q4_0:
	////{
	////ASSERT(false);
	////} break;
	////case TYPE_Q4_1:
	////{
	////ASSERT(false);
	////} break;
	////case TYPE_I8:
	////{
	////assert(tensor.nb[0] == sizeof(int8_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i8(nc, (int8_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_I16:
	////{
	////assert(tensor.nb[0] == sizeof(int16_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i16(nc, (int16_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_I32:
	////{
	////assert(tensor.nb[0] == sizeof(int32_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_i32(nc, (int32_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_F16:
	////{
	////assert(tensor.nb[0] == sizeof(ggml_fp16_t));
	////for (int i = 0; i < n; i++) {
	////ggml_vec_set_f16(nc, (ggml_fp16_t *)(data + i*n1), value);
	////}
	////} break;
	////case TYPE_F32:
	////{
	////assert(tensor.nb[0] == sizeof(float));

	// FIXME Optimize with mem zeroing
	n := tensor.Nelements()
	for i := uint32(0); i < n; i++ {
		////ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
		tensor.Data[i] = value
	}

	////} break;
	////case TYPE_COUNT:
	////{
	////ASSERT(false);
	////} break;
	////}

	return tensor
}

// ggml_scale

func ScaleImpl(ctx *Context, a, b *Tensor, inplace bool) *Tensor {
	////ASSERT(ggml_is_scalar(b));
	////ASSERT(ggml_is_padded_1d(a));

	////bool is_node = false;

	if !inplace && (a.grad != nil || b.grad != nil) {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] ScaleImpl : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SCALE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

func Scale(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, false)
}

func ScaleInplace(ctx *Context, a, b *Tensor) *Tensor {
	return ScaleImpl(ctx, a, b, true)
}

// ggml_diag_mask_inf

func DiagMaskInf(ctx *Context, a *Tensor, past uint32) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] DiagMaskInf : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)
	//// FIXME
	//// b := NewI32(ctx, past)
	b := NewFP32(ctx, float32(past))

	result.op = OP_DIAG_MASK_INF
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = b

	return result
}

// ggml_soft_max

func SoftMax(ctx *Context, a *Tensor) *Tensor {
	////bool is_node = false;

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
		fmt.Printf("\n[STOP] SoftMax : assertion failed")
		os.Exit(1)
	}

	// TODO: when implement backward, fix this:
	//struct ggml_tensor * result = inplace ? ggml_view_tensor(ctx, a) : ggml_dup_tensor(ctx, a);
	result := ViewTensor(ctx, a)

	result.op = OP_SOFT_MAX
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// ggml_silu

func SiluImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	////bool is_node = false;

	////if (!inplace && (a.grad)) {
	////is_node = true;
	////}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_SILU
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

func Silu(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, false)
}

func SiluInplace(ctx *Context, a *Tensor) *Tensor {
	return SiluImpl(ctx, a, true)
}

// ggml_step

func StepImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		isNode = true
	}

	var result *Tensor
	if inplace {
		result = ViewTensor(ctx, a)
	} else {
		result = DupTensor(ctx, a)
	}

	result.op = OP_STEP
	result.src0 = a
	result.src1 = nil

	if isNode {
		result.grad = DupTensor(ctx, result)
	} else {
		result.grad = nil
	}

	return result
}

func Step(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, false)
}

func StepInplace(ctx *Context, a *Tensor) *Tensor {
	return StepImpl(ctx, a, true)
}

// ggml_transpose

func Transpose(ctx *Context, a *Tensor) *Tensor {
	////isNode := false

	if a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		////is_node = true;
	}

	result := ViewTensor(ctx, a)

	result.NE[0] = a.NE[1]
	result.NE[1] = a.NE[0]

	result.NB[0] = a.NB[1]
	result.NB[1] = a.NB[0]

	result.op = OP_TRANSPOSE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

/*
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	////UNUSED(n0); FIXME ASAP

	VisitParents(graph, tensor)

	newCount := graph.NodesCount - n0
	////PRINT_DEBUG("%s: visited %d new nodes\n", __func__, n_new);

	if newCount > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
	}
}

func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}*/

func BuildForward(tensor *Tensor) *Graph {

	result := Graph{
		NodesCount: 0,
		LeafsCount: 0,
		// .threads    = 0,
		// .work_size    = 0,
		// *.work         = NULL,

		// FIXME Do use [4096] or [] with append?
		//Nodes: make([4096]*Tensor, 0),
		//Grads: nil,
		//Leafs: nil,

		//.perf_runs    = 0,
		//.perf_cycles  = 0,
		//.perf_time_us = 0,
	}

	BuildForwardImpl(&result, tensor, false)

	return &result
}

func BuildBackward(ctx *Context, gf *Graph, keep bool) Graph {
	////result = *gf
	result := *gf

	////ASSERT(gf.n_nodes > 0);

	// if we are keeping the gradient graph, we have to detach the gradient nodes from the original graph
	if keep {
		for i := uint32(0); i < gf.NodesCount; i++ {
			node := gf.Nodes[i]

			if node.grad != nil {
				node.grad = DupTensor(ctx, node)
				gf.Grads[i] = node.grad
			}
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		// because we detached the grad nodes from the original graph, we can afford inplace operations
		if node.grad != nil {
			ComputeBackward(ctx, node, keep)
		}
	}

	for i := gf.NodesCount - 1; i >= 0; i-- {
		node := gf.Nodes[i]

		if node.isParam {
			////PRINT_DEBUG("%s: found root node %p\n", __func__, (void *) node);
			BuildForwardImpl(&result, node.grad, true)
		}
	}

	return result
}

////////////////////////////////////////////////////////////////////////////////

func ComputeBackward(ctx *Context, tensor *Tensor, inplace bool) {

	src0 := tensor.src0
	src1 := tensor.src1

	switch tensor.op {

	case OP_DUP:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_ADD:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = AddImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_SUB:
		if src0.grad != nil {
			src0.grad = AddImpl(ctx, src0.grad, tensor.grad, inplace)
		}
		if src1.grad != nil {
			src1.grad = SubImpl(ctx, src1.grad, tensor.grad, inplace)
		}
	case OP_MUL:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx, src1, tensor.grad),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					Mul(ctx, src0, tensor.grad),
					inplace)
		}
	case OP_DIV:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx, tensor.grad, src1),
					inplace)
		}
		if src1.grad != nil {
			src1.grad =
				SubImpl(ctx,
					src1.grad,
					Mul(ctx,
						tensor.grad,
						Div(ctx, tensor, src1)),
					inplace)
		}
	case OP_SQR:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Mul(ctx, src0, tensor.grad),
						Repeat(ctx, NewFP32(ctx, 2.0), src0)),
					inplace)
		}
	case OP_SQRT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Div(ctx,
						Repeat(ctx, NewFP32(ctx, 0.5), tensor),
						tensor),
					inplace)
		}
	case OP_SUM:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Repeat(ctx, tensor.grad, src0.grad),
					inplace)
		}
	case OP_MEAN:
		//// ASSERT(false); // TODO: implement
	case OP_REPEAT:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Sum(ctx, tensor.grad),
					inplace)
		}
	case OP_ABS:
		if src0.grad != nil {
			src0.grad =
				AddImpl(ctx,
					src0.grad,
					Mul(ctx,
						Sgn(ctx, src0),
						tensor.grad),
					inplace)
		}
	case OP_SGN:
		if src0.grad != nil {
			// noop
		}
	case OP_NEG:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx, src0.grad, tensor.grad, inplace)
		}
	case OP_STEP:
		if src0.grad != nil {
			// noop
		}
	case OP_RELU:
		if src0.grad != nil {
			src0.grad = SubImpl(ctx,
				src0.grad,
				Mul(ctx,
					Step(ctx, src0),
					tensor.grad),
				inplace)
		}
	case OP_GELU:
		//// ASSERT(false); // TODO: not implemented
	case OP_SILU:
		//// ASSERT(false); // TODO: not implemented
	case OP_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_RMS_NORM:
		//// ASSERT(false); // TODO: not implemented
	case OP_MUL_MAT:
		if src0.grad != nil {
			// TODO: this requires outer product - ggml_out_prod(ctx, src1, tensor.grad);
			//// ASSERT(false);
			fmt.Printf("\n[HALT] ComputeBackward : OP_MUL_MAT with src0.grad!")
			os.Exit(1)
		}
		if src1.grad != nil {
			src1.grad =
				AddImpl(ctx,
					src1.grad,
					// TODO: fix transpose, the node will break the graph connections
					MulMat(ctx, Transpose(ctx, src0), tensor.grad),
					inplace)
		}
	case OP_SCALE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CPY:
		//// ASSERT(false); // TODO: not implemented
	case OP_RESHAPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_VIEW:
		//// ASSERT(false); // not supported
	case OP_PERMUTE:
		//// ASSERT(false); // TODO: not implemented
	case OP_TRANSPOSE:
		//// ASSERT(false); // TODO: not implemented
	case OP_GET_ROWS:
		//// ASSERT(false); // TODO: not implemented
	case OP_DIAG_MASK_INF:
		//// ASSERT(false); // TODO: not implemented
	case OP_SOFT_MAX:
		//// ASSERT(false); // TODO: not implemented
	case OP_ROPE:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_1S:
		//// ASSERT(false); // TODO: not implemented
	case OP_CONV_1D_2S:
		//// ASSERT(false); // TODO: not implemented
	case OP_FLASH_ATTN:
		//// ASSERT(false); // not supported
	case OP_FLASH_FF:
		//// ASSERT(false); // not supported
	case OP_NONE:
		// nop
	case OP_COUNT:
		//// ASSERT(false);
	}
}

// ---

type TaskType uint8

const (
	TASK_INIT     TaskType = 0
	TASK_COMPUTE  TaskType = 1
	TASK_FINALIZE TaskType = 2
)

type ComputeParams struct {
	Type TaskType
	ith  uint32
	nth  uint32
	// work buffer for all threads
	wsize uint32
	wdata []float32 // byte // FIXME *void
}

type ComputeStateShared struct {
	////threads uint32
	////ggml_lock_t spin;
	// synchronization primitives
	////atomic_int  n_ready;
	////atomic_bool has_work;
	////atomic_bool stop; // stop all threads
}

type ComputeState struct {
	threads uint32
	params  *ComputeParams
	node    *Tensor
	shared  *ComputeStateShared
}

// Golang doesn’t have unary Bitwise NOT(~) like other programming languages
// Here, you have to use Bitwise XOR(^) operator as Bitwise NOT
func up32(n uint32) uint32 {
	return uint32(n+31) & ^uint32(31)
}

func up(n, m uint32) uint32 {
	// assert m is a power of 2
	////GGML_ASSERT((m & (m - 1)) == 0);
	return uint32(n+m-1) & ^uint32(m-1)
}

func max(a, b int) int {
	if a >= b {
		return a
	}
	return b
}

func GraphCompute(ctx *Context, graph *Graph) {

	//fmt.Printf("\n\n === GraphCompute : %d nodes ===\n\n", graph.NodesCount) // DEBUG

	threads := graph.ThreadsCount

	////struct ggml_compute_state_shared state_shared = {
	////    spin      = LOCK_INITIALIZER,
	////    threads = threads,
	////    n_ready   = 0,
	////    has_work  = false,
	////    stop      = false,
	////};

	var workers []ComputeState
	if threads > 1 {
		//////workers = alloca(sizeof(struct ggml_compute_state)*(threads - 1))
		fmt.Printf("\n[HALT] Parallelism is not allowed!")
		os.Exit(1)
		workers = make([]ComputeState, graph.ThreadsCount)
	}

	// create thread pool
	if threads > 1 {
		////ggml_lock_init(&state_shared.spin);

		////atomic_store(&state_shared.has_work, true);

		////for (int j = 0; j < threads - 1; j++) {
		////    workers[j] = (struct ggml_compute_state) {
		////        .thrd   = 0,
		////        .params = {
		////           .type  = TASK_COMPUTE,
		////           .ith   = j + 1,
		////           .nth   = threads,
		////           .wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
		////           .wdata = cgraph->work ? cgraph->work->data : NULL,
		////       },
		////       .node   = NULL,
		////       .shared = &state_shared,
		////   };

		////   int rc = ggml_thread_create(&workers[j].thrd, NULL, ggml_graph_compute_thread, &workers[j]);
		////   ASSERT(rc == 0);
		////   UNUSED(rc);
		////}
	}

	fmt.Printf("\n\n === GraphCompute INIT : %d nodes ===\n\n", graph.NodesCount) // DEBUG
	// initialize tasks + work buffer
	{
		workSize := 0

		// thread scheduling for the different operations
		// TasksCount might be 0, 1, or ThreadsCount
		for i := uint32(0); i < graph.NodesCount; i++ {

			////struct ggml_tensor * node = cgraph->nodes[i];
			node := graph.Nodes[i]

			// DEBUG
			fmt.Printf("\n\n###### #%d - %d - %d [ %d,%d ] %.4f \n", i, node.op, node.Type, node.NE[1], node.NE[2], node.Data[0])

			switch node.op {

			case OP_DUP:
				node.TasksCount = 1
			case OP_ADD:
				node.TasksCount = threads
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
				node.TasksCount = 1
			case OP_GELU:
				node.TasksCount = threads
			case OP_SILU:
				node.TasksCount = threads
			case OP_NORM:
			case OP_RMS_NORM:
				node.TasksCount = threads
			case OP_MUL_MAT:
				node.TasksCount = threads
				// TODO: use different scheduling for different matrix sizes
				//const int nr0 = ggml_nrows(node->src0);
				//const int nr1 = ggml_nrows(node->src1);
				//node->n_tasks = MIN(threads, MAX(1, nr0/128));
				//printf("nr0 = %8d, nr1 = %8d, nr0*nr1 = %8d, n_tasks = %d\n", nr0, nr1, nr0*nr1, node->n_tasks);
				cur := 0
				if node.src0.Type == TYPE_F16 && node.src1.Type == TYPE_F32 {
					fmt.Printf("\n[HALT] GraphCompute : data types are not supprted!")
					os.Exit(1)
					////#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
					////				if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
					////					node->n_tasks = 1; // TODO: this actually is doing nothing
					////									   //       the threads are still spinning
					////					cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
					////					//printf("src0: ne0 = %d, ne1 = %d, ne = %d\n", node->src0->ne[0], node->src0->ne[1], node->src0->ne[0]*node->src0->ne[1]);
					////					//printf("src1: ne0 = %d, ne1 = %d, ne = %d\n", node->src1->ne[0], node->src1->ne[1], node->src1->ne[0]*node->src1->ne[1]);
					////					//printf("cur = %zu\n", cur);
					////				} else {
					////					cur = GGML_TYPE_SIZE[GGML_TYPE_F16]*ggml_nelements(node->src1);
					////				}
					////#else
					////cur = TYPE_SIZE[TYPE_F16] * node.src1.Nelements()
					////#endif
				} else if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
					cur = 0 // FIXME WHY ??
				} else if node.src0.Type == TYPE_Q4_0 && node.src1.Type == TYPE_F32 {
					fmt.Printf("\n[HALT] GraphCompute : data types are not supprted!")
					os.Exit(1)
					////#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
					////				if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
					////					node->n_tasks = 1;
					////					cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
					////				} else {
					////					cur = (GGML_TYPE_SIZE[GGML_TYPE_Q4_0]*ggml_nelements(node->src1))/GGML_BLCK_SIZE[GGML_TYPE_Q4_0];
					////				}
					////#else
					////cur = TYPE_SIZE[TYPE_Q4_0] * node.src1.Nelements() / BLCK_SIZE[TYPE_Q4_0]
					////#endif
				} else if node.src0.Type == TYPE_Q4_1 && node.src1.Type == TYPE_F32 {
					fmt.Printf("\n[HALT] GraphCompute : data types are not supprted!")
					os.Exit(1)
					////#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)
					////				if (ggml_compute_forward_mul_mat_use_blas(node->src0, node->src1, node)) {
					////					node->n_tasks = 1;
					////					cur = GGML_TYPE_SIZE[GGML_TYPE_F32]*(node->src0->ne[0]*node->src0->ne[1]);
					////				} else {
					////					cur = (GGML_TYPE_SIZE[GGML_TYPE_Q4_1]*ggml_nelements(node->src1))/GGML_BLCK_SIZE[GGML_TYPE_Q4_1];
					////				}
					////#else
					////cur = TYPE_SIZE[TYPE_Q4_1] * node.src1.Nelements() / BLCK_SIZE[TYPE_Q4_1]
					////#endif
				} else {
					fmt.Printf("\n[HALT] GraphCompute : data types are not supprted!")
					os.Exit(1)
				}
				workSize = max(workSize, cur)
			case OP_SCALE:
				node.TasksCount = threads
			case OP_CPY:
			case OP_RESHAPE:
			case OP_VIEW:
			case OP_PERMUTE:
			case OP_TRANSPOSE:
			case OP_GET_ROWS:
			case OP_DIAG_MASK_INF:
				node.TasksCount = 1
			case OP_SOFT_MAX:
				node.TasksCount = threads
			case OP_ROPE:
				////node.TasksCount = 1
			case OP_CONV_1D_1S:
			case OP_CONV_1D_2S:
				node.TasksCount = threads
				////ASSERT(node->src0->ne[3] == 1);
				////ASSERT(node->src1->ne[2] == 1);
				////ASSERT(node->src1->ne[3] == 1);
				cur := 0
				nk := node.src0.NE[0]
				////if node.src0.Type == TYPE_F16 && node.src1.Type == TYPE_F32 {
				////cur = sizeof(ggml_fp16_t)*(
				////    nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +
				////( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]
				////);
				////fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
				////os.Exit(1)
				////} else if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
				if node.src0.Type == TYPE_F32 && node.src1.Type == TYPE_F32 {
					// FIXME Check up32() vs ggml_up32
					////cur = sizeof(float)*(nk*ggml_up32(node->src0->ne[1])*node->src0->ne[2] +( 2*(nk/2) + node->src1->ne[0])*node->src1->ne[1]);
					cur = int(4 * (nk*up32(node.src0.NE[1])*node.src0.NE[2] + (2*(nk/2)+node.src1.NE[0])*node.src1.NE[1]))
				} else {
					////ASSERT(false);
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}
				workSize = max(workSize, cur)
			case OP_FLASH_ATTN:
				node.TasksCount = threads
				cur := 0
				////ne11 := Up(node.src1.NE[1], SOFT_MAX_UNROLL)
				const SOFT_MAX_UNROLL = uint32(4)
				ne11 := up(node.src1.NE[1], SOFT_MAX_UNROLL)
				if node.src1.Type == TYPE_F32 {
					////cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
					cur = 4 * int(ne11) * int(node.TasksCount)
					cur += 4 * int(ne11) * int(node.TasksCount) // this is overestimated by x2
				}
				if node.src1.Type == TYPE_F16 {
					////cur  = sizeof(float)*ne11*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*ne11*node->n_tasks; // this is overestimated by x2
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}
				workSize = max(workSize, cur)
			case OP_FLASH_FF:
				node.TasksCount = threads
				cur := 0
				if node.src1.Type == TYPE_F32 {
					////cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
					cur = int(4 * node.src1.NE[1] * node.TasksCount)  // TODO: this can become (n_tasks-1)
					cur += int(4 * node.src1.NE[1] * node.TasksCount) // this is overestimated by x2
				}
				if node.src1.Type == TYPE_F16 {
					////cur  = sizeof(float)*node->src1->ne[1]*node->n_tasks; // TODO: this can become (n_tasks-1)
					////cur += sizeof(float)*node->src1->ne[1]*node->n_tasks; // this is overestimated by x2
					fmt.Printf("\n[HALT] Mismatch of data within compute graph!")
					os.Exit(1)
				}
				workSize = max(workSize, cur)
			case OP_NONE:
				node.TasksCount = 1
			case OP_COUNT:
				fmt.Printf("\n[HALT] Something wrong with compute graph!")
				os.Exit(1)
			}
		}

		if graph.Work != nil && workSize > int(graph.WorkSize) {
			////ASSERT(false); // TODO: better handling
			fmt.Printf("\n[HALT] Something wrong with work size of compute graph!")
			os.Exit(1)
		}

		const CACHE_LINE_SIZE = uint32(64)
		if workSize > 0 && graph.Work == nil {
			graph.WorkSize = uint32(workSize) + CACHE_LINE_SIZE*(threads-1)
			////PRINT_DEBUG("%s: allocating work buffer for graph (%zu bytes)\n", __func__, cgraph->work_size);
			////graph.Work = NewTensor1D(ctx, TYPE_I8, graph.WorkSize)
			graph.Work = NewTensor1D(ctx, TYPE_F32 /*TYPE_I8*/, graph.WorkSize)
			fmt.Printf("\n[COMPUTE] graph.WorkSize = %d", graph.WorkSize)
		}
	}

	////const int64_t perf_start_cycles  = ggml_perf_cycles();
	////const int64_t perf_start_time_us = ggml_perf_time_us();

	fmt.Printf("\n\n === GraphCompute START : %d nodes ===\n\n", graph.NodesCount) // DEBUG

	for i := uint32(0); i < graph.NodesCount; i++ {
		////PRINT_DEBUG_5("%s: %d/%d\n", __func__, i, cgraph->n_nodes);

		node := graph.Nodes[i]

		// DEBUG
		fmt.Printf("\n\n###### #%d - %d - %d [%d,%d] %.4f \n", i, node.op, node.Type, node.NE[1], node.NE[2], node.Data[0])

		// TODO: this could be used to avoid unnecessary computations, but it needs to be improved
		//if (node->grad == NULL && node->perf_runs > 0) {
		//    continue;
		//}

		////const int64_t perf_node_start_cycles  = ggml_perf_cycles();
		////const int64_t perf_node_start_time_us = ggml_perf_time_us();

		var wsize uint32
		var wdata []float32
		if graph.Work != nil {
			wsize = graph.Work.Nbytes()
			wdata = graph.Work.Data
		}

		params := ComputeParams{
			Type: TASK_INIT,
			ith:  0,
			nth:  node.TasksCount,
			////wsize: graph.work ? ggml_nbytes(cgraph->work) : 0,
			////wdata: graph.work ? cgraph->work->data : NULL,
			wsize: wsize,
			wdata: wdata,
		}

		//fmt.Printf("\n[COMPUTE] ComputeForward | TASK_INIT | ...")
		ComputeForward(&params, node) // TASK_INIT

		// --- COMPUTE

		if node.TasksCount > 1 {

			////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
			////atomic_store(&state_shared.has_work, false);
			////}

			////while (atomic_load(&state_shared.has_work)) {
			////ggml_lock_lock  (&state_shared.spin);
			////ggml_lock_unlock(&state_shared.spin);
			////}

			var wsize uint32
			var wdata []float32
			if graph.Work != nil {
				wsize = graph.Work.Nbytes()
				wdata = graph.Work.Data
			}

			// launch thread pool
			for j := uint32(0); j < threads-1; j++ {
				workers[j].params = &ComputeParams{
					Type: TASK_COMPUTE,
					ith:  j + 1,
					nth:  node.TasksCount,
					////.wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
					////.wdata = cgraph->work ? cgraph->work->data : NULL,
					wsize: wsize,
					wdata: wdata,
				}
				workers[j].node = node
			}

			////atomic_fetch_sub(&state_shared.n_ready, 1);

			////while (atomic_load(&state_shared.n_ready) > 0) {
			////ggml_lock_lock  (&state_shared.spin);
			////ggml_lock_unlock(&state_shared.spin);
			////}

			////atomic_store(&state_shared.has_work, true);
		}

		//fmt.Printf("\n[COMPUTE] ComputeForward | TASK_COMPUTE | ...")
		params.Type = TASK_COMPUTE
		ComputeForward(&params, node)

		// wait for thread pool
		////if (node->n_tasks > 1) {
		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) != 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}
		////}

		// --- FINALIZE

		if node.TasksCount > 1 {
			////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
			////atomic_store(&state_shared.has_work, false);
			////}

			////while (atomic_load(&state_shared.has_work)) {
			////ggml_lock_lock  (&state_shared.spin);
			////ggml_lock_unlock(&state_shared.spin);
			////}

			var wsize uint32
			var wdata []float32
			if graph.Work != nil {
				wsize = graph.Work.Nbytes()
				wdata = graph.Work.Data
			}

			// launch thread pool
			for j := uint32(0); j < threads-1; j++ {
				workers[j].params = &ComputeParams{
					Type: TASK_FINALIZE,
					ith:  j + 1,
					nth:  node.TasksCount,
					////.wsize = cgraph->work ? ggml_nbytes(cgraph->work) : 0,
					////.wdata = cgraph->work ? cgraph->work->data : NULL,
					wsize: wsize,
					wdata: wdata,
				}
				workers[j].node = node
			}

			////atomic_fetch_sub(&state_shared.n_ready, 1);

			////while (atomic_load(&state_shared.n_ready) > 0) {
			////ggml_lock_lock  (&state_shared.spin);
			////ggml_lock_unlock(&state_shared.spin);
			////}

			////atomic_store(&state_shared.has_work, true);
		}

		//fmt.Printf("\n[COMPUTE] ComputeForward | TASK_FINALIZE | ...")
		params.Type = TASK_FINALIZE
		ComputeForward(&params, node)

		// wait for thread pool
		////if node.TasksCount > 1 {
		////if (atomic_fetch_add(&state_shared.n_ready, 1) == threads - 1) {
		////atomic_store(&state_shared.has_work, false);
		////}

		////while (atomic_load(&state_shared.has_work)) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}

		////atomic_fetch_sub(&state_shared.n_ready, 1);

		////while (atomic_load(&state_shared.n_ready) != 0) {
		////ggml_lock_lock  (&state_shared.spin);
		////ggml_lock_unlock(&state_shared.spin);
		////}
		////}

		// performance stats (node)
		////{
		////int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_node_start_cycles;
		////int64_t perf_time_us_cur = ggml_perf_time_us() - perf_node_start_time_us;

		////node->perf_runs++;
		////node->perf_cycles  += perf_cycles_cur;
		////node->perf_time_us += perf_time_us_cur;
		////}
	}

	// join thread pool
	//if (threads > 1) {
	////atomic_store(&state_shared.stop, true);
	////atomic_store(&state_shared.has_work, true);

	////for (int j = 0; j < threads - 1; j++) {
	////int rc = ggml_thread_join(workers[j].thrd, NULL);
	////ASSERT(rc == 0);
	////UNUSED(rc);
	////}

	////ggml_lock_destroy(&state_shared.spin);
	////}

	// performance stats (graph)
	////{
	////int64_t perf_cycles_cur  = ggml_perf_cycles()  - perf_start_cycles;
	////int64_t perf_time_us_cur = ggml_perf_time_us() - perf_start_time_us;

	////cgraph->perf_runs++;
	////cgraph->perf_cycles  += perf_cycles_cur;
	////cgraph->perf_time_us += perf_time_us_cur;

	////PRINT_DEBUG("%s: perf (%d) - cpu = %.3f / %.3f ms, wall = %.3f / %.3f ms\n",
	////        __func__, cgraph->perf_runs,
	////        (double) perf_cycles_cur      / (double) ggml_cycles_per_ms(),
	////        (double) cgraph->perf_cycles  / (double) ggml_cycles_per_ms() / (double) cgraph->perf_runs,
	////        (double) perf_time_us_cur     / 1000.0,
	////        (double) cgraph->perf_time_us / 1000.0 / cgraph->perf_runs);
	////}
}

/////////////////////////////////

func ComputeForward(params *ComputeParams, tensor *Tensor) {

	//fmt.Printf("\n[COMPUTE] ComputeForward...")
	////ASSERT(params);

	switch tensor.op {

	case OP_DUP:
		////ggml_compute_forward_dup(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_dup")
		os.Exit(1)
	case OP_ADD:
		////ggml_compute_forward_add(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_add")
		////os.Exit(1)
		ComputeForwardAddFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_SUB:
		////ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sub")
		os.Exit(1)
	case OP_MUL:
		////ggml_compute_forward_mul(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mul")
		////os.Exit(1)
		ComputeForwardMulFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_DIV:
		////ggml_compute_forward_div(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_div")
		os.Exit(1)
	case OP_SQR:
		////ggml_compute_forward_sqr(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqr")
		os.Exit(1)
	case OP_SQRT:
		////ggml_compute_forward_sqrt(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sqrt")
		os.Exit(1)
	case OP_SUM:
		////ggml_compute_forward_sum(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sum")
		os.Exit(1)
	case OP_MEAN:
		////ggml_compute_forward_mean(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mean")
		os.Exit(1)
	case OP_REPEAT:
		////ggml_compute_forward_repeat(params, tensor->src0, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_repeat")
		////os.Exit(1)
		ComputeForwardRepeatFP32(params, tensor.src0, tensor)
	case OP_ABS:
		////ggml_compute_forward_abs(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_abs")
		os.Exit(1)
	case OP_SGN:
		////ggml_compute_forward_sgn(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sgn")
		os.Exit(1)
	case OP_NEG:
		////ggml_compute_forward_neg(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_neg")
		os.Exit(1)
	case OP_STEP:
		////ggml_compute_forward_step(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_step")
		os.Exit(1)
	case OP_RELU:
		////ggml_compute_forward_relu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_relu")
		os.Exit(1)
	case OP_GELU:
		////ggml_compute_forward_gelu(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_gelu")
		os.Exit(1)
	case OP_SILU:
		////ggml_compute_forward_silu(params, tensor->src0, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_silu")
		////os.Exit(1)
		ComputeForwardSiluFP32(params, tensor.src0, tensor)
	case OP_NORM:
		////ggml_compute_forward_norm(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_norm")
		os.Exit(1)
	case OP_RMS_NORM:
		////ggml_compute_forward_rms_norm(params, tensor->src0, tensor);
		//fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rms_norm")
		//os.Exit(1)
		ComputeForwardRMSNormFP32(params, tensor.src0, tensor)
	case OP_MUL_MAT:
		////ggml_compute_forward_mul_mat(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_mul_mat")
		////os.Exit(1)
		ComputeForwardMulMatFP32(params, tensor.src0, tensor.src1, tensor)
		////fmt.Printf("[ return ]")
	case OP_SCALE:
		////ggml_compute_forward_scale(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_scale")
		////os.Exit(1)
		ComputeForwardScaleFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_CPY:
		////ggml_compute_forward_cpy(params, tensor->src0, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_cpy")
		////os.Exit(1)
		ComputeForwardDupFP32(params, tensor.src0, tensor) // FIXME Double Check
	case OP_RESHAPE:
		////ggml_compute_forward_reshape(params, tensor->src0, tensor);
		///fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_reshape")
		////os.Exit(1)
		ComputeForwardReshape(params, tensor.src0, tensor) // NOP
	case OP_VIEW:
		////ggml_compute_forward_view(params, tensor->src0);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_view")
		////os.Exit(1)
		ComputeForwardView(params, tensor.src0) // NOP
	case OP_PERMUTE:
		////ggml_compute_forward_permute(params, tensor->src0);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_permute")
		////os.Exit(1)
		ComputeForwardPermute(params, tensor.src0) // NOP
	case OP_TRANSPOSE:
		////ggml_compute_forward_transpose(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_transpose")
		os.Exit(1)
	case OP_GET_ROWS:
		////ggml_compute_forward_get_rows(params, tensor->src0, tensor->src1, tensor);
		//fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rows")
		//os.Exit(1)
		ComputeForwardGetRows(params, tensor.src0, tensor.src1, tensor)
	case OP_DIAG_MASK_INF:
		////ggml_compute_forward_diag_mask_inf(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_diag_mask_inf")
		////os.Exit(1)
		ComputeForwardDiagMaskInfFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_SOFT_MAX:
		////ggml_compute_forward_soft_max(params, tensor->src0, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_soft_max")
		////os.Exit(1)
		ComputeForwardSoftMaxFP32(params, tensor.src0, tensor)
	case OP_ROPE:
		////ggml_compute_forward_rope(params, tensor->src0, tensor->src1, tensor);
		////fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_rope")
		////os.Exit(1)
		ComputeForwardRopeFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_CONV_1D_1S:
		////ggml_compute_forward_conv_1d_1s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_1s")
		os.Exit(1)
	case OP_CONV_1D_2S:
		////ggml_compute_forward_conv_1d_2s(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_conv_1d_2s")
		os.Exit(1)
	case OP_FLASH_ATTN:
		////int32_t t = ggml_get_i32_1d(tensor->opt[1], 0);
		////ASSERT(t == 0 || t == 1);
		////bool masked = t != 0;
		////ggml_compute_forward_flash_attn(params, tensor->src0, tensor->src1, tensor->opt[0], masked, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_attn")
		os.Exit(1)
	case OP_FLASH_FF:
		////ggml_compute_forward_flash_ff(params, tensor->src0, tensor->src1, tensor->opt[0], tensor->opt[1], tensor->opt[2], tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_flash_ff")
		os.Exit(1)
	case OP_NONE:
		// nop
	case OP_COUNT:
		////ASSERT(false);
		fmt.Printf("\n[HALT] ComputeForward got OP_COUNT method!")
		os.Exit(1)
	}
}

////////////////////////////////////////////////////////////////////////////////

// ---

// FIXME ASAP Play with it!
func VecCopyFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] = x[i]
	}
}

// NB! Only FP32
// ggml_compute_forward_get_rows_f32
func ComputeForwardGetRows(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardGetRows ] ")
	////assert(params->ith == 0);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	nc := src0.NE[0]
	nr := src1.Nelements()

	////assert( dst->ne[0] == nc);
	////assert( dst->ne[1] == nr);
	////assert(src0->nb[0] == sizeof(float));

	if dst.NE[0] != nc || dst.NE[1] != nr || src0.NB[0] != TYPE_SIZE[TYPE_F32] /*TYPE_SIZE[TYPE_I32]*/ {
		fmt.Printf("[HALT]ComputeForwardGetRows : wrong dimensions!")
		os.Exit(1)
	}

	// FIXME Speed-up
	////for row := uint32(0); row < nr; row++ {
	////	for column := uint32(0); column < nc; column++ {
	////		(*dst.Data)[row*nr+column] = (*src0.Data)[row*nr+column]
	////	}
	////}

	// DEBUG
	fmt.Printf("\n\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %f |", ii, dst.Data[ii])
	}

	for i := uint32(0); i < nr; i++ {
		////const int r = ((int32_t *) src1->data)[i];
		r := uint32(src1.Data[i]) // FIXME WTF ??

		fmt.Printf(" [ r = %d | dst = %d | src = %d ]", r, i*dst.NE[0], r*src0.NE[0])

		////ggml_vec_cpy_f32(nc,
		////        (float *) ((char *)  dst->data + i*dst->nb[1]),
		////        (float *) ((char *) src0->data + r*src0->nb[1]));

		// FIXME ASAP and double check!
		// VecCopyFP32(nc, (*dst.Data)[i*dst.NE[0]:], (*src0.Data)[uint32(r)*src0.NE[0]:])
		// VecCopyFP32(nc, dst.Data[i*dst.NB[1]/4:], src0.Data[r*src0.NB[1]/4:])
		VecCopyFP32(nc, dst.Data[i*dst.NE[0]:], src0.Data[r*src0.NE[0]:])
	}

	// DEBUG
	fmt.Printf("\n\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %f |", ii, dst.Data[ii])
	}
}

// NB! FP32 Only
// ggml_compute_forward_rms_norm_f32
func ComputeForwardRMSNormFP32(params *ComputeParams, src0, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardRMSNormFP32 ] ")
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));
	////GGML_ASSERT(src0->nb[0] == sizeof(float));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	ith := params.ith
	nth := params.nth

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]

	nb1 := dst.NB[1]
	nb2 := dst.NB[2]
	nb3 := dst.NB[3]

	eps := 1e-5 // TODO: make this a parameter

	// TODO: optimize
	for i03 := uint32(0); i03 < ne03; i03++ {
		for i02 := uint32(0); i02 < ne02; i02++ {
			for i01 := uint32(ith); i01 < ne01; i01 += nth {

				////const float * x = (float *) ((char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03);
				x := src0.Data[i01*nb01/4+i02*nb02/4+i03*nb03/4:]

				////ggml_float mean = 0.0;
				mean := 0.0
				////for (int i00 = 0; i00 < ne00; i00++) {
				// TODO Simplify to directly access [src]
				for i00 := uint32(0); i00 < ne00; i00++ {
					////mean += x[i00] * x[i00];
					mean += float64(x[i00] * x[i00])
				}

				////mean /= ne00;
				mean /= float64(ne00)

				////const float scale = 1.0/sqrt(mean + eps);
				scale := float32(1.0 / math.Sqrt(mean+eps))

				// TODO Simplify to directly update [dst]
				////float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
				y := dst.Data[i01*nb1/4+i02*nb2/4+i03*nb3/4:]
				/*
					////memcpy(y, x, ne00 * sizeof(float));
					for i := uint32(0); i < ne00*4/4; i++ {
						y[i] = x[i]
					}

					////ggml_vec_scale_f32(ne00, y, scale);
					VecScaleFP32(ne00, y, float32(scale))
				*/

				for i := uint32(0); i < ne00; i++ {
					y[i] = x[i] * scale
				}
			}
		}
	}
}

// inline static void ggml_vec_scale_f32(const int n, float * y, const float   v) {
func VecScaleFP32(n uint32, y []float32, v float32) {
	////#if defined(GGML_SIMD)
	////const int np = (n & ~(GGML_F32_STEP - 1));

	////GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	////GGML_F32_VEC ay[GGML_F32_ARR];

	////for (int i = 0; i < np; i += GGML_F32_STEP) {
	////for (int j = 0; j < GGML_F32_ARR; j++) {
	////ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
	////ay[j] = GGML_F32_VEC_MUL(ay[j], vx);

	////GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
	////}
	////}

	// leftovers
	////for (int i = np; i < n; ++i) {
	////y[i] *= v;
	////}
	////#else
	// scalar
	for i := uint32(0); i < n; i++ {
		y[i] *= v
	}
	////#endif
}

// NB! FP32 Only
// ggml_compute_forward_repeat
func ComputeForwardRepeatFP32(params *ComputeParams, src0, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardRepeatFP32 ] ")
	////assert(params->ith == 0);
	////assert(ggml_can_repeat(src0, dst));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// TODO: implement support for rank > 2 tensors
	////assert(src0->ne[2] == 1);
	////assert(src0->ne[3] == 1);
	////assert( dst->ne[2] == 1);
	////assert( dst->ne[3] == 1);

	nc := dst.NE[0]
	nr := dst.NE[1]
	nc0 := src0.NE[0]
	nr0 := src0.NE[1]
	ncr := nc / nc0 // guaranteed to be an integer due to the check in ggml_can_repeat
	nrr := nr / nr0 // guaranteed to be an integer due to the check in ggml_can_repeat

	// TODO: support for transposed / permuted tensors
	////assert( dst->nb[0] == sizeof(float));
	////assert(src0->nb[0] == sizeof(float));

	// TODO: maybe this is not optimal?
	for i := uint32(0); i < nrr; i++ {
		for j := uint32(0); j < ncr; j++ {
			for k := uint32(0); k < nr0; k++ {

				////VecCopyFP32(nc0,
				////	(*dst.Data)[i*nr0+k+j*nc0:],
				////	(*src0.Data)[k:])

				////ggml_vec_cpy_f32(nc0,
				////(float *) ((char *)  dst->data + (i*nr0 + k)*( dst->nb[1]) + j*nc0*( dst->nb[0])),
				////(float *) ((char *) src0->data + (        k)*(src0->nb[1])));

				// FIXME ASAP Double Check !!
				VecCopyFP32(nc0,
					dst.Data[(i*nr0+k)*dst.NB[1]/4+j*nc0*dst.NB[0]/4:],
					src0.Data[k*src0.NB[1]/4:])
			}
		}
	}
}

func VecMulFP32(n uint32, z, x, y []float32) {
	for i := uint32(0); i < n; i++ {
		z[i] = x[i] * y[i]
	}
}

// NB! FP32 Only
// ggml_compute_forward_mul
func ComputeForwardMulFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardMulFP32 ] ")
	////assert(params->ith == 0);
	////assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	n := src0.Nrows()
	nc := src0.NE[0]

	////assert( dst->nb[0] == sizeof(float));
	////assert(src0->nb[0] == sizeof(float));
	////assert(src1->nb[0] == sizeof(float));

	for i := uint32(0); i < n; i++ {

		// !!! BUG !!!
		//VecMulFP32(nc, dst.Data[i:], src0.Data[i:], src1.Data[i:])
		VecMulFP32(nc, dst.Data[i*dst.NE[0]:], src0.Data[i*src0.NE[0]:], src1.Data[i*src1.NE[0]:])

		////ggml_vec_mul_f32(nc,
		////(float *) ((char *) dst->data  + i*( dst->nb[1])),
		////(float *) ((char *) src0->data + i*(src0->nb[1])),
		////(float *) ((char *) src1->data + i*(src1->nb[1])));
	}
}

// inline static void ggml_vec_dot_f32(const int n, float * restrict s, const float * restrict x, const float * restrict y) {
func VecDotFP32(n uint32, x, y []float32) float32 {

	sumf := float32(0.0)

	////#ifdef GGML_SIMD
	////    const int np = (n & ~(GGML_F32_STEP - 1));

	////    GGML_F32_VEC sum[GGML_F32_ARR] = { GGML_F32_VEC_ZERO };

	////    GGML_F32_VEC ax[GGML_F32_ARR];
	////    GGML_F32_VEC ay[GGML_F32_ARR];

	////    for (int i = 0; i < np; i += GGML_F32_STEP) {
	////        for (int j = 0; j < GGML_F32_ARR; j++) {
	////            ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
	////            ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);

	////            sum[j] = GGML_F32_VEC_FMA(sum[j], ax[j], ay[j]);
	////        }
	////    }

	////    // reduce sum0..sum3 to sum0
	////    GGML_F32_VEC_REDUCE(sumf, sum);

	////    // leftovers
	////    for (int i = np; i < n; ++i) {
	////        sumf += x[i]*y[i];
	////    }
	////#else

	// scalar
	for i := uint32(0); i < n; i++ {
		sumf += x[i] * y[i]
	}

	////#endif

	////*s = sumf;
	return sumf
}

// inline static void ggml_vec_mad_f32(const int n, float * restrict y, const float * restrict x, const float v) {
func VecMadFP32(n uint32, y, x []float32, v float32) {
	////		#if defined(GGML_SIMD)
	////		const int np = (n & ~(GGML_F32_STEP - 1));

	////		GGML_F32_VEC vx = GGML_F32_VEC_SET1(v);

	////		GGML_F32_VEC ax[GGML_F32_ARR];
	////		GGML_F32_VEC ay[GGML_F32_ARR];

	////		for (int i = 0; i < np; i += GGML_F32_STEP) {
	////			for (int j = 0; j < GGML_F32_ARR; j++) {
	////				ax[j] = GGML_F32_VEC_LOAD(x + i + j*GGML_F32_EPR);
	////				ay[j] = GGML_F32_VEC_LOAD(y + i + j*GGML_F32_EPR);
	////				ay[j] = GGML_F32_VEC_FMA(ay[j], ax[j], vx);

	////				GGML_F32_VEC_STORE(y + i + j*GGML_F32_EPR, ay[j]);
	////			}
	////		}

	////		// leftovers
	////		for (int i = np; i < n; ++i) {
	////			y[i] += x[i]*v;
	////		}
	////	#else

	// scalar
	for i := uint32(0); i < n; i++ {
		y[i] += x[i] * v
	}

	////	#endif
}

// inline static void ggml_vec_acc_f32 (const int n, float * y, const float * x)                  { for (int i = 0; i < n; ++i) y[i] += x[i];        }
func VecAccFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] += x[i]
	}
}

// NB! FP32 Only
// ggml_compute_forward_mul_mat_f32
func ComputeForwardMulMatFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardMulMatFP32 ] ")
	////int64_t t0 = ggml_perf_time_us();
	////UNUSED(t0);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	fmt.Printf("\n\n>>> ComputeForwardMulMatFP32 IN <<<\n")

	fmt.Printf("\n=== SRC0 === %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	//ne10 := src1.NE[0] // for BLAS only
	ne11 := src1.NE[1]
	//ne12 := src1.NE[2]
	//ne13 := src1.NE[3]

	//ne0 := dst.NE[0]
	//ne1 := dst.NE[1]
	//ne2 := dst.NE[2]
	//ne3 := dst.NE[3]
	//ne := ne0 * ne1 * ne2 * ne3

	//nb00 := src0.NB[0]
	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]

	//nb10 := src1.NB[0]
	nb11 := src1.NB[1]
	nb12 := src1.NB[2]
	nb13 := src1.NB[3]

	nb0 := dst.NB[0]
	nb1 := dst.NB[1]
	nb2 := dst.NB[2]
	nb3 := dst.NB[3]

	ith := params.ith
	nth := params.nth

	////assert(ne02 == ne12);
	////assert(ne03 == ne13);
	////assert(ne2  == ne12);
	////assert(ne3  == ne13);

	// TODO: we don't support permuted src0
	////assert(nb00 == sizeof(float) || nb01 == sizeof(float));

	// dst cannot be transposed or permuted
	////assert(nb0 == sizeof(float));
	////assert(nb0 <= nb1);
	////assert(nb1 <= nb2);
	////assert(nb2 <= nb3);

	////assert(ne0 == ne01);
	////assert(ne1 == ne11);
	////assert(ne2 == ne02);
	////assert(ne3 == ne03);

	// nb01 >= nb00 - src0 is not transposed
	//   compute by src0 rows

	/*
		////#if defined(GGML_USE_ACCELERATE) || defined(GGML_USE_OPENBLAS)

		////if (ggml_compute_forward_mul_mat_use_blas(src0, src1, dst)) {
		////GGML_ASSERT(nb10 == sizeof(float));

		if params.ith != 0 {
		return
		}

		if params.Type == TASK_INIT {
		return
		}

		if params.Type == TASK_FINALIZE {
		return
		}

		for i03 := uint32(0); i03 < ne03; i03++ {
		for i02 := uint32(0); i02 < ne02; i02++ {

		const float * x = (float *) (src0->data);

		////const float * y = (float *) ((char *) src1->data + i02*nb12 + i03*nb13);

		////float * d = (float *) ((char *) dst->data + i02*nb2 + i03*nb3);

		// zT = y * xT
		////{
		////cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
		////ne11, ne01, ne10,
		////1.0f,    y, ne10,
		////         x, ne10,
		////0.0f,    d, ne01);
		////}
		////}
		////}

		//printf("CBLAS F32 = %f ms, %d x %d x %d x %d\n", (ggml_perf_time_us() - t0)/1000.0, ne0, ne1, ne2, ne3);

		////return;
		////}
		////#endif
	*/

	// TODO: do not support transposed src1
	////assert(nb10 == sizeof(float));
	////if nb10 == 4 {
	////	fmt.Printf("\n[HALT] Do not support transposed src1")
	////	os.Exit(1)
	////}

	// parallelize by src0 rows using ggml_vec_dot_f32

	// total rows in src0
	nr := ne01 * ne02 * ne03

	// rows per thread
	dr := (nr + nth - 1) / nth

	// row range for this thread
	ir0 := dr * ith
	ir1 := min32(ir0+dr, nr)

	////void * wdata = params->wdata;

	for ir := uint32(ir0); ir < ir1; ir++ {

		// src0 indices
		i03 := ir / (ne02 * ne01)
		i02 := (ir - i03*ne02*ne01) / ne01
		i01 := (ir - i03*ne02*ne01 - i02*ne01)

		// src1 indices
		i13 := i03
		i12 := i02
		//i11 := ic

		// dst indices
		i0 := i01
		//i1 := i11
		i2 := i02
		i3 := i03

		for ic := uint32(0); ic < ne11; ic++ {

			i11 := ic
			i1 := i11

			////ggml_vec_dot_f32(ne00,
			////	(float *) ((char *)  dst->data + (i0*nb0 + i1*nb1 + i2*nb2 + i3*nb3)),
			////	(float *) ((char *) src0->data + (i01*nb01 + i02*nb02 + i03*nb03)),
			////	(float *) ((char *) src1->data + (i11*nb11 + i12*nb12 + i13*nb13)));

			////(*dst.Data)[i0*nb0+i1*nb1+i2*nb2+i3*nb3] =
			////	VecDotFP32(ne00,
			////		(*src0.Data)[i01*nb01+i02*nb02+i03*nb03:],
			////		(*src1.Data)[i11*nb11+i12*nb12+i13*nb13:])

			dst.Data[i0*nb0/4+i1*nb1/4+i2*nb2/4+i3*nb3/4] =
				VecDotFP32(ne00,
					src0.Data[i01*nb01/4+i02*nb02/4+i03*nb03/4:],
					src1.Data[i11*nb11/4+i12*nb12/4+i13*nb13/4:])

			//fmt.Printf(" # %f = %f * %f # ",
			//	(*dst.Data)[i0*nb0+i1*nb1+i2*nb2+i3*nb3],
			//	(*src0.Data)[i01*nb01+i02*nb02+i03*nb03],
			//	(*src1.Data)[i11*nb11+i12*nb12+i13*nb13])
		}
	}

	// DEBUG IDEAL

	fmt.Printf("\n\n>>> ComputeForwardMulMatFP32 OUT <<<\n")

	fmt.Printf("\n=== SRC0 === %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	//fmt.Printf("\n>>> ComputeForwardMulMatFP32 <<<")
	//fmt.Printf("\n\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	//for ii := 0; ii < 8; ii++ {
	//	fmt.Printf("| DST[%d] = %f |", ii, dst.Data[ii])
	//}

	//os.Exit(0)

	//int64_t t1 = ggml_perf_time_us();
	//static int64_t acc = 0;
	//acc += t1 - t0;
	//if (t1 - t0 > 10) {
	//    printf("\n");
	//    printf("ne00 = %5d, ne01 = %5d, ne02 = %5d, ne03 = %5d\n", ne00, ne01, ne02, ne03);
	//    printf("nb00 = %5d, nb01 = %5d, nb02 = %5d, nb03 = %5d\n", nb00, nb01, nb02, nb03);
	//    printf("ne10 = %5d, ne11 = %5d, ne12 = %5d, ne13 = %5d\n", ne10, ne11, ne12, ne13);
	//    printf("nb10 = %5d, nb11 = %5d, nb12 = %5d, nb13 = %5d\n", nb10, nb11, nb12, nb13);

	//	   printf("XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX task %d/%d: %d us, acc = %d\n", ith, nth, (int) (t1 - t0), (int) acc);
	//}
}

// ggml_compute_forward_view

func ComputeForwardView(params *ComputeParams, src0 *Tensor) {
	// NOP
	////UNUSED(params);
	////UNUSED(src0);
}

func ComputeForwardCopy(params *ComputeParams, src0, dst *Tensor) {
	////ggml_compute_forward_dup(params, src0, dst);
	ComputeForwardDupFP32(params, src0, dst)
}

// FIXME ASAP
// FIXME [dst] IS main tensor and [src0] IS inside
// ggml_compute_forward_dup_f32
func ComputeForwardDupFP32(params *ComputeParams, src0, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardDupFP32 ] ")
	////GGML_ASSERT(params->ith == 0);
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_nelements(dst) == ggml_nelements(src0));

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardDupFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if dst.Nelements() != src0.Nelements() {
		fmt.Printf("[HALT] ComputeForwardDupFP32 : [dst] and [src0] capacities are different!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	fmt.Printf("\n\n>>> ml.Copy <<< >>> ComputeForwardDupFP32 IN <<<\n")
	fmt.Printf("\n=== SRC === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 12; ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 12; ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	nb00 := src0.NB[0]
	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]

	////if (ggml_is_contiguous(src0) && src0->type == dst->type) {
	if src0.IsContiguous() && src0.Type == dst.Type {
		////memcpy(dst->data, src0->data, ggml_nelements(dst) * GGML_TYPE_SIZE[src0->type]);
		////return;
		////copy(dst.Data, src0.Data)
		n := dst.Nelements()
		for i := uint32(0); i < n; i++ {
			if i == 28672 && (len(dst.Data) <= 28672 || len(src0.Data) <= 28672) {
				fmt.Printf("THATS-IT")
				return
			}

			dst.Data[i] = src0.Data[i]
		}

		fmt.Printf("\nCONTIGIOUS")
		fmt.Printf("\n\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
		for ii := 0; ii < 12; ii++ {
			fmt.Printf("%.4f  ", dst.Data[ii])
		}
		//os.Exit(0);

		return
	}

	// src0 is NOT contigious

	//fmt.Printf("[HALT] ComputeForwardDupFP32 for NOT contiguous is NOT implemented yet!")
	//os.Exit(1)

	// --- supporting only 4-bytes data for [src0] and FP32 for [dst]

	//if src0.Type == TYPE_F32 && dst.Type == TYPE_F32 {

	if src0.NB[0] == TYPE_SIZE[TYPE_F32] {

		if dst.Type == TYPE_F32 {

			id := uint32(0) // Row number ??
			//// rs := ne00 * nb00
			rs := ne00 * nb00 / 4 // FIXME Row size in 4-bytes elements

			for i03 := uint32(0); i03 < ne03; i03++ {
				for i02 := uint32(0); i02 < ne02; i02++ {
					for i01 := uint32(0); i01 < ne01; i01++ {
						////const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
						src0Ptr := src0.Data[i01*nb01/4+i02*nb02/4+i03*nb03/4:]
						////char * dst_ptr = (char *) dst->data + id*rs;
						dstPtr := dst.Data[id*rs:]

						////memcpy(dst_ptr, src0_ptr, rs);
						for i := uint32(0); i < rs; i++ {
							dstPtr[i] = src0Ptr[i] // FIXME ASAP / Double Check
						}

						id++
					}
				}
			}
			////} else if (dst->type == GGML_TYPE_F16) {
			////    int id = 0;
			////    ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

			////    for (int i03 = 0; i03 < ne03; i03++) {
			////        for (int i02 = 0; i02 < ne02; i02++) {
			////            for (int i01 = 0; i01 < ne01; i01++) {
			////                for (int i00 = 0; i00 < ne00; i00++) {
			////                    const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

			////                    dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
			////                    id++;
			////                }
			////            }
			////        }
			////    }
		} else {
			////GGML_ASSERT(false); // TODO: implement
			fmt.Printf("[HALT] ComputeForwardDupFP32 : not supported tensor type!")
			os.Exit(1)
		}
	} else {

		//printf("%s: this is not optimal - fix me\n", __func__);

		if dst.Type == TYPE_F32 {

			id := 0
			////dstPtr = (float *) dst->data;

			for i03 := uint32(0); i03 < ne03; i03++ {
				for i02 := uint32(0); i02 < ne02; i02++ {
					for i01 := uint32(0); i01 < ne01; i01++ {
						for i00 := uint32(0); i00 < ne00; i00++ {
							//src0Ptr := src0.Data[i00*nb00/4 + i01*nb01/4 + i02*nb02/4 + i03*nb03/4:]
							//dstPtr[id] = *src0_ptr;
							// FIXME DoubleCheck
							dst.Data[id] = src0.Data[i00*nb00/4+i01*nb01/4+i02*nb02/4+i03*nb03/4]
							id++
						}
					}
				}
			}
			////} else if (dst->type == GGML_TYPE_F16) {
			////    int id = 0;
			////    ggml_fp16_t * dst_ptr = (ggml_fp16_t *) dst->data;

			////    for (int i03 = 0; i03 < ne03; i03++) {
			////        for (int i02 = 0; i02 < ne02; i02++) {
			////            for (int i01 = 0; i01 < ne01; i01++) {
			////                for (int i00 = 0; i00 < ne00; i00++) {
			////                    const float * src0_ptr = (float *) ((char *) src0->data + i00*nb00 + i01*nb01 + i02*nb02 + i03*nb03);

			////                    dst_ptr[id] = GGML_FP32_TO_FP16(*src0_ptr);
			////                    id++;
			////                }
			////            }
			////        }
			////    }
		} else {
			////GGML_ASSERT(false) // TODO: implement
			fmt.Printf("[HALT] ComputeForwardDupFP32 : not supported tensor type!")
			os.Exit(1)
		}
	}

	//fmt.Printf("\n\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	//for ii := 0; ii < 8; ii++ {
	//	fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	//}
	//os.Exit(0);

	fmt.Printf("\n\n>>> COPY <<< >>> ComputeForwardDupFP32 OUT <<<\n")
	fmt.Printf("\nNOT CONTIGIOUS")
	fmt.Printf("\n=== SRC === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 12; ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 12; ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}
}

// ggml_compute_forward_reshape
func ComputeForwardReshape(params *ComputeParams, src0, dst *Tensor) {
	// NOP
	////UNUSED(params);
	////UNUSED(src0);
	////UNUSED(dst);
}

// ggml_compute_forward_permute
func ComputeForwardPermute(params *ComputeParams, src0 *Tensor) {
	// NOP
	////UNUSED(params);
	////UNUSED(src0);
}

// ggml_compute_forward_rope
func ComputeForwardRopeFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardRopeFP32 ] ")
	////assert(params->ith == 0);
	////assert(src1->type == GGML_TYPE_I32);
	////assert(ggml_nelements(src1) == 3);

	if src1.Nelements() != 3 {
		fmt.Printf("\n[HALT] ComputeForwardRopeFP32 : src1 has NOT EXACT 3 elements!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	fmt.Printf("\n\n>>> ComputeForwardRopeFP32 <<<")
	fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	}
	fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f %f %f\n",
		src1.NE[0], src1.NE[1], src1.NE[2],
		src1.Data[0], src1.Data[1], src1.Data[2]) // DEBUG
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	}

	pastCount := uint32(src1.Data[0])
	dims := uint32(src1.Data[1])
	mode := uint32(src1.Data[2])

	//const int ne0 = src0->ne[0];
	ne1 := src0.NE[1]
	ne2 := src0.NE[2]
	ne3 := src0.NE[3]

	nb0 := src0.NB[0]
	nb1 := src0.NB[1]
	nb2 := src0.NB[2]
	nb3 := src0.NB[3]

	//printf("ne0: %d, ne1: %d, ne2: %d, ne3: %d\n", ne0, ne1, ne2, ne3);
	//printf("n_past = %d, ne2 = %d\n", n_past, ne2);

	////assert(nb0 == sizeof(float));

	var modeCount uint32
	if mode == 0 {
		modeCount = 0
	} else {
		modeCount = pastCount
	}

	// TODO: optimize
	for i3 := uint32(0); i3 < ne3; i3++ {
		for i2 := modeCount; i2 < ne2; i2++ {

			////const int p = (mode == 0 ? n_past + i2 : i2);
			var p uint32
			if mode == 0 {
				p = pastCount + i2
			} else {
				p = i2
			}

			for i1 := uint32(0); i1 < ne1; i1++ {
				for i0 := 0; i0 < int(dims); i0 += 2 { // WHY 2 ??

					////const double theta = pow(10000.0, ((double)-i0)/n_dims);
					theta := math.Pow(10000.0, float64(-i0)/float64(dims))

					cosTheta := math.Cos(float64(p) * theta)
					sinTheta := math.Sin(float64(p) * theta)

					////const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
					offset := i3*nb3/4 + i2*nb2/4 + i1*nb1/4 + uint32(i0)*nb0/4
					src := src0.Data[offset:]
					////   float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
					dstData := dst.Data[offset:]

					//if len(src) <= 0 {
					//	fmt.Printf("THATS-IT-02")
					//return
					//}

					x0 := float64(src[0])
					x1 := float64(src[1])

					dstData[0] = float32(x0*cosTheta - x1*sinTheta)
					dstData[1] = float32(x0*sinTheta + x1*cosTheta)

					//x0 = 0.0 // DEBUG
				}
			}
		}
	}

	fmt.Printf("\n=== SRC === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	}
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	}
	//os.Exit(1);
}

// ggml_compute_forward_scale_f32
func ComputeForwardScaleFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardScaleFP32 ] ")
	////GGML_ASSERT(ggml_is_contiguous(src0));
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));
	////GGML_ASSERT(ggml_is_scalar(src1));

	if !src0.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardScaleFP32 : [src0] is NOT contiguous!")
		os.Exit(1)
	}

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardScaleFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	//fmt.Printf("\n\n>>> ComputeForwardScaleFP32 <<<")
	//fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
	//src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
	//src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	//for ii := 0; ii < 8; ii++ {
	//	fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	//}
	//fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f \n",
	//	src1.NE[0], src1.NE[1], src1.NE[2], src1.Data[0]) // DEBUG
	//fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	//for ii := 0; ii < 8; ii++ {
	//	fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	//}
	//os.Exit(1)

	// scale factor
	v := src1.Data[0]

	ith := params.ith
	nth := params.nth

	nc := src0.NE[0]
	nr := src0.Nrows()

	// rows per thread
	dr := (nr + nth - 1) / nth

	// row range for this thread
	ir0 := dr * ith
	ir1 := min(int(ir0)+int(dr), int(nr))

	for i1 := ir0; int(i1) < ir1; i1++ {
		////ggml_vec_scale_f32(nc, (float *) ((char *) dst->data + i1*(dst->nb[1])), v);
		////VecScaleFP32(nc, (*dst.Data)[i1*dst.NE[0]:], v)
		VecScaleFP32(nc, dst.Data[i1*dst.NB[1]/4:], v)
	}

	fmt.Printf("\n---")
	fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	}
	fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f \n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.Data[0]) // DEBUG
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	}
	//os.Exit(1)
}

// ggml_compute_forward_diag_mask_inf
func ComputeForwardDiagMaskInfFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardDiagMaskInfFP32 ] ")
	////assert(params->ith == 0);
	////assert(src1->type == GGML_TYPE_I32);
	////assert(ggml_nelements(src1) == 1);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	fmt.Printf("\n\n>>> ComputeForwardDiagMaskInfFP32 IN <<<\n")

	fmt.Printf("\n=== SRC0 === %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(10, len(src0.Data)); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < min(10, len(src1.Data)); ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(10, len(dst.Data)); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	////const int n_past = ((int32_t *) src1->data)[0];
	pastCount := uint32(src1.Data[0])

	// TODO: handle transposed/permuted matrices

	n := src0.Nrows()
	nc := src0.NE[0]
	nr := src0.NE[1]
	nz := n / nr

	////assert( dst->nb[0] == sizeof(float));
	////assert(src0->nb[0] == sizeof(float));

	for k := uint32(0); k < nz; k++ {
		for j := uint32(0); j < nr; j++ {
			for i := pastCount; i < nc; i++ {
				if i > pastCount+j {
					////*(float *)((char *) dst->data + k*dst->nb[2] + j*dst->nb[1] + i*dst->nb[0]) = -INFINITY;
					////(*dst.Data)[k*dst.NE[0]*dst.NE[1]+j*dst.NE[0]+i] = float32(math.Inf(-1)) // TODO Use const
					dst.Data[k*dst.NB[2]/4+j*dst.NB[1]/4+i*dst.NB[0]/4] = float32(math.Inf(-1)) // TODO Use const
					// FIXME ^^^ SRC and DST Data slices are the same! Both will be overwritten here
				}
			}
		}
	}

	fmt.Printf("\n\n>>> ComputeForwardDiagMaskInfFP32 OUT <<<\n")

	fmt.Printf("\n=== SRC0 === %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(12, len(src0.Data)); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < min(12, len(src1.Data)); ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(12, len(dst.Data)); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

}

func maxFloat(x, y float32) float32 {
	if x >= y {
		return x
	}
	return y
}

/*
// inline static void ggml_vec_max_f32(const int n, float * s, const float * x) {
func VecMaxFP32(n uint32, s *float32, x []float32) {
	////#ifndef GGML_USE_ACCELERATE
	max := float32(math.Inf(-1))
	for i := uint32(0); i < n; i++ {
		max = maxFloat(max, x[i])
	}
	////*s = max;
	*s = max
	// //#else
	// //vDSP_maxv(x, 1, s, n);
	// //#endif
}
*/

func VecMaxFP32(n uint32, x []float32) float32 {
	max := float32(math.Inf(-1)) // TODO use constant
	for i := uint32(0); i < n; i++ {
		max = maxFloat(max, x[i])
	}
	return max
}

// ggml_compute_forward_soft_max
func ComputeForwardSoftMaxFP32(params *ComputeParams, src0, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardSoftMaxFP32 ] ")
	////GGML_ASSERT(ggml_is_contiguous(src0));
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));

	if !src0.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSoftMaxFP32 : [src0] is NOT contiguous!")
		os.Exit(1)
	}

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSoftMaxFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	fmt.Printf("\n\n>>> ComputeForwardSoftMaxFP32 IN <<<\n")
	fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	}
	//fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f %f %f\n",
	//	src1.NE[0], src1.NE[1], src1.NE[2],
	//	src1.Data[0], src1.Data[1], src1.Data[2]) // DEBUG
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	}

	//os.Exit(0)

	negInf := float32(math.Inf(-1)) // TODO use constant

	// TODO: handle transposed/permuted matrices

	ith := params.ith
	nth := params.nth

	nc := src0.NE[0]
	nr := src0.Nrows()

	// rows per thread
	dr := (nr + nth - 1) / nth

	// row range for this thread
	ir0 := dr * ith
	ir1 := min(int(ir0+dr), int(nr))

	for i1 := ir0; int(i1) < ir1; i1++ {
		////float *p = (float *)((char *) dst->data + i1*dst->nb[1]);

		p := dst.Data[i1*dst.NB[1]/4:]

		////#ifndef NDEBUG
		////for (int i = 0; i < nc; ++i) {
		//printf("p[%d] = %f\n", i, p[i]);
		////assert(!isnan(p[i]));
		////}
		////#endif

		//////////////////////////////////////////////////////////max := negInf
		//VecMaxFP32(nc, &max, p)
		////////////////////////////////////////////VecMaxFP32(nc, &max, p)
		max := VecMaxFP32(nc, p)
		sum := float32(0.0)
		//var bits uint16
		for i := 0; i < int(nc); i++ {
			if p[i] == negInf { // TODO use constant
				p[i] = 0.0
			} else {
				//const float val = (p[i] == -INFINITY) ? 0.0 : exp(p[i] - max);

				////ggml_fp16_t s = GGML_FP32_TO_FP16(p[i] - max);
				//s := FP32_TO_FP16(p[i] - max)
				////memcpy(&scvt, &s, sizeof(scvt));
				////const float val = GGML_FP16_TO_FP32(table_exp_f16[scvt]);

				//////////////////////////fp16 := float16.Fromfloat32(p[i] - max)
				//////////////////////////bits := fp16.Bits()
				//////////////////////////exp := TableExpFP16[bits] // FIXME table_exp_f16 ASAP Initialize first!
				//////////////////////////val := exp.Float32()

				val := float32(math.Exp(float64(p[i] - max)))
				sum += val
				p[i] = val
			}
		}

		////assert(sum > 0.0f);

		sum = 1.0 / sum
		VecScaleFP32(nc, p, sum)

		////#ifndef NDEBUG
		////for (int i = 0; i < nc; ++i) {
		////assert(!isnan(p[i]));
		////assert(!isinf(p[i]));
		////}
		////#endif
	}

	fmt.Printf("\n\n>>> ComputeForwardSoftMaxFP32 OUT <<<\n")
	fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
	}
	//fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f %f %f\n",
	//	src1.NE[0], src1.NE[1], src1.NE[2],
	//	src1.Data[0], src1.Data[1], src1.Data[2]) // DEBUG
	fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
	for ii := 0; ii < 8; ii++ {
		fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
	}

	//os.Exit(0)
}

// inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
func VecAddFP32(n uint32, z, x, y []float32) {
	for i := uint32(0); i < n; i++ {
		z[i] = x[i] + y[i]
	}
}

// ggml_compute_forward_add
func ComputeForwardAddFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardAddFP32 ] ")
	////GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	if src1.NB[0] != TYPE_SIZE[TYPE_F32] {
		fmt.Printf("[HALT] ComputeForwardAddFP32 : [src1] is NOT contiguous!")
		os.Exit(1)
	}

	// IDEAL3

	fmt.Printf("\n\n>>> IN <<< ComputeForwardAddFP32 <<<")

	fmt.Printf("\n=== SRC0 | %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src0.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src1.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(dst.Nelements())); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	//os.Exit(0)

	// FIXME Works only for 1 thread
	//VecAddFP32(dst.NE[0], dst.Data, src0.Data, src1.Data)
	//return

	ith := params.ith
	nth := params.nth

	n := src0.Nrows()
	nc := src0.NE[0]

	//nb00 := src0.NB[0]
	nb01 := src0.NB[1]

	nb10 := src1.NB[0]
	nb11 := src1.NB[1]

	//nb0 := dst.NB[0]
	nb1 := dst.NB[1]

	////GGML_ASSERT( nb0 == sizeof(float));
	////GGML_ASSERT(nb00 == sizeof(float));

	if nb10 == TYPE_SIZE[TYPE_F32] {

		fmt.Printf("\nCONTIGIOUS")

		j0 := (n / nth) * ith

		// j1 := ith == nth - 1 ? n : (n/nth)*(ith + 1)
		var j1 uint32
		if ith == nth-1 {
			j1 = n
		} else {
			j1 = (n / nth) * (ith + 1)
		}

		for j := j0; j < j1; j++ {
			////ggml_vec_add_f32(nc,
			////        (float *) ((char *) dst->data  + j*nb1),
			////        (float *) ((char *) src0->data + j*nb01),
			////        (float *) ((char *) src1->data + j*nb11));

			////VecAddFP32(nc, dst.Data[j], src0.Data[j], src1.Data[j])
			//VecAddFP32(nc, dst.Data[j:j+nc], src0.Data[j:j+nc], src1.Data[j:j+nc])
			VecAddFP32(nc, dst.Data[j*nb1/4:], src0.Data[j*nb01/4:], src1.Data[j*nb11/4:])
		}

	} else {

		// src1 is not contiguous

		fmt.Printf("\nNON-CONTIGIOUS")

		for j := ith; j < n; j += nth {
			////float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
			dstPtr := dst.Data[j*nb1/4:]
			////float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
			src0Ptr := src0.Data[j*nb01/4:]
			for i := uint32(0); i < nc; i++ {
				////float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);
				src1Ptr := src1.Data[j*nb11/4+i*nb10/4]
				////dst_ptr[i] = src0_ptr[i] + *src1_ptr;
				dstPtr[i] = src0Ptr[i] + src1Ptr
			}
		}
	}

	/*
		fmt.Printf("\n\n>>> ComputeForwardAddFP32 <<<")
		fmt.Printf("\n=== SRC0 === LEN = %d %d %d %d - %d %d %d %d\n",
			src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
			src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
		for ii := 0; ii < 8; ii++ {
			fmt.Printf("| SRC[%d] = %.4f |", ii, src0.Data[ii])
		}
		fmt.Printf("\n=== SRC1 === [ %d %d %d ] %f %f %f\n",
			src1.NE[0], src1.NE[1], src1.NE[2],
			src1.Data[0], src1.Data[1], src1.Data[2]) // DEBUG
		fmt.Printf("\n=== DST === LEN = %d * %d\n", dst.NE[0], dst.NE[1]) // DEBUG
		for ii := 0; ii < 8; ii++ {
			fmt.Printf("| DST[%d] = %.4f |", ii, dst.Data[ii])
		}*/

	fmt.Printf("\n\n>>> OUT <<< ComputeForwardAddFP32 <<<")

	fmt.Printf("\n=== SRC0 | %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src1.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== SRC1 === %d %d %d %d === %d %d %d %d ===\n",
		src1.NE[0], src1.NE[1], src1.NE[2], src1.NE[3],
		src1.NB[0], src1.NB[1], src1.NB[2], src1.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src1.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src1.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src1.Nelements())); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	//os.Exit(0)
}

// Sigmoid Linear Unit (SiLU) function
func SiluFP32(x float32) float32 {
	return x / float32(1.0+math.Exp(float64(-x)))
}

// inline static void ggml_vec_silu_f32(const int n, float * y, const float * x) {
func VecSiluFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] = SiluFP32(x[i]) // ggml_silu_f32
	}
}

// FIXME ASAP
// ggml_compute_forward_silu
func ComputeForwardSiluFP32(params *ComputeParams, src0, dst *Tensor) {

	//fmt.Printf(" [ ComputeForwardSiluFP32 ] ")
	////GGML_ASSERT(ggml_is_contiguous(src0));
	////GGML_ASSERT(ggml_is_contiguous(dst));
	////GGML_ASSERT(ggml_are_same_shape(src0, dst));

	if !src0.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSiluFP32 : [src0] is NOT contiguous!")
		os.Exit(1)
	}

	if !dst.IsContiguous() {
		fmt.Printf("[HALT] ComputeForwardSiluFP32 : [dst] is NOT contiguous!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	// IDEAL4

	fmt.Printf("\n\n>>> IN <<< ComputeForwardSiluFP32 <<<")

	fmt.Printf("\n=== SRC0 | %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src0.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(dst.Nelements())); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	//os.Exit(0)

	// FIXME Works only for 1 thread
	VecSiluFP32(dst.NE[0], dst.Data, src0.Data)
	//return
	/*
	   ith := params.ith
	   nth := params.nth

	   nc := int(src0.NE[0])
	   nr := src0.Nrows()

	   // rows per thread
	   dr := int((nr + nth - 1) / nth)

	   // row range for this thread
	   ir0 := dr * int(ith)
	   ir1 := int(min(int(ir0+dr), int(nr)))

	   	for i1 := ir0; i1 < ir1; i1++ {
	   		////ggml_vec_silu_f32(nc,
	   		////        (float *) ((char *) dst->data  + i1*( dst->nb[1])),
	   		////        (float *) ((char *) src0->data + i1*(src0->nb[1])));

	   		dsttmp := dst.Data[i1*nc : i1*nc+ir1]   // FIXME ??
	   		src0tmp := src0.Data[i1*nc : i1*nc+ir1] // FIXME ??

	   		//VecSiluFP32(nc, dst.Data[i1*dst.NB[1]], src0.Data[i1*src0.NB[1]])
	   		VecSiluFP32(nc, dsttmp, src0tmp)

	   		////#ifndef NDEBUG

	   		////for (int k = 0; k < nc; k++) {
	   		////const float x = ((float *) ((char *) dst->data + i1*( dst->nb[1])))[k];
	   		////UNUSED(x);
	   		////assert(!isnan(x));
	   		////assert(!isinf(x));
	   		////}

	   		////#endif
	   	}
	*/

	fmt.Printf("\n\n>>> OUT <<< ComputeForwardSiluFP32 <<<")

	fmt.Printf("\n=== SRC0 | %d %d %d %d === %d %d %d %d ===\n",
		src0.NE[0], src0.NE[1], src0.NE[2], src0.NE[3],
		src0.NB[0], src0.NB[1], src0.NB[2], src0.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(src0.Nelements())); ii++ {
		fmt.Printf("%.4f  ", src0.Data[ii])
	}

	fmt.Printf("\n=== DST === %d %d %d %d === %d %d %d %d ===\n",
		dst.NE[0], dst.NE[1], dst.NE[2], dst.NE[3],
		dst.NB[0], dst.NB[1], dst.NB[2], dst.NB[3]) // DEBUG
	for ii := 0; ii < min(10, int(dst.Nelements())); ii++ {
		fmt.Printf("%.4f  ", dst.Data[ii])
	}

	//os.Exit(0)
}

// ---

/*
struct llama_vocab {
    using id    = int32_t;
    using token = std::string;

    struct token_score {
        token tok;
        float score;
    };

    std::unordered_map<token, id> token_to_id;
    std::vector<token_score> id_to_token;
};*/

type TokenScore struct {
	Token string
	Score float32
}

type Vocab struct {
	Token2ID map[string]uint32
	ID2Token []TokenScore
}

func NewVocab(size uint32) *Vocab {
	return &Vocab{
		Token2ID: make(map[string]uint32, size),
		ID2Token: make([]TokenScore, size, size),
	}
}

func min(a, b int) int {
	if a <= b {
		return a
	}
	return b
}

func min32(a, b uint32) uint32 {
	if a <= b {
		return a
	}
	return b
}

// ---- SentencePiece Tokenizer

// struct llama_sp_symbol {
type Symbol struct {
	////using index = int;

	// NB! Allow -1
	Prev int
	Next int

	Text string //const char * text;
	N    uint32 // size_t n;
}

// struct llama_sp_bigram {
type Bigram struct {
	////struct comparator {
	////bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
	////return (l.score < r.score) || (l.score == r.score && l.left > r.left);
	////}
	////};
	////using queue_storage = std::vector<llama_sp_bigram>;
	////using queue = std::priority_queue<llama_sp_bigram, queue_storage, comparator>;

	// NB! Allow -1
	Left  int // llama_sp_symbol::index left;
	Right int // llama_sp_symbol::index

	Score float32
	Size  uint32
}

func utf8Len(src byte) uint32 {
	lookup := []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}
	highbits := uint8(src) >> 4 // static_cast<uint8_t>(src) >> 4;
	return lookup[highbits]
}

//type Tokenizer struct {
//vocab *Vocab
//symbols []Symbol // std::vector<llama_sp_symbol> symbols_;
//work_queue []Bigram // llama_sp_bigram::queue work_queue_; // std::priority_queue<llama_sp_bigram, queue_storage, comparator>;
//}

//func NewTokenizer() *Tokenizer {
//  return &{

//}
//}

func Token2Str(vocab *Vocab, token uint32) string {
	if int(token) >= len(vocab.ID2Token) {
		return ""
	}

	return vocab.ID2Token[token].Token
}

func PopMax(queue *[]Bigram) Bigram {

	max := 0 // index of max score element in queue
	for cur := 1; cur < len(*queue); cur++ {
		if ((*queue)[max].Score < (*queue)[cur].Score) ||
			((*queue)[max].Score == (*queue)[cur].Score &&
				(*queue)[max].Left > (*queue)[cur].Left) {
			max = cur // FIXME Double Check
		}
	}

	pop := (*queue)[max]

	// replace max element with last and shrink slice (if max == last, then just remove it)
	(*queue)[max] = (*queue)[len(*queue)-1]
	*queue = (*queue)[:len(*queue)-1]

	return pop
}

////struct comparator {
////bool operator()(llama_sp_bigram & l, llama_sp_bigram & r) {
////return (l.score < r.score) || (l.score == r.score && l.left > r.left);
////}
////};

func TryAddBigram(vocab *Vocab, symbols []Symbol, workQueue *[]Bigram, left, right int) {

	//fmt.Printf("\n* left = %d | right = %d * ", left, right) // DEBUG

	if left == -1 || right == -1 {
		return
	}

	//fmt.Printf("\n** symbols[left].Text = %s | N = %d ** ", symbols[left].Text, symbols[left].N) // DEBUG
	//fmt.Printf("\n** symbols[right].Text = %s | N = %d ** ", symbols[right].Text, symbols[right].N) // DEBUG

	////const std::string text = std::string(symbols_[left].text, symbols_[left].n + symbols_[right].n);
	token := symbols[left].Text[:symbols[left].N+symbols[right].N]
	//fmt.Printf(" !! token = %s !! ", token) // DEBUG
	id, ok := vocab.Token2ID[token]
	////if token == vocab.Token2ID.end()) {
	//if (static_cast<size_t>((*token).second) >= vocab_.id_to_token.size()) {
	if !ok || int(id) >= len(vocab.ID2Token) {
		return
	}

	tokenScore := vocab.ID2Token[id]

	//fmt.Printf(" [ token = %s | token id = %d | score = %f | len = %d ] ", token, id, tokenScore.Score, len(token)) // DEBUG

	bigram := Bigram{Left: left, Right: right, Score: tokenScore.Score, Size: uint32(len(token))}
	////bigram.left = left
	////bigram.right = right;
	/////bigram.score = ;
	////bigram.size = text.size();
	////workQueue_.push(bigram);
	*workQueue = append(*workQueue, bigram)
}

// void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
func Tokenize(vocab *Vocab, text string, bos bool) []uint32 {

	output := make([]uint32, 0)
	symbols := make([]Symbol, 0)   // std::vector<llama_sp_symbol> symbols_;
	workQueue := make([]Bigram, 0) // llama_sp_bigram::queue work_queue_; // std::priority_queue<llama_sp_bigram, queue_storage, comparator>;

	if bos {
		output = append(output, 1) // TODO: replace with vocab.bos
	}

	// split string into utf8 chars
	index := 0
	offs := 0
	for offs < len(text) {
		var sym Symbol
		charLen := min(len(text)-offs, int(utf8Len(text[offs])))
		sym.Text = text[offs:] // text.c_str() + offs // FIXME ASAP
		sym.N = uint32(charLen)
		offs += charLen
		sym.Prev = index - 1
		if offs == len(text) {
			sym.Next = -1
		} else {
			sym.Next = index + 1
		}
		index++
		symbols = append(symbols, sym) ////symbols_.emplace_back(std::move(sym));
	}

	// seed the work queue with all possible 2-character tokens
	for i := 1; i < len(symbols); i++ {
		//fmt.Printf(" [ sym[%d] = %s ] ", i, symbols[i].Text) // DEBUG
		TryAddBigram(vocab, symbols, &workQueue, i-1, i)
	}

	// keep substituting the highest frequency pairs for as long as we can
	for len(workQueue) > 0 {
		////bigram := work_queue_.top();
		////work_queue_.pop();
		bigram := PopMax(&workQueue)

		leftSym := &symbols[bigram.Left]
		rightSym := &symbols[bigram.Right]

		// if one of the symbols already got merged, skip it
		if leftSym.N == 0 || rightSym.N == 0 || leftSym.N+rightSym.N != bigram.Size {
			continue
		}

		// merge the right sym into the left one
		leftSym.N += rightSym.N
		rightSym.N = 0

		//printf("left = '%*s' size = %zu\n", (int) left_sym.n, left_sym.text, bigram.size);

		// remove the right sym from the chain
		leftSym.Next = rightSym.Next
		if rightSym.Next >= 0 {
			symbols[rightSym.Next].Prev = bigram.Left
		}

		// find more substitutions
		////try_add_bigram(left_sym.prev, bigram.left);
		TryAddBigram(vocab, symbols, &workQueue, leftSym.Prev, bigram.Left)
		////try_add_bigram(bigram.left, left_sym.next);
		TryAddBigram(vocab, symbols, &workQueue, bigram.Left, leftSym.Next)
	}

	for i := 0; i != -1; i = symbols[i].Next {
		symbol := symbols[i]
		id, ok := vocab.Token2ID[symbol.Text[:symbol.N]]

		////if (token == vocab_.token_to_id.end()) {
		if !ok {
			// output any symbols that did not form tokens as bytes.
			for j := uint32(0); j < symbol.N; j++ {
				////llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
				tokenID := uint32(symbol.Text[j] + 3) // FIXME ASAP
				////output.push_back(token_id);
				output = append(output, tokenID)
			}
		} else {
			////output.push_back((*token).second);
			output = append(output, id)
		}
	}

	////private:

	////const llama_vocab & vocab_;
	////std::vector<llama_sp_symbol> symbols_;
	////llama_sp_bigram::queue work_queue_;

	return output

}

// FIXME Would it work with UTF-8? Rewrite for runes
// SentencePiece implementation after https://guillaume-be.github.io/2020-05-30/sentence_piece
// std::vector<gpt_vocab::id> llamaTokenize(const gpt_vocab & vocab, const std::string & text, bool bos) {
func TokenizeOld(vocab *Vocab, text string, bos bool) []uint32 {

	// TODO: Calculate this constant from the vocabulary
	MAX_TOKEN_LEN := 18
	length := len(text)

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
	for i := 0; i < length; i++ {
		maxLen := min(length-i, MAX_TOKEN_LEN)
		for subLen := 1; subLen <= maxLen; subLen++ {
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
		token := vocab.ID2Token[tokenID].Token
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

	////isFirstCall := true // FIXME static ??

	// FIXME Init only once !!
	////if isFirstCall {

	// ---- initialize GELU, SILU and EXP F32 tables
	////{
	////const uint64_t t_start = ggml_time_us(); UNUSED(t_start);

	/////////////////////////////////////////var ii uint16
	/////////////////////////////////////////for i := uint32(0); i < (1 << 16); i++ {
	/////////////////////////////////////////ui := uint16(i)

	////memcpy(&ii, &ui, sizeof(ii));
	////const float f = table_f32_f16[i] = COMPUTE_FP16_TO_FP32(ii);
	/////////////////////////////////////////fp32 := float32()

	////table_gelu_f16[i] = FP32_TO_FP16(ggml_gelu_f32(f));
	////table_silu_f16[i] = FP32_TO_FP16(ggml_silu_f32(f));

	////TableExpFP16[i]  = FP32_TO_FP16(exp(f));
	/////////////////////////////////////////exp := float32(math.Exp(fp32))
	/////////////////////////////////////////TableExpFP16[i] = float16.Fromfloat32(exp)

	/////////////////////////////////////////}

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

	////isFirstCall = false
	////}

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

	//var buf []byte
	//if params.MemBuffer == nil {
	//	buf = make([]byte, params.MemSize)
	//} else {
	//	buf = params.MemBuffer
	//}

	ctx = &Context{
		//MemSize:        params.MemSize,
		//MemBuffer:      buf,
		//MemBufferOwned: params.MemBuffer != nil,
		//Objects:        0,
		//Objects:      make([]Object, 0),
		//ObjectsBegin: nil,
		//ObjectsEnd:   nil,
		//Scratch:      Scratch{0, 0, nil},
		//ScratchSave:  Scratch{0, 0, nil},
	}

	////ggml_assert_aligned(ctx.mem_buffer);

	////PRINT_DEBUG("%s: context initialized\n", __func__);

	////ggml_critical_section_end();

	return ctx
}
