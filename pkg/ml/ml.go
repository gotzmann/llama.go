package ml

import (
	"fmt"
	"math"
	"os"
	"reflect"
	"runtime"
	"sync"
	"time"
	"unsafe"

	"github.com/x448/float16"
)

const (
	DEBUG = false

	MAX_DIMS   = 4
	MAX_NODES  = 4096
	MAX_PARAMS = 16
	MAX_OPT    = 4

	QK = 32 // quantization

	TOKEN_BOS = 1
	TOKEN_EOS = 2
)

// computation graph
type Graph struct {
	//MaxThreads int

	//UseAVX  bool
	//UseNEON bool

	NodesCount uint32
	LeafsCount uint32

	Jobs chan *ComputeParams

	Nodes [MAX_NODES]*Tensor
	Grads [MAX_NODES]*Tensor
	Leafs [MAX_NODES]*Tensor
}

type InitParams struct {
}

type Context struct {
	MaxThreads int
	UseAVX     bool
	UseNEON    bool
	//Graph      *Graph
	Compute   chan *ComputeParams
	Allocator *Allocator
}

func NewContext(maxThreads int, useAVX, useNEON bool) *Context {

	ch := make(chan *ComputeParams, maxThreads) // TODO: +1 for safety?

	for i := 0; i < maxThreads; i++ {
		go Job(ch, i)
	}

	return &Context{
		MaxThreads: maxThreads,
		UseAVX:     useAVX,
		UseNEON:    useNEON,
		Compute:    ch,
		Allocator:  NewAllocator(),
	}
}

// ReleaseContext frees all context resources - channel will be closed and goroutines stopped
func (ctx *Context) ReleaseContext() {
	close(ctx.Compute)
	// TODO: Maybe some steps for Allocator too
}

type DType uint8

// Data types are the same as in llama.cpp so full compatibility there
const (
	TYPE_F32   DType = 0
	TYPE_F16   DType = 1
	TYPE_Q4_0  DType = 2
	TYPE_Q4_1  DType = 3
	TYPE_I8    DType = 4
	TYPE_I16   DType = 5
	TYPE_I32   DType = 6
	TYPE_COUNT DType = 8
)

func printTensor(tensor *Tensor, name string) {

	var dt string
	switch tensor.Type {
	case TYPE_F16:
		dt = "FP16"
	case TYPE_F32:
		dt = "FP32"
	case TYPE_Q4_0:
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

// precomputed exp table for f16 (128 KB)
// static ggml_fp16_t table_exp_f16[1 << 16];
var TableExpFP16 [1 << 16]float16.Float16

var BLCK_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{1, 1, QK, QK, 1, 1, 1, 0}
var TYPE_SIZE [TYPE_COUNT]uint32 = [TYPE_COUNT]uint32{4, 2, 4 + QK/2, 4*2 + QK/2, 1, 2, 4, 0}

func TypeSizeFloat(dt DType) float32 {
	return float32(TYPE_SIZE[dt]) / float32(BLCK_SIZE[dt])
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
	OP_NORM
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

// Tensor of up to 4x dimensions
// The multi-dimensional tensors are stored in row-major order
// and the array indexes are written row-first (lexicographical access order)

type Tensor struct {
	Type DType

	Reusable bool // this tensor Data buffer might be reused with pooling

	Dims uint32

	NE [MAX_DIMS]uint32 // number of elements
	NB [MAX_DIMS]uint32 // stride in bytes

	op optype

	isParam bool

	grad *Tensor
	src0 *Tensor
	src1 *Tensor

	opt [MAX_OPT]*Tensor // FIXME: Do we need this?

	TasksCount int

	Data []float32
}

// ggml_is_contiguous
func (tensor *Tensor) IsContiguous() bool {
	return tensor.NB[0] == TYPE_SIZE[tensor.Type] &&
		tensor.NB[1] == tensor.NB[0]*tensor.NE[0]/BLCK_SIZE[tensor.Type] &&
		tensor.NB[2] == tensor.NB[1]*tensor.NE[1] &&
		tensor.NB[3] == tensor.NB[2]*tensor.NE[2]
}

func AreSameShape(a, b *Tensor) bool {
	return (a.NE[0] == b.NE[0]) && (a.NE[1] == b.NE[1]) && (a.NE[2] == b.NE[2]) && (a.NE[3] == b.NE[3])
}

func (t *Tensor) Nelements() uint32 {
	return t.NE[0] * t.NE[1] * t.NE[2] * t.NE[3]
}

func (t *Tensor) Nrows() uint32 {
	return t.NE[1] * t.NE[2] * t.NE[3]
}

// ggml_nbytes
func (t *Tensor) Nbytes() uint32 {
	return (t.Nelements() * TYPE_SIZE[t.Type]) / BLCK_SIZE[t.Type]
}

// ggml_view_tensor
func ViewTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], src.Data)
}

// ggml_dup_tensor
func DupTensor(ctx *Context, src *Tensor) *Tensor {
	return NewTensor(ctx, src.Type, src.Dims, src.NE[0], src.NE[1], src.NE[2], src.NE[3], nil) // Reusbale OK
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

// ggml_can_mul_mat
func CanMulMat(t0, t1 *Tensor) bool {
	return (t0.NE[0] == t1.NE[0]) && (t0.NE[2] == t1.NE[2]) && (t0.NE[3] == t1.NE[3])
}

// ggml_mul_mat
func MulMat(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_can_mul_mat(a, b));
	////GGML_ASSERT(!ggml_is_transposed(a));

	isNode := false

	if a.grad != nil || b.grad != nil {
		isNode = true
	}

	result := NewTensor(ctx, TYPE_F32, min32(a.Dims, b.Dims), a.NE[1], b.NE[1], a.NE[2], b.NE[3], nil) // Reusable OK

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

	result := NewTensor1D(ctx, a.Type, 1) // Reusable OK

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

	result := NewTensor(ctx, a.Type, b.Dims, b.NE[0], b.NE[1], b.NE[2], b.NE[3], nil) // Reusable OK

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
	return tensor.NE[0] == 1 && tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsVector(tensor *Tensor) bool {
	return tensor.NE[1] == 1 && tensor.NE[2] == 1 && tensor.NE[3] == 1
}

func IsMatrix(tensor *Tensor) bool {
	return tensor.NE[2] == 1 && tensor.NE[3] == 1
}

// ggml_get_rows
func GetRows(ctx *Context, a, b *Tensor) *Tensor {
	////ASSERT(ggml_is_matrix(a) && ggml_is_vector(b) && b.type == TYPE_I32);
	//if !IsMatrix(a) || !IsVector(b) /* || b.Type != TYPE_I32 */ {
	//	fmt.Printf("\n[ERROR] GetRows fail basic assertions")
	//	os.Exit(1)
	//}

	isNode := false

	if a.grad != nil || b.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows")
		os.Exit(1)
	}

	result := NewTensor2D(ctx, TYPE_F32, a.NE[0], b.NE[0]) // Reusable OK

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

// ggml_rms_norm_impl
func RMSNormImpl(ctx *Context, a *Tensor, inplace bool) *Tensor {
	isNode := false

	if !inplace && a.grad != nil {
		////ASSERT(false); // TODO: implement backward
		isNode = true
		fmt.Printf("\n[STOP] ml.GetRows")
		os.Exit(1)
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
	result := NewTensor(ctx, a.Type, 1, ne0, 1, 1, 1, slice)

	result.op = OP_VIEW
	result.grad = nil
	result.src0 = a
	result.src1 = nil // TODO: maybe store the offset here?

	return result
}

// ggml_build_forward_impl
func BuildForwardImpl(graph *Graph, tensor *Tensor, expand bool) {

	if !expand {
		graph.NodesCount = 0
		graph.LeafsCount = 0
	}

	n0 := graph.NodesCount
	VisitParents(graph, tensor)
	n_new := graph.NodesCount - n0

	if n_new > 0 {
		// the last added node should always be starting point
		////ASSERT(cgraph.nodes[cgraph.n_nodes - 1] == tensor);
		if !(graph.Nodes[graph.NodesCount-1] == tensor) {
			fmt.Printf("\n[STOP] BuildForwardImpl : the last added node should always be starting point!")
			os.Exit(1)
		}
	}
}

// ggml_build_forward_expand
func BuildForwardExpand(graph *Graph, tensor *Tensor) {
	BuildForwardImpl(graph, tensor, true)
}

// ggml_visit_parents
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
	//if a.Nelements() != b.Nelements() {
	//	fmt.Printf("\n[HALT] Copy tensors of different dimensions!")
	//	os.Exit(1)
	//}

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

// ggml_new_tensor_1d
func NewTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne0, 1, 1, 1, nil)
}

// ggml_new_tensor_2d
func NewTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil)
}

func NewTensor3D(ctx *Context, dt DType, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil)
}

func NewTensor4D(ctx *Context, dt DType, ne0, ne1, ne2, ne3 uint32) *Tensor {
	return NewTensor(ctx, dt, 4, ne0, ne1, ne2, ne3, nil)
}

// ggml_new_tensor_impl
func NewTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	// TODO: Check allowed data types on graph creation
	//if dt != TYPE_F32 && dt != TYPE_I32 {
	//	fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
	//	os.Exit(1)
	//}

	////ggml_assert_aligned(result);

	if data == nil {
		total := ne0 * ne1 * ne2 * ne3
		data = make([]float32, total, total)
	}

	return &Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		NB:   [4]uint32{4, ne0 * 4, ne0 * ne1 * 4, ne0 * ne1 * ne2 * 4},
		op:   OP_NONE,
		Data: data,
	}
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
	b.Data[0] = float32(past)
	b.Data[1] = float32(dims)
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

	//if !a.IsContiguous() {
	//	fmt.Printf("\n[STOP] Reshape3D : tensor is NOT contiguous!")
	//	os.Exit(1)
	//}

	//if a.Nelements() != ne0*ne1*ne2 {
	//	fmt.Printf("\n[STOP] Reshape3D : different elements number!")
	//	os.Exit(1)
	//}

	////bool is_node = false;

	////if (a.grad) {
	////   //// ASSERT(false); // TODO: implement backward
	////    is_node = true;
	////}

	result := NewTensor(ctx, a.Type, 3, ne0, ne1, ne2, 1, a.Data) // Reusable OK

	result.op = OP_RESHAPE
	////result.grad = is_node ? ggml_dup_tensor(ctx, result) : NULL;
	result.grad = nil
	result.src0 = a
	result.src1 = nil

	return result
}

// ggml_new_f32
func NewFP32(ctx *Context, value float32) *Tensor {
	result := NewTensor1D(ctx, TYPE_F32, 1) // Reusable OK
	SetFP32(result, value)
	return result
}

// ggml_set_f32
func SetFP32(tensor *Tensor, value float32) *Tensor {
	// FIXME Optimize with mem zeroing
	n := tensor.Nelements()
	for i := uint32(0); i < n; i++ {
		////ggml_vec_set_f32(nc, (float *)(data + i*n1), value);
		tensor.Data[i] = value
	}
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
	b := NewFP32(ctx, float32(past)) // FIXME NewI32(ctx, past)

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

func BuildForward(tensor *Tensor) *Graph {
	result := Graph{}
	BuildForwardImpl(&result, tensor, false)
	return &result
}

func BuildBackward(ctx *Context, gf *Graph, keep bool) Graph {

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

	ith uint32
	nth uint32

	tensor *Tensor

	wg *sync.WaitGroup

	UseAVX  bool
	UseNEON bool
}

// Golang doesn’t have unary Bitwise NOT(~) like other programming languages
// Here, you have to use Bitwise XOR(^) operator as Bitwise NOT
func up32(n uint32) uint32 { // FIXME Not needed ?
	return uint32(n+31) & ^uint32(31)
}

func up(n, m uint32) uint32 { // FIXME Not needed ?
	// assert m is a power of 2
	////GGML_ASSERT((m & (m - 1)) == 0);
	return uint32(n+m-1) & ^uint32(m-1)
}

func max(a, b int) int { // FIXME Not needed ?
	if a >= b {
		return a
	}
	return b
}

// Job is goroutine existing while the computation loop is active
// The main purpose of the Job is to perform some part
// of time consuming matrix multiplications
// TODO: Investigate https://pkg.go.dev/runtime#LockOSThread
func Job(listen <-chan *ComputeParams, id int) {
	runtime.LockOSThread()
	for params := range listen {
		ComputeForwardMulMatFP32(
			params,
			params.tensor.src0,
			params.tensor.src1,
			params.tensor)
		params.wg.Done()
	}
}

// Do is an experimental alternative for always waiting Job threads
func Do(params *ComputeParams, id int) {
	ComputeForwardMulMatFP32(
		params,
		params.tensor.src0,
		params.tensor.src1,
		params.tensor)
	params.wg.Done()
}

func GraphCompute(ctx *Context, graph *Graph) {

	//maxThreads := graph.MaxThreads
	maxThreads := ctx.MaxThreads

	// --- init N job goroutines and channel to send tasks for them

	//graph.Jobs = make(chan *ComputeParams, maxThreads) // TODO Right place to init? +1 for safety?
	//defer close(graph.Jobs)

	//for i := 0; i < maxThreads; i++ {
	//	go Job(graph.Jobs, i)
	//}

	// --- initialize tasks

	{
		// thread scheduling for the different operations
		// TasksCount might be 0, 1, or ThreadsCount
		for i := uint32(0); i < graph.NodesCount; i++ {

			node := graph.Nodes[i]

			if DEBUG {
				fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
			}

			switch node.op {

			case OP_DUP:
				node.TasksCount = 1
			case OP_ADD:
				node.TasksCount = 1 // TODO threads
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
				node.TasksCount = 1 // TODO threads
			case OP_SILU:
				node.TasksCount = 1 // TODO threads
			case OP_NORM:
			case OP_RMS_NORM:
				node.TasksCount = 1 // TODO threads
			case OP_MUL_MAT:
				node.TasksCount = maxThreads
				// TODO: use different scheduling for different matrix sizes
			case OP_SCALE:
				node.TasksCount = 1 // TODO threads
			case OP_CPY:
			case OP_RESHAPE:
			case OP_VIEW:
			case OP_PERMUTE:
			case OP_TRANSPOSE:
			case OP_GET_ROWS:
			case OP_DIAG_MASK_INF:
				node.TasksCount = 1
			case OP_SOFT_MAX:
				node.TasksCount = 1 // TODO threads
			case OP_ROPE:
				////node.TasksCount = 1
			case OP_CONV_1D_1S:
			case OP_CONV_1D_2S:
				node.TasksCount = 1 // TODO threads
				////ASSERT(node->src0->ne[3] == 1);
				////ASSERT(node->src1->ne[2] == 1);
				////ASSERT(node->src1->ne[3] == 1);
			case OP_FLASH_ATTN:
				node.TasksCount = 1 // TODO threads
			case OP_FLASH_FF:
				node.TasksCount = 1 // TODO threads
			case OP_NONE:
				node.TasksCount = 1
			case OP_COUNT:
				fmt.Printf("\n[HALT] Something wrong with compute graph!")
				os.Exit(1)
			}
		}
	}

	for i := uint32(0); i < graph.NodesCount; i++ {

		node := graph.Nodes[i]

		if DEBUG {
			fmt.Printf("\n\n### STEP #%d ### %d - %d [ %d:%d:%d:%d ]", i, node.op, node.Type, node.NE[0], node.NE[1], node.NE[2], node.NE[3])
		}

		params := &ComputeParams{
			Type: TASK_INIT,
			ith:  0,
			nth:  uint32(node.TasksCount),
		}

		ComputeForward(ctx, graph, params, node) // TASK_INIT

		// --- COMPUTE

		params.Type = TASK_COMPUTE
		ComputeForward(ctx, graph, params, node)

		// --- FINALIZE

		params.Type = TASK_FINALIZE
		ComputeForward(ctx, graph, params, node)
	}

}

// =======================================================================

func ComputeForward(ctx *Context, graph *Graph, params *ComputeParams, tensor *Tensor) {

	switch tensor.op {

	case OP_DUP:
		////ggml_compute_forward_dup(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_dup")
		os.Exit(1)
	case OP_ADD:
		ComputeForwardAddFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_SUB:
		////ggml_compute_forward_sub(params, tensor->src0, tensor->src1, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_sub")
		os.Exit(1)
	case OP_MUL:
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
		ComputeForwardSiluFP32(params, tensor.src0, tensor)
	case OP_NORM:
		////ggml_compute_forward_norm(params, tensor->src0, tensor);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_norm")
		os.Exit(1)
	case OP_RMS_NORM:
		ComputeForwardRMSNormFP32(params, tensor.src0, tensor)
	case OP_MUL_MAT:

		// TODO Optimize this
		if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
			return
		}

		// FIXME: Need better heuristic for how many threads to use there
		// But not more than minimal dimension of tensors involved!
		// Like if there dim = 8, it safe to use only 8 or less threads, not 12

		// TODO: There might be small architectures where not reasonable to spin up
		// all available threads, so better to limit parallelism here
		// But that's not the case for LLMs and particularly LLaMA, thus commented

		// totalRows := tensor.src0.NE[1] * tensor.src0.NE[2] * tensor.src0.NE[3]
		// maxThreads := min(graph.MaxThreads, int(totalRows))

		//maxThreads := graph.MaxThreads
		maxThreads := ctx.MaxThreads

		wg := new(sync.WaitGroup)
		wg.Add(maxThreads)

		for i := 0; i < maxThreads; i++ {

			//graph.Jobs <- &ComputeParams{
			ctx.Compute <- &ComputeParams{
				Type:   TASK_COMPUTE,
				ith:    uint32(i),
				nth:    uint32(maxThreads),
				tensor: tensor,
				//UseNEON: graph.UseNEON,
				UseNEON: ctx.UseNEON,
				//UseAVX:  graph.UseAVX,
				UseAVX: ctx.UseAVX,
				wg:     wg,
			}

			/* go Do(&ComputeParams{
				Type:    TASK_COMPUTE,
				ith:     uint32(i),
				nth:     uint32(maxThreads),
				tensor:  tensor,
				UseNEON: graph.UseNEON,
				UseAVX:  graph.UseAVX,
				wg:      wg,
			}, i) */
		}

		wg.Wait()

	case OP_SCALE:
		ComputeForwardScaleFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_CPY:
		ComputeForwardDupFP32(params, tensor.src0, tensor)
	case OP_RESHAPE:
		ComputeForwardReshape(params, tensor.src0, tensor) // NOP
	case OP_VIEW:
		ComputeForwardView(params, tensor.src0) // NOP
	case OP_PERMUTE:
		ComputeForwardPermute(params, tensor.src0) // NOP
	case OP_TRANSPOSE:
		////ggml_compute_forward_transpose(params, tensor->src0);
		fmt.Printf("\n[HALT] Please implement : ggml_compute_forward_transpose")
		os.Exit(1)
	case OP_GET_ROWS:
		ComputeForwardGetRows(params, tensor.src0, tensor.src1, tensor)
	case OP_DIAG_MASK_INF:
		ComputeForwardDiagMaskInfFP32(params, tensor.src0, tensor.src1, tensor)
	case OP_SOFT_MAX:
		ComputeForwardSoftMaxFP32(params, tensor.src0, tensor)
	case OP_ROPE:
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

func VecCopyFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] = x[i]
	}
}

// ggml_compute_forward_get_rows_f32
func ComputeForwardGetRows(params *ComputeParams, src0, src1, dst *Tensor) {

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

	for i := uint32(0); i < nr; i++ {
		r := uint32(src1.Data[i])

		////ggml_vec_cpy_f32(nc,
		////        (float *) ((char *)  dst->data + i*dst->nb[1]),
		////        (float *) ((char *) src0->data + r*src0->nb[1]));

		// FIXME ASAP and double check!
		// VecCopyFP32(nc, (*dst.Data)[i*dst.NE[0]:], (*src0.Data)[uint32(r)*src0.NE[0]:])
		// VecCopyFP32(nc, dst.Data[i*dst.NB[1]/4:], src0.Data[r*src0.NB[1]/4:])
		VecCopyFP32(nc, dst.Data[i*dst.NE[0]:], src0.Data[r*src0.NE[0]:]) // TODO copy()
	}
}

// ggml_compute_forward_rms_norm_f32
func ComputeForwardRMSNormFP32(params *ComputeParams, src0, dst *Tensor) {

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

				mean := 0.0
				// TODO Simplify to directly access [src]
				for i00 := uint32(0); i00 < ne00; i00++ {
					////mean += x[i00] * x[i00];
					mean += float64(x[i00] * x[i00])
				}

				mean /= float64(ne00)

				scale := float32(1.0 / math.Sqrt(mean+eps))

				// TODO Simplify to directly update [dst]
				////float * y = (float *) ((char *) dst->data + i01*nb1 + i02*nb2 + i03*nb3);
				y := dst.Data[i01*nb1/4+i02*nb2/4+i03*nb3/4:]

				////memcpy(y, x, ne00 * sizeof(float));
				//VecScaleFP32(ne00, y, float32(scale))

				for i := uint32(0); i < ne00; i++ {
					y[i] = x[i] * scale
				}
			}
		}
	}
}

// ggml_vec_scale_f32
func VecScaleFP32(n uint32, y []float32, v float32) {
	for i := uint32(0); i < n; i++ {
		y[i] *= v
	}
}

// ggml_compute_forward_repeat
func ComputeForwardRepeatFP32(params *ComputeParams, src0, dst *Tensor) {

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

				////ggml_vec_cpy_f32(nc0,
				////(float *) ((char *)  dst->data + (i*nr0 + k)*( dst->nb[1]) + j*nc0*( dst->nb[0])),
				////(float *) ((char *) src0->data + (        k)*(src0->nb[1])));

				VecCopyFP32(nc0,
					dst.Data[(i*nr0+k)*dst.NB[1]/4+j*nc0*dst.NB[0]/4:],
					src0.Data[k*src0.NB[1]/4:])
			}
		}
	}

	if DEBUG {
		printTensor(src0, "REPEAT SRC0")
		printTensor(dst, "REPEAT DST")
	}
}

func VecMulFP32(n uint32, z, x, y []float32) {
	for i := uint32(0); i < n; i++ {
		z[i] = x[i] * y[i]
	}
}

// ggml_compute_forward_mul
func ComputeForwardMulFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	////assert(params->ith == 0);
	////assert(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if !AreSameShape(src0, src1) || !AreSameShape(src0, dst) {
		fmt.Printf("\n[HALT] ComputeForwardMulFP32 : different shapes!")
		os.Exit(1)
	}

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	n := src0.Nrows()
	nc := src0.NE[0]

	////assert( dst->nb[0] == sizeof(float));
	////assert(src0->nb[0] == sizeof(float));
	////assert(src1->nb[0] == sizeof(float));

	for i := uint32(0); i < n; i++ {

		////ggml_vec_mul_f32(nc,
		////(float *) ((char *) dst->data  + i*( dst->nb[1])),
		////(float *) ((char *) src0->data + i*(src0->nb[1])),
		////(float *) ((char *) src1->data + i*(src1->nb[1])));

		// FIXME NE vs NB
		VecMulFP32(nc, dst.Data[i*dst.NE[0]:], src0.Data[i*src0.NE[0]:], src1.Data[i*src1.NE[0]:])
	}

	if DEBUG {
		printTensor(src0, "MUL SRC0")
		printTensor(src1, "MUL SRC1")
		printTensor(dst, "MUL DST")
	}
}

// ggml_vec_dot_f32
func VecDotFP32(n uint32, x, y []float32) float32 {
	sumf := float32(0.0)
	for i := uint32(0); i < n; i++ {
		sumf += x[i] * y[i]
	}
	return sumf
}

// ggml_vec_mad_f32
func VecMadFP32(n uint32, y, x []float32, v float32) {
	for i := uint32(0); i < n; i++ {
		y[i] += x[i] * v
	}
}

// ggml_vec_acc_f32
func VecAccFP32(n uint32, y, x []float32) {
	for i := uint32(0); i < n; i++ {
		y[i] += x[i]
	}
}

// TODO: Implement all the tensor asserts BEFORE the real computing
func CheckGraph() {

	// --- ComputeForwardMulMatFP32(params *ComputeParams, src0, src1, dst *Tensor)

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

	// TODO: do not support transposed src1
	////assert(nb10 == sizeof(float));
	////if nb10 == 4 {
	////	fmt.Printf("\n[HALT] Do not support transposed src1")
	////	os.Exit(1)
	////}

}

// ggml_compute_forward_mul_mat_f32
func ComputeForwardMulMatFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	// This extra check is not needed (moved to control loop)
	// if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
	// 	return
	// }

	// TODO: Precompute some numbers like ir0..ir1 within main thread and pass them into Job threads?
	// --- Copy tensor parameters to local vars for compact fitting in CPU cache lines

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	ne11 := src1.NE[1]

	nb01 := src0.NB[1]
	nb02 := src0.NB[2]
	nb03 := src0.NB[3]

	nb11 := src1.NB[1]
	nb12 := src1.NB[2]
	nb13 := src1.NB[3]

	nb0 := dst.NB[0]
	nb1 := dst.NB[1]
	nb2 := dst.NB[2]
	nb3 := dst.NB[3]

	src0Data := unsafe.Pointer(&src0.Data[0])
	src1Data := unsafe.Pointer(&src1.Data[0])
	dstData := unsafe.Pointer(&dst.Data[0])

	nr := ne01 * ne02 * ne03                 // total rows in src0
	dr := (nr + params.nth - 1) / params.nth // rows per thread
	ir0 := dr * params.ith                   // row range...
	ir1 := min32(ir0+dr, nr)                 // ...for this thread

	// Optimized math for x64 AVX2 and ARM NEON
	// Works well both for 2D and 3D tensors (it's possible to remove extra math for 2D matrix)

	if (params.UseAVX || params.UseNEON) && src0.IsContiguous() && src1.IsContiguous() {

		srcStride := nb01 // common dimension size between src0 and src1
		dstStride := nb1

		for ir := ir0; ir < ir1; ir++ {

			step3D := ir / ne01
			stepPos := ir % ne01

			src0Ptr := unsafe.Add(src0Data, ir*srcStride)
			src1Ptr := unsafe.Add(src1Data, step3D*nb12)
			dstPtr := unsafe.Add(dstData, step3D*nb2+stepPos*4)

			for ic := uint32(0); ic < ne11; ic++ {
				vdot(src0Ptr, src1Ptr, uint64(ne00), dstPtr)
				src1Ptr = unsafe.Add(src1Ptr, srcStride)
				dstPtr = unsafe.Add(dstPtr, dstStride)
			}
		}

	} else {

		mult := ne02 * ne01
		for ir := ir0; ir < ir1; ir++ {

			// original GGML indices math + bit optimizations
			//i03 := ir / (ne02 * ne01)
			i03 := ir / mult
			//i02 := (ir - i03*ne02*ne01) / ne01
			diff := ir - i03*mult
			//i02 := (ir - i03*mult) / ne01
			i02 := diff / ne01
			//i01 := (ir - i03*ne02*ne01 - i02*ne01)
			//i01 := ir - i03*mult - i02*ne01
			i01 := diff - i02*ne01

			src0Offset := i01*nb01 + i02*nb02 + i03*nb03

			for ic := uint32(0); ic < ne11; ic++ {

				//dst.Data[i0*nb0+ic*nb1+i2*nb2+i3*nb3] =
				//	VecDotFP32(ne00,
				//		src0.Data[i01*nb01+i02*nb02+i03*nb03:],
				//		src1.Data[ic*nb11+i12*nb12+i13*nb13:])

				// --- inline VecDotFP32

				src1Offet := ic*nb11 + i02*nb12 + i03*nb13
				dstOffset := i01*nb0 + ic*nb1 + i02*nb2 + i03*nb3

				if params.UseAVX || params.UseNEON {

					src0Ptr := unsafe.Add(src0Data, src0Offset)
					src1Ptr := unsafe.Add(src1Data, src1Offet)
					dstPtr := unsafe.Add(dstData, dstOffset)

					vdot(src0Ptr, src1Ptr, uint64(ne00), dstPtr)

				} else { // scalar CPU math

					src0Ptr := src0.Data[src0Offset/4:]
					src1Ptr := src1.Data[src1Offet/4:]

					sum := float32(0.0)
					for i := uint32(0); i < ne00; i++ {
						sum += src0Ptr[i] * src1Ptr[i]
					}

					dst.Data[dstOffset/4] = sum
				}
			}
		}
	}

	if DEBUG {
		fmt.Printf("\n\n>>> ComputeForwardMulMatFP32 OUT <<<\n")
		printTensor(dst, "DST CPU")
	}

}

// ggml_compute_forward_view
func ComputeForwardView(params *ComputeParams, src0 *Tensor) {
	// NOP
}

func ComputeForwardCopy(params *ComputeParams, src0, dst *Tensor) {
	ComputeForwardDupFP32(params, src0, dst)
}

// ggml_compute_forward_dup_f32
func ComputeForwardDupFP32(params *ComputeParams, src0, dst *Tensor) {

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

	ne00 := src0.NE[0]
	ne01 := src0.NE[1]
	ne02 := src0.NE[2]
	ne03 := src0.NE[3]

	nb00 := src0.NB[0] / 4
	nb01 := src0.NB[1] / 4
	nb02 := src0.NB[2] / 4
	nb03 := src0.NB[3] / 4

	////if (ggml_is_contiguous(src0) && src0->type == dst->type) {
	if src0.IsContiguous() && src0.Type == dst.Type {
		////memcpy(dst->data, src0->data, ggml_nelements(dst) * GGML_TYPE_SIZE[src0->type]);
		copy(dst.Data, src0.Data)
		return
	}

	// --- src0 is NOT contigious
	// --- supporting only 4-bytes data for [src0] and FP32 for [dst]

	if src0.NB[0] == TYPE_SIZE[TYPE_F32] {
		if dst.Type == TYPE_F32 {

			id := uint32(0)
			rs := ne00 * nb00

			for i03 := uint32(0); i03 < ne03; i03++ {
				for i02 := uint32(0); i02 < ne02; i02++ {
					for i01 := uint32(0); i01 < ne01; i01++ {

						////const char * src0_ptr = (char *) src0->data + i01*nb01 + i02*nb02 + i03*nb03;
						src0Ptr := src0.Data[i01*nb01+i02*nb02+i03*nb03 : i01*nb01+i02*nb02+i03*nb03+rs]
						////char * dst_ptr = (char *) dst->data + id*rs;
						dstPtr := dst.Data[id*rs : id*rs+rs]
						////memcpy(dst_ptr, src0_ptr, rs);
						copy(dstPtr, src0Ptr)

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

		if dst.Type == TYPE_F32 {

			id := 0
			////dstPtr = (float *) dst->data;

			for i03 := uint32(0); i03 < ne03; i03++ {
				for i02 := uint32(0); i02 < ne02; i02++ {
					for i01 := uint32(0); i01 < ne01; i01++ {
						for i00 := uint32(0); i00 < ne00; i00++ {

							//src0Ptr := src0.Data[i00*nb00/4 + i01*nb01/4 + i02*nb02/4 + i03*nb03/4:]
							//dstPtr[id] = *src0_ptr;

							dst.Data[id] = src0.Data[i00*nb00+i01*nb01+i02*nb02+i03*nb03]

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

	if DEBUG {
		fmt.Printf("\n\n>>> ComputeForwardDupFP32 OUT <<<\n")
	}
}

// ggml_compute_forward_reshape
func ComputeForwardReshape(params *ComputeParams, src0, dst *Tensor) {
	// NOP
}

// ggml_compute_forward_permute
func ComputeForwardPermute(params *ComputeParams, src0 *Tensor) {
	// NOP
}

// ggml_compute_forward_rope
func ComputeForwardRopeFP32(params *ComputeParams, src0, src1, dst *Tensor) {

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
				for i0 := 0; i0 < int(dims); i0 += 2 {

					////const double theta = pow(10000.0, ((double)-i0)/n_dims);
					theta := math.Pow(10000.0, float64(-i0)/float64(dims))

					cosTheta := math.Cos(float64(p) * theta)
					sinTheta := math.Sin(float64(p) * theta)

					////const float * const src = (float *)((char *) src0->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
					offset := i3*nb3/4 + i2*nb2/4 + i1*nb1/4 + uint32(i0)*nb0/4
					src := src0.Data[offset:]
					////   float * dst_data  = (float *)((char *)  dst->data + i3*nb3 + i2*nb2 + i1*nb1 + i0*nb0);
					dstData := dst.Data[offset:]

					x0 := float64(src[0])
					x1 := float64(src[1])

					dstData[0] = float32(x0*cosTheta - x1*sinTheta)
					dstData[1] = float32(x0*sinTheta + x1*cosTheta)
				}
			}
		}
	}

}

// ggml_compute_forward_scale_f32
func ComputeForwardScaleFP32(params *ComputeParams, src0, src1, dst *Tensor) {

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

}

// ggml_compute_forward_diag_mask_inf
func ComputeForwardDiagMaskInfFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	////assert(params->ith == 0);
	////assert(src1->type == GGML_TYPE_I32);
	////assert(ggml_nelements(src1) == 1);

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

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
					dst.Data[k*dst.NB[2]/4+j*dst.NB[1]/4+i*dst.NB[0]/4] = float32(math.Inf(-1)) // TODO Use const
				}
			}
		}
	}

	if DEBUG {
		fmt.Printf("\n\n>>> ComputeForwardDiagMaskInfFP32 OUT <<<\n")
	}

}

func maxFloat(x, y float32) float32 {
	if x >= y {
		return x
	}
	return y
}

func VecMaxFP32(n uint32, x []float32) float32 {
	max := float32(math.Inf(-1)) // TODO use constant
	for i := uint32(0); i < n; i++ {
		max = maxFloat(max, x[i])
	}
	return max
}

// ggml_compute_forward_soft_max
func ComputeForwardSoftMaxFP32(params *ComputeParams, src0, dst *Tensor) {

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
	}

	if DEBUG {
		fmt.Printf("\n\n>>> ComputeForwardSoftMaxFP32 OUT <<<\n")
	}
}

// inline static void ggml_vec_add_f32 (const int n, float * z, const float * x, const float * y) { for (int i = 0; i < n; ++i) z[i]  = x[i] + y[i]; }
func VecAddFP32(n uint32, z, x, y []float32) {
	for i := uint32(0); i < n; i++ {
		z[i] = x[i] + y[i]
	}
}

// ggml_compute_forward_add
func ComputeForwardAddFP32(params *ComputeParams, src0, src1, dst *Tensor) {

	////GGML_ASSERT(ggml_are_same_shape(src0, src1) && ggml_are_same_shape(src0, dst));

	if params.Type == TASK_INIT || params.Type == TASK_FINALIZE {
		return
	}

	if src1.NB[0] != TYPE_SIZE[TYPE_F32] {
		fmt.Printf("[HALT] ComputeForwardAddFP32 : [src1] is NOT contiguous!")
		os.Exit(1)
	}

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

			VecAddFP32(nc, dst.Data[j*nb1/4:], src0.Data[j*nb01/4:], src1.Data[j*nb11/4:])
		}

	} else { // src1 is not contiguous
		for j := ith; j < n; j += nth {
			////float * dst_ptr  = (float *) ((char *) dst->data  + j*nb1);
			dstPtr := dst.Data[j*nb1/4:]
			////float * src0_ptr = (float *) ((char *) src0->data + j*nb01);
			src0Ptr := src0.Data[j*nb01/4:]
			for i := uint32(0); i < nc; i++ {
				////float * src1_ptr = (float *) ((char *) src1->data + j*nb11 + i*nb10);
				src1Ptr := src1.Data[j*nb11/4+i*nb10/4]
				dstPtr[i] = src0Ptr[i] + src1Ptr
			}
		}
	}

	if DEBUG {
		fmt.Printf("\n\n>>> OUT <<< ComputeForwardAddFP32 <<<")
	}
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

// ggml_compute_forward_silu
func ComputeForwardSiluFP32(params *ComputeParams, src0, dst *Tensor) {

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

	ith := params.ith
	nth := params.nth

	nc := src0.NE[0]
	nr := src0.Nrows()

	// rows per thread
	dr := (nr + nth - 1) / nth

	// row range for this thread
	ir0 := dr * ith
	ir1 := uint32(min(int(ir0+dr), int(nr)))

	for i1 := ir0; i1 < ir1; i1++ {
		////ggml_vec_silu_f32(nc,
		////        (float *) ((char *) dst->data  + i1*( dst->nb[1])),
		////        (float *) ((char *) src0->data + i1*(src0->nb[1])));

		VecSiluFP32(nc, dst.Data[i1*dst.NB[1]/4:], src0.Data[i1*src0.NB[1]/4:])
	}

	if DEBUG {
		printTensor(src0, "SRC SILI")
		printTensor(dst, "DST SILI")
	}
}

// ---

type TokenScore struct {
	Token string
	Score float32
}

type Vocab struct {
	Size     uint32
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

	Text string
	N    uint32
}

// struct llama_sp_bigram {
type Bigram struct {

	// NB! Allow -1
	Left  int
	Right int

	Score float32
	Size  uint32
}

func utf8Len(src byte) uint32 {
	lookup := []uint32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 3, 4}
	highbits := uint8(src) >> 4
	return lookup[highbits]
}

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
			max = cur
		}
	}

	pop := (*queue)[max]

	// replace max element with last and shrink slice (if max == last, then just remove it)
	(*queue)[max] = (*queue)[len(*queue)-1]
	*queue = (*queue)[:len(*queue)-1]

	return pop
}

func TryAddBigram(vocab *Vocab, symbols []Symbol, workQueue *[]Bigram, left, right int) {

	if left == -1 || right == -1 {
		return
	}

	token := symbols[left].Text[:symbols[left].N+symbols[right].N]
	id, ok := vocab.Token2ID[token]

	if !ok || int(id) >= len(vocab.ID2Token) {
		return
	}

	tokenScore := vocab.ID2Token[id]

	bigram := Bigram{Left: left, Right: right, Score: tokenScore.Score, Size: uint32(len(token))}
	*workQueue = append(*workQueue, bigram)
}

const NewLineToken = 13 // ml.Tokenize(Ctx.Vocab, "\n", false)[0]

// void tokenize(const std::string & text, std::vector<llama_vocab::id> & output) {
func Tokenize(vocab *Vocab, text string, bos bool) []uint32 {

	output := make([]uint32, 0)
	symbols := make([]Symbol, 0)   // std::vector<llama_sp_symbol> symbols_;
	workQueue := make([]Bigram, 0) // llama_sp_bigram::queue work_queue_; // std::priority_queue<llama_sp_bigram, queue_storage, comparator>;

	if bos {
		output = append(output, 1) // TODO: replace with vocab.bos
	}

	// --- split string into utf8 chars

	index := 0
	offs := 0
	for offs < len(text) {
		var sym Symbol
		charLen := min(len(text)-offs, int(utf8Len(text[offs])))
		sym.Text = text[offs:]
		sym.N = uint32(charLen)
		offs += charLen
		sym.Prev = index - 1
		if offs == len(text) {
			sym.Next = -1
		} else {
			sym.Next = index + 1
		}
		index++
		symbols = append(symbols, sym)
	}

	// seed the work queue with all possible 2-character tokens
	for i := 1; i < len(symbols); i++ {
		TryAddBigram(vocab, symbols, &workQueue, i-1, i)
	}

	// keep substituting the highest frequency pairs for as long as we can
	for len(workQueue) > 0 {
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

		// remove the right sym from the chain
		leftSym.Next = rightSym.Next
		if rightSym.Next >= 0 {
			symbols[rightSym.Next].Prev = bigram.Left
		}

		// find more substitutions
		TryAddBigram(vocab, symbols, &workQueue, leftSym.Prev, bigram.Left)
		TryAddBigram(vocab, symbols, &workQueue, bigram.Left, leftSym.Next)
	}

	for i := 0; i != -1; i = symbols[i].Next {
		symbol := symbols[i]
		id, ok := vocab.Token2ID[symbol.Text[:symbol.N]]

		if !ok {
			// output any symbols that did not form tokens as bytes.
			for j := uint32(0); j < symbol.N; j++ {
				////llama_vocab::id token_id = static_cast<uint8_t>(symbol.text[j]) + 3;
				tokenID := uint32(symbol.Text[j] + 3)
				output = append(output, tokenID)
			}
		} else {
			output = append(output, id)
		}
	}

	if DEBUG {
		fmt.Printf("\n\n=== TOKENIZER ===\n\n%+v", output)
		for i := 0; i < len(output); i++ {
			fmt.Printf("%d:'%s'  ", output[i], Token2Str(vocab, output[i]))
		}
	}

	return output

}

// TODO Do we need this?
func Init(params InitParams) {

	// ---- initialize GELU, SILU and EXP F32 tables

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

}

// Allocator is an experimental memory pool for FP32 slices
// TODO: Investigate https://github.com/valyala/bytebufferpool
type Allocator struct {
	sync.Mutex

	// TODO: [][]float32 vs []*[]float32
	// Used map[uint32][]*[]float32
	// Free map[uint32][]*[]float32

	PoolSize int
	MemSize  int

	Pool []byte
	Mem  []byte
}

// TODO: Precompute max needed RAM size
const MaxPool = 0 // 2_000_000_000
const MaxMem = 0  // 28_000_000_000

func NewAllocator() *Allocator {
	return &Allocator{
		// Used: make(map[uint32][]*[]float32),
		// Free: make(map[uint32][]*[]float32),
		Pool: make([]byte, MaxPool),
		Mem:  make([]byte, MaxMem),
	}
}

// Get new or reuse memory buffer of size bytes
func (a *Allocator) Get(size uint32) *[]float32 {
	//gcSlice := make([]float32, size, size)
	//return &gcSlice

	a.Lock()
	byteSize := int(size * 4)

	if a.PoolSize+byteSize >= MaxPool {
		fmt.Printf("[ HALT ] Allocator go over free POOL MEM")
		os.Exit(0)
	}

	cur := a.PoolSize
	a.PoolSize += byteSize
	a.Unlock()

	var slice []float32
	head := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	head.Len = int(size)
	head.Cap = int(size)
	head.Data = uintptr(unsafe.Pointer(&a.Pool[cur]))

	return &slice

	/*
		head := reflect.SliceHeader{
			Len:  size,
			Cap:  size,
			Data: (*[]float32)(unsafe.Pointer(&a.Mem[cur])),
		}*/

	/*
	   _, ok := a.Free[size]

	   	if !ok {
	   		a.Used[size] = make([]*[]float32, 0, 1024) // Which CAP default?
	   		a.Free[size] = make([]*[]float32, 0, 1024) // Which CAP default?
	   	}

	   available := len(a.Free[size])

	   	if available > 0 {
	   		slice := a.Free[size][available-1]
	   		a.Free[size] = a.Free[size][:available-1]
	   		a.Used[size] = append(a.Used[size], slice)
	   		return slice
	   	}

	   ///slice := make([]float32, size, size)
	   a.Used[size] = append(a.Used[size], &slice)
	   return &slice
	*/
}

// Get fixed memory buffer of size bytes
func (a *Allocator) GetFixed(size uint32) *[]float32 {
	//gcSlice := make([]float32, size, size)
	//return &gcSlice

	a.Lock()
	byteSize := int(size * 4)

	if a.MemSize+byteSize >= MaxMem {
		fmt.Printf("[ HALT ] Allocator go over free FIXED MEM")
		os.Exit(0)
	}

	cur := a.MemSize
	a.MemSize += byteSize
	a.Unlock()

	var slice []float32
	head := (*reflect.SliceHeader)(unsafe.Pointer(&slice))
	head.Len = int(size)
	head.Cap = int(size)
	head.Data = uintptr(unsafe.Pointer(&a.Mem[cur]))

	return &slice

	/*
		head := reflect.SliceHeader{
			Len:  size,
			Cap:  size,
			Data: (*[]float32)(unsafe.Pointer(&a.Mem[cur])),
		}*/
}

func (a *Allocator) Reset() {
	a.Lock()
	a.PoolSize = 0
	a.Unlock()
	runtime.GC()

	// var rtm runtime.MemStats
	// runtime.ReadMemStats(&rtm)
	// printMemStats("Start", rtm)

	/*
	   	for size, _ := range a.Used {
	   		a.Free[size] = append(a.Free[size], a.Used[size]...)
	   		a.Used[size] = a.Used[size][:0]
	   	}

	   fmt.Printf("")
	*/
}

func printMemStats(message string, rtm runtime.MemStats) {
	fmt.Println("\n===", message, "===")
	fmt.Println("Mallocs: ", rtm.Mallocs)
	fmt.Println("Frees: ", rtm.Frees)
	fmt.Println("LiveObjects: ", rtm.Mallocs-rtm.Frees)
	fmt.Println("PauseTotalNs: ", rtm.PauseTotalNs)
	fmt.Println("NumGC: ", rtm.NumGC)
	fmt.Println("LastGC: ", time.UnixMilli(int64(rtm.LastGC/1_000_000)))
	fmt.Println("HeapObjects: ", rtm.HeapObjects)
	fmt.Println("HeapAlloc: ", rtm.HeapAlloc)
}

/*
// Release memory buffer back
func (a *Allocator) Put(size uint32, slice []float32) {

}

// Release memory buffer back
func (a Allocator) PutTensor(tensor *Tensor) {
	size := tensor.NE[0] * tensor.NE[1] * tensor.NE[2] * tensor.NE[3]
	_, ok := a.Pool[size]
	if !ok {
		a.Pool[size] = make([][]float32, 0, 64) // Which CAP default?
	}
	a.Pool[size] = append(a.Pool[size], tensor.Data)
	tensor.Data = nil
}
*/

/*
func NewReusableTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewTensor(ctx, dt, 1, ne0, 1, 1, 1, nil) // Reusable OK
}

func NewReusableTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil) // Reusable OK
}

func NewReusableTensor3D(ctx *Context, dt DType, ne0, ne1, ne2 uint32) *Tensor {
	return NewTensor(ctx, dt, 3, ne0, ne1, ne2, 1, nil) // Reusable OK
}

func NewFixedTensor1D(ctx *Context, dt DType, ne0 uint32) *Tensor {
	return NewFixedTensor(ctx, dt, 1, ne0, 1, 1, 1, nil)
}

func NewFixedTensor2D(ctx *Context, dt DType, ne0, ne1 uint32) *Tensor {
	return NewFixedTensor(ctx, dt, 2, ne0, ne1, 1, 1, nil)
}
*/
/*
// ggml_new_tensor_impl
func NewReusableTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	fmt.Printf("NewReusableTensor")
	os.Exit(1)

	// Reusable OK
	if data == nil {
		data = *ctx.Allocator.Get(ne0 * ne1 * ne2 * ne3)
	}

	//if data == nil {
	//	total := ne0 * ne1 * ne2 * ne3
	//	data = make([]float32, total, total)
	//}

	return &Tensor{
		Type:     dt,
		Reusable: true,
		Dims:     dims,
		NE:       [4]uint32{ne0, ne1, ne2, ne3},
		NB:       [4]uint32{4, ne0 * 4, ne0 * ne1 * 4, ne0 * ne1 * ne2 * 4},
		op:       OP_NONE,
		Data:     data,
	}
}
*/
/*
// ggml_new_tensor_impl
func NewFixedTensor(ctx *Context, dt DType, dims uint32, ne0, ne1, ne2, ne3 uint32, data []float32) *Tensor {

	fmt.Printf("NewFixedTensor")
	os.Exit(1)

	// TODO: Check allowed data types on graph creation
	//if dt != TYPE_F32 && dt != TYPE_I32 {
	//	fmt.Printf("\n[ERROR] NewTensorImpl got not supported type : %d", dt)
	//	os.Exit(1)
	//}

	////ggml_assert_aligned(result);

	if data == nil {
		total := ne0 * ne1 * ne2 * ne3
		data = make([]float32, total, total)

		// Reusable OK ???
		// data = *ctx.Allocator.GetFixed(ne0 * ne1 * ne2 * ne3)
	}

	return &Tensor{
		Type: dt,
		Dims: dims,
		NE:   [4]uint32{ne0, ne1, ne2, ne3},
		NB:   [4]uint32{4, ne0 * 4, ne0 * ne1 * 4, ne0 * ne1 * ne2 * 4},
		op:   OP_NONE,
		Data: data,
	}
}
*/
