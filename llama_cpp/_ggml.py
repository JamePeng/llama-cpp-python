"""Internal module use at your own risk

This module provides a minimal interface for working with ggml tensors from llama-cpp-python
"""
import ctypes
import enum
import os
import pathlib

from llama_cpp._ctypes_extensions import (
    load_shared_library,
    byref,
    ctypes_function_for_shared_library,
)

from typing import (
    Callable,
    Union,
    NewType,
    Optional,
    TYPE_CHECKING,
)

libggml_base_path = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
libggml_base_paths = [
    libggml_base_path / "lib",
    libggml_base_path / "bin",
]

libggml = load_shared_library("ggml", libggml_base_paths)

ggml_function = ctypes_function_for_shared_library(libggml)

libggml_base = load_shared_library("ggml-base", libggml_base_paths)

ggml_base_function = ctypes_function_for_shared_library(libggml_base)

# // ====== ggml.h ======

GGML_FILE_MAGIC = 0x67676d6c # b"ggml"
GGML_FILE_VERSION = 2

GGML_QNT_VERSION = 2    # bump this on quantization format changes
GGML_QNT_VERSION_FACTOR = 1000 # do not change this

GGML_MAX_DIMS       =  4
GGML_MAX_PARAMS     =  2048
GGML_MAX_SRC        =  10
GGML_MAX_N_THREADS  =  512
GGML_MAX_OP_PARAMS  =  64

GGML_MAX_NAME = 64

GGML_DEFAULT_N_THREADS = 4
GGML_DEFAULT_GRAPH_SIZE = 2048

GGML_EXIT_SUCCESS = 0
GGML_EXIT_ABORTED = 1

GGML_ROPE_TYPE_NORMAL = 0
GGML_ROPE_TYPE_NEOX   = 2
GGML_ROPE_TYPE_MROPE  = 8
GGML_ROPE_TYPE_VISION = 24
GGML_ROPE_TYPE_IMROPE = 40 # binary: 101000

GGML_MROPE_SECTIONS = 4

# enum ggml_status {
#     GGML_STATUS_ALLOC_FAILED = -2,
#     GGML_STATUS_FAILED = -1,
#     GGML_STATUS_SUCCESS = 0,
#     GGML_STATUS_ABORTED = 1,
# };
class GGMLStatus(enum.IntEnum):
    GGML_STATUS_ALLOC_FAILED = -2
    GGML_STATUS_FAILED       = -1
    GGML_STATUS_SUCCESS      = 0
    GGML_STATUS_ABORTED      = 1


# // NOTE: always add types at the end of the enum to keep backward compatibility
# enum ggml_type {
#     GGML_TYPE_F32     = 0,
#     GGML_TYPE_F16     = 1,
#     GGML_TYPE_Q4_0    = 2,
#     GGML_TYPE_Q4_1    = 3,
#     // GGML_TYPE_Q4_2 = 4, support has been removed
#     // GGML_TYPE_Q4_3 = 5, support has been removed
#     GGML_TYPE_Q5_0    = 6,
#     GGML_TYPE_Q5_1    = 7,
#     GGML_TYPE_Q8_0    = 8,
#     GGML_TYPE_Q8_1    = 9,
#     GGML_TYPE_Q2_K    = 10,
#     GGML_TYPE_Q3_K    = 11,
#     GGML_TYPE_Q4_K    = 12,
#     GGML_TYPE_Q5_K    = 13,
#     GGML_TYPE_Q6_K    = 14,
#     GGML_TYPE_Q8_K    = 15,
#     GGML_TYPE_IQ2_XXS = 16,
#     GGML_TYPE_IQ2_XS  = 17,
#     GGML_TYPE_IQ3_XXS = 18,
#     GGML_TYPE_IQ1_S   = 19,
#     GGML_TYPE_IQ4_NL  = 20,
#     GGML_TYPE_IQ3_S   = 21,
#     GGML_TYPE_IQ2_S   = 22,
#     GGML_TYPE_IQ4_XS  = 23,
#     GGML_TYPE_I8      = 24,
#     GGML_TYPE_I16     = 25,
#     GGML_TYPE_I32     = 26,
#     GGML_TYPE_I64     = 27,
#     GGML_TYPE_F64     = 28,
#     GGML_TYPE_IQ1_M   = 29,
#     GGML_TYPE_BF16    = 30,
#     // GGML_TYPE_Q4_0_4_4 = 31, support has been removed from gguf files
#     // GGML_TYPE_Q4_0_4_8 = 32,
#     // GGML_TYPE_Q4_0_8_8 = 33,
#     GGML_TYPE_TQ1_0   = 34,
#     GGML_TYPE_TQ2_0   = 35,
#     // GGML_TYPE_IQ4_NL_4_4 = 36,
#     // GGML_TYPE_IQ4_NL_4_8 = 37,
#     // GGML_TYPE_IQ4_NL_8_8 = 38,
#     GGML_TYPE_MXFP4   = 39, // MXFP4 (1 block)
#     GGML_TYPE_NVFP4   = 40, // NVFP4 (4 blocks, E4M3 scale)
#     GGML_TYPE_Q1_0    = 41,
#     GGML_TYPE_COUNT   = 42,
# };
class GGMLType(enum.IntEnum):
    GGML_TYPE_F32  = 0
    GGML_TYPE_F16  = 1
    GGML_TYPE_Q4_0 = 2
    GGML_TYPE_Q4_1 = 3
    GGML_TYPE_Q5_0 = 6
    GGML_TYPE_Q5_1 = 7
    GGML_TYPE_Q8_0 = 8
    GGML_TYPE_Q8_1 = 9
    GGML_TYPE_Q2_K = 10
    GGML_TYPE_Q3_K = 11
    GGML_TYPE_Q4_K = 12
    GGML_TYPE_Q5_K = 13
    GGML_TYPE_Q6_K = 14
    GGML_TYPE_Q8_K = 15
    GGML_TYPE_IQ2_XXS = 16
    GGML_TYPE_IQ2_XS  = 17
    GGML_TYPE_IQ3_XXS = 18
    GGML_TYPE_IQ1_S  = 19
    GGML_TYPE_IQ4_NL = 20
    GGML_TYPE_IQ3_S  = 21
    GGML_TYPE_IQ2_S  = 22
    GGML_TYPE_IQ4_XS = 23
    GGML_TYPE_I8  = 24
    GGML_TYPE_I16 = 25
    GGML_TYPE_I32 = 26
    GGML_TYPE_I64 = 27
    GGML_TYPE_F64 = 28
    GGML_TYPE_IQ1_M = 29
    GGML_TYPE_BF16  = 30
    GGML_TYPE_TQ1_0 = 34
    GGML_TYPE_TQ2_0 = 35
    GGML_TYPE_MXFP4 = 39
    GGML_TYPE_NVFP4 = 40
    GGML_TYPE_Q1_0 = 41
    GGML_TYPE_COUNT = 42


# // precision
# enum ggml_prec {
#     GGML_PREC_DEFAULT =  0, // stored as ggml_tensor.op_params, 0 by default
#     GGML_PREC_F32     = 10,
# };
class GGMLPrec(enum.IntEnum):
    GGML_PREC_DEFAULT =  0
    GGML_PREC_F32     = 10


# // model file types
# enum ggml_ftype {
#     GGML_FTYPE_UNKNOWN        = -1,
#     GGML_FTYPE_ALL_F32        = 0,
#     GGML_FTYPE_MOSTLY_F16     = 1,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_0    = 2,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1    = 3,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4, // tok_embeddings.weight and output.weight are F16
#     GGML_FTYPE_MOSTLY_Q8_0    = 7,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_0    = 8,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_1    = 9,  // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q2_K    = 10, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q3_K    = 11, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q4_K    = 12, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q5_K    = 13, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q6_K    = 14, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XXS = 15, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_XS  = 16, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_XXS = 17, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_S   = 18, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_NL  = 19, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ3_S   = 20, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ2_S   = 21, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ4_XS  = 22, // except 1d tensors
#     GGML_FTYPE_MOSTLY_IQ1_M   = 23, // except 1d tensors
#     GGML_FTYPE_MOSTLY_BF16    = 24, // except 1d tensors
#     GGML_FTYPE_MOSTLY_MXFP4   = 25, // except 1d tensors
#     GGML_FTYPE_MOSTLY_NVFP4   = 26, // except 1d tensors
#     GGML_FTYPE_MOSTLY_Q1_0    = 27, // except 1d tensors
# };
class GGMLFType(enum.IntEnum):
    GGML_FTYPE_UNKNOWN        = -1
    GGML_FTYPE_ALL_F32        = 0
    GGML_FTYPE_MOSTLY_F16     = 1
    GGML_FTYPE_MOSTLY_Q4_0    = 2
    GGML_FTYPE_MOSTLY_Q4_1    = 3
    GGML_FTYPE_MOSTLY_Q4_1_SOME_F16 = 4
    GGML_FTYPE_MOSTLY_Q8_0    = 7
    GGML_FTYPE_MOSTLY_Q5_0    = 8
    GGML_FTYPE_MOSTLY_Q5_1    = 9
    GGML_FTYPE_MOSTLY_Q2_K    = 10
    GGML_FTYPE_MOSTLY_Q3_K    = 11
    GGML_FTYPE_MOSTLY_Q4_K    = 12
    GGML_FTYPE_MOSTLY_Q5_K    = 13
    GGML_FTYPE_MOSTLY_Q6_K    = 14
    GGML_FTYPE_MOSTLY_IQ2_XXS = 15
    GGML_FTYPE_MOSTLY_IQ2_XS  = 16
    GGML_FTYPE_MOSTLY_IQ3_XXS = 17
    GGML_FTYPE_MOSTLY_IQ1_S   = 18
    GGML_FTYPE_MOSTLY_IQ4_NL  = 19
    GGML_FTYPE_MOSTLY_IQ3_S   = 20
    GGML_FTYPE_MOSTLY_IQ2_S   = 21
    GGML_FTYPE_MOSTLY_IQ4_XS  = 22
    GGML_FTYPE_MOSTLY_IQ1_M   = 23
    GGML_FTYPE_MOSTLY_BF16    = 24
    GGML_FTYPE_MOSTLY_MXFP4   = 25
    GGML_FTYPE_MOSTLY_NVFP4   = 26
    GGML_FTYPE_MOSTLY_Q1_0    = 27


# // available tensor operations:
# enum ggml_op {
#     GGML_OP_NONE = 0,

#     GGML_OP_DUP,
#     GGML_OP_ADD,
#     GGML_OP_ADD_ID,
#     GGML_OP_ADD1,
#     GGML_OP_ACC,
#     GGML_OP_SUB,
#     GGML_OP_MUL,
#     GGML_OP_DIV,
#     GGML_OP_SQR,
#     GGML_OP_SQRT,
#     GGML_OP_LOG,
#     GGML_OP_SIN,
#     GGML_OP_COS,
#     GGML_OP_SUM,
#     GGML_OP_SUM_ROWS,
#     GGML_OP_CUMSUM,
#     GGML_OP_MEAN,
#     GGML_OP_ARGMAX,
#     GGML_OP_COUNT_EQUAL,
#     GGML_OP_REPEAT,
#     GGML_OP_REPEAT_BACK,
#     GGML_OP_CONCAT,
#     GGML_OP_SILU_BACK,
#     GGML_OP_NORM, // normalize
#     GGML_OP_RMS_NORM,
#     GGML_OP_RMS_NORM_BACK,
#     GGML_OP_GROUP_NORM,
#     GGML_OP_L2_NORM,

#     GGML_OP_MUL_MAT,
#     GGML_OP_MUL_MAT_ID,
#     GGML_OP_OUT_PROD,

#     GGML_OP_SCALE,
#     GGML_OP_SET,
#     GGML_OP_CPY,
#     GGML_OP_CONT,
#     GGML_OP_RESHAPE,
#     GGML_OP_VIEW,
#     GGML_OP_PERMUTE,
#     GGML_OP_TRANSPOSE,
#     GGML_OP_GET_ROWS,
#     GGML_OP_GET_ROWS_BACK,
#     GGML_OP_SET_ROWS,
#     GGML_OP_DIAG,
#     GGML_OP_DIAG_MASK_INF,
#     GGML_OP_DIAG_MASK_ZERO,
#     GGML_OP_SOFT_MAX,
#     GGML_OP_SOFT_MAX_BACK,
#     GGML_OP_ROPE,
#     GGML_OP_ROPE_BACK,
#     GGML_OP_CLAMP,
#     GGML_OP_CONV_TRANSPOSE_1D,
#     GGML_OP_IM2COL,
#     GGML_OP_IM2COL_BACK,
#     GGML_OP_IM2COL_3D,
#     GGML_OP_CONV_2D,
#     GGML_OP_CONV_3D,
#     GGML_OP_CONV_2D_DW,
#     GGML_OP_CONV_TRANSPOSE_2D,
#     GGML_OP_POOL_1D,
#     GGML_OP_POOL_2D,
#     GGML_OP_POOL_2D_BACK,
#     GGML_OP_UPSCALE,
#     GGML_OP_PAD,
#     GGML_OP_PAD_REFLECT_1D,
#     GGML_OP_ROLL,
#     GGML_OP_ARANGE,
#     GGML_OP_TIMESTEP_EMBEDDING,
#     GGML_OP_ARGSORT,
#     GGML_OP_TOP_K,
#     GGML_OP_LEAKY_RELU,
#     GGML_OP_TRI,
#     GGML_OP_FILL,

#     GGML_OP_FLASH_ATTN_EXT,
#     GGML_OP_FLASH_ATTN_BACK,
#     GGML_OP_SSM_CONV,
#     GGML_OP_SSM_SCAN,
#     GGML_OP_WIN_PART,
#     GGML_OP_WIN_UNPART,
#     GGML_OP_GET_REL_POS,
#     GGML_OP_ADD_REL_POS,
#     GGML_OP_RWKV_WKV6,
#     GGML_OP_GATED_LINEAR_ATTN,
#     GGML_OP_RWKV_WKV7,
#     GGML_OP_SOLVE_TRI,
#     GGML_OP_GATED_DELTA_NET,

#     GGML_OP_UNARY,

#     GGML_OP_MAP_CUSTOM1,
#     GGML_OP_MAP_CUSTOM2,
#     GGML_OP_MAP_CUSTOM3,

#     GGML_OP_CUSTOM,

#     GGML_OP_CROSS_ENTROPY_LOSS,
#     GGML_OP_CROSS_ENTROPY_LOSS_BACK,
#     GGML_OP_OPT_STEP_ADAMW,
#     GGML_OP_OPT_STEP_SGD,

#     GGML_OP_GLU,

#     GGML_OP_COUNT,
# };
class GGML_OP(enum.IntEnum):
    GGML_OP_NONE = 0

    GGML_OP_DUP = 1
    GGML_OP_ADD = 2
    GGML_OP_ADD_ID = 3
    GGML_OP_ADD1 = 4
    GGML_OP_ACC = 5
    GGML_OP_SUB = 6
    GGML_OP_MUL = 7
    GGML_OP_DIV = 8
    GGML_OP_SQR = 9
    GGML_OP_SQRT = 10
    GGML_OP_LOG = 11
    GGML_OP_SIN = 12
    GGML_OP_COS = 13
    GGML_OP_SUM = 14
    GGML_OP_SUM_ROWS = 15
    GGML_OP_CUMSUM = 16
    GGML_OP_MEAN = 17
    GGML_OP_ARGMAX = 18
    GGML_OP_COUNT_EQUAL = 19
    GGML_OP_REPEAT = 20
    GGML_OP_REPEAT_BACK = 21
    GGML_OP_CONCAT = 22
    GGML_OP_SILU_BACK = 23
    GGML_OP_NORM = 24 # // normalize
    GGML_OP_RMS_NORM = 25
    GGML_OP_RMS_NORM_BACK = 26
    GGML_OP_GROUP_NORM = 27
    GGML_OP_L2_NORM = 28

    GGML_OP_MUL_MAT = 29
    GGML_OP_MUL_MAT_ID = 30
    GGML_OP_OUT_PROD = 31

    GGML_OP_SCALE = 32
    GGML_OP_SET = 33
    GGML_OP_CPY = 34
    GGML_OP_CONT = 35
    GGML_OP_RESHAPE = 36
    GGML_OP_VIEW = 37
    GGML_OP_PERMUTE = 38
    GGML_OP_TRANSPOSE = 39
    GGML_OP_GET_ROWS = 40
    GGML_OP_GET_ROWS_BACK = 41
    GGML_OP_SET_ROWS = 42
    GGML_OP_DIAG = 43
    GGML_OP_DIAG_MASK_INF = 44
    GGML_OP_DIAG_MASK_ZERO = 45
    GGML_OP_SOFT_MAX = 46
    GGML_OP_SOFT_MAX_BACK = 47
    GGML_OP_ROPE = 48
    GGML_OP_ROPE_BACK = 49
    GGML_OP_CLAMP = 50
    GGML_OP_CONV_TRANSPOSE_1D = 51
    GGML_OP_IM2COL = 52
    GGML_OP_IM2COL_BACK = 53
    GGML_OP_IM2COL_3D = 54
    GGML_OP_CONV_2D = 55
    GGML_OP_CONV_3D = 56
    GGML_OP_CONV_2D_DW = 57
    GGML_OP_CONV_TRANSPOSE_2D = 58
    GGML_OP_POOL_1D = 59
    GGML_OP_POOL_2D = 60
    GGML_OP_POOL_2D_BACK = 61
    GGML_OP_UPSCALE = 62
    GGML_OP_PAD = 63
    GGML_OP_PAD_REFLECT_1D = 64
    GGML_OP_ROLL = 65
    GGML_OP_ARANGE = 66
    GGML_OP_TIMESTEP_EMBEDDING = 67
    GGML_OP_ARGSORT = 68
    GGML_OP_TOP_K = 69
    GGML_OP_LEAKY_RELU = 70
    GGML_OP_TRI = 71
    GGML_OP_FILL = 72

    GGML_OP_FLASH_ATTN_EXT = 73
    GGML_OP_FLASH_ATTN_BACK = 74
    GGML_OP_SSM_CONV = 75
    GGML_OP_SSM_SCAN = 76
    GGML_OP_WIN_PART = 77
    GGML_OP_WIN_UNPART = 78
    GGML_OP_GET_REL_POS = 79
    GGML_OP_ADD_REL_POS = 80
    GGML_OP_RWKV_WKV6 = 81
    GGML_OP_GATED_LINEAR_ATTN = 82
    GGML_OP_RWKV_WKV7 = 83
    GGML_OP_SOLVE_TRI = 84
    GGML_OP_GATED_DELTA_NET = 85

    GGML_OP_UNARY = 86

    GGML_OP_MAP_CUSTOM1 = 87
    GGML_OP_MAP_CUSTOM2 = 88
    GGML_OP_MAP_CUSTOM3 = 89

    GGML_OP_CUSTOM = 90

    GGML_OP_CROSS_ENTROPY_LOSS = 91
    GGML_OP_CROSS_ENTROPY_LOSS_BACK = 92
    GGML_OP_OPT_STEP_ADAMW = 93
    GGML_OP_OPT_STEP_SGD = 94

    GGML_OP_GLU = 95

    GGML_OP_COUNT = 96

# enum ggml_unary_op {
#     GGML_UNARY_OP_ABS,
#     GGML_UNARY_OP_SGN,
#     GGML_UNARY_OP_NEG,
#     GGML_UNARY_OP_STEP,
#     GGML_UNARY_OP_TANH,
#     GGML_UNARY_OP_ELU,
#     GGML_UNARY_OP_RELU,
#     GGML_UNARY_OP_SIGMOID,
#     GGML_UNARY_OP_GELU,
#     GGML_UNARY_OP_GELU_QUICK,
#     GGML_UNARY_OP_SILU,
#     GGML_UNARY_OP_HARDSWISH,
#     GGML_UNARY_OP_HARDSIGMOID,
#     GGML_UNARY_OP_EXP,
#     GGML_UNARY_OP_EXPM1,
#     GGML_UNARY_OP_SOFTPLUS,
#     GGML_UNARY_OP_GELU_ERF,
#     GGML_UNARY_OP_XIELU,
#     GGML_UNARY_OP_FLOOR,
#     GGML_UNARY_OP_CEIL,
#     GGML_UNARY_OP_ROUND,
#     GGML_UNARY_OP_TRUNC,

#     GGML_UNARY_OP_COUNT,
# };
class GGMLUnaryOp(enum.IntEnum):
    GGML_UNARY_OP_ABS = 0
    GGML_UNARY_OP_SGN = 1
    GGML_UNARY_OP_NEG = 2
    GGML_UNARY_OP_STEP = 3
    GGML_UNARY_OP_TANH = 4
    GGML_UNARY_OP_ELU = 5
    GGML_UNARY_OP_RELU = 6
    GGML_UNARY_OP_SIGMOID = 7
    GGML_UNARY_OP_GELU = 8
    GGML_UNARY_OP_GELU_QUICK = 9
    GGML_UNARY_OP_SILU = 10
    GGML_UNARY_OP_HARDSWISH = 11
    GGML_UNARY_OP_HARDSIGMOID = 12
    GGML_UNARY_OP_EXP = 13
    GGML_UNARY_OP_EXPM1 = 14
    GGML_UNARY_OP_SOFTPLUS = 15
    GGML_UNARY_OP_GELU_ERF = 16
    GGML_UNARY_OP_XIELU = 17
    GGML_UNARY_OP_FLOOR = 18
    GGML_UNARY_OP_CEIL = 19
    GGML_UNARY_OP_ROUND = 20
    GGML_UNARY_OP_TRUNC = 21

    GGML_UNARY_OP_COUNT = 22

# enum ggml_glu_op {
#     GGML_GLU_OP_REGLU,
#     GGML_GLU_OP_GEGLU,
#     GGML_GLU_OP_SWIGLU,
#     GGML_GLU_OP_SWIGLU_OAI,
#     GGML_GLU_OP_GEGLU_ERF,
#     GGML_GLU_OP_GEGLU_QUICK,
#     GGML_GLU_OP_COUNT,
# };
class GGMLGluOp(enum.IntEnum):
    GGML_GLU_OP_REGLU = 0
    GGML_GLU_OP_GEGLU = 1
    GGML_GLU_OP_SWIGLU = 2
    GGML_GLU_OP_SWIGLU_OAI = 3
    GGML_GLU_OP_GEGLU_ERF = 4
    GGML_GLU_OP_GEGLU_QUICK = 5

    GGML_GLU_OP_COUNT = 6

# //
# // ggml object
# //

# enum ggml_object_type {
#     GGML_OBJECT_TYPE_TENSOR,
#     GGML_OBJECT_TYPE_GRAPH,
#     GGML_OBJECT_TYPE_WORK_BUFFER
# };
class GGMLObjectType(enum.IntEnum):
    GGML_OBJECT_TYPE_TENSOR      = 0
    GGML_OBJECT_TYPE_GRAPH       = 1
    GGML_OBJECT_TYPE_WORK_BUFFER = 2


# struct ggml_object {
#     size_t offs;
#     size_t size;
#     struct ggml_object * next;
#     enum ggml_object_type type;
#     char padding[4];
# };
class ggml_object(ctypes.Structure):
    if TYPE_CHECKING:
        offs: ctypes.c_size_t
        size: ctypes.c_size_t
        next: "ctypes.POINTER(ggml_object)"
        type: int
        padding: ctypes.Array[ctypes.c_char]

ggml_object_p = ctypes.POINTER(ggml_object)

ggml_object._fields_ = [
    ("offs", ctypes.c_size_t),
    ("size", ctypes.c_size_t),
    ("next", ggml_object_p),
    ("type", ctypes.c_int),
    ("padding", ctypes.c_char * 4),
]

GGML_OBJECT_SIZE = ctypes.sizeof(ggml_object)


# //
# // ggml context
# //

# struct ggml_context {
#     size_t mem_size;
#     void * mem_buffer;
#     bool   mem_buffer_owned;
#     bool   no_alloc;
#     int    n_objects;
#     struct ggml_object * objects_begin;
#     struct ggml_object * objects_end;
# };
class ggml_context(ctypes.Structure):

    if TYPE_CHECKING:
        mem_size: ctypes.c_size_t
        mem_buffer: ctypes.c_void_p
        mem_buffer_owned: bool
        no_alloc: bool
        n_objects: int
        objects_begin: ggml_object_p
        objects_end: ggml_object_p

    _fields_ = [
        ("mem_size", ctypes.c_size_t),
        ("mem_buffer", ctypes.c_void_p),
        ("mem_buffer_owned", ctypes.c_bool),
        ("no_alloc", ctypes.c_bool),
        ("n_objects", ctypes.c_int),
        ("objects_begin", ggml_object_p),
        ("objects_end", ggml_object_p),
    ]

ggml_context_p = ctypes.POINTER(ggml_context)


# enum ggml_log_level {
#     GGML_LOG_LEVEL_NONE  = 0,
#     GGML_LOG_LEVEL_DEBUG = 1,
#     GGML_LOG_LEVEL_INFO  = 2,
#     GGML_LOG_LEVEL_WARN  = 3,
#     GGML_LOG_LEVEL_ERROR = 4,
#     GGML_LOG_LEVEL_CONT  = 5, // continue previous log
# };
class GGMLLogLevel(enum.IntEnum):
    GGML_LOG_LEVEL_NONE  = 0
    GGML_LOG_LEVEL_DEBUG = 1
    GGML_LOG_LEVEL_INFO  = 2
    GGML_LOG_LEVEL_WARN  = 3
    GGML_LOG_LEVEL_ERROR = 4
    GGML_LOG_LEVEL_CONT  = 5 # continue previous log


# // this tensor...
# enum ggml_tensor_flag {
#     GGML_TENSOR_FLAG_INPUT  =  1, // ...is an input for the GGML compute graph
#     GGML_TENSOR_FLAG_OUTPUT =  2, // ...is an output for the GGML compute graph
#     GGML_TENSOR_FLAG_PARAM  =  4, // ...contains trainable parameters
#     GGML_TENSOR_FLAG_LOSS   =  8, // ...defines loss for numerical optimization (multiple loss tensors add up)
# };
class GGMLTensorFlag(enum.IntEnum):
    GGML_TENSOR_FLAG_INPUT  = 1  # ...is an input for the GGML compute graph
    GGML_TENSOR_FLAG_OUTPUT = 2  # ...is an output for the GGML compute graph
    GGML_TENSOR_FLAG_PARAM  = 4  # ...contains trainable parameters
    GGML_TENSOR_FLAG_LOSS   = 8  # ...defines loss for numerical optimization (multiple loss tensors add up)


# enum ggml_tri_type {
#     GGML_TRI_TYPE_UPPER_DIAG = 0,
#     GGML_TRI_TYPE_UPPER      = 1,
#     GGML_TRI_TYPE_LOWER_DIAG = 2,
#     GGML_TRI_TYPE_LOWER      = 3
# };
class GGMLTriType(enum.IntEnum):
    GGML_TRI_TYPE_UPPER_DIAG = 0
    GGML_TRI_TYPE_UPPER      = 1
    GGML_TRI_TYPE_LOWER_DIAG = 2
    GGML_TRI_TYPE_LOWER      = 3


# struct ggml_init_params {
#     // memory pool
#     size_t mem_size;   // bytes
#     void * mem_buffer; // if NULL, memory will be allocated internally
#     bool   no_alloc;   // don't allocate memory for the tensor data
# };
class ggml_init_params(ctypes.Structure):
    _fields_ = [
        ('mem_size', ctypes.c_size_t),
        ('mem_buffer', ctypes.c_void_p),
        ('no_alloc', ctypes.c_bool),
    ]


# // n-dimensional tensor
# struct ggml_tensor {
#     enum ggml_type type;
#     struct ggml_backend_buffer * buffer;
#     int64_t ne[GGML_MAX_DIMS]; // number of elements
#     size_t  nb[GGML_MAX_DIMS]; // stride in bytes:
#                                 // nb[0] = ggml_type_size(type)
#                                 // nb[1] = nb[0]   * (ne[0] / ggml_blck_size(type)) + padding
#                                 // nb[i] = nb[i-1] * ne[i-1]
#     // compute data
#     enum ggml_op op;
#     // op params - allocated as int32_t for alignment
#     int32_t op_params[GGML_MAX_OP_PARAMS / sizeof(int32_t)];
#     int32_t flags;
#     struct ggml_tensor * src[GGML_MAX_SRC];
#     // source tensor and offset for views
#     struct ggml_tensor * view_src;
#     size_t               view_offs;
#     void * data;
#     char name[GGML_MAX_NAME];
#     void * extra; // extra things e.g. for ggml-cuda.cu
#     char padding[8];
# };
class ggml_tensor(ctypes.Structure):
    """n-dimensional tensor"""

    if TYPE_CHECKING:
        type: int
        buffer: ctypes.c_void_p
        ne: ctypes.Array[ctypes.c_int64]
        nb: ctypes.Array[ctypes.c_size_t]
        op: int
        op_params: ctypes.Array[ctypes.c_int32]
        flags: int
        src: "ctypes.Array[ctypes.POINTER(ggml_tensor)]"
        view_src: "ctypes.POINTER(ggml_tensor)"
        view_offs: ctypes.c_size_t
        data: ctypes.c_void_p
        name: ctypes.Array[ctypes.c_char]
        extra: ctypes.c_void_p
        padding: ctypes.Array[ctypes.c_char]

ggml_tensor_p = ctypes.POINTER(ggml_tensor)

ggml_tensor._fields_ = [
        ("type", ctypes.c_int),
        ("buffer", ctypes.c_void_p),
        ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
        ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
        ("op", ctypes.c_int),
        ("op_params", ctypes.c_int32 * (GGML_MAX_OP_PARAMS // ctypes.sizeof(ctypes.c_int32))),
        ("flags", ctypes.c_int32),
        ("src", ggml_tensor_p * GGML_MAX_SRC),
        ("view_src", ggml_tensor_p),
        ("view_offs", ctypes.c_size_t),
        ("data", ctypes.c_void_p),
        ("name", ctypes.c_char * GGML_MAX_NAME),
        ("extra", ctypes.c_void_p),
        ("padding", ctypes.c_char * 8),
]

# // Abort callback
# // If not NULL, called before ggml computation
# // If it returns true, the computation is aborted
# typedef bool (*ggml_abort_callback)(void * data);
ggml_abort_callback = ctypes.CFUNCTYPE(ctypes.c_bool, ctypes.c_void_p)


# // TODO these functions were sandwiched in the old optimization interface, is there a better place for them?
# typedef void (*ggml_log_callback)(enum ggml_log_level level, const char * text, void * user_data);
ggml_log_callback = ctypes.CFUNCTYPE(
    None, ctypes.c_int, ctypes.c_char_p, ctypes.c_void_p
)

# // Set callback for all future logging events.
# // If this is not called, or NULL is supplied, everything is output on stderr.
# // The logger state is global so these functions are NOT thread safe.
# GGML_API void ggml_log_get(ggml_log_callback * log_callback, void ** user_data);
@ggml_base_function(
    "ggml_log_get",
    [ctypes.POINTER(ggml_log_callback), ctypes.POINTER(ctypes.c_void_p)],
    None,
)
def ggml_log_get(
    log_callback: Optional[ctypes.POINTER(ggml_log_callback)],
    user_data: ctypes.POINTER(ctypes.c_void_p),
    /,
):
    """
    Get callback for all future logging events.
    If this is not called, or NULL is supplied, everything is output on stderr.
    The logger state is global so these functions are NOT thread safe.
    """
    ...


# GGML_API void ggml_log_set(ggml_log_callback   log_callback, void *  user_data);
@ggml_base_function(
    "ggml_log_set",
    [ggml_log_callback, ctypes.c_void_p],
    None,
)
def ggml_log_set(
    log_callback: Optional[ggml_log_callback],
    user_data: ctypes.c_void_p,
    /,
):
    """
    Set callback for all future logging events.
    If this is not called, or NULL is supplied, everything is output on stderr.
    The logger state is global so these functions are NOT thread safe.
    """
    ...


# GGML_API struct ggml_tensor * ggml_set_zero(struct ggml_tensor * tensor);
@ggml_base_function(
    "ggml_set_zero",
    [ggml_tensor_p],
    ggml_tensor_p,
)
def ggml_set_zero(
    tensor: ctypes.c_void_p,
    /,
) -> ctypes.c_void_p:
    """
    Memset tensor data to zero
    """
    ...


# // ====== ggml-opt.h ======

# // built-in loss types, i.e. the built-in quantities minimized by the optimizer
# // custom loss types can be defined via mean or sum which simply reduce the outputs for all datapoints to a single value
# enum ggml_opt_loss_type {
#     GGML_OPT_LOSS_TYPE_MEAN,
#     GGML_OPT_LOSS_TYPE_SUM,
#     GGML_OPT_LOSS_TYPE_CROSS_ENTROPY,
#     GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR,
# };
class GGMLOptLossType(enum.IntEnum):
    GGML_OPT_LOSS_TYPE_MEAN               = 0
    GGML_OPT_LOSS_TYPE_SUM                = 1
    GGML_OPT_LOSS_TYPE_CROSS_ENTROPY      = 2
    GGML_OPT_LOSS_TYPE_MEAN_SQUARED_ERROR = 3


# enum ggml_opt_build_type {
#     GGML_OPT_BUILD_TYPE_FORWARD = 10,
#     GGML_OPT_BUILD_TYPE_GRAD    = 20,
#     GGML_OPT_BUILD_TYPE_OPT     = 30,
# };
class GGMLOptBuildType(enum.IntEnum):
    GGML_OPT_BUILD_TYPE_FORWARD = 10
    GGML_OPT_BUILD_TYPE_GRAD    = 20
    GGML_OPT_BUILD_TYPE_OPT     = 30


# enum ggml_opt_optimizer_type {
#     GGML_OPT_OPTIMIZER_TYPE_ADAMW,
#     GGML_OPT_OPTIMIZER_TYPE_SGD,

#     GGML_OPT_OPTIMIZER_TYPE_COUNT
# };
class GGMLOptBuildType(enum.IntEnum):
    GGML_OPT_OPTIMIZER_TYPE_ADAMW = 0
    GGML_OPT_OPTIMIZER_TYPE_SGD   = 1
    GGML_OPT_OPTIMIZER_TYPE_COUNT = 2


# // parameters that control which optimizer is used and how said optimizer tries to find the minimal loss
# struct ggml_opt_optimizer_params {
#     struct {
#         float alpha; // learning rate
#         float beta1; // first AdamW momentum
#         float beta2; // second AdamW momentum
#         float eps;   // epsilon for numerical stability
#         float wd;    // weight decay - 0.0f to disable
#     } adamw;
#     struct {
#         float alpha; // learning rate
#         float wd;    // weight decay
#     } sgd;
# };
class ggml_opt_adamw_params(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float), # learning rate
        ('beta1', ctypes.c_float), # first AdamW momentum
        ('beta2', ctypes.c_float), # second AdamW momentum
        ('eps',   ctypes.c_float), # epsilon for numerical stability
        ('wd',    ctypes.c_float), # weight decay - 0.0f to disable
    ]

class ggml_opt_sgd_params(ctypes.Structure):
    _fields_ = [
        ('alpha', ctypes.c_float), # learning rate
        ('wd',    ctypes.c_float), # weight decay
    ]

class ggml_opt_optimizer_params(ctypes.Structure):
    _fields_ = [
        ('adamw', ggml_opt_adamw_params), # Nested AdamW parameters
        ('sgd', ggml_opt_sgd_params), # Nested SGD parameters
    ]


# // callback to calculate optimizer parameters prior to a backward pass
# // userdata can be used to pass arbitrary data
# typedef struct ggml_opt_optimizer_params (*ggml_opt_get_optimizer_params)(void * userdata);
ggml_opt_get_optimizer_params = ctypes.CFUNCTYPE(
    ctypes.POINTER(ggml_opt_optimizer_params), ctypes.c_void_p
)


# from ggml-backend.h
# // Evaluation callback for each node in the graph (set with ggml_backend_sched_set_eval_callback)
# // when ask == true, the scheduler wants to know if the user wants to observe this node
# // this allows the scheduler to batch nodes together in order to evaluate them in a single call
# //
# // when ask == false, the scheduler is passing the node tensor to the user for observation
# // if the user returns false, the scheduler will cancel the graph compute
# //
# typedef bool (*ggml_backend_sched_eval_callback)(struct ggml_tensor * t, bool ask, void * user_data);
ggml_backend_sched_eval_callback = ctypes.CFUNCTYPE(
    ctypes.c_bool, ctypes.c_void_p, ctypes.c_bool, ctypes.c_void_p
)

# //
# // Backend registry
# //

# // Load all known backends from dynamic libraries
# GGML_API void               ggml_backend_load_all(void);
@ggml_function("ggml_backend_load_all", [], None)
def ggml_backend_load_all():
    """Load all known backends from dynamic libraries"""
    ...

# GGML_API void               ggml_backend_load_all_from_path(const char * dir_path);
@ggml_function("ggml_backend_load_all_from_path", [ctypes.c_char_p], None)
def ggml_backend_load_all_from_path(dir_path: ctypes.c_char_p):
    """Load all known backends from path"""
    ...

# //
# // GGML internal header from ggml-impl.h
# //

# typedef uint32_t ggml_bitset_t;
ggml_bitset_t = ctypes.c_uint32

# // computation graph

# enum ggml_cgraph_eval_order {
#     GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0,
#     GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT,
#     GGML_CGRAPH_EVAL_ORDER_COUNT
# };
class GGMLCgraphEvalOrder(enum.IntEnum):
    GGML_CGRAPH_EVAL_ORDER_LEFT_TO_RIGHT = 0
    GGML_CGRAPH_EVAL_ORDER_RIGHT_TO_LEFT = 1
    GGML_CGRAPH_EVAL_ORDER_COUNT = 2


# struct ggml_hash_set {
#     size_t size;
#     ggml_bitset_t * used;       // whether or not the keys are in use i.e. set
#     struct ggml_tensor ** keys; // actual tensors in the set, keys[i] is only defined if ggml_bitset_get(used, i)
# };
class ggml_hash_set(ctypes.Structure):
    if TYPE_CHECKING:
        size: int
        used: ctypes.POINTER(ggml_bitset_t)
        keys: "ctypes.POINTER(ggml_tensor_p)"

    _fields_ = [
        ("size", ctypes.c_size_t),
        ("used", ctypes.POINTER(ggml_bitset_t)),
        ("keys", ctypes.POINTER(ggml_tensor_p)),
    ]


# struct ggml_cgraph {
#     int size;    // maximum number of nodes/leafs/grads/grad_accs
#     int n_nodes; // number of nodes currently in use
#     int n_leafs; // number of leafs currently in use

#     struct ggml_tensor ** nodes;     // tensors with data that can change if the graph is evaluated
#     struct ggml_tensor ** grads;     // the outputs of these tensors are the gradients of the nodes
#     struct ggml_tensor ** grad_accs; // accumulators for node gradients
#     struct ggml_tensor ** leafs;     // tensors with constant data
#     int32_t             * use_counts;// number of uses of each tensor, indexed by hash table slot

#     struct ggml_hash_set visited_hash_set;

#     enum ggml_cgraph_eval_order order;
# };
class ggml_cgraph(ctypes.Structure):
    if TYPE_CHECKING:
        size: int
        n_nodes: int
        n_leafs: int
        nodes: "ctypes.POINTER(ggml_tensor_p)"
        grads: "ctypes.POINTER(ggml_tensor_p)"
        grad_accs: "ctypes.POINTER(ggml_tensor_p)"
        leafs: "ctypes.POINTER(ggml_tensor_p)"
        use_counts: ctypes.POINTER(ctypes.c_int32)
        visited_hash_set: ggml_hash_set
        order: int

    _fields_ = [
        ("size", ctypes.c_int),
        ("n_nodes", ctypes.c_int),
        ("n_leafs", ctypes.c_int),
        ("nodes", ctypes.POINTER(ggml_tensor_p)),
        ("grads", ctypes.POINTER(ggml_tensor_p)),
        ("grad_accs", ctypes.POINTER(ggml_tensor_p)),
        ("leafs", ctypes.POINTER(ggml_tensor_p)),
        ("use_counts", ctypes.POINTER(ctypes.c_int32)),
        ("visited_hash_set", ggml_hash_set),
        ("order", ctypes.c_int),
    ]
