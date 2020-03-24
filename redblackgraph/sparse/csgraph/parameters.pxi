
DTYPE = np.int32
ctypedef np.int32_t DTYPE_t

UDTYPE = np.uint32
ctypedef np.uint32_t UDTYPE_t

ITYPE = np.int32
ctypedef np.int32_t ITYPE_t

# Fused type for int32 and int64
ctypedef fused int32_or_int64:
    np.int32_t
    np.int64_t

# EPS is the precision of DTYPE
DEF DTYPE_EPS = 0

# NULL_IDX is the index used in predecessor matrices to store a non-path
DEF NULL_IDX = -9999
