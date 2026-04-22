module LabelPropCUDAjl

using CUDA, MAT, SparseArrays, BenchmarkTools

include("kernels/kernel1_basic.jl")
include("kernels/kernel2_grid_stride.jl")
include("kernels/kernel3_warps_shmem.jl")
include("kernels/kernel4_warps_regs.jl")

include("sequential.jl")

include("utils.jl")
include("wrappers.jl")

export exec_kernel1, exec_kernel2, exec_kernel3, exec_kernel4, labelprop_seq!, load_matrix

end
