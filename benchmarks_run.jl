using BenchmarkTools
using CUDA
using Printf
using MAT
using SparseArrays

#include("src/utils.jl")
include("src/CudaProject.jl")

using .LabelPropCUDAjl

#Configuration
const MAT_FILE = "SNAP/com-Orkut.mat"
const MAX_BW = 320 # Change this based on your specific GPU (for Tesla T4 it is 320 GB/s)

function run_full_suite()
    println("="^50)
    println("  LABEL PROPAGATION BENCHMARK ")
    println("="^50)

    #Data Loading
    println("Loading matrix: $MAT_FILE...")
    n, m, h_row_ptr, h_col_idx = load_matrix(MAT_FILE)
    

    # Move graph data to GPU
    d_row_ptr = CuArray(h_row_ptr)
    d_col_idx = CuArray(h_col_idx)

    println("Graph Stats: Nodes = $n, Edges = $m")
    println("-"^50)

    #Sequential Baseline
    println("Running Sequential Baseline...")
    h_labels = collect(Int32, 1:n)
    seq_time = @elapsed iters_seq = labelprop_seq!(h_labels, h_row_ptr, h_col_idx, n)
    println("Sequential: $iters_seq iterations in $(round(seq_time, digits=3))s")
    println("-"^50)

    # GPU Benchmarking
    #We'll store results in a dictionary for easy printing
    results = []

    # Case 1
    println("Executing Case 1 (Basic Active Mask)...")
    it1, t1, sccs1, edges1 = exec_kernel1(MAT_FILE, n, m, d_row_ptr, d_col_idx)
    push!(results, ("Case 1", it1, t1, sccs1, edges1))

    # Case 2
    println("Executing Case 2 (Grid-Stride Loop)...")
    it2, t2, sccs2 = exec_kernel2(MAT_FILE, n, m, d_row_ptr, d_col_idx)
    push!(results, ("Case 2", it2, t2, sccs2, m * it2)) # Assuming m edges per iter

    # Case 3
    println("Executing Case 3 (Shared Memory)...")
    it3, t3, sccs3 = exec_kernel3(MAT_FILE, n, m, d_row_ptr, d_col_idx)
    push!(results, ("Case 3", it3, t3, sccs3, m * it3))

    # Case 4
    println("Executing Case 4 (Warp Shuffle)...")
    it4, t4, sccs4, edges4 = exec_kernel4(MAT_FILE, n, m, d_row_ptr, d_col_idx)
    push!(results, ("Case 4", it4, t4, sccs4, edges4))

    #Print Comparison Table
    println("\n" * "="^80)
    @printf("%-10s | %-6s | %-8s | %-6s | %-8s\n", 
	    "Kernel", "Iters", "Time(s)", "SCCs", "GTEPS")
    println("-"^80)

    for (name, it, t, sccs, edges) in results
        #Calculations
        gteps = edges / (t * 1e9)
        

        @printf("%-10s | %-6d | %-8.4f | %-6d | %-8.3f\n", 
                name, it, t, sccs, gteps)
    end
    println("="^80)
end


run_full_suite()
