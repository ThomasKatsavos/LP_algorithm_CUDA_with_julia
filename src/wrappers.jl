#######################################################
# Function for running CASE 1 method
function exec_kernel1(matname, n, m, row_ptr, col_idx)
  labels = CuArray{Int32}(1:n)
  changed = CuArray(Int32[0])

  active_curr = CUDA.ones(Bool,n)
  active_next = CUDA.zeros(Bool,n)

  active_counter = CuArray(Int64[0])
  edges_accessed = 0

  println("Starting compilation...")

  kernel = @cuda launch=false labelprop_case1_kernel!(labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter)

  config = launch_configuration(kernel.fun)
  threads = config.threads
  blocks = cld(n, threads)

  println("Running LP/CC on $matname on Tesla T4 GPU...")

  iterations =0
  time_dur = CUDA.@elapsed begin

    while true
      CUDA.fill!(changed,0)
      CUDA.fill!(active_next,false)

      kernel(labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter; threads=threads, blocks=blocks)

     iterations +=1

      if CUDA.@allowscalar(changed[1]) == 0
        break
      end

      active_curr,active_next = active_next, active_curr
    end
  end
  #actual number of traversed edges
  edges_accessed = CUDA.@allowscalar(active_counter[1])

  println("Case 1 Converged!")
  sccs = length(unique(Array(labels)))
  println("Converged after $iterations iterations, time needed: $time_dur seconds. Graph $matname has $sccs SCCs")

  bytes_tot = (
   n * iterations * sizeof(Bool) +
   edges_accessed* sizeof(eltype(col_idx)) +
   edges_accessed* sizeof(eltype(labels)) +  #neighbour search
   n*iterations* sizeof(eltype(labels))  #present node's read-write
  )

  bytes_p_iter = bytes_tot/iterations

  return iterations, time_dur, bytes_p_iter, edges_accessed

end

#######################################################
# Function for running CASE 2 method


function exec_kernel2(matname, n, m, row_ptr, col_idx)
  labels = CuArray{Int32}(1:n)
  changed = CuArray(Int32[0])

  println("Starting compilation...")

  kernel = @cuda launch=false labelprop_case2_kernel!(labels, row_ptr, col_idx, n, changed)

  config = launch_configuration(kernel.fun)
  threads = config.threads
  blocks = cld(n, threads)

  println("Running LP/CC on $matname on Tesla T4 GPU...")

  iterations =0
  time_dur = CUDA.@elapsed begin

    while true
      CUDA.fill!(changed,0)
      kernel(labels, row_ptr, col_idx, n, changed, threads=threads, blocks=blocks)

      iterations +=1
      if CUDA.@allowscalar(changed[1]) == 0
        break
      end
    end
  end

  println("Case 2 Converged!")
  sccs = length(unique(Array(labels)))
  println("Converged after $iterations iterations, time needed: $time_dur seconds. Graph $matname has $sccs SCCs")

  bytes_p_iter = (
   (n+1)* sizeof(eltype(row_ptr)) +
   m* sizeof(eltype(col_idx)) +
   m* sizeof(eltype(labels)) +  #neighbour search
   n*2* sizeof(eltype(labels))  #present node's read-write
  )

  return iterations, time_dur, bytes_p_iter

end

#######################################################
# Function for running CASE 3 method

function exec_kernel3(matname, n, m, row_ptr, col_idx)
  labels = CuArray{Int32}(1:n)
  changed = CuArray(Int32[0])

  println("Starting compilation...")

  threads_per_block = 256
  warps_per_block = div(threads_per_block, 32)
  num_blocks = div(n + warps_per_block - 1, warps_per_block)
  shmem_size = threads_per_block * sizeof(Int32)

  kernel = @cuda launch=false labelprop_case3_kernel!(labels, row_ptr, col_idx, n, changed)


  println("Running LP/CC on $matname on Tesla T4 GPU...")

  iterations =0
  time_dur = CUDA.@elapsed begin

   while true
     CUDA.fill!(changed,0)

     kernel(
      labels, row_ptr, col_idx, n, changed;
      threads=threads_per_block,
      blocks=num_blocks,
      shmem=shmem_size
     )

     iterations +=1

     if CUDA.@allowscalar(changed[1]) == 0
        break
      end
    end
  end

  println("Case 3 Converged!")
  sccs = length(unique(Array(labels)))
  println("Converged after $iterations iterations, time needed: $time_dur seconds. Graph $matname has $sccs SCCs")

  bytes_p_iter = (
    (n+1)* sizeof(eltype(row_ptr)) +
    m* sizeof(eltype(col_idx)) +
    m* sizeof(eltype(labels)) +  #neighbour search
    n*2* sizeof(eltype(labels))  #present node's read-write
  )

  return iterations, time_dur, bytes_p_iter

end

#######################################################
# Function for running CASE 4 method

function exec_kernel4(matname, n, m, row_ptr, col_idx)

  # CUDA Arrays fro labels vector and 'changed' flag variable
  labels = CuArray{Int32}(1:n)
  changed = CuArray(Int32[0])

  active_counter = CuArray(Int64[0])
  edges_accessed = 0

  active_curr = CUDA.ones(Bool, n)
  active_next = CUDA.zeros(Bool, n)

  println("Starting compilation...")

  threads_per_block = 256
  warps_per_block = div(threads_per_block, 32)
  num_blocks = div(n + warps_per_block - 1, warps_per_block)


  kernel = @cuda launch=false labelprop_case4_kernel!(labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter)
  println("Running LP/CC on $matname on Tesla T4 GPU...")

  iterations =0
  time_dur = CUDA.@elapsed begin

    while true
      CUDA.fill!(changed,0)
      CUDA.fill!(active_next, false)

      kernel(
    labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter;
    threads=threads_per_block,
    blocks=num_blocks
   )

      iterations +=1

      if CUDA.@allowscalar(changed[1]) == 0
        break
      end

      active_curr, active_next = active_next, active_curr
   end
  end

  edges_accessed = CUDA.@allowscalar(active_counter[1])

  println("Case 4 Converged!")
  sccs = length(unique(Array(labels)))
  println("Converged after $iterations iterations, time needed: $time_dur seconds. Graph $matname has $sccs SCCs")

  bytes_tot = (
   n * iterations * sizeof(Bool) +
   edges_accessed* sizeof(eltype(col_idx)) +
   edges_accessed* sizeof(eltype(labels)) +  #neighbour search
   n*iterations* sizeof(eltype(labels))  #present node's read-write
  )

  bytes_p_iter = bytes_tot/iterations

  return iterations, time_dur, bytes_p_iter, edges_accessed
end
