#######################################################
# CASE 2

function labelprop_case2_kernel!(labels, row_ptr, col_idx, n, changed)

  index = (blockIdx().x-1) * blockDim().x + threadIdx().x #thread indexing
  incr = gridDim().x * blockDim().x #define incrementation 'step'

  #utilize enough threads for covering 'chunks'/teams of nodes
  for idx in index:incr:n
    curr_label = labels[idx]
    new_label = curr_label
    #work on neighbors of each node
    for i in row_ptr[idx]:(row_ptr[idx+1]-1)
      neighbour = col_idx[i]
      new_label = min(new_label, labels[neighbour])
    end

    if new_label < curr_label
      CUDA.atomic_min!(pointer(labels, idx), new_label)
      if changed[1]==0
        CUDA.atomic_xchg!(pointer(changed,1), Int32(1))
      end
    end
  end

  return nothing
end
