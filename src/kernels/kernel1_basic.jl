###########################################################
#CASE 1
function labelprop_case1_kernel!(labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter)
  idx = (blockIdx().x-1) * blockDim().x + threadIdx().x #thread indexing

  if idx <=n && active_curr[idx]#until covering all (active) nodes with a thread
    curr_label = labels[idx]
    new_label = curr_label

    degree = row_ptr[idx+1]- row_ptr[idx]
    CUDA.@atomic active_counter[1]+= degree

    #neighbouring nodes access
    for i in row_ptr[idx]:(row_ptr[idx+1]-1)
      neighbour = col_idx[i]
      new_label = min(new_label, labels[neighbour])
    end
    #label atomic update
    if new_label < curr_label
      CUDA.atomic_min!(pointer(labels, idx), new_label)
      if changed[1]==0 #updating 'changed' variable, atomic oper.
        CUDA.atomic_xchg!(pointer(changed,1), Int32(1))
      end
      #update activation mask
      for i in row_ptr[idx]:(row_ptr[idx+1]-1)
        active_next[col_idx[i]]= true
      end
    end
  end

  return nothing
end


