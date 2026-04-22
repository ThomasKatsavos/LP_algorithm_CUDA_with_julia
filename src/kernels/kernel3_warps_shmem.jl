#########################################################
#CASE 3

function labelprop_case3_kernel!(labels, row_ptr, col_idx, n, changed)
    #Shared memory allocation-the size is the number of threads inside a block
    shmem = CuDynamicSharedArray(Int32, blockDim().x)

    threads_per_warp = 32 #define thread-inside-warp count

    #Warp ID definition - div(x,y) -> returns integer part of x/y
    warp_id = div((blockIdx().x - 1) * blockDim().x + threadIdx().x - 1, threads_per_warp) + 1

    #Lane ID - thread identificiation inside a warp
    lane_id = (threadIdx().x - 1) % threads_per_warp + 1

    #Position inside the whole block
    tid = threadIdx().x

    #We use warps until we have 'covered' all nodes
    if warp_id <= n
        row_start = row_ptr[warp_id]
        row_end = row_ptr[warp_id + 1] - 1

        curr_label = labels[warp_id]
        min_label = curr_label  #this is the minimum of each THREAD, a local minimum

        # Every thread inside a warp reads some neighboring vertices/nodes.
        for i in (row_start + lane_id - 1):threads_per_warp:row_end
            neighbor_label = labels[col_idx[i]]
            if neighbor_label < min_label
                min_label = neighbor_label
            end
        end


        # We save every thread-local minimum label inside shared memory
        shmem[tid] = min_label

        #synchronize warps
        CUDA.sync_warp()

        # Compare minimums inside shared memory
        #We start comparing 1-17, 2-18, ...
        #...then we compare the new minimums the same way until one remains!
        offset = 16
        while offset > 0
            if lane_id <= offset
                remote_val = shmem[tid + offset]
                if remote_val < shmem[tid]
                    shmem[tid] = remote_val
                end
            end

            # Syncronize in every iteration for avoiding race conditions
            CUDA.sync_warp()
            offset = div(offset, 2)
        end

        #Thread 1 of warp now has the best label
        if lane_id == 1
            final_min = shmem[tid]
            if final_min < curr_label#update min
                CUDA.atomic_min!(pointer(labels, warp_id), final_min)
                CUDA.atomic_xchg!(pointer(changed, 1), Int32(1))
            end
        end
    end
    return nothing
end

