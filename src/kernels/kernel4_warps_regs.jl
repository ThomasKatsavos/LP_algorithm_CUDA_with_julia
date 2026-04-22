############################################################################
#CASE 4 (EXTRA)

function labelprop_case4_kernel!(labels, row_ptr, col_idx, n, changed, active_curr, active_next, active_counter)
    threads_per_warp = 32

    #Position inside the whole block
    tid = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    #Lane ID - thread identificiation inside a warp
    lane_id = (threadIdx().x - 1) % threads_per_warp + 1
    #Warp ID definition - div(x,y) -> returns integer part of x/y
    warp_id = div(tid - 1, threads_per_warp) + 1

    #activation flag initially false
    node_is_active = false

    if warp_id <= n
        if lane_id == 1
            node_is_active = active_curr[warp_id]
        end
        node_is_active = CUDA.shfl_sync(0xffffffff, node_is_active, 1)

        if node_is_active
            row_start = row_ptr[warp_id]
            row_end_ptr = row_ptr[warp_id + 1]
            row_end = row_end_ptr - 1
            #logic for counting total traversed edges-the active ones for each iter.
            if lane_id == 1
                degree = Int64(row_end_ptr - row_start)
                CUDA.atomic_add!(pointer(active_counter, 1), degree)
            end

            curr_label = labels[warp_id]
            min_label = curr_label

            for i in (row_start + lane_id - 1):threads_per_warp:row_end
                neighbor_idx = col_idx[i]
                if neighbor_idx >= 1 && neighbor_idx <= n
                    neighbor_label = labels[neighbor_idx]
                    if neighbor_label < min_label
                        min_label = neighbor_label
                    end
                end
            end

            # Compare minimums - Warp-Shuffle function
            mask = 0xffffffff
            offset = 16
            while offset > 0
                min_gen = CUDA.shfl_down_sync(mask, min_label, offset)
                if min_gen < min_label
                    min_label = min_gen
                end
                offset = div(offset, 2)
            end
            #Broadcast new min. label
            min_label = CUDA.shfl_sync(mask, min_label, 1)

            if min_label < curr_label
                if lane_id == 1
                    CUDA.atomic_min!(pointer(labels, warp_id), min_label)
                    CUDA.atomic_xchg!(pointer(changed, 1), Int32(1))
                end
                #update activation mask
                for i in (row_start + lane_id - 1):threads_per_warp:row_end
                    neighbor_idx = col_idx[i]
                    if neighbor_idx >= 1 && neighbor_idx <= n
                        active_next[neighbor_idx] = true
                    end
                end
            end
        end
    end
    return nothing
end
