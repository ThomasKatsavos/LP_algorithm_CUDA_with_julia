function labelprop_seq!(labels, row_ptr, col_idx, n)
    iters = 0
    chgd = true
    while chgd
        iters += 1
        chgd = false
        for v in 1:n
            best = labels[v]
            for e in row_ptr[v]:(row_ptr[v+1]-1)
                u = col_idx[e]
                lu = labels[u]
                if lu < best
                    best = lu
                end
            end
            if best < labels[v]
                labels[v] = best
                chgd = true
            end
        end
    end
    return iters
end
