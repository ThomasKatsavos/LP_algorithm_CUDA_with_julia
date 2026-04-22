using MAT
using SparseArrays

function load_matrix(name)
    path = joinpath("graphs", name)
    
    if !isfile(path)
        run(`wget -P graphs/ https://suitesparse-collection-website.herokuapp.com/$name`)
    end
    
    file = matopen(path)
    mat_csc = read(file, "Problem")["A"] #load from disk in csc format
    mat_csr = SparseMatrixCSC(mat_csc')#convert to csr

    n = length(mat_csr.colptr)-1 #number of non-zeros

    #CUDA Arrays for row pointers and columns indices vectors
    row_ptr = mat_csr.colptr
    col_idx = mat_csr.rowval

    m = length(col_idx)

    #clear no-more useful elements
    mat_csc = nothing
    mat_csr = nothing
    close(file)

    return n, m, row_ptr, col_idx 
end
