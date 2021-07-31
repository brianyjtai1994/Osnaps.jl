function lupf!(lu::MatIO, p::VecIO{Int}) # @code_warntype ✓
    nrow, ncol = size(lu)
    @inbounds p[end] = ncol
    for kx in 1:ncol # column-wise
        #### Find pivoting row
        kpiv = kx  # kth column pivot
        amax = 0.0 # max. abs. of kth column
        for ix in kx:nrow
            @inbounds temp = abs(lu[ix,kx])
            if temp > amax
                kpiv = ix
                amax = temp
            end
        end
        @inbounds p[kx] = kpiv
        #### Check singularity
        iszero(amax) && error("lup_decomp!: Singular matrix!")
        #### Row-swap
        if kx ≠ kpiv
            for jx in eachindex(1:ncol)
                swap!(lu, kx, jx, kpiv, jx) # interchange
            end
            @inbounds p[end] += 1
        end
        #### Scale the 1st column
        @inbounds lukkinv = inv(lu[kx,kx])
        iszero(lukkinv) && (lukkinv = eps())
        @simd for ix in kx+1:nrow
            @inbounds lu[ix,kx] *= lukkinv
        end
        #### Update the rest block
        @inbounds for jx in kx+1:ncol, ix in kx+1:nrow
            lu[ix,jx] -= lu[ix,kx] * lu[kx,jx]
        end
    end
    return nothing
end

function lups!(x::VecIO, lu::MatI, p::VecI{Int})
    @inbounds for i in eachindex(x)
        swap!(x, i, p[i])
    end
    trsv!('L', 'N', 'U', lu, x)
    trsv!('U', 'N', 'N', lu, x)
    return nothing
end
