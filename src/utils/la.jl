# export nrm2, swap!

function nrm2(x::VecI{Tx}, y::VecI{Ty}, b::VecB{Tb}) where {Tx<:Real,Ty<:Real,Tb<:Real} # @code_warntype ✓
    @simd for i in eachindex(b)
        @inbounds b[i] = abs2(x[i] - y[i])
    end
    return sqrt(sum(b))
end

function swap!(v::VecIO, i::Int, j::Int) # @code_warntype ✓
    @inbounds temp = v[i]
    @inbounds v[i] = v[j]
    @inbounds v[j] = temp
    return nothing
end

function swap!(m::MatI, i1::Int, j1::Int, i2::Int, j2::Int) # @code_warntype ✓
    @inbounds temp     = m[i1,j1]
    @inbounds m[i1,j1] = m[i2,j2]
    @inbounds m[i2,j2] = temp
    return nothing
end
