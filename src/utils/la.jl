function apy2(x::Real, y::Real)
    isnan(x) && return x
    isnan(y) && return y
    # general case
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    iszero(z) && return w
    return w * sqrt(1.0 + abs2(z / w))
end

function unsafe_copy!(des::AbstractArray, src::AbstractArray)
    @simd for i in eachindex(des)
        @inbounds des[i] = src[i]
    end
    return nothing
end

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

function _tr(A::MatI)
    ret = 0.0
    @inbounds for i in axes(A, 1)
        ret += A[i,i]
    end
    return ret
end

"""
    tr(A)

Matrix trace. Sums the diagonal elements of `A`. 
`A` must be a square matrix.
"""
function tr(A::MatI)
    M, N = size(A)
    M == N || error("tr: A should be a square matrix")
    return _tr(A)
end

function _dot(x::VecI, A::MatI, y::VecI)
    ret = 0.0
    for j in eachindex(y)
        @inbounds yj = y[j]
        if !iszero(yj)
            tmp = 0.0
            @simd for i in eachindex(x)
                @inbounds tmp += A[i,j] * x[i]
            end
            ret += tmp * yj
        end
    end
    return ret
end

# x - y → z
function xmy2z!(x::VecI, y::VecI, z::VecIO)
    @simd for i in eachindex(z)
        @inbounds z[i] = x[i] - y[i]
    end
    return nothing
end

"""
    dot(x, A, y)

Compute the dot product `x' * A * y` between two vectors `x` and `y`.
"""
function dot(x::VecI, A::MatI, y::VecI)
    M, N = size(A)
    M == length(x) || error("dot: length(x) ≠ size(A, 1) = $M.")
    N == length(y) || error("dot: length(y) ≠ size(A, 2) = $N.")
    return _dot(x, A, y)
end
