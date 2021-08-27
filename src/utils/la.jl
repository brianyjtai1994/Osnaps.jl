cubic(x::Real) = x * x * x

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

function sqr2(x::Real, y::Real)
    isnan(x) && return x
    isnan(y) && return y
    # general case
    xabs = abs(x)
    yabs = abs(y)
    w = max(xabs, yabs)
    z = min(xabs, yabs)
    iszero(z) && return abs2(w)
    return abs2(w) * (1.0 + abs2(z / w))
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

function _checksquare(A::MatI)
    m, n = size(A)
    m == n || error("tr: A should be a square matrix")
    return n
end

function _diagmax(A::MatI, n::Int)
    r = -Inf
    @inbounds for i in eachindex(1:n)
        r = max(r, A[i,i])
    end
    return r
end

diagmax(A::MatI) = (n = _checksquare(A); return _diagmax(A, n))

function _tr(A::MatI, n::Int)
    ret = 0.0
    @inbounds for i in eachindex(1:n)
        ret += A[i,i]
    end
    return ret
end

"""
    tr(A)

Matrix trace. Sums the diagonal elements of `A`. 
`A` must be a square matrix.
"""
tr(A::MatI) = (n = _checksquare(A); return _tr(A, n))

function _dot(x::VecI, y::VecI, n::Int)
    r = 0.0
    m = mod(n, 5)
    if m ≠ 0
        for i in 1:m
            r += x[i] * y[i]
        end
        n < 5 && return r
    end
    m += 1
    @inbounds for i in m:5:n
        r += x[i] * y[i] + x[i+1] * y[i+1] + x[i+2] * y[i+2] + x[i+3] * y[i+3] + x[i+4] * y[i+4]
    end
    return r
end

function _dot(x::VecI, A::MatI, y::VecI)
    ret = 0.0
    for j in eachindex(y)
        @inbounds yj = y[j]
        if !iszero(yj)
            tmp = 0.0
            @inbounds for i in eachindex(x)
                tmp += A[i,j] * x[i]
            end
            ret += tmp * yj
        end
    end
    return ret
end

function dot(x::VecI, y::VecI)
    n = length(x)
    n == length(y) || error("dot: length(y) ≠ length(x) = $n.")
    return _dot(x, y, n)
end

"""
    dot(x, A, y)

Compute the dot product `x' * A * y` between two vectors `x` and `y`.
"""
function dot(x::VecI, A::MatI, y::VecI)
    m, n = size(A)
    m == length(x) || error("dot: length(x) ≠ size(A, 1) = $m.")
    n == length(y) || error("dot: length(y) ≠ size(A, 2) = $n.")
    return _dot(x, A, y)
end

function logdet(A::MatI)
    n = _checksquare(A)
    r = 0.0
    @inbounds for i in eachindex(1:n)
        r += log(abs(A[i,i]))
    end
    return r
end
