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

nrm2(x::VecI{Tx}) where Tx<:Real = sqrt(dot(x, x))

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

function checksquare(A::MatI)
    m, n = size(A)
    m == n || error("tr: A should be a square matrix")
    return n
end

function diagmax(A::MatI, n::Int)
    r = -Inf
    @inbounds for i in eachindex(1:n)
        r = max(r, A[i,i])
    end
    return r
end

diagmax(A::MatI) = (n = checksquare(A); return diagmax(A, n))

function tr(A::MatI, n::Int)
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
tr(A::MatI) = (n = checksquare(A); return tr(A, n))

function dot(x::VecI, y::VecI, n::Int)
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

function dot(x::VecI, m::Int, A::MatI, y::VecI, n::Int)
    ret = 0.0
    for j in eachindex(1:n)
        @inbounds yj = y[j]
        if !iszero(yj)
            tmp = 0.0
            @inbounds for i in eachindex(1:m)
                tmp += A[i,j] * x[i]
            end
            ret += tmp * yj
        end
    end
    return ret
end

"""
    dot(x, y)
Compute the dot product `x' * y` between two vectors `x` and `y`.
"""
function dot(x::VecI, y::VecI)
    n = length(x)
    n == length(y) || error("dot: length(y) ≠ length(x) = $n.")
    return dot(x, y, n)
end

"""
    dot(x, A, y)
Compute the dot product `x' * A * y` between two vectors `x` and `y`.
"""
function dot(x::VecI, A::MatI, y::VecI)
    m, n = size(A)
    m == length(x) || error("dot: length(x) ≠ size(A, 1) = $m.")
    n == length(y) || error("dot: length(y) ≠ size(A, 2) = $n.")
    return dot(x, m, A, y, n)
end

function logdet(A::MatI, n::Int)
    r = 0.0
    @inbounds for i in eachindex(1:n)
        r += log(abs(A[i,i]))
    end
    return r
end

"""
    logdet(A)
Compute the log-determinant `log(det(A))` where `A` is a triangular matrix.
"""
logdet(A::MatI) = (n = checksquare(A); return logdet(A, n))
