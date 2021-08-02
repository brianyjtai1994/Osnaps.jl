export flp2, clp2

#### flooring `x` to a power-of-2 integer
function flp2(x::Int)
    x = x | (x >>  1)
    x = x | (x >>  2)
    x = x | (x >>  4)
    x = x | (x >>  8)
    x = x | (x >> 16)
    x = x | (x >> 32)
    return x - x >> 1
end

#### ceiling `x` to a power-of-2 integer
function clp2(x::Int)
    x == 0 && return 1
    x == 1 && return 2
    x = x - 1
    x = x | (x >> 1)
    x = x | (x >> 2)
    x = x | (x >> 4)
    x = x | (x >> 8)
    x = x | (x >> 16)
    return x + 1
end

#### Find the exponent of a power-of-2 integer 
function Base.log2(x::Int)
    x == 0 && error("_log2(x): x cannot be zero!")
    r = 0
    x > 4294967295 && (x >>= 32; r += 32)
    x >      65535 && (x >>= 16; r += 16)
    x >        255 && (x >>=  8; r +=  8)
    x >         15 && (x >>=  4; r +=  4)
    x >          3 && (x >>=  2; r +=  2)
    x >          1 && (x >>=  1; r +=  1)
    return r
end

#### Generic logistic function
logistic(x::Real, x0::Real, a::Real, k::Real, c::Real) = a / (1.0 + exp(k * (x0 - x))) + c

#### Perform a single step of Wolford algorithm
function welford_step(μ::Real, s::Real, v::Real, c::Real)
    isone(c) && return v, zero(v)
    s = s * (c - 1)
    m = μ + (v - μ) / c
    s = s + (v - μ) * (v - m)
    μ = m
    return μ, s / (c - 1)
end
