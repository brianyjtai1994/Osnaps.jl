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
