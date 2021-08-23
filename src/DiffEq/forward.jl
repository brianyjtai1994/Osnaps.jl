export Dual, duals

struct Partial{N}
    p::NTuple{N,Float64}

    Partial(p::NTuple{N})          where N     = new{N}(p)
    Partial(n::Val{N}, at::Val{I}) where {N,I} = new{N}(tup_seed(n, at))
end

function tup_add(x::NTuple{N}, y::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(x[$i] + y[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> x[i] + y[i], N)
    end
end

function tup_sub(x::NTuple{N}, y::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(x[$i] - y[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> x[i] - y[i], N)
    end
end

function tup_scal(a::Real, x::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(a * x[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> a * x[i], N)
    end
end

function tup_scal(x::NTuple{N}, a::Real) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(x[$i] / a)
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> x[i] / a, N)
    end
end

function tup_seed(::Val{N}, ::Val{I}) where {N,I}
    if @generated
        e = Vector{Float64}(undef, N)
        @inbounds for i in 1:N
            e[i] = ifelse(i == I, 1.0, 0.0)
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> ifelse(i == I, 1.0, 0.0), N)
    end
end

function tup_minus(x::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(-x[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> -x[i], N)
    end
end

function tup_axpby(a::Real, x::NTuple{N}, b::Real, y::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        @inbounds for i in 1:N
            e[i] = :(a * x[$i] + b * y[$i])
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> a * x[i] + b * y[i], N)
    end
end

axpby(a::Real, x::Partial{N}, b::Real, y::Partial{N}) where N = Partial(tup_axpby(a, x.p, b, y.p))

Base.:-(x::Partial)                           = Partial(tup_minus(x.p))
Base.:+(x::Partial{N}, y::Partial{N}) where N = Partial(tup_add(x.p, y.p))
Base.:-(x::Partial{N}, y::Partial{N}) where N = Partial(tup_sub(x.p, y.p))
Base.:*(a::Real,       x::Partial)            = Partial(tup_scal(a, x.p))
Base.:/(x::Partial,    a::Real)               = Partial(tup_scal(x.p, a))
Base.getindex(x::Partial, i::Int) = getindex(x.p, i)

struct Dual{N} <: Real
    v::Float64; p::Partial{N}

    Dual(v::T, p::Partial{N})         where {T,N}   = new{N}(v, p)
    Dual(v::T, p::NTuple{N})          where {T,N}   = new{N}(v, Partial(p))
    Dual(v::T, n::Val{N}, at::Val{I}) where {T,N,I} = new{N}(v, Partial(n, at))
end

duals(xs::Real...) = duals(xs)

function duals(xs::NTuple{N}) where N
    if @generated
        e = Vector{Expr}(undef, N)
        p = Vector{Float64}(undef, N)
        for j in 1:N
            @simd for i in 1:N
                @inbounds p[i] = ifelse(i == j, 1.0, 0.0)
            end
            @inbounds e[j] = Expr(:call, :Dual, :(xs[$j]), Expr(:tuple, p...))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, e...))
        end
    else
        return ntuple(i -> Dual(xs[i], Val(N), Val(i)), N)
    end
end

Base.:-(x::Dual)                         = Dual(-x.v, -x.p)
Base.:<(x::Dual,    y::Real)             = x.v < y
Base.:<(x::Real,    y::Dual)             = x < y.v
Base.:<(x::Dual{N}, y::Dual{N})  where N = x.v < y.v
Base.:>(x::Dual{N}, y::Dual{N})  where N = x.v > y.v

Base.:<=(x::Dual{N}, y::Dual{N}) where N = x.v <= y.v

Base.:+(a::Real,    x::Dual)            = Dual(a + x.v, x.p)
Base.:+(x::Dual,    a::Real)            = Dual(x.v + a, x.p)
Base.:+(x::Dual{N}, y::Dual{N}) where N = Dual(x.v + y.v, x.p + y.p)

Base.:-(a::Real,    x::Dual)            = Dual(a - x.v, -x.p)
Base.:-(x::Dual,    a::Real)            = Dual(x.v - a,  x.p)
Base.:-(x::Dual{N}, y::Dual{N}) where N = Dual(x.v - y.v, x.p - y.p)

Base.:*(a::Real,    x::Dual)            = Dual(a * x.v, a * x.p)
Base.:*(x::Dual,    a::Real)            = Dual(a * x.v, a * x.p)
Base.:*(x::Dual{N}, y::Dual{N}) where N = (xv = x.v; yv = y.v; return Dual(xv * yv, axpby(yv, x.p, xv, y.p)))

Base.:/(x::Dual{N}, y::Dual{N}) where N = (xv = x.v; yv = y.v; de = abs2(yv); return Dual(xv / yv, axpby(yv / de, x.p, -xv / de, y.p)))

Base.isinf(x::Dual)    = isinf(x.v)
Base.isfinite(x::Dual) = isfinite(x.v)
Base.sin(x::Dual)      = Dual(sin(x.v),  cos(x.v) * x.p)
Base.cos(x::Dual)      = Dual(cos(x.v), -sin(x.v) * x.p)
Base.exp(x::Dual)      = Dual(exp(x.v),  exp(x.v) * x.p)
Base.abs(x::Dual)      = Dual(abs(x.v), sign(x.v) * x.p)
Base.sqrt(x::Dual)     = (v = sqrt(x.v); return Dual(v, x.p / (2.0 * v)))

Base.atan(y::Dual{N}, x::Dual{N}) where N = (xv = x.v; yv = y.v; de = abs2(xv) + abs2(yv); return Dual(atan(yv, xv), axpby(-yv / de, x.p, xv / de, y.p)))

function apy2(x::Dual{N}, y::Dual{N}) where N
    xv = x.v
    yv = y.v
    de = apy2(xv, yv)
    return Dual(de, axpby(xv / de, x.p, yv / de, y.p))
end
