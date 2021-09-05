macro narray(dims::Expr, vars::Expr, T::Symbol=:Float64)
    d = dims.args
    a = vars.args
    m = length(d)
    n = length(a)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        e[i] = Expr(:(=), a[i], Expr(:call, :(Array{$T,$m}), :undef, d...))
    end
    return Expr(:escape, Expr(:block, e...))
end

macro nget(obj::Symbol, vars::Expr)
    a = vars.args
    n = length(a)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        e[i] = :($(a[i]) = $(obj).$(a[i]))
    end
    return Expr(:escape, Expr(:block, e...))
end

macro cpy!(de::Expr, se::Expr)
    n = length(de.args)
    n ≠ length(se.args) && error("@cpy!: length($se) ≠ length($de) = $n.")
    d = de.args
    s = se.args
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        e[i] = Expr(:call, :cpy!, d[i], s[i])
    end
    return Expr(:escape, Expr(:block, e...))
end
