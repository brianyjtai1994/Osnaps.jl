macro def(genre::Symbol, ex::Union{Expr,Symbol}, vars::Symbol...)
    op = genre == :prop ? :(::) : genre == :vars ? :(=) :
         error("@ndef(genre = $genre, ...) is invalid.")
    n = length(vars)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        e[i] = Expr(op, vars[i], ex)
    end
    return Expr(:escape, Expr(:block, e...))
end

macro get(obj::Symbol, vars::Symbol...)
    n = length(vars)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        vari = vars[i]
        e[i] = :($vari = $obj.$vari)
    end
    return Expr(:escape, Expr(:block, e...))
end

macro copy!(ex::Expr...)
    n = length(ex)
    e = Vector{Expr}(undef, n)
    @inbounds for i in 1:n
        ei = ex[i]
        ai = ei.args
        if ai[1] == :(=>)
            ai[1] = :unsafe_copy!
            ai[2], ai[3] = ai[3], ai[2]
        else
            error("$ei has invalid symbol.")
        end
        e[i] = ei
    end
    return Expr(:escape, Expr(:block, e...))
end
