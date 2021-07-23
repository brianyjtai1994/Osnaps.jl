struct BoxBound a::Float64; b::Float64; i::Int end # a * x[i] + b

resolve_lb(lb::Real) = iszero(lb) ? (-1.0, 0.0) : (-abs(inv(lb)),  1.0 * sign(lb)) # @code_warntype ✓
resolve_ub(ub::Real) = iszero(ub) ? ( 1.0, 0.0) : ( abs(inv(ub)), -1.0 * sign(ub)) # @code_warntype ✓

function boxbounds(lb::NTuple{ND}, ub::NTuple{ND}) where ND
    if @generated
        a = Vector{Expr}(undef, 2*ND)
        @inbounds for i in 1:ND
            a[i]    = :(BoxBound(resolve_lb(lb[$i])..., $i))
            a[i+ND] = :(BoxBound(resolve_ub(ub[$i])..., $i))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:tuple, a...))
        end
    else
        return ntuple(i -> i > ND ? BoxBound(resolve_ub(ub[i - ND])..., i - ND) : BoxBound(resolve_lb(lb[i])..., i), 2*ND)
    end
end

eval_violation(x::VecI, bb::BoxBound) = max(0.0, bb.a * x[bb.i] + bb.b) # @code_warntype ✓
function eval_violation(x::VecI, cons::NTuple{NB,BoxBound}) where NB    # @code_warntype ✓
    if @generated
        a = Vector{Expr}(undef, NB)
        @inbounds for i in eachindex(a)
            a[i] = :(eval_violation(x, cons[$i]))
        end
        return quote
            $(Expr(:meta, :inline))
            @inbounds return $(Expr(:call, :+, a...))
        end
    else
        ret = 0.0
        @inbounds for i in eachindex(cons)
            ret += eval_violation(x, cons[i])
        end
        return ret
    end
end

# check feasibility of agent in throng, @code_warntype ✓
function check!(xnew::VecI, agents::VecIO{Agent}, elites::VecIO{Agent}, throng::VecIO{Agent}, edx::Int, tdx::Int, fn::Function, cons::NTuple)
    violation = eval_violation(xnew, cons)
    violation > 0.0 && return @inbounds check!(xnew, violation, throng[tdx]) # x[new] is infeasible
    return check!(xnew, fcall(fn, xnew), agents, elites, throng, edx, tdx)   # x[new] is feasible
end

# Matchup for a feasible x[new] agent in throng, @code_warntype ✓
function check!(xnew::VecI, fnew::Real, agents::VecIO{Agent}, elites::VecIO{Agent}, throng::VecIO{Agent}, edx::Int, tdx::Int)
    @inbounds xold = throng[tdx]
    # x[old] is infeasible
    if !xold.v
        xold.f = fnew
        xold.v = true
        xold.c = 0.0
        copy!(xold.x, xnew)
        return nothing
    end
    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    if @inbounds !(elites[edx].f < fnew) 
        xold.f = fnew
        copy!(xold.x, xnew)
        swap!(agents, edx, length(elites) + tdx)
        return nothing
    end
    # x[old], x[new] are feasible
    if !(xold.f < fnew) 
        xold.f = fnew
        copy!(xold.x, xnew)
        return nothing
    end
end

# check feasibility of agent in elites, @code_warntype ✓
function check!(xnew::VecI, elites::VecIO{Agent}, edx::Int, fn::Function, cons::NTuple)
    violation = eval_violation(xnew, cons)
    violation > 0.0 && return @inbounds check!(xnew, violation, elites[edx]) # x[new] is infeasible
    return check!(xnew, fcall(f, xnew), elites, edx)                         # x[new] is feasible
end

# Matchup for a feasible x[new] trial in elites, @code_warntype ✓
function check!(xnew::VecI, fnew::Real, elites::VecIO{Agent}, edx::Int)
    @inbounds elite = elites[edx]
    # x[old] is infeasible
    if !elite.v
        elite.f = fnew
        elite.v = true
        elite.c = 0.0
        copy!(elite.x, xnew)
        return nothing
    end
    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    if @inbounds !(elites[1].f < fnew)
        elite.f = fnew
        copy!(elite.x, xnew)
        swap!(elites, 1, edx)
        return nothing
    end
    # x[old], x[new] are feasible
    if !(elite.f < fnew) 
        elite.f = fnew
        copy!(elite.x, xnew)
        return nothing
    end
end

# Matchup for an infeasible x[new] trial, here "fnew = violation", @code_warntype ✓
function check!(xnew::VecI, violation::Real, agent::Agent)
    # x[old], x[new] are infeasible, compare violation
    # There is no `else` condition, if x[old] is feasible, then a matchup is unnecessary.
    if !agent.v && !(agent.c < violation)
        agent.c = violation
        copy!(agent.x, xnew)
    end
    return nothing
end
