include("./agent.jl")
include("./constraint.jl")
include("./optimizer.jl")

struct GlobalMinimizer <: AbstractOptimizer
    xsol::Vector{Float64}
    xerr::Vector{Float64}
    buff::Vector{Float64}
    fork::Vector{Int64}
    pool::Vector{Agent}
    NP::Int
    NE::Int

    function GlobalMinimizer(ND::Int, NP::Int, NE::Int)
        xsol = Vector{Float64}(undef, ND)
        xerr = Vector{Float64}(undef, ND)
        buff = Vector{Float64}(undef, ND)
        fork = Vector{Int}(undef, NE)
        pool = return_agents(ND, NP)
        return new(xsol, xerr, buff, fork, pool, NP, NE)
    end
end

# @code_warntype ✓
function inits!(agents::VecIO{Agent}, lb::NTuple, ub::NTuple)
    for agent in agents
        born!(agent.x, lb, ub)
    end
end

# @code_warntype ✓
function inits!(agents::VecIO{Agent}, f::Function, cons::NTuple)
    fmax = -Inf
    for agent in agents
        violation = eval_violation(agent.x, cons)
        violation > 0.0 && (agent.c = violation; continue) # agent is infeasible
        agent.v = true
        agent.f = fcall(f, agent.x)
        fmax    = max(fmax, agent.f)
    end
    for agent in agents
        !agent.v && (agent.f = agent.c + fmax) # agent is infeasible
    end
end

# @code_warntype ✓
function group!(fork::VecIO{Int}, agents::VecI{Agent}, NE::Int, NC::Int)
    diversity = 0.0
    @inbounds for i in eachindex(fork)
        diversity += agents[NE + 1].f - agents[i].f
    end
    if iszero(diversity) || isnan(diversity)
        fill!(fork, 1)
    else
        @inbounds for i in eachindex(fork)
            fork[i] = max(1, round(Int, NC * (agents[NE + 1].f - agents[i].f) / diversity))
        end
    end
    res = NC - sum(fork) # residue
    idx = 2
    while res > 0
        @inbounds fork[idx] += 1; res -= 1
        idx < NE ? idx += 1 : idx = 2
    end
    while res < 0
        @inbounds fork[idx] = max(1, fork[idx] - 1); res += 1
        idx < NE ? idx += 1 : idx = 2
    end
end

# @code_warntype ✓
function minimize!(o::GlobalMinimizer, fn::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}, itmax::Int, dmax::Real, avgtimes::Int) where {ND,T<:Real}
    NP = o.NP
    NE = o.NE
    NC = NP - NE

    cons = boxbounds(lb, ub)
    xsol = o.xsol
    xerr = o.xerr
    buff = o.buff
    fork = o.fork

    agents = o.pool
    elites = return_elites(agents, NE)
    throng = return_throng(agents, NE, NP)

    generation = 0
    while generation < avgtimes
        generation += 1
        itcount     = 0

        inits!(agents, lb, ub)
        inits!(agents, fn, cons)
        binsort!(agents)
        @inbounds while itcount < itmax
            itcount += 1
            ss = logistic(itcount, 0.5 * itmax, -0.618, 20.0 / itmax, 2.0)
            group!(fork, agents, NE, NC)
            #### Moves throng/elites to elites/the best
            rx = 1
            fx = fork[rx]
            # move agents in throng toward elites
            for ix in eachindex(throng)
                sco_move!(buff, elites[rx].x, throng[ix].x, ss)
                check!(buff, agents, elites, throng, rx, ix, fn, cons)
                fx -= 1
                iszero(fx) && (rx += 1; fx = fork[rx])
            end
            # move agents in elites and find the best one
            for rx in 2:NE
                sco_move!(buff, elites[1].x, elites[rx].x, ss)
                check!(buff, elites, rx, fn, cons)
            end
            #### Random searching process
            for ix in 1:fork[1]
                if !(dmax < nrm2(agents[1].x, throng[ix].x, buff))
                    wca_move!(buff, agents[1].x)
                    check!(buff, agents, elites, throng, 1, ix, fn, cons)
                end
            end
            for rx in 2:NE
                if !(dmax < nrm2(agents[1].x, elites[rx].x, buff)) || !(0.1 < rand())
                    born!(buff, lb, ub)
                    check!(buff, elites, rx, fn, cons)
                end
            end
            #### Update the function-value of infeasible candidates
            fmax = -Inf
            for agent in agents
                agent.v && (fmax = max(fmax, agent.f))
            end
            for agent in agents
                !agent.v && (agent.f = agent.c + fmax)
            end

            binsort!(agents)
            dmax -= dmax / itmax
        end
        @inbounds xnew = agents[1].x
        @inbounds for i in eachindex(xsol)
            xsol[i], xerr[i] = welford_step(xsol[i], xerr[i], xnew[i], generation)
        end
    end
end
