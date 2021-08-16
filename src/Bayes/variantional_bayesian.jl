# compute: Λ(n+1) ← Λ(0) + J(n)' * Λy * J(n)
function varbayes_update!(Λn::MatIO, Λ0::MatI, Λy::MatI, Jn::MatI, An::MatI)
    symm!('L', 'U', 1.0, Λy, Jn, 0.0, An)
    gemm!('T', 'N', 1.0, An, Jn, 1.0, Λn)
    return nothing
end

# compute: Δ(n) ← A(n)' * k(n) - Λ(0) * [m(n) - m(0)]
function varbayes_lmstep!(Δn::VecB, Λ0::MatI, mb::VecB, An::MatB, kn::VecI)
    symv!('U', 1.0, Λ0, mb,  0.0, Δn) # Δ(n) ← Λ(0) * [m(n) - m(0)]
    gemv!('T', 1.0, An, kn, -1.0, Δn) # Δ(n) ← A(n) * k(n) - Δ(n)
    return nothing
end

# compute: m(n+1) ← m(n) + Δ(n) \ [Λ(n+1) + αI]
function varbayes_lmstep!(mn::VecIO, Λf::MatB, α::Real, Δn::VecB)
    @simd for i in eachindex(mn)
        @inbounds Λf[i] += α
    end
    _, cholesky_state = potrf!('L', Λf)
    trsv!('L', 'N', 'N', Λf, Δn)
    trsv!('L', 'T', 'N', Λf, Δn)
    axpy!(1.0, Δn, mn)
    return cholesky_state
end

# compute lower bound
function varbayes_lowbnd!(Λy::MatI, kn::VecI, Λ0::MatI, mb::VecB, Λf::MatB, Λe::MatB)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    ret = 1.0
    @inbounds for i in eachindex(mb)
        ret *= abs2(Λf[i,i])
    end
    ret += dot(kn, Λy, kn) + dot(mb, Λ0, mb) + tr(Λe)
    return ret, cholesky_state
end

# Unfinished version
function varbayes(f, g, m0::VecI, Λ0::MatI, xs::VecI, ys::VecI; σ::Real=1.0, itmax::Int=50)
    M  = length(ps)
    N  = length(xs)
    #### Allocation of vectors
    ms = similar(m0)       # solution
    mn = similar(m0)       # trial of solution
    mb = similar(m0)       # buffer for computing evidence
    Δn = similar(m0)       # buffer for LM step
    kn = similar(ys)       # deviation from model
    #### Allocation of matrices
    Λs = similar(Λ0)       # solution
    Λn = similar(Λ0)       # trail of solution
    Λf = similar(Λ0)       # buffer to be factorized
    Λe = similar(Λ0)       # buffer for computing evidence
    Λb = similar(Λ0)       # buffer for backup
    Jn = similar(ys, N, M) # Jacobian matrix
    An = similar(ys, N, M) # buffer for Λy * Jn
    #### Initialization
    unsafe_copy!(mn, m0)
    unsafe_copy!(Λn, Λ0)
    LowBnd_best = Inf      # evidence of solution
    cholesky_state = 0.0   # monitor if Λn is positive definite
    isfound = 0
    it = 0
    while it < itmax
        it += 1
        # compute deviation
        fcall!(kn, f,  mn)
        gcall!(Jn, g,  mn)
        xmy2z!(ys, kn, kn)
        # update Λn
        unsafe_copy!(Λb, Λn)                 # Λb     ← Λ(n), copy
        unsafe_copy!(Λf, Λn)                 # Λf     ← Λ(n), copy
        unsafe_copy!(Λn, Λ0)                 # Λ(n)   ← Λ(0), copy
        varbayes_update!(Λn, Λ0, Λy, Jn, An) # Λ(n+1) ← Λ(n), copy
        # check evidence saturated
        xmy2z!(mn, m0, mb)                   # mb ← m(n) - m(0)
        unsafe_copy!(Λe, Λn)                 # Λe ← Λ(n+1), copy
        Ln, cholesky_state = varbayes_lowbnd!(Λy, kn, Λ0, mb, Λf, Λe)
        if Ln < LowBnd_best && cholesky_state == 0.0
            LowBnd_best = Ln
            unsafe_copy!(ms, mn)
            unsafe_copy!(Λs, Λb)
            isfound = 0
        else
            isfound += 1
            isfound > 9 && break
        end
        # update mn
        unsafe_copy!(Λf, Λn)                 # Λf ← Λ(n+1), copy
        varbayes_lmstep!(Δn, Λ0, mb, An, kn) # Δ(n) ← A(n)' * k(n) - Λ(0) * [m(n) - m(0)]
        cholesky_state = varbayes_lmstep!(mn, Λf, 0.01, Δn)
    end
    return ms, Λs
end
