export varbayes

function varbayes_update!(ks::VecIO, ys::VecI)
    @simd for i in eachindex(ks)
        @inbounds ks[i] = ys[i] - ks[i]
    end
    return nothing
end

function varbayes_update!(Δm::VecIO, ms::VecI, m0::VecI)
    @simd for i in eachindex(Δm)
        @inbounds Δm[i] = ms[i] - m0[i]
    end
    return nothing
end

# compute: Λn ← Λ0 + Js' * Λy * Js
function varbayes_update!(Λn::MatIO, Λy::MatI, Λ0::MatI, Js::MatI, As::MatB)
    unsafe_copy!(Λn, Λ0)                  # Λn ← Λ0
    symm!('L', 'U', 1.0, Λy, Js, 0.0, As) # As ← Λy  * Js
    gemm!('T', 'N', 1.0, As, Js, 1.0, Λn) # Λs ← As' * Js + Λ0
    return nothing
end

function varbayes_energy!(Λy::MatI, ks::VecI, Λ0::MatI, Δm::VecB, Λf::MatB, Λe::MatB)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    return -0.5 * (dot(ks, Λy, ks) + dot(Δm, Λ0, Δm) + logdet(Λf) + tr(Λe))
end

# barebone version
function varbayes(m0::VecI, Λ0::MatI, ys::VecI, Λy::MatI, mo::Function; τ::Real=1e-3, α::Real=0.1, itmax::Int=100)
    np = length(m0) # dims of params
    nd = length(ys) # dims of data

    #### Allocations ####
    ks = Vector{Float64}(undef, nd)
    kn = Vector{Float64}(undef, nd)
    hn = Vector{Float64}(undef, np)
    Δm = Vector{Float64}(undef, np)
    ms = Vector{Float64}(undef, np)
    mn = Vector{Float64}(undef, np)

    Js = Matrix{Float64}(undef, nd, np)
    Jn = Matrix{Float64}(undef, nd, np)
    As = Matrix{Float64}(undef, nd, np)
    An = Matrix{Float64}(undef, nd, np)
    Λs = Matrix{Float64}(undef, np, np)
    Λn = Matrix{Float64}(undef, np, np)
    Λe = Matrix{Float64}(undef, np, np)
    Λf = Matrix{Float64}(undef, np, np)

    invα1 = inv(α)
    invα2 = abs2(invα1)

    unsafe_copy!(ms, m0)
    unsafe_copy!(Λs, Λ0)

    fcall!(ks, mo, ms) # use current best `ms` to compute `ks`
    gcall!(Js, mo, ms) # use current best `ms` to compute `Js`
    varbayes_update!(Λn, Λy, Λ0, Js, As)
    μ = diagmax(Λn) * τ
    ν = 2.0

    varbayes_update!(ks, ys)
    varbayes_update!(Δm, ms, m0)
    unsafe_copy!(Λf, Λs) # Λf ← Λs for factorization
    unsafe_copy!(Λe, Λn) # Λe ← Λn for free energy
    _, cholesky_state = potrf!('L', Λf)
    Fn = varbayes_energy!(Λy, ks, Λ0, Δm, Λf, Λe)

    isfound = 0; it = 0
    while isfound < 5 && it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        λ = diagmax(Λn) * μ
        unsafe_copy!(Λf, Λn) # Λf ← Λn
        @simd for i in eachindex(mn)
            @inbounds Λf[i,i] += λ # Λf ← Λn + λI
        end
        _, cholesky_state = potrf!('L', Λf)
        # Levenberg-Marquardt step
        symv!('U', 1.0, Λ0, Δm,  0.0, hn) # hn ← Λ0 * Δm
        gemv!('T', 1.0, As, ks, -1.0, hn) # hn ← As' * ks - hn
        trsv!('L', 'N', 'N', Λf, hn)
        trsv!('L', 'T', 'N', Λf, hn)
        # Finite-difference approximation for Hessian
        @simd for i in eachindex(mn)
            @inbounds mn[i] = ms[i] + α * hn[i]
        end
        fcall!(kn, mo, mn)
        @simd for i in eachindex(kn)
            @inbounds kn[i] = ys[i] - kn[i] - ks[i]
        end
        gemv!('N', invα1, Js, hn, invα2, kn)
        # Geodesic acceleration step
        gemv!('T', 1.0, As, kn, 0.0, mn) # use mn for geodesic step
        trsv!('L', 'N', 'N', Λf, mn)
        trsv!('L', 'T', 'N', Λf, mn)
        @simd for i in eachindex(hn)
            @inbounds hn[i] += mn[i]
        end
        # compute predicted gain of the free energy by linear approximation
        Ln = 0.5 * (dot(hn, Λn, hn) + μ * dot(hn, hn))
        # to peek if the free energy is increased by `hn`
        @simd for i in eachindex(mn)
            @inbounds mn[i] = ms[i] + hn[i]
        end
        fcall!(kn, mo, mn) # use new trial `mn` to compute `kn`
        gcall!(Jn, mo, mn) # use new trial `mn` to compute `Jn`
        varbayes_update!(Λe, Λy, Λ0, Jn, An)
        varbayes_update!(kn, ys)
        varbayes_update!(hn, mn, m0)
        unsafe_copy!(Λf, Λn) # Λf ← Λn
        _, cholesky_state = potrf!('L', Λf)
        Ft = varbayes_energy!(Λy, kn, Λ0, hn, Λf, Λe)
        ρ  = (Ft - Fn) / Ln # gain ratio

        if ρ > 0
            Fn = Ft
            unsafe_copy!(ms, mn)
            unsafe_copy!(Λs, Λn)
            unsafe_copy!(ks, kn)
            unsafe_copy!(Js, Jn)
            unsafe_copy!(As, An)
            unsafe_copy!(Δm, hn)
            varbayes_update!(Λn, Λy, Λ0, Js, As)
            μ = ρ < 0.9367902323681495 ? μ * (1.0 - cubic(2.0 * ρ - 1.0)) : μ / 3.0
            ν = 2.0; isfound = 0
        else
            μ *= ν; ν *= 2.0; isfound += 1
        end
    end
    return ms, Λs
end
