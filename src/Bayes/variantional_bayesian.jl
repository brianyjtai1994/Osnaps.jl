struct VarBayesInfMinimizer <: AbstractMinimizer
    xs::Vector{Float64}
    Σs::Matrix{Float64}
    rs::Vector{Float64}
    rc::Vector{Float64}
    θs::Vector{Float64}
    θc::Vector{Float64}
    Δθ::Vector{Float64}
    dθ::Vector{Float64}
    δθ::Vector{Float64}
    Js::Matrix{Float64}
    Jc::Matrix{Float64}
    As::Matrix{Float64}
    Ac::Matrix{Float64}
    Λs::Matrix{Float64}
    Λc::Matrix{Float64}
    Λt::Matrix{Float64}
    Λf::Matrix{Float64}
    ny::Int
    nd::Int

    function VarBayesInfMinimizer(nd::Int, ny::Int)
        @narray (ny,)    (rs, rc)
        @narray (nd,)    (xs, θs, θc, Δθ, dθ, δθ)
        @narray (ny, nd) (Js, Jc, As, Ac)
        @narray (nd, nd) (Σs, Λs, Λc, Λt, Λf)
        return new(xs, Σs, rs, rc, θs, θc, Δθ, dθ, δθ, Js, Jc, As, Ac, Λs, Λc, Λt, Λf, ny, nd)
    end
end

Base.show(io::IO, o::VarBayesInfMinimizer) = print(io, "Variational Bayesian Inference Minimizer(nd = $(o.nd), nd = $(o.np))")

function minimize!(o::VarBayesInfMinimizer, fn::Function, θ0::VecI, Λ0::MatI, x::VecI, y::VecI, Λy::MatI, τ::Real, h::Real, itmax::Int)
    invh1 = inv(h)
    invh2 = invh1 * invh1

    nd = o.nd; one2nd = eachindex(1:nd)
    ny = o.ny; one2ny = eachindex(1:ny)
    #### Allocations ####
    @nget o (xs, Σs, rs, rc, θs, θc, Δθ, dθ, δθ, Js, Jc, As, Ac, Λs, Λc, Λt, Λf)
    #### Initialization
    @cpy! (θs, Λs) (θ0, Λ0)
    #### Compute precision matrix
    rsd!(rs, fn, θs; x, y)
    jac!(Js, fn, θs; x)
    cpy!(Λc, Λ0)
    symm!('L', 'U', 1.0, Λy, Js, 0.0, As) # As ← Λy  * Js
    gemm!('T', 'N', 1.0, As, Js, 1.0, Λc) # Λc ← As' * Js + Λ0
    μ = diagmax(Λc, nd) * τ
    ν = 2.0
    #### Compute free energy
    fill!(Δθ, 0.0)
    @cpy! (Λf, Σs) (Λs, Λc)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
    Fnow = 0.5 * (dot(rs, ny, Λy, rs, ny) + tr(Σs, nd)) + logdet(Λf, nd)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        cpy!(Λf, Λc)
        λ = μ * diagmax(Λc, nd)
        @simd for i in one2nd
            @inbounds Λf[i,i] += λ
        end
        _, cholesky_state = potrf!('L', Λf)
        # Levenberg-Marquardt step by 1st order linearization
        symv!('U', 1.0, Λ0, Δθ,  0.0, dθ) # dθ ← Λ0  * Δθ
        gemv!('T', 1.0, As, rs, -1.0, dθ) # dθ ← As' * rs - dθ
        trsv!('L', 'N', 'N', Λf, dθ)
        trsv!('L', 'T', 'N', Λf, dθ)
        # Finite-difference of 2nd order directional derivative
        @simd for i in one2nd
            @inbounds θc[i] = θs[i] + h * dθ[i]
        end
        rsd!(rc, fn, θc; x, y)
        axpy!(-1.0, rs, rc)
        gemv!('N', invh1, Js, dθ, invh2, rc)
        # Geodesic acceleration step by 2nd order linearization
        gemv!('T', 1.0, As, rc, 0.0, δθ)
        trsv!('L', 'N', 'N', Λf, δθ)
        trsv!('L', 'T', 'N', Λf, δθ)
        gratio = 2.0 * nrm2(δθ) / nrm2(dθ)
        gratio > 0.75 && (μ *= ν; ν *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        axpy!(1.0, δθ, dθ)
        # Compute predicted gain of the free energy by linear approximation
        LinApprox = 0.5 * dot(dθ, nd, Λc, dθ, nd) + μ * dot(dθ, dθ, nd)
        # Compute gain ratio
        @simd for i in one2nd
            @inbounds θc[i] = θs[i] + dθ[i]
        end
        rsd!(rc, fn, θc; x, y)
        jac!(Jc, fn, θc; x)
        cpy!(Λt, Λ0)
        symm!('L', 'U', 1.0, Λy, Jc, 0.0, Ac) # Ac ← Λy  * Jc
        gemm!('T', 'N', 1.0, Ac, Jc, 1.0, Λt) # Λt ← Ac' * Jc + Λ0
        @simd for i in one2nd
            @inbounds δθ[i] = θc[i] - θ0[i]
        end
        @cpy! (Λf, Σs) (Λc, Λt)
        _, cholesky_state = potrf!('L', Λf)
        trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
        trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
        Fnew = 0.5 * (dot(rc, ny, Λy, rc, ny) + dot(δθ, nd, Λ0, δθ, nd) + tr(Σs, nd)) + logdet(Λf, nd)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew
            @cpy! (θs, Λs) (θc, Λc)
            (maximum(δθ) < 1e-16 || Δ < 1e-10) && break # Check localized convergence
            @cpy! (rs, Λc, Js, As, Δθ) (rc, Λt, Jc, Ac, δθ)
            μ = ρ < 0.9367902323681495 ? μ * (1.0 - cubic(2.0 * ρ - 1.0)) : μ / 3.0
            ν = 2.0
        else
            μ > 1e10 && break # Check localized divergence
            μ *= ν; ν *= 2.0
        end
    end
    # compute solution parameters and covariance
    cpy!(xs, θs)
    fill!(Σs, 0.0)
    @simd for i in one2nd
        @inbounds Σs[i,i] = 1.0
    end
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
    return nothing
end
