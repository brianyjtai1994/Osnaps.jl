struct VarInfOptimizer <: AbstractOptimizer
    @def prop Vector{Float64} rs rc μs μc Δμ dμ δμ
    @def prop Matrix{Float64} Js Jc As Ac Λs Λc Λt Λe Λf
    @def prop Int nd ny

    function VarInfOptimizer(nd::Int, ny::Int)
        @def vars Vector{Float64}(undef, ny) rs rc
        @def vars Vector{Float64}(undef, nd) μs μc Δμ dμ δμ
        @def vars Matrix{Float64}(undef, ny, nd) Js Jc As Ac
        @def vars Matrix{Float64}(undef, nd, nd) Λs Λc Λt Λe Λf
        return new(rs, rc, μs, μc, Δμ, dμ, δμ, Js, Jc, As, Ac, Λs, Λc, Λt, Λe, Λf, nd, ny)
    end
end

Base.show(io::IO, o::VarInfOptimizer) = print(io, "Variational Inference Optimizer(nd = $(o.nd), ny = $(o.ny))")

struct InferredPosterior
    @def prop Vector{Float64} μ g
    @def prop Matrix{Float64} Σ Λ

    function InferredPosterior(o::VarInfOptimizer)
        @get o nd μs Λs Λf
        @def vars Vector{Float64}(undef, nd)     μ g
        @def vars Matrix{Float64}(undef, nd, nd) Σ Λ
        @copy!(μs => μ, Λs => Λ, Λs => Λf)
        for j in eachindex(μ)
            @simd for i in eachindex(μ)
                @inbounds Σ[i,j] = ifelse(i == j, 1.0, 0.0)
            end
        end
        potrf!('L', Λf)
        trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σ)
        trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σ)
        return new(μ, g, Σ, Λ)
    end
end

Base.show(io::IO, o::InferredPosterior) = print(io, "Inferred Posterior of Variational Inference (dim = $(length(o.μ)))")

function inference!(o::VarInfOptimizer, fn::Function, μ0::VecI, Λ0::MatI, x::VecI, y::VecI, Λy::MatI, τ::Real, h::Real, itmax::Int)
    #### Allocations ####
    @get o nd ny rs rc μs μc Δμ dμ δμ
    @get o Js Jc As Ac Λs Λc Λt Λe Λf
    #### Initialization
    invh1  = inv(h)
    invh2  = invh1 * invh1
    one2nd = eachindex(1:nd)
    @copy!(μ0 => μs, Λ0 => Λs)
    #### Compute precision matrix
    rsd!(rs, fn, μs; x, y)
    jac!(Js, fn, μs; x)
    @copy!(Λ0 => Λc)
    symm!('L', 'U', 1.0, Λy, Js, 0.0, As) # As ← Λy  * Js
    gemm!('T', 'N', 1.0, As, Js, 1.0, Λc) # Λc ← As' * Js + Λ0
    μ = diagmax(Λc, nd) * τ
    ν = 2.0
    #### Compute free energy
    fill!(Δμ, 0.0)
    @copy!(Λs => Λf, Λc => Λe)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    Fnow = 0.5 * (dot(rs, ny, Λy, rs, ny) + tr(Λe, nd)) + logdet(Λf, nd)
    #### Iteration
    it = 0
    while it < itmax
        it += 1
        # Damped Gauss-Newton approximation
        @copy!(Λc => Λf)
        λ = μ * diagmax(Λc, nd)
        @simd for i in one2nd
            @inbounds Λf[i,i] += λ
        end
        _, cholesky_state = potrf!('L', Λf)
        # Levenberg-Marquardt step by 1st order linearization
        symv!('U', 1.0, Λ0, Δμ,  0.0, dμ) # dμ ← Λ0  * Δμ
        gemv!('T', 1.0, As, rs, -1.0, dμ) # dμ ← As' * rs - dμ
        trsv!('L', 'N', 'N', Λf, dμ)
        trsv!('L', 'T', 'N', Λf, dμ)
        # Finite-difference of 2nd order directional derivative
        @simd for i in one2nd
            @inbounds μc[i] = μs[i] + h * dμ[i]
        end
        rsd!(rc, fn, μc; x, y)
        axpy!(-1.0, rs, rc)
        gemv!('N', invh1, Js, dμ, invh2, rc)
        # Geodesic acceleration step by 2nd order linearization
        gemv!('T', 1.0, As, rc, 0.0, δμ)
        trsv!('L', 'N', 'N', Λf, δμ)
        trsv!('L', 'T', 'N', Λf, δμ)
        gratio = 2.0 * nrm2(δμ) / nrm2(dμ)
        gratio > 0.75 && (μ *= ν; ν *= 2.0; continue)
        # (Levenberg-Marquardt velocity) + (Geodesic acceleration)
        axpy!(1.0, δμ, dμ)
        # Compute predicted gain of the free energy by linear approximation
        LinApprox = 0.5 * dot(dμ, nd, Λc, dμ, nd) + μ * dot(dμ, dμ, nd)
        # Compute gain ratio
        @simd for i in one2nd
            @inbounds μc[i] = μs[i] + dμ[i]
        end
        rsd!(rc, fn, μc; x, y)
        jac!(Jc, fn, μc; x)
        @copy!(Λ0 => Λt)
        symm!('L', 'U', 1.0, Λy, Jc, 0.0, Ac) # Ac ← Λy  * Jc
        gemm!('T', 'N', 1.0, Ac, Jc, 1.0, Λt) # Λt ← Ac' * Jc + Λ0
        @simd for i in one2nd
            @inbounds δμ[i] = μc[i] - μ0[i]
        end
        @copy!(Λc => Λf, Λt => Λe)
        _, cholesky_state = potrf!('L', Λf)
        trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
        trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
        Fnew = 0.5 * (dot(rc, ny, Λy, rc, ny) + dot(δμ, nd, Λ0, δμ, nd) + tr(Λe, nd)) + logdet(Λf, nd)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew
            @copy!(rc => rs, μc => μs, Λc => Λs, Jc => Js, Ac => As)
            (maximum(δμ) < 1e-16 || Δ < 1e-10) && break # Check localized convergence
            @copy!(Λt => Λc, δμ => Δμ)
            μ = ρ < 0.9367902323681495 ? μ * (1.0 - cubic(2.0 * ρ - 1.0)) : μ / 3.0
            ν = 2.0
        else
            μ > 1e10 && break # Check localized divergence
            μ *= ν; ν *= 2.0
        end
    end
    return InferredPosterior(o)
end

function inference!(y::VecIO, yerr::VecIO, f::Function, o::InferredPosterior, σ::Real; x::VecI)
    @get o μ g Σ
    fnc!(y, f, μ; x)
    @inbounds for i in eachindex(yerr)
        jac!(g, f, μ; x=x[i])
        yerr[i] = sqrt(σ + dot(g, Σ, g))
    end
end
