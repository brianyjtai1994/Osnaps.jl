# Levenberg-Marquardt step by 1st order linearization
function lmstep!(d::VecIO, A::MatI, β::Real, J::MatI, rs::VecI, Λ0::MatI, Δμ::VecI)
    _, cholesky_state = potrf!('L', A)
    symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    gemv!('T',   β,  J, rs, -1.0, d) # d ← βJ' * rs - d
    trsv!('L', 'N', 'N', A, d)
    trsv!('L', 'T', 'N', A, d)
    return cholesky_state
end

function lmstep!(d::VecIO, A::MatI, ΛJ::MatI, rs::VecI, Λ0::MatI, Δμ::VecI)
    _, cholesky_state = potrf!('L', A)
    symv!('U', 1.0, Λ0, Δμ,  0.0, d) # d ← Λ0  * Δμ
    gemv!('T', 1.0, ΛJ, rs, -1.0, d) # d ← ΛJ' * rs - d
    trsv!('L', 'N', 'N', A, d)
    trsv!('L', 'T', 'N', A, d)
    return cholesky_state
end
# Geodesic acceleration step by 2nd order linearization
function lmstep!(d::VecIO, A::MatI, β::Real, J::MatI, rs::VecI)
    gemv!('T', β, J, rs, 0.0, d)
    trsv!('L', 'N', 'N', A, d)
    trsv!('L', 'T', 'N', A, d)
    return nothing
end

function lmstep!(d::VecIO, A::MatI, ΛJ::MatI, rs::VecI)
    gemv!('T', 1.0, ΛJ, rs, 0.0, d)
    trsv!('L', 'N', 'N', A, d)
    trsv!('L', 'T', 'N', A, d)
    return nothing
end

function varinf_posterior!(Λn::MatI, Λ0::MatI, J::MatI, β::Real)
    unsafe_copy!(Λn, Λ0)
    syrk!('U', 'T', β, J, 1.0, Λn) # Λn ← β * J'J + Λ0
    @inbounds for j in axes(Λn, 1), i in 1:j-1
        Λn[j,i] = Λn[i,j]
    end
    return nothing
end
function varinf_posterior!(Λn::MatI, Λ0::MatI, J::MatI, Λy::MatI, ΛJ::MatB)
    unsafe_copy!(Λn, Λ0)
    symm!('L', 'U', 1.0, Λy, J, 0.0, ΛJ) # ΛJ ← Λy * J
    gemm!('T', 'N', 1.0, ΛJ, J, 1.0, Λn) # Λn ← JΛ * J + Λ0
    return nothing
end

function varinf_lowerbound(rs::VecI, β::Real, ny::Int, nd::Int, Λf::MatB, Λe::MatB)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    ret = 0.5 * (β * dot(rs, rs, ny) + tr(Λe, nd)) + logdet(Λf, nd)
    return ret, cholesky_state
end

function varinf_lowerbound(rs::VecI, Λy::MatI, ny::Int, nd::Int, Λf::MatB, Λe::MatB)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    ret = 0.5 * (dot(rs, ny, Λy, rs, ny) + tr(Λe, nd)) + logdet(Λf, nd)
    return ret, cholesky_state
end

function varinf_lowerbound(rs::VecI, β::Real, ny::Int, Λ0::MatI, Δμ::VecI, nd::Int, Λf::MatB, Λe::MatB)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    ret = 0.5 * (β * dot(rs, rs, ny) + dot(Δμ, nd, Λ0, Δμ, nd) + tr(Λe, nd)) + logdet(Λf, nd)
    return ret, cholesky_state
end

function varinf_lowerbound(rs::VecI, Λy::MatI, ny::Int, Λ0::MatI, Δμ::VecI, nd::Int, Λf::MatB, Λe::MatB)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Λe)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Λe)
    ret = 0.5 * (dot(rs, ny, Λy, rs, ny) + dot(Δμ, nd, Λ0, Δμ, nd) + tr(Λe, nd)) + logdet(Λf, nd)
    return ret, cholesky_state
end

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

function inference!(o::VarInfOptimizer, fn::Function, μ0::VecI, Λ0::MatI, x::VecI, y::VecI, β::Real, τ::Real, h::Real, itmax::Int)
    #### Allocations ####
    @get o nd ny rs rc μs μc Δμ dμ δμ
    @get o Js Jc Λs Λc Λt Λe Λf
    #### Initialization
    invh1  = inv(h)
    invh2  = invh1 * invh1
    one2nd = eachindex(1:nd)
    @copy!(μ0 => μs, Λ0 => Λs)
    #### Compute precision matrix
    rsd!(rs, fn, μs; x, y)
    jac!(Js, fn, μs; x)
    varinf_posterior!(Λc, Λ0, Js, β)
    μ = diagmax(Λc, nd) * τ
    ν = 2.0
    #### Compute free energy
    fill!(Δμ, 0.0)
    @copy!(Λs => Λf, Λc => Λe)
    Fnow, cholesky_state = varinf_lowerbound(rs, β, ny, nd, Λf, Λe)
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
        # Levenberg-Marquardt step by 1st order linearization
        cholesky_state = lmstep!(dμ, Λf, β, Js, rs, Λ0, Δμ)
        # Finite-difference of 2nd order directional derivative
        @simd for i in one2nd
            @inbounds μc[i] = μs[i] + h * dμ[i]
        end
        rsd!(rc, fn, μc; x, y)
        axpy!(-1.0, rs, rc)
        gemv!('N', invh1, Js, dμ, invh2, rc)
        # Geodesic acceleration step by 2nd order linearization
        lmstep!(δμ, Λf, β, Js, rc)
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
        varinf_posterior!(Λt, Λ0, Jc, β)
        @simd for i in one2nd
            @inbounds δμ[i] = μc[i] - μ0[i]
        end
        @copy!(Λc => Λf, Λt => Λe)
        Fnew, cholesky_state = varinf_lowerbound(rc, β, ny, Λ0, δμ, nd, Λf, Λe)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew; @copy!(μc => μs, Λc => Λs)
            (maximum(δμ) < 1e-16 || Δ < 1e-10) && break # Check localized convergence
            @copy!(rc => rs, Λt => Λc, Jc => Js, δμ => Δμ)
            μ = ρ < 0.9367902323681495 ? μ * (1.0 - cubic(2.0 * ρ - 1.0)) : μ / 3.0
            ν = 2.0
        else
            μ > 1e10 && break # Check localized divergence
            μ *= ν; ν *= 2.0
        end
    end
    return InferredPosterior(o)
end

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
    varinf_posterior!(Λc, Λ0, Js, Λy, As)
    μ = diagmax(Λc, nd) * τ
    ν = 2.0
    #### Compute free energy
    fill!(Δμ, 0.0)
    @copy!(Λs => Λf, Λc => Λe)
    Fnow, cholesky_state = varinf_lowerbound(rs, Λy, ny, nd, Λf, Λe)
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
        # Levenberg-Marquardt step by 1st order linearization
        cholesky_state = lmstep!(dμ, Λf, As, rs, Λ0, Δμ)
        # Finite-difference of 2nd order directional derivative
        @simd for i in one2nd
            @inbounds μc[i] = μs[i] + h * dμ[i]
        end
        rsd!(rc, fn, μc; x, y)
        axpy!(-1.0, rs, rc)
        gemv!('N', invh1, Js, dμ, invh2, rc)
        # Geodesic acceleration step by 2nd order linearization
        lmstep!(δμ, Λf, As, rc)
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
        varinf_posterior!(Λt, Λ0, Jc, Λy, Ac)
        @simd for i in one2nd
            @inbounds δμ[i] = μc[i] - μ0[i]
        end
        @copy!(Λc => Λf, Λt => Λe)
        Fnew, cholesky_state = varinf_lowerbound(rc, Λy, ny, Λ0, δμ, nd, Λf, Λe)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew; @copy!(μc => μs, Λc => Λs)
            (maximum(δμ) < 1e-16 || Δ < 1e-10) && break # Check localized convergence
            @copy!(rc => rs, Λt => Λc, Jc => Js, Ac => As, δμ => Δμ)
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
