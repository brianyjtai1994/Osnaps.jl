struct VarInference <: AbstractMinimizer
    @def prop Vector{Float64} xs rs rc θs θc Δθ dθ δθ
    @def prop Matrix{Float64} Σs Js Jc As Ac Λs Λc Λt Λf
    @def prop Int ny nd

    function VarInference(nd::Int, ny::Int)
        @def vars Vector{Float64}(undef, ny) rs rc
        @def vars Vector{Float64}(undef, nd) xs θs θc Δθ dθ δθ
        @def vars Matrix{Float64}(undef, ny, nd) Js Jc As Ac
        @def vars Matrix{Float64}(undef, nd, nd) Σs Λs Λc Λt Λf
        return new(xs, rs, rc, θs, θc, Δθ, dθ, δθ, Σs, Js, Jc, As, Ac, Λs, Λc, Λt, Λf, ny, nd)
    end
end

Base.show(io::IO, o::VarInference) = print(io, "Variational Inference Optimizer(nd = $(o.nd), ny = $(o.ny))")

function minimize!(o::VarInference, fn::Function, θ0::VecI, Λ0::MatI, x::VecI, y::VecI, Λy::MatI, τ::Real, h::Real, itmax::Int)
    #### Allocations ####
    @get o nd ny
    @get o xs rs rc θs θc Δθ dθ δθ
    @get o Σs Js Jc As Ac Λs Λc Λt Λf
    #### Initialization
    invh1  = inv(h)
    invh2  = invh1 * invh1
    one2nd = eachindex(1:nd)
    @copy!(θ0 => θs, Λ0 => Λs)
    #### Compute precision matrix
    rsd!(rs, fn, θs; x, y)
    jac!(Js, fn, θs; x)
    @copy!(Λ0 => Λc)
    symm!('L', 'U', 1.0, Λy, Js, 0.0, As) # As ← Λy  * Js
    gemm!('T', 'N', 1.0, As, Js, 1.0, Λc) # Λc ← As' * Js + Λ0
    μ = diagmax(Λc, nd) * τ
    ν = 2.0
    #### Compute free energy
    fill!(Δθ, 0.0)
    @copy!(Λs => Λf, Λc => Σs)
    _, cholesky_state = potrf!('L', Λf)
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
    Fnow = 0.5 * (dot(rs, ny, Λy, rs, ny) + tr(Σs, nd)) + logdet(Λf, nd)
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
        @copy!(Λ0 => Λt)
        symm!('L', 'U', 1.0, Λy, Jc, 0.0, Ac) # Ac ← Λy  * Jc
        gemm!('T', 'N', 1.0, Ac, Jc, 1.0, Λt) # Λt ← Ac' * Jc + Λ0
        @simd for i in one2nd
            @inbounds δθ[i] = θc[i] - θ0[i]
        end
        @copy!(Λc => Λf, Λt => Σs)
        _, cholesky_state = potrf!('L', Λf)
        trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
        trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
        Fnew = 0.5 * (dot(rc, ny, Λy, rc, ny) + dot(δθ, nd, Λ0, δθ, nd) + tr(Σs, nd)) + logdet(Λf, nd)

        Δ = Fnow - Fnew; ρ = Δ / LinApprox
        if ρ > 0.0
            Fnow = Fnew
            @copy!(θc => θs, Λc => Λs)
            (maximum(δθ) < 1e-16 || Δ < 1e-10) && break # Check localized convergence
            @copy!(rc => rs, Λt => Λc, Jc => Js, Ac => As, δθ => Δθ)
            μ = ρ < 0.9367902323681495 ? μ * (1.0 - cubic(2.0 * ρ - 1.0)) : μ / 3.0
            ν = 2.0
        else
            μ > 1e10 && break # Check localized divergence
            μ *= ν; ν *= 2.0
        end
    end
    # compute solution parameters and covariance
    @copy!(θs => xs)
    fill!(Σs, 0.0)
    @simd for i in one2nd
        @inbounds Σs[i,i] = 1.0
    end
    trsm!('L', 'L', 'N', 'N', 1.0, Λf, Σs)
    trsm!('L', 'L', 'T', 'N', 1.0, Λf, Σs)
    return nothing
end
