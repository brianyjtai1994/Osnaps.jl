using Osnaps, Test

print_head(s::String) = println(" \033[1m\033[32mTest Osnaps\033[0m \033[33m" * s * "\033[0m")
print_body(s::String) = println(" "^13 * s)

@testset "Variational Inference: NIST Dataset(BoxBOD)" begin
    X = [1., 2., 3., 5., 7., 10.]
    Y = [109., 149., 149., 191., 213., 224.]
    Λ = reshape([ifelse(i == j, 1.0, 0.0) for i in 1:6, j in 1:6], (6, 6))
    o = minimizer(2; ny=6, method="varinf")

    fn(x::Real, θ1::Real, θ2::Real) = θ1 * (1.0 - exp(-θ2 * x))
    fn(x::Real, θ::AbstractVector)  = fn(x, θ[1], θ[2])

    function Osnaps.fnc!(y::AbstractVector, f::typeof(fn), θ::AbstractVector; x::AbstractVector)
        @inbounds for i in eachindex(y)
            y[i] = f(x[i], θ)
        end
        return nothing
    end

    function Osnaps.rsd!(r::AbstractVector, f::typeof(fn), θ::AbstractVector; x::AbstractVector, y::AbstractVector)
        Osnaps.fnc!(r, f, θ; x)
        @simd for i in eachindex(r)
            @inbounds r[i] = y[i] - r[i]
        end
        return nothing
    end

    function Osnaps.jac!(J::AbstractMatrix, f::typeof(fn), θ::AbstractVector; x::AbstractVector)
        @inbounds θ1 = θ[1]
        @inbounds θ2 = θ[2]
        for i in eachindex(x)
            @inbounds tmp = exp(-θ2 * x[i])
            @inbounds J[i,1] = 1.0 - tmp
            @inbounds J[i,2] = θ1 * x[i] * tmp
        end
        return nothing
    end

    minimize!(o, fn, [10., 5.5], [1e-5 0.; 0. 1e-5], X, Y, Λ)
    print_head("Variational Inference: NIST Dataset(BoxBOD)")
    print_body("xsol[1] = $(round(o.xs[1],       digits=3))")
    print_body("xsol[2] = $(round(o.xs[2],       digits=3))")
    print_body("xerr[1] = $(round(sqrt(o.Σs[1]), digits=3))")
    print_body("xerr[2] = $(round(sqrt(o.Σs[4]), digits=3))")
    @test o.xs[1] ≈ 213.80 atol=0.01
    @test o.xs[2] ≈   0.55 atol=0.01
end
