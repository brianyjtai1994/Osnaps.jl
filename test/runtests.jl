using Osnaps, Test

print_head(s::String) = println("\n \033[1m\033[32mTest Osnaps\033[0m \033[33m" * s * "\033[0m")
print_body(s::String) = println(" "^13 * s)

@testset "Variational Inference: NIST Dataset(BoxBOD)" begin
    X = [1., 2., 3., 5., 7., 10.]
    Y = [109., 149., 149., 191., 213., 224.]
    Λ = reshape([ifelse(i == j, 1.0, 0.0) for i in 1:6, j in 1:6], (6, 6))
    o = optimizer(2; ny=6, method="varinf")

    fn(x::Real, θ1::Real, θ2::Real) = θ1 * (1.0 - exp(-θ2 * x))
    fn(x::Real, θ::AbstractVector)  = @inbounds fn(x, θ[1], θ[2])

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

    p = inference!(o, fn, [10., 5.5], reshape([ifelse(i == j, 1e-5, 0.0) for i in 1:2, j in 1:2], (2, 2)), X, Y, Λ)
    print_head("Variational Inference: NIST Dataset(BoxBOD)")
    print_body("xsol[1] = $(round(p.μ[1],       digits=3))")
    print_body("xsol[2] = $(round(p.μ[2],       digits=3))")
    print_body("xerr[1] = $(round(sqrt(p.Σ[1]), digits=3))")
    print_body("xerr[2] = $(round(sqrt(p.Σ[4]), digits=3))")
    @test p.μ[1] ≈ 213.80 atol=0.01
    @test p.μ[2] ≈   0.55 atol=0.01
end

@testset "Variational Inference: NIST Dataset(Eckerle4)" begin
    X = [400.0, 405.0, 410.0, 415.0, 420.0, 425.0, 430.0,
         435.0, 436.5, 438.0, 439.5, 441.0, 442.5, 444.0,
         445.5, 447.0, 448.5, 450.0, 451.5, 453.0, 454.5,
         456.0, 457.5, 459.0, 460.5, 462.0, 463.5, 465.0,
         470.0, 475.0, 480.0, 485.0, 490.0, 495.0, 500.0]
    Y = [1.575000e-04, 1.699000e-04, 2.350000e-04, 3.102000e-04, 4.917000e-04,
         8.710000e-04, 1.741800e-03, 4.640000e-03, 6.589500e-03, 9.730200e-03,
         1.490020e-02, 2.373100e-02, 4.016830e-02, 7.125590e-02, 1.264458e-01,
         2.073413e-01, 2.902366e-01, 3.445623e-01, 3.698049e-01, 3.668534e-01,
         3.106727e-01, 2.078154e-01, 1.164354e-01, 6.167640e-02, 3.372000e-02,
         1.940230e-02, 1.178310e-02, 7.435700e-03, 2.273200e-03, 8.800000e-04,
         4.579000e-04, 2.345000e-04, 1.586000e-04, 1.143000e-04, 7.100000e-05]
    Λ = reshape([ifelse(i == j, 1.0, 0.0) for i in 1:35, j in 1:35], (35, 35))
    o = optimizer(3; ny=35, method="varinf")

    fn(x::Real, θ1::Real, θ2::Real, θ3::Real) = θ1 * exp(-0.5 * abs2((x - θ3) / θ2)) / θ2
    fn(x::Real, θ::AbstractVector)            = @inbounds fn(x, θ[1], θ[2], θ[3])

    function Osnaps.jac!(J::AbstractMatrix, f::typeof(fn), θ::AbstractVector; x::AbstractVector)
        @inbounds θ1 = θ[1]
        @inbounds θ2 = θ[2]
        @inbounds θ3 = θ[3]
        @inbounds for i in eachindex(x)
            tmp1 = abs2(x[i] - θ3) / abs2(θ2)
            tmp2 = exp(-0.5 * tmp1)
            J[i,1] = tmp2 / θ2
            J[i,2] = θ1 * tmp2 * (tmp1 - 1.0) / abs2(θ2)
            J[i,3] = θ1 * (x[i] - θ3) * tmp2 / Osnaps.cubic(θ2)
        end
        return nothing
    end

    p = inference!(o, fn, [1.5, 5.0, 450.0], reshape([ifelse(i == j, 1e-5, 0.0) for i in 1:3, j in 1:3], (3, 3)), X, Y, Λ)
    print_head("Variational Inference: NIST Dataset(Eckerle4)")
    print_body("xsol[1] = $(round(p.μ[1],       digits=3))")
    print_body("xsol[2] = $(round(p.μ[2],       digits=3))")
    print_body("xsol[3] = $(round(p.μ[3],       digits=3))")
    print_body("xerr[1] = $(round(sqrt(p.Σ[1]), digits=3))")
    print_body("xerr[2] = $(round(sqrt(p.Σ[5]), digits=3))")
    print_body("xerr[3] = $(round(sqrt(p.Σ[9]), digits=3))")
    @test p.μ[1] ≈   1.554 atol=0.005
    @test p.μ[2] ≈   4.088 atol=0.005
    @test p.μ[3] ≈ 451.541 atol=0.005
end
