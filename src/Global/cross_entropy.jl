function fnc!(des::VecIO, f::Function, src::MatI)
    @inbounds for i in eachindex(des)
        des[i] = f(view(src, i, :))
    end
    return nothing
end

function gaussian_sampling!(X::MatIO, μ::VecI, σ::VecI)
    for j in eachindex(μ)
        @inbounds μj = μ[j]
        @inbounds σj = σ[j]
        @simd for i in axes(X, 1)
            @inbounds X[i,j] = μj + σj * randn()
        end
    end
    return nothing
end

struct CrossEntropyOptimizer
    @def prop Vector{Float64} xs
    @def prop Matrix{Float64} Xs
    @def prop Vector{Float64} μc μn σc σn fs
    @def prop Int nd np

    function CrossEntropyOptimizer(nd::Int, np::Int)
        @def vars Vector{Float64}(undef, np) fs
        @def vars Matrix{Float64}(undef, np, nd) Xs
        @def vars Vector{Float64}(undef, nd) xs μc μn σc σn
        return new(xs, Xs, μc, μn, σc, σn, fs, nd, np)
    end
end

function minimize!(o::CrossEntropyOptimizer, f::Function, μ::VecI, σ::VecI;
                   ρ::Real=0.01, N::Int=2000, α::Real=0.7, β::Real=0.5, λ::Real=1e-5, itmax::Int=500)
    M = length(μ)
    N ≠ o.np && error("minimize!: CrossEntropyOptimizer.np ≠ N = $N.")
    M ≠ o.nd && error("minimize!: CrossEntropyOptimizer.nd ≠ length(μ).")
    E = floor(Int, ρ * N)

    @get o xs Xs μc μn σc σn fs

    toM = eachindex(1:M)
    toE = eachindex(1:E) # elite index

    @copy!(μ => μc, σ => σc)

    fnow = f(μc)
    ν = 1.0 + λ
    itcount = 0
    isfound = 0
    while itcount < itmax
        itcount += 1
        gaussian_sampling!(Xs, μc, σc)
        fnc!(fs, f, Xs)

        #### Sorting (min → max)
        @inbounds for n in 2:N
            fn = fs[n]
            lc = biinsert(fs, fn, 1, n)
            ix = n
            while ix > lc
                for jx in toM
                    swap!(Xs, ix, jx, ix-1, jx)
                end
                swap!(fs, ix, ix-1)
                ix -= 1
            end
        end

        @inbounds fnew = fs[E]

        if fnew < fnow
            fnow = fnew

            for j in toM
                nu = de = 0.0
                @inbounds μj = μc[j]
                @inbounds for i in toE
                    fi  = fs[i]
                    nu += fi * Xs[i,j]
                    de += fi
                end
                @inbounds μn[j] = μj = nu / de

                nu = 0.0
                @inbounds for i in toE
                    nu += fs[i] * abs2(Xs[i,j] - μj)
                end
                @inbounds σn[j] = nu / de
            end

            axpby!(1.0 - α, μn, α, μc)
            @simd for i in toM
                @inbounds σc[i] = sqrt((1.0 - β) * abs2(σc[i]) + β * σn[i])
            end
            β = β * (1.0 - (0.9 - inv(itcount))^5)
            ν = 1.0 + λ
            isfound = 0
        else
            isfound += 1
            isfound == 40 && break
            @simd for i in toM
                @inbounds σc[i] *= ν
            end
            ν += λ * ν
        end
    end
    gaussian_sampling!(Xs, μc, σc)
    fnc!(fs, f, Xs)

    @inbounds for n in 2:N
        fn = fs[n]
        lc = biinsert(fs, fn, 1, n)
        ix = n
        while ix > lc
            for jx in toM
                swap!(Xs, ix, jx, ix-1, jx)
            end
            swap!(fs, ix, ix-1)
            ix -= 1
        end
    end

    @simd for i in eachindex(xs)
        @inbounds xs[i] = Xs[1,i]
    end
    return xs
end
