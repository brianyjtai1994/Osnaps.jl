module Osnaps

const VecI  = AbstractVector # Input  Vector
const VecO  = AbstractVector # Output Vector
const VecB  = AbstractVector # Buffer Vector
const VecIO = AbstractVector # In/Out Vector
const MatI  = AbstractMatrix # Input  Matrix
const MatO  = AbstractMatrix # Output Matrix
const MatB  = AbstractMatrix # Buffer Matrix
const MatIO = AbstractMatrix # In/Out Matrix

const 𝚷 = 2.0 * π

using LinearAlgebra.BLAS: axpy!, gemv!, symv!, trsv!, gemm!, symm!, trsm!
using LinearAlgebra.LAPACK: potrf!

abstract type AbstractMinimizer end

fcall(f::Function, x::VecI) = f(x)
gcall(g::Function, x::VecI) = g(x)

function fnc! end # fnc!(y, f, θ; x)
function jac! end # jac!(J, f, θ; x)
function rsd! end # rsd!(r, f, θ; x, y)

include("./utils/la.jl")
include("./utils/lu.jl")
include("./utils/fft.jl")
include("./utils/stats.jl")
include("./utils/macros.jl")
include("./utils/sorting.jl")
include("./utils/interpolation.jl")
include("./Bayes/variantional_bayesian.jl")
include("./Global/minimize.jl")
include("./DiffEq/forward.jl")
include("./minimizer.jl")

end # module
