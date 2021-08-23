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

fcall(fn::Function, x::VecI) = fn(x)

include("./utils/la.jl")
include("./utils/lu.jl")
include("./utils/fft.jl")
include("./utils/stats.jl")
include("./utils/sorting.jl")
include("./utils/interpolation.jl")
include("./optimizer/minimizer.jl")
include("./DiffEq/forward.jl")

export fcall, 𝚷

end # module
