export minimizer, minimize!
"""
    minimizer(nd::Int; np::Int=35*ND, ne::Int=ND+1, ny::Int=0, method::String="global")

An interface function to create/initialize an object for the minimization.

Arguments:
---
- `nd`:     Dimension of parameters to be minimized.
- `np`:     Desired population size (*optional*).
- `ne`:     Desired size of elites (*optional*).
- `ny`:     Size of data space (required for *variational-inference*).
- `method`: Desired algorithm ("global", "variational-inference").
"""
function minimizer(nd::Int; np::Int=35*nd, ne::Int=nd+1, ny::Int=0, method::String="global")
    method == "global" && return GenericMinimizer(nd, np, ne)
    if method == "varinf" || method == "varbayes" || method == "variational-inference"
        iszero(ny) && error("minimizer(..., ny): ny should be provided for variational inference.")
        return VarBayesInfMinimizer(nd, ny)
    end
end

"""
    minimize!(o::GenericMinimizer, fn::Function, lb::NTuple{ND}, ub::NTuple{ND}; itmax::Int=210*ND, dmax::Real=1e-7, avgtimes::Int=1)

An interface function to proceed the "global" minimization.

Arguments:
---
- `o`:        An object for the minimization created by `minimizer(..., method="global")`.
- `fn`:       Objective function to be minimized. 
              `fn` should be callable with only one argument of `fn(x::Vector)`. 
              If you have any additional arguments need to pass into it, 
              dispatch the function by `fcall(fn, x) = fn(x; kwargs...)`
- `lb`:       Lower bounds of minimization which are forced to be feasible.
- `ub`:       Upper bounds of minimization which are forced to be feasible.
- `itmax`:    Maximum of minimizing iteration (*optional*).
- `dmax`:     An Euclidean distance acts as a criterion to
              prevent the population falling into local minimal (*optional*).
- `avgtimes`: Number of average times of the whole minimization process (*optional*).
"""
minimize!(o::GenericMinimizer, fn::Function, lb::NTuple{ND}, ub::NTuple{ND}; itmax::Int=210*ND, dmax::Real=1e-7, avgtimes::Int=1) where ND = minimize!(o, fn, lb, ub, itmax, dmax, avgtimes)

"""
    minimize!(o::VarBayesInfMinimizer, fn::Function, θ0::VecI, Λ0::MatI, x::VecI, y::VecI, Λy::MatI; τ::Real=1e-3, h::Real=0.1, itmax::Int=100)

An interface function to proceed the variational Bayesian inference.

Arguments:
---
- `o`:      An object for the minimization created by `minimizer(..., method="variational-inference")`.
- `fn`:     Objective function to be minimized. `fn` should be callable by `fnc!(y, fn, θ; x)`. 
            Dispatch the functions `Osnaps.\$f(y, fn, θ; x)` where `f = fnc!, jac!, rsd!`.
- `θ0`:     Initial guess of prior distribution mean.
- `Λ0`:     Initial guess of prior distribution covariance.
- `τ`:      A Levenberg-Marquardt method parameters,
            pass a smaller number (e.g. 1e-5) when `θ0` is close to the ground truth,
            and pass a larger number (e.g. 1e-1) otherwise.
- `h`:      Step size of finite-difference for computing geodesic acceleration.
- `itmax`:  Maximum of minimizing iteration (*optional*).
"""
minimize!(o::VarBayesInfMinimizer, fn::Function, θ0::VecI, Λ0::MatI, x::VecI, y::VecI, Λy::MatI; τ::Real=1e-3, h::Real=0.1, itmax::Int=100) = minimize!(o, fn, θ0, Λ0, x, y, Λy, τ, h, itmax)
