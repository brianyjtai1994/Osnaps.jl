#=
    Sine-Cosine Optimizer (https://doi.org/10.1016/j.knosys.2015.12.022)

    params:
    -------
    * Xb := buffer
    * Xn := n-th solution in the pool
    * Xr := referred solution
    * ss := step size
=#
function sco_move!(Xb::VecB, Xn::VecI, Xr::VecI, ss::Real) # @code_warntype ✓
    r = 2.0 * rand()
    s = sinpi(r)
    c = cospi(r)
    @simd for i in eachindex(Xb)
        @inbounds Xb[i] = Xn[i] + ss * abs(Xn[i] - Xr[i]) * ifelse(rand() < 0.5, s, c)
    end
end
#=
    Water-Cycle Algorithm Optimizer (https://doi.org/10.1016/j.compstruc.2012.07.010)

    params:
    -------
    * Xb    := buffer
    * Xbest := the best solution currently
=#
function wca_move!(Xb::VecB, Xbest::VecI) # @code_warntype ✓
    scaled_rand = randn() * 0.31622776601683794 # sqrt(0.1)
    @simd for i in eachindex(Xb)
        @inbounds Xb[i] = Xbest[i] + scaled_rand
    end
end
