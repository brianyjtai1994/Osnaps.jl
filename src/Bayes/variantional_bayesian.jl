function vbi_update!(Λn::MatIO, mn::VecIO, Jn::MatI, kn::VecI, Λy::MatI, An::MatB, Λb::MatB, mb::VecB)
    #### Update covariance matrix of params.
    gemv!('N', true, Jn, mn, true,  kn) # Jn * mn + kn → kn
    symv!('U', true, Λn, mb, false, mn) # Λ0 * m0 → mn
    #### Update covariance matrix of params.
    symm!('L', 'U', true, Λy, Jn, false, An)
    gemm!('T', 'N', true, An, Jn, true,  Λn)
    copy!(Λb, Λn)                       # copy to buffer matrix
    #### Update mean vector of params. by Cholesky factorization
    gemv!('T', true, An, kn, true, mn)  # An * kn + mn → mn
    _, cholesky_state = potrf!('L', Λb)
    trsv!('L', 'N', 'N', Λb, mn)
    trsv!('L', 'T', 'N', Λb, mn)
    return cholesky_state
end

function vbi_update!(Λn::MatIO, mn::VecIO, Jn::MatI, kn::VecI, Λy::MatI, pv::VecI{Int}, An::MatB, Λb::MatB, mb::VecB)
    #### Update covariance matrix of params.
    gemv!('N', true, Jn, mn, true,  kn) # Jn * mn + kn → kn
    symv!('U', true, Λn, mb, false, mn) # Λ0 * m0 → mn
    #### Update covariance matrix of params.
    symm!('L', 'U', true, Λy, Jn, false, An)
    gemm!('T', 'N', true, An, Jn, true,  Λn)
    copy!(Λb, Λn)                       # copy to buffer matrix
    #### Update mean vector of params. by LUP factorization
    gemv!('T', true, An, kn, true, mn)  # An * kn + mn → mn
    lupf!(Λb, pv)
    lups!(mn, Λb, pv)
    return nothing
end
