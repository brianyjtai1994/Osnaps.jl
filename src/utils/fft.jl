function twiddle(sz::Int)
    ws = Matrix{Float64}(undef, sz, 2)
    hz = sz >> 1
    ix = 1
    rc = 1.0
    rs = 0.0
    cf = cospi(inv(sz))
    sf = sinpi(inv(sz))
    #### initialize: θ = 0
    @inbounds ws[ix,1] = rc
    @inbounds ws[ix,2] = rs
    @inbounds while ix < hz
        ix += 1
        tc  = cf * rc - sf * rs
        ts  = sf * rc + cf * rs
        rc  = tc
        rs  = ts
        ws[ix,1] = rc
        ws[ix,2] = rs
    end
    #### specific case: θ = π / 2
    ix += 1
    @inbounds ws[ix,1] = 0.0
    @inbounds ws[ix,2] = 1.0
    #### remaining cases: θ > π / 2
    jx = ix
    @inbounds while ix < sz
        ix += 1
        jx -= 1
        ws[ix,1] = -ws[jx,1]
        ws[ix,2] =  ws[jx,2]
    end
    return ws
end

function butterfly!(y::MatO, x::MatI, w::MatI, jx::Int, kx::Int, wx::Int, H::Int, h::Int, S::Int, s::Int)
    @inbounds for _ in eachindex(1:h)
        jy = jx + s

        wr = w[wx,1]
        wi = w[wx,2]
        xr = x[kx,1]
        xi = x[kx,2]
        Xr = x[kx+H,1]
        Xi = x[kx+H,2]

        y[jx,1] = xr + Xr
        y[jx,2] = xi + Xi
        y[jy,1] = wr * (xr - Xr) - wi * (xi - Xi)
        y[jy,2] = wr * (xi - Xi) + wi * (xr - Xr)

        jx += S
        kx += s
        wx += s
    end
end

function difnn!(a::MatIO, b::MatIO, w::MatI, H::Int)
    h = H
    s = 1
    S = 2
    r = false

    while h > 0
        for ix in eachindex(1:s)
            r ? butterfly!(a, b, w, ix, ix, 1, H, h, S, s) : butterfly!(b, a, w, ix, ix, 1, H, h, S, s)
        end

        h >>= 1
        s <<= 1
        S <<= 1
        r = !r
    end
    return nothing
end
