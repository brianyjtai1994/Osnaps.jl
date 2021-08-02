export polyinterp!, polyinterp

function polyinterp!(ydes::VecO{Ty}, xdes::VecO{Tx}, ysrc::VecI, xsrc::VecI; order::Int=3) where {Tx<:Real,Ty<:Real}
    lx = 1                     # left index
    rx = sz = order + 1        # right index
    ia = sz >> 1 + 1           # anchor index
    mx = size(xsrc, 1)         # upper bound of window
    xb = Vector{Tx}(undef, sz) # xsrc buffer
    yb = Vector{Ty}(undef, sz) # ysrc buffer
    tb = Vector{Ty}(undef, sz) # interpolation buffer
    @inbounds for i in 1:sz
        xb[i] = xsrc[i]
        yb[i] = ysrc[i]
    end
    @inbounds xanchor = xb[ia]
    @inbounds for i in eachindex(xdes)
        xi = xdes[i]
        #### moving interpolation window
        while xi > xanchor && rx < mx
            lx += 1
            rx += 1
            ix  = 1
            jx  = 2
            while ix < sz
                @inbounds xb[ix] = xb[jx]
                @inbounds yb[ix] = yb[jx]
                ix += 1
                jx += 1
            end
            @inbounds xb[ix]  = xsrc[rx]
            @inbounds yb[ix]  = ysrc[rx]
            @inbounds xanchor = xb[ia]
        end
        ydes[i] = polyinterp(xi, xb, yb, tb, sz)
    end
    return nothing
end

function polyinterp(x::Real, xv::VecI, yv::VecI, bv::VecB, n::Int)
    one2n = eachindex(1:n)
    @inbounds for i in one2n
        x == xv[i] && return yv[i]
    end
    Δx = Inf
    yp = 0.0
    @inbounds for i in one2n
        δx = abs(x - xv[i])
        δx < Δx && (Δx = δx; yp = yv[i])
    end
    @simd for i in one2n
        @inbounds bv[i] = yv[i] - yp
    end
    @inbounds for k in 1:n-1, i in 1:n-k
        bv[i] += (bv[i] - bv[i+1]) * (x - xv[i]) / (xv[i] - xv[i+k])
    end
    @inbounds return yp + bv[1]
end
