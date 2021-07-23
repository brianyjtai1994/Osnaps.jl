biinsert(arr::VecI, val::T) where T = biinsert(arr, val, 1, length(arr)) # @code_warntype ✓
function biinsert(arr::VecI, val::T, lx::Int, rx::Int) where T           # @code_warntype ✓
    lx ≥ rx && return lx
    ub = rx # upper bound
    while lx < rx
        mx = (lx + rx) >> 1                                    # midpoint (binary search)
        @inbounds isless(val, arr[mx]) ? rx = mx : lx = mx + 1 # arr[mx].f == val in this case
    end
    @inbounds lx == ub && !isless(val, arr[lx]) && (lx += 1)   # lx = upper bound && arr[lx] ≤ val
    return lx
end

binsort!(arr::VecI) = binsort!(arr, 1, length(arr)) # @code_warntype ✓
function binsort!(arr::VecI, lx::Int, rx::Int)      # @code_warntype ✓
    for ix in lx+1:rx
        @inbounds val = arr[ix]
        jx = ix
        lc = biinsert(arr, val, lx, ix) # location
        while jx > lc
            swap!(arr, jx, jx - 1)
            jx -= 1
        end
    end
end
