module heterogeneousCoreCUDAUtilitiesInterfaceCudaCompat

module cms

module cudacompat

    function atomicCAS(address, compare, val)
        old = address[1]
        if old == compare
            address[1] = val
        end
        return old
    end
    
    function atomicInc(a, b)
        ret = a[1]
        if ret < b
            a[1] += 1
        end
        return ret
    end

    function atomicAdd(a, b)
        ret = a[1]
        a[1] += b
        return ret
    end
    
    function atomicSub(a, b)
        ret = a[1]
        a[1] -= b
        return ret
    end

    function atomicMin(a, b)
        ret = a[1]
        a[1] = min(ret, b)
        return ret
    end

    function atomicMax(a, b)
        ret = a[1]
        a[1] = max(ret, b)
        return ret
    end

end

end

end
