include("estructuras.jl")

function coso(x, v)
    if length(v) == 1
        return x * v[1]
    else
        coso(x * v[1], v[2:end])
    end
end
