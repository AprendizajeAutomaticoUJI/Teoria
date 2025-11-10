using Random

mutable struct Neurona
    nentradas::Int64
    pesos::Matrix{Float64}

    function Neurona(nentradas::Int64)::Neurona
        new(nentradas, randn(nentradas)')
    end
end

mutable struct Capa
    nentradas::Int64
    nneuronas::Int64
    neuronas::Vector{Neurona}
    activacion::Function

    function Capa(nentradas::Int64, nneuronas::Int64, activacion::Function)::Capa
        new(nentradas, nneuronas, [Neurona(nentradas) for _ in 1:nneuronas], activacion)
    end
end

function salida(entrada::Vector{Float64}, neurona::Neurona)
    (neurona.pesos * entrada)[1]
end

function salida(entrada::Vector{Float64}, capa::Capa)
    [salida(entrada, neurona) for neurona in capa.neuronas]
end
