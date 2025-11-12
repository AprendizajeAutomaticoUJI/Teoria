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
    factivacion::Function

    function Capa(nentradas::Int64, nneuronas::Int64, factivacion::Function)::Capa
        new(nentradas, nneuronas, [Neurona(nentradas) for _ in 1:nneuronas], factivacion)
    end
end

mutable struct RedNeuronal
    capas::Vector{Capa}
    
    function RedNeuronal(capas::Capa...)
        new([c for c in capas])
    end
        
end

σ(x) = 1 / (1 + exp(-x))
dσ(x) = σ(x)*(1 - σ(x))

function activacion(entrada::Vector{Float64}, neurona::Neurona)
    (neurona.pesos * entrada)[1]
end

function salida(entrada::Vector{Float64}, capa::Capa)
    [activacion(entrada, neurona) for neurona in capa.neuronas]
end

function forward(entrada::Vector{Float64}, red::RedNeuronal)
    x = salida(entrada, red.capas[1])
    for capa in red.capas[2:end]
        x = salida(x, capa)
    end
    return x
end

function backprop(entrada::Vector{Float64}, red::RedNeuronal, y::Vector{Float64})
    prediccion = forward(entrada, red)
    error = prediccion - y
    return error
end
