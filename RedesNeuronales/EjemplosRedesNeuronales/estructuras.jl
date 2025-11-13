using Random

mutable struct Neurona
    nentradas::Int64
    entradas::Vector{Float64}
    pesos::Matrix{Float64}
    activacion::Float64
    error::Float64

    function Neurona(nentradas::Int64)::Neurona
        new(nentradas, [], randn(nentradas)', 0.0, 0.0)
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
η = 0.01

function activacion(entradas::Vector{Float64}, neurona::Neurona)
    neurona.entradas = entradas
    neurona.activacion = (neurona.pesos * entradas)[1]
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

function errorespesos(capa::Capa, i::Int64)
    errores = [neurona.error for neurona in capa.neuronas]
    pesos = [neurona.pesos[i] for neurona in capa.neuronas]
    return errores' * pesos
end

function backpropneurona(neurona::Neurona, siguiente::Capa, i::Int64)
    neurona.error = dσ(neurona.activacion) * errorespesos(siguiente, i)
    # actualizacion = dσ(neurona.activacion) * errorespesos(siguiente, i) * neurona.entradas
    actualizacion = neurona.error * neurona.entradas
    neurona.pesos -= η * actualizacion'
end

function backpropcapa(actual::Capa, siguiente::Capa)
    for (i, neurona) in enumerate(actual.neuronas)
        backpropneurona(neurona, siguiente, i)
    end
end

function backprop(entrada::Vector{Float64}, red::RedNeuronal, y::Vector{Float64})
    prediccion = forward(entrada, red)
    error = prediccion - y
    red.capas[end].neuronas[end].error = error[1]
    # Ahora propago hacia atrás
    for i in length(red.capas):-1:2
        backpropcapa(red.capas[i-1], red.capas[i])
    end
end

function creared()::RedNeuronal
    RedNeuronal(
                Capa(1, 3, σ),
                Capa(3, 3, σ),
                Capa(3, 1, x -> x)
               )
end
