using Random
using Plots
using DataFrames

mutable struct Neurona
    nentradas::Int64
    entradas::Vector{Float64}
    pesos::Matrix{Float64}
    bias::Float64
    activacion::Float64
    error::Float64

    function Neurona(nentradas::Int64)::Neurona
        inicializacion = randn(nentradas + 1)
        new(nentradas, [], inicializacion[1:end - 1]', inicializacion[end], 0.0, 0.0)
    end
end

mutable struct Capa
    nentradas::Int64
    nneuronas::Int64
    neuronas::Vector{Neurona}
    factivacion::Function
    dfactivacion::Function

    function Capa(nentradas::Int64, nneuronas::Int64, factivacion::Function, dfactivacion::Function)::Capa
        new(nentradas, nneuronas, [Neurona(nentradas) for _ in 1:nneuronas], factivacion, dfactivacion)
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
dtanh(x) = 1 - tanh(x)^2
η = 0.02

function activacion(entradas::Vector{Float64}, neurona::Neurona)
    neurona.entradas = entradas
    neurona.activacion = (neurona.pesos * entradas)[1] + neurona.bias
end

function forward(entrada::Vector{Float64}, capa::Capa)
    [capa.factivacion(activacion(entrada, neurona)) for neurona in capa.neuronas]
end

function forward(entrada::Vector{Float64}, red::RedNeuronal)
    x = forward(entrada, red.capas[1])
    for capa in red.capas[2:end]
        x = forward(x, capa)
    end
    return x
end

function errorespesos(capa::Capa, i::Int64)
    errores = [neurona.error for neurona in capa.neuronas]
    pesos = [neurona.pesos[i] for neurona in capa.neuronas]
    return errores' * pesos
end

function backpropneurona(neurona::Neurona, siguiente::Capa, i::Int64)
    neurona.error = siguiente.dfactivacion(neurona.activacion) * errorespesos(siguiente, i)
    actualizacion = neurona.error * neurona.entradas
    neurona.pesos -= η * actualizacion'
    neurona.bias -= η * neurona.error
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

function backprop(entrada::Matrix{Float64}, red::RedNeuronal, y::Matrix{Float64})
    for i in 1:size(entrada, 2)
        backprop(entrada[1:end, i], red, y[1:end, i])
    end
end

function error(entradas::Matrix{Float64}, red::RedNeuronal, y::Matrix{Float64})
    error = 0
    for i in 1:size(entradas, 2)
        error += ((forward(entradas[1:end, i], red) - y[1:end, i])[1])^2
    end
    return sqrt(error) / size(entradas, 2)
end

function entrena(epocas::Int64, entradas::Matrix{Float64}, red::RedNeuronal, y::Matrix{Float64})
    errores = []
    for _ in 1:epocas
        backprop(entradas, red, y)
        push!(errores, error(entradas, red, y))
    end
    return errores
end

# A partir de aquí, funciones de utilidad

function crearedtangente(nneuronas::Int64)::RedNeuronal
    RedNeuronal(
                Capa(1, nneuronas, tanh, dtanh),
                Capa(nneuronas, nneuronas, tanh, dtanh),
                Capa(nneuronas, 1, x -> x, x -> 1)
               )
end

function crearedsigmoide(nneuronas::Int64)::RedNeuronal
    RedNeuronal(
                Capa(1, nneuronas, σ, dσ),
                Capa(nneuronas, nneuronas, σ, dσ),
                Capa(nneuronas, 1, x -> x, x -> 1)
               )
end

function generadatoscuadrado()
    x = collect(-1:0.05:1)
    y = x.^2
    df = DataFrame(x = x, y = y)
    df = shuffle(df)
    x = collect(df[:, :x]')
    y = collect(df[:, :y]')
    return x, y
end
    
function generadatosseno()
    x = [i for i in 0:0.1:2π]
    y = sin.(x)
    df = DataFrame(x = x, y = y)
    df = shuffle(df)
    x = collect(df[:, :x]')
    y = collect(df[:, :y]')
    return x, y
end

X, y = generadatosseno();
# red = crearedsigmoide(3);
# X, y = generadatoscuadrado()
red = crearedtangente(3);
perdidas = entrena(1000, X, red, y);
plot(perdidas, label = "Pérdidas");gui()

estimadas = [forward(X[1:end, i], red)[1] for i in 1:size(X, 2)];
scatter(X[1, 1:end], y[1, 1:end], label="Real")
scatter!(X[1, 1:end], estimadas, label = "Estimada")
gui()
