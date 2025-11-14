using Random
using Plots
using DataFrames

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
dtanh(x) = 1 - tanh(x)^2
η = 0.001

function activacion(entradas::Vector{Float64}, neurona::Neurona)
    neurona.entradas = entradas
    neurona.activacion = (neurona.pesos * entradas)[1]
end

function forward(entrada::Vector{Float64}, capa::Capa)
    # [activacion(entrada, neurona) for neurona in capa.neuronas]
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
    # neurona.error = dσ(neurona.activacion) * errorespesos(siguiente, i)
    neurona.error = dtanh(neurona.activacion) * errorespesos(siguiente, i)
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

function backprop(entrada::Matrix{Float64}, red::RedNeuronal, y::Matrix{Float64})
    for i in 1:size(entrada, 2)
        backprop(entrada[1:end, i], red, y[1:end, i])
    end
end

function creared()::RedNeuronal
    RedNeuronal(
                # Capa(2, 3, σ),
                # Capa(3, 3, σ),
                Capa(2, 3, tanh),
                Capa(3, 3, tanh),
                Capa(3, 1, x -> x)
               )
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
    for i in 1:epocas
        backprop(entradas, red, y)
        push!(errores, error(entradas, red, y))
    end
    return errores
end
        

# A partir de aquí es código de prueba
# Primero voy a crear los datos
x = [i for i in 0:0.1:2π]
y = sin.(x)
df = DataFrame(x = x, y = y)
df = shuffle(df)
m = ones(2, length(x))
m[1, 1:end] = df[:,:x]
y = collect(df[:, :y]')
# Ya tengo los datos barajados

red = creared()
perdidas = entrena(10000, m, red, y);
plot(perdidas);gui()

estimadas = [forward(m[1:end, i], red)[1] for i in 1:size(m, 2)]
scatter(m[1, 1:end], y[1, 1:end])
scatter!(m[1, 1:end], estimadas)
gui()
