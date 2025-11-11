### A Pluto.jl notebook ###
# v0.20.20

using Markdown
using InteractiveUtils

# ╔═╡ 49bc4ae1-cc37-4de1-8616-d5bd2cc740ad
using PlutoUI

# ╔═╡ 2537da9f-b15c-448f-805b-8ea2e6bd5996
using PlutoTeachingTools

# ╔═╡ 98d86666-f5c0-4389-8b8c-79291db1b57d
using ShortCodes

# ╔═╡ 17163d32-2fd3-11f0-053d-7b01c1dc1e5d
html"""
<link rel="stylesheet" type="text/css" href="https://belmonte.uji.es/Docencia/IR2130/Teoria/mi_estilo.css" media="screen" />
"""

# ╔═╡ 98925bf9-6cfa-4046-b0c7-de0a0ad286ad
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ f747d2c2-a8ec-46ab-9705-b11b3d48dfe1
url_imagenes = "https://belmonte.uji.es/Docencia/IR2130/Teoria/RedesNeuronales/Imagenes/";

# ╔═╡ bc7e1700-5d03-4493-84d7-b96a7da32d1b
md"""
# Redes neuronales convolucionales

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ ab845847-31e8-49e6-bab2-d8aad2bc6e51
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ c2db677d-1029-47ff-ac13-309d586ecc69
md"""
## Introducción

* Las redes neuronales convolucionales están inspiradas en cómo funciona la visión humana.
* Las redes convolucionales trabajan con imágenes, aunque se pueden utilizar en otros ámbitos.
* Las redes convolucionales son capaces de «reconocer» la relación entre regiones de una imagen, también entre secuencias temporales de imágenes si lo que se analiza es una secuencia de vídeo.
"""

# ╔═╡ d8736fb8-a6a9-4e4d-bd28-509b2fdb6336
md"""
## Objetivos de aprendizaje

* Resumir en qué consisten las principales tareas de la visión por computador.
* Definir las bases de las redes neuronales convolucionales: convoluciones y pooling.
* Construir una red neuronal convolucional utilizando Flux.
* Valorar si existe sobreajuste durante el proceso de entrenamiento de una red convolucional.
* Describir algunas de las arquitecturas de redes neuronales convolucionales más importantes.
"""

# ╔═╡ ce478b27-9250-41ff-898b-c887aa166ed3
md"""
## Objetivos de aprendizaje

* Utilizar la arquitectura YOLO para tareas de visión por computador.
* Resumir en qué consiste la transferencia de aprendizaje.
* Construir un modelo utilizando transferencia de aprendizaje.
"""

# ╔═╡ 7900f0f7-3ec0-434e-a6a8-2b6f381d3328
md"""
## Referencias

1. [Deep Learning](http://www.deeplearningbook.org), Ian Goodfellow, Joshua Bengio and Aaron Courbille. MIT Press. 2016.
1. [Dive into deep learning](https://d2l.ai/index.html), Aston Zhang et al. Cambridge University Press. 2023.
"""

# ╔═╡ a0bbbc75-a852-47cf-88d5-982d03eb5c06
md"""
# Redes neuronales en visión por ordenador
"""

# ╔═╡ 50a1c924-0a84-4cf0-99b7-476cb4b4afe7
md"""
## Introducción

La visión por computador es una disciplina con una gran trayectoria en las Ciencias de la Computación.
"""

# ╔═╡ 8bdb7e0f-dece-4b09-a4e1-f32ee3f7601f
Resource(
	url_imagenes * "computer_vision.jpg",
	:alt => "Computer Vision Book",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 0dd08d60-2b9d-41d0-bcc8-927853f7c962
md"""
## Introducción

Hasta ahora, en los ejemplos de redes neuronales que hemos visto, hemos utilizados capas densas. Sin embargo, utilizar redes con capas densas en visión por computador es impracticable. El tamaño de una imagen, en píxeles, determina el número de neuronas en la capa de entrada y posteriores (una neurona de entrada por cada píxel de la imagen).

El número de parámetros a entrenar sería muy elevado.
"""

# ╔═╡ 799ac9d4-2151-4474-a98e-1476e8bb1033
md"""
## Introducción

Por otro lado, las redes neuronales convolucionales están _inspiradas_ en el funcionamiento de la percepción de la visión:
"""

# ╔═╡ a6aa59f8-f25f-402f-a9e7-9553235f7303
Resource(
	url_imagenes * "funcionamiento_vision.png",
	:alt => "Esquema del funcionamiento de la visión.",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ ee29cc12-7007-4f16-88a0-1dd4ed894e78
md"""
## Introducción

Las redes convolucionales proponen una solución basada en dos elementos:

1. Utilizar convoluciones para **aprender** las características relevantes de las imágenes.
1. Utilizar **pooling** para reducir el tamaño de las imágenes.

Además, en muchos casos las últimas capas de una red convolucional, las que contienen menos neuronas, están formadas por capas densas.

Estos dos elementos permiten utilizar redes neuronales en las tareas típicas de la visión por computador.
"""

# ╔═╡ 4e1c05fb-ebf4-4397-9ceb-838586369223
md"""
## Introducción

Las tareas típicas dentro de la visión por computador son:

* Clasificación de imágenes. Por ejemplo indicar si la imagen es de una persona, un coche o una señal de tráfico.
* Detección de objetos en una imagen. Enumerar todos los objetos que aparecen en una imagen
* Ubicación de objetos dentro de una imagen. Encuadrar cada uno de los objetos reconocidos en una imagen.
* Seguimiento de objetos en secuencias de imágenes. Seguir un determinado objeto (por ejemplo un coche) en un vídeo.
* Segmentación semántica de los objetos de una imagen. Asignar cada pixel de una imagen a cada uno de los objetos que se han reconocido en la imagen.
"""

# ╔═╡ d59746f5-4d77-45cc-b57e-f70c7bf12c2e
md"""
## Clasificación

Un modelo de clasificación de visión por ordenador está entrenado de tal modo que 
es capaz de asignar una clase, o etiqueta, a una nueva imagen.
"""

# ╔═╡ ba6966b3-72d3-4a53-8a37-371a4ff65eaf
Columns(
	Resource(
		url_imagenes * "gato.jpg",
		:alt => "Logo UJI",
		:width => 400,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "french-bulldog.jpg",
		:alt => "Logo UJI",
		:width => 400,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ 9f669593-4b2b-4eab-b33d-5d27aca05f2c
Columns("Gato", "Perro")

# ╔═╡ 57dad7fb-d2cf-4063-a7df-1d66216ec1c6
md"""
## Detección de objetos

Encontrar todos los objetos dentro de una imagen:
"""

# ╔═╡ eef3fbbe-99bc-435d-823b-9964c55cbc93
Resource(
	url_imagenes * "gato_ubicado.png",
	:alt => "Detección de objetos dentro de una imagen.",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 0eaa405e-3a79-4c17-b9ea-52b16c1be54b
md"""
## Detección de objetos
Encontrar todos los objetos dentro de una imagen:
"""

# ╔═╡ b35d2727-4270-495d-9cc8-b60db5ebce48
Resource(
	url_imagenes * "familia.png",
	:alt => "Ejemplo de detección de objetos en una imagen.",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 60734849-05bb-4c6a-a8e3-9bde5ab210b4
md"""
## Seguimiento de objetos
"""

# ╔═╡ 6d895aff-98da-485a-9a4e-43f9862a5630
YouTube("cHDLvp_NPOk")

# ╔═╡ 1b8609a0-11b8-4695-9de2-bc87dbc7ec92
md"""
## Segmentación semántica

Asigna cada pixel a uno de los objetos detectados.
"""

# ╔═╡ fa2e68a3-2921-4694-a203-79d63c744a73
Resource(
	url_imagenes * "familia_segmentada.png",
	:alt => "Imagen segmentada semánticamente.",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 96ada7c9-c5d1-48b9-a76f-bcc41cee12a7
md"""
# Fundamentos de la CNN
"""

# ╔═╡ ab9c62de-7781-4876-a2cc-7a42c501b447
md"""
## Convoluciones

Una convolución es una operación sobre los píxeles de una imagen. La operación se aplica a grupos contiguos de píxeles en la imagen, y la operación se va _desplazando_ sobre los píxeles de la imagen.

En los dos casos siguientes, el desplazamiento es de 1 pixel sobre la imagen. La figura de la derecha muestra la opción con márgenes añadidos.

"""

# ╔═╡ b7fb90d0-e601-44fe-9490-fd9fa89e6e31
Columns(
	Resource(
		url_imagenes * "no_padding_no_strides.gif",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "same_padding_no_strides.gif",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ 2a12e686-6cb7-40a8-9907-90a970ef175b
html"""
Fuente: Vincent Dumoulin, Francesco Visin - A guide to convolution arithmetic for deep learning
"""

# ╔═╡ 4f342e4f-8cb2-4888-a238-ae4e024ad881
md"""
## Convoluciones

En este otro caso, el desplazamiento entre conjuntos de píxeles sobre los que se aplica la convolución es de 2 píxeles.
"""

# ╔═╡ 0a008000-3033-4dd6-8bc4-bfcba76bc5d7
Columns(
	Resource(
		url_imagenes * "no_padding_strides.gif",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "padding_strides.gif",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ 9d24e406-7115-4879-9541-09db6339fd49
md"""
## Convoluciones

Las imágenes en color tienen tres canales: rojo, verde y azul. Las convoluciones que se aplican a la imagen pueden ser distintas para cada uno de los canales.
"""

# ╔═╡ 34ca8253-22af-480e-8aba-9071bf85fd1c
Resource(
	url_imagenes * "convolucion_imagen_color.gif",
	:alt => "Convolución imagen color",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 1ed0f5a2-754f-41ee-a362-6884c7ab20cf
html"""
Fuente: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
"""

# ╔═╡ 22d64831-55af-4883-8013-91d7d1aed6f0
md"""
## Convoluciones

Ejemplo de convolución que _emborrona_ una imagen:

```math
\begin{equation}
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
\end{bmatrix}
\end{equation}
```

"""

# ╔═╡ 48d90adc-5dda-47b4-8703-48783f3284c0
Columns(
	Resource(
		url_imagenes * "damero.jpg",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "damero_emborronado.png",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ 28b85f5d-ad91-4ac6-9874-552ae80fd5bf
html"""
Fuente: Wikipedia
"""

# ╔═╡ 47934a59-a752-4b5c-9ac3-ca603aa6a639
md"""
## Convoluciones

Ejemplo de convolución que _resalta_ los bordes en una imagen:

```math
\begin{equation}
\frac{1}{9}
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0 \\
\end{bmatrix}
\end{equation}
```
"""

# ╔═╡ e77701ee-8989-48b2-9150-b93a230267fb
Columns(
	Resource(
		url_imagenes * "damero.jpg",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "damero_resaltado.png",
		:alt => "Convolución",
		:width => 300,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ e6c81043-fc99-4367-9999-be72c91d9e5f
html"""
<font-size = 2>
Fuente: Wikipedia
</font-size>
"""

# ╔═╡ 1ea47d14-3d4e-4414-808a-ca6c0e6a4f89
md"""
## Convoluciones

Lo que va a _aprender_ la red convolucional es a ajustar los valores de las matrices de convolución para llevar a cabo la tarea para la que sea entrenada: clasificación, segmentación, etc.

```math
\begin{equation}
\begin{bmatrix}
m_{1,1} & m_{1,2} & m_{1,3} \\
m_{2,1} & m_{2,2} & m_{2,3}\\
m_{3,1} & m_{3,2} & m_{3,3} \\
\end{bmatrix}
\end{equation}
```
"""

# ╔═╡ 2df53044-e390-43dc-b7f0-c34f7bd523c0
md"""
## Show me the code

Para instanciar una capa convolucional en Flux

```julia
Conv(filter, in => out, σ = identity;
     stride = 1, pad = 0, dilation = 1, groups = 1, [bias, init])

Conv((5,5), 3 => 7, relu; stride=2, pad=SamePad())
```
"""

# ╔═╡ 80727712-f9a7-44c4-9996-bd6d15f6819c
Resource(
	url_imagenes * "capas_convoluciones.png",
	:alt => "Capas convolucionales.",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 0da704ef-90ea-43ff-ac22-7c84d84c0ed0
md"""
## Show me the code

**pad**: se añaden píxeles nulos para que la imagen de la salida sea 
``ancho_{entrada}/salto``.
"""

# ╔═╡ c19b317f-e5c7-44bc-a802-a76f48f79355
Resource(
	url_imagenes * "padding.png",
	:alt => "Ejemplos de padding.",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ ecdc9b78-1779-4341-9d5f-f25761270f9a
html"""
Fuente: Hands-on Machine Learning... Aurélien Gèron
"""

# ╔═╡ b8d7a851-babb-4faf-b7a5-a3150b13a516
md"""
## Pooling

El número de parámetros que hay que entrenar crece al ir aplicando sucesivas capas convolucionales. Una manera de reducir el número de parámetros es aplicar capas de pooling tras las capas convolucionales:
"""

# ╔═╡ eb7e1c09-f662-4870-8e22-c91124512eaf
Resource(
	url_imagenes * "pooling.png",
	:alt => "Ejemplos de pooling máxio y promedio.",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 60c48cb2-29b1-4007-83ee-c8e282e03003
md"""
De cada una de las zonas recuadradas con un color diferente, se elige un representante para que forme parte de la nueva imagen de tamaño reducido. En el ejemplo, pasamos de una imagen de tamaño 4x4 a tamaño 2x2.

Max pooling elige el valor más alto de cada recuadro. Average pooling elige el promedio de los píxeles del recuadro.
"""

# ╔═╡ 5e47956e-2e46-44c0-bf55-74e030139565
md"""
## Show me the code

Flux nos proporciona distintas capas de Pooling. Para utilizar polling máximo:

```julia
MaxPool((n,m))
```

Para utilizar pooling promedio:

```julia
MeanPool((n,m))
```

donde (n,m) es el tamaño de la región sobre la que se hace el pooling.
"""

# ╔═╡ 8aea4bec-4f53-406b-8880-ce00746d08ec
md"""
# Arquitecturas
"""

# ╔═╡ 574816f0-f6d9-4208-a1fd-12cabf07df40
md"""
## Arquitectura básica de una CNN

Esta es la arquitectura básica de una CNN. Observa cómo se van alternando las 
capas convolucionales con las capas de pooling.
"""

# ╔═╡ 3b3f8538-00ea-49ed-84ec-cff6934e6f27
Resource(
	url_imagenes * "arquitectura_cnn.png",
	:alt => "Arquitectura básica de una red convolucional.",
	:width => 700,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 96babbc2-f033-48e4-9349-68710bdfb067
md"""
Al final hay una estructura de capas totalmente conectadas.
"""

# ╔═╡ a4920e1e-1a26-4580-9f32-17b73e3faf4a
html"""
Fuente: Hands-on machine learning, Aurélien Géron.
"""

# ╔═╡ 410abb9f-56d4-4252-a353-4c1d550f36cd
md"""
## LeNet5

Esta es la arquitectura original presentada por 
[Yann Le-Cunn et al](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).
"""

# ╔═╡ 14eabc76-22bc-4b7d-b26d-e0c18095056e
Resource(
	url_imagenes * "lenet5.png",
	:alt => "Arquitectura de la red LeNet5.",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ af5ae3f6-522c-4a9f-9ff8-9f73161dad9a
md"""
Las **Gaussian connections** del final son la activación **softmax**.
"""

# ╔═╡ aa1559b9-0b7f-4973-93a5-bbbdf4ea0ab0
html"""
Fuente: Yann LeCunn et al.
"""

# ╔═╡ 8692bcdf-059e-45d6-baab-1af4705a72d4
md"""
## LeNet5

La implementación de la red utilizando Flux:

```julia
lenet5 = Chain(
	Conv((5, 5), 1=>6, activacion),
	MeanPool((2, 2)),
    Conv((5, 5), 6=>16, activacion),
	MeanPool((2, 2)),
    Flux.flatten,
    Dense(256 => 120, activacion),
    Dense(120 => 84, activacion), 
    Dense(84 => 10),
)
```
"""

# ╔═╡ 85382e11-cb0f-4e85-bb48-ec21714a230c
md"""
## LeNet5

MINIST es el conjunto con el que se entrenó LeNet5:
"""

# ╔═╡ d01effc2-4200-499f-93d8-a5bb5638449c
Resource(
	url_imagenes * "mnist.png",
	:alt => "Conjunto de datos MNIST.",
	:width => 600,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 43fdbfe8-d55c-4474-a358-fa82768e5db1
html"""
Fuente: Wikipedia
"""

# ╔═╡ d23085f7-1f1e-4abf-8c96-a64a08988328
md"""
## LeNet5

Si entrenamos LeNet5 con el conjunto MNIST, utilizando como función de activación la función tanh y como optimizador Stocastic Gradient Descent (SGD), durante 20 épocas, obtenemos una precisión sobre el conjunto de pruebas del 97,01%!!!.
"""

# ╔═╡ a072564c-33a9-4141-8df0-1fa926a4702c
Columns(
	Resource(
		url_imagenes * "lenet5_precision.png",
		:alt => "Precisión LeNet5 durante entrenamiento",
		:width => 500,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "lenet5_perdidas.png",
		:alt => "Precisión LeNet5 durante entrenamiento",
		:width => 500,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ fda8f26d-087b-406c-9cc1-c63b9d8bd1a5
md"""
Fíjate en que, para este caso, las pérdidas son menores para el conjunto de entrenamiento que para el conjunto de pruebas.
"""

# ╔═╡ 72ec224f-54ca-4acb-a4d1-c3f9881cf299
md"""
## LeNet5

La matriz de confusión:
"""

# ╔═╡ ed9d8902-ddd6-4099-9861-ca8a242e2d4b
Resource(
	url_imagenes * "matriz_confusion_lenet5_tanh_sgd.png",
	:alt => "Matriz de confusión LeNet5 sobre MNIST tanh, sgd, 20 épocas",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 793035a9-f596-491d-bfd9-ed418c171b7d
md"""
## LeNet5

Si utilizamos como función de activación 
la función _relu_ y como optimizador Adam, durante 
20 épocas, obtenemos una precisión sobre el conjunto de pruebas de casi el 98,74%. 
"""

# ╔═╡ 180d751e-1486-4058-acaa-0e1859d76b36
Columns(
	Resource(
		url_imagenes * "lenet5_precision_relu_adam.png",
		:alt => "Precisión durante el entrenamiento de LeNet5 con RELU y Adam.",
		:width => 600,
		:style => "display: block; margin: auto;",
	),
	Resource(
		url_imagenes * "lenet5_perdidas_relu_adam.png",
		:alt => "Precisión durante el entrenamiento de LeNet5 con RELU y Adam.",
		:width => 600,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ d91c36f5-5aaa-4086-8bfb-384ec9a1a200
md"""
## LeNet5

La matriz de confusión:
"""

# ╔═╡ cca48072-8bd6-4993-aba2-d4e8861cb3a1
Resource(
	url_imagenes * "matriz_confusion_lenet5_relu_adam.png",
	:alt => "Matriz de confusión LeNet5 sobre MNIST relu, adam, 20 épocas",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 83ee5b2f-c44f-464b-b3c6-747ed061aad1
md"""
## LeNet5

Este es el vídeo donde un joven LeCun y su equipo muestran cómo trabaja la rede neuronal.
"""

# ╔═╡ 528c23a5-3a0a-4fcc-99eb-90d9a01ff3df
YouTube("FwFduRA_L6Q")

# ╔═╡ f265ac2e-2014-4827-be72-3422ef6a2f11
md"""
En este enlace tienes una [implementación de LeNet5](https://github.com/AprendizajeAutomaticoUJI/Teoria/blob/main/RedesNeuronales/EjemplosRedesNeuronales/lenet5.jl).
"""

# ╔═╡ e35e23ec-fb8a-480b-9904-c0fd95e5fe73
md"""
## ResNet

[ResNet](https://arxiv.org/pdf/1512.03385) introdujo una novedad en su diseño 
para reducir el problema del desvanecimiento del gradiente. La novedad es 
en _saltar_ capas:
"""

# ╔═╡ 348a29c8-c562-4158-a26c-da3b27cb4f84
Resource(
	url_imagenes * "resnet.png",
	:alt => "Detalle arquitectura ResNet.",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ f63ed416-2350-4790-b324-842a69fb429e
md"""
$H(x) = F(x) + x \rightarrow F(x) = H(x) - x$
"""

# ╔═╡ d2f9757f-68eb-4c4c-8589-4df4bafa830e
md"""
## ResNet

Lo que aprenden las dos capas intermedias de la figura (representadas por la función $F(x)$) es a ajustar la diferencia entre la entrada real a la capa y el resultado de la aplicación de la capa.
"""

# ╔═╡ ffc23bf9-987f-4392-8671-f6a692d12926
Resource(
	url_imagenes * "resnet.png",
	:alt => "Detalle arquitectura ResNet.",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ cdf52389-528b-409e-abf0-2306d71d2303
md"""
## ResNet


"""

# ╔═╡ 54c73cae-750c-47ec-8ad2-7943177370d5
Columns(
	Resource(
		url_imagenes * "resnet_vs_full.png",
		:alt => "Detalle arquitectura ResNet.",
		:height => 800,
		:style => "display: block; margin: auto;",
	),
	md"""
	Después de aplicar dos capas convolucionales, se ajusta con la técnica de ResNet.
	"""
)

# ╔═╡ 89386d4f-4d89-4868-917d-71a59f124018
md"""
## ResNet
"""

# ╔═╡ 9b38cde2-1c0a-4871-ab3a-874fca8ac403
md"""
### Metalhead

[Metalhead](https://fluxml.ai/Metalhead.jl/stable/) es una biblioteca en Julia que implementa algunos de los algoritmos más conocidos en visión por computador. Esta es la [lista de modelos](https://fluxml.ai/Metalhead.jl/0.8.0-DEV/README.html#available-models) implementados en Metalhead.

Los modelos de Metalhead se han entrenado con la base de datos [ImageNet](https://www.image-net.org/) que es un conjunto de datos con
1.000 clases , 1.281.167 imágenes de entrenamiento, 50.000
imágenes de validación y 100.000 imágenes de prueba.
"""

# ╔═╡ 0c1094af-ddf9-4206-bb79-0a5146f792e1
md"""
## Show me the code

```julia
# Cargamos el modelo con los parámetros pre-entrenados
model = ResNet(18; pretrain=true)

# Clasificamos el modelo
output = model(img_data)

# Obtenemos el vector de probabilidades con softmax
probabilities = softmax(vec(output))

# Seleccionamos el índice con máxima probabilidad
top_class_idx = argmax(probabilities)

# Motramos resultados
println("Predicted class: $(labels[top_class_idx]) with probability $(probabilities[top_class_idx])")
```

En el [github](https://github.com/AprendizajeAutomaticoUJI/Teoria/blob/main/RedesNeuronales/EjemplosRedesNeuronales/resnet_metalhead.jl) de la asignatura tienes un ejemplo de cómo utiliar ResNet.
"""

# ╔═╡ 9bbd7f5a-7279-42c4-977f-753cee52be32
md"""
# YOLO - You Only Look Once
"""

# ╔═╡ bea79739-21ec-4ccf-9f55-af442c1dba7f
md"""
## YOLO

You Only Look Once (YOLO) es una red convolucional con un gran desempeño en muchas de la tareas típicas en visión por computador.

YOLO es mantenida por la empresa [Ultralytics](https://www.ultralytics.com/es).

Este es el [artículo](https://arxiv.org/abs/1506.02640) donde se presentó esta arquitectura de red neuronal.
"""

# ╔═╡ 85c037d0-c584-4cfe-a3af-726810a9660e
md"""
## YOLO

Y esta es la arquitectura tal y como se presenta en el artículo:
"""

# ╔═╡ 5db3fc14-31c7-40f6-abe3-f2ee3f4f0bf7
Resource(
	url_imagenes * "yolo.png",
	:alt => "Arquitectura de YOLO",
	:width => 700,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 8f4a4084-ea4c-4111-89e9-4c9cd476d975
md"""
## YOLO

La función de pérdidas que se minimiza es:
"""

# ╔═╡ 608ca8ac-4dd9-484c-b2e7-52dceb9c1461
Resource(
	url_imagenes * "yolo_funcion_perdidas.png",
	:alt => "Función de pérdidas de YOLO",
	:width => 700,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 6915ef51-2ea5-4705-bc52-42c89205b48a
md"""
## YOLO
Una de la novedades de YOLO es que a la salida propociona la caja envolvente de los objetos detectados, así como la probabilidad de su clasificación, y también es capaz de realizar segmentación semántica.

YOLO se presenta en varias versiones, basándose en el número de parámetros presente en la red (tamaño). Las versiones más pequeñas tienen un desempeño más bajo que las versiones grandes, pero, como contrapartida, se pueden ejecutar en dispositivos con hardware más modesto.

Todas las versiones están entrenadas con el conjunto de datos [COCO](https://cocodataset.org/#home).

Afortunadamente, existe una implementación de YOLO en Julia, llamada [ObjectDetector.jl](https://github.com/r3tex/ObjectDetector.jl), pero sólo implementa hasta la versión 7 de YOLO.
"""

# ╔═╡ a5882dd9-f4d3-46c9-9096-743671bf13fb
md"""
## Show me the code

Primero, cargamos el modelo YOLO que vamos a utilizar:

```julia
function prepara_yolo()
    yolomod = YOLO.v7_416_COCO(batch=1, silent=true)
    batch = emptybatch(yolomod)

    return yolomod, batch
end
```

Después preparamos la imagen:

```julia
function prepara_imagen(path, yolomod, batch)
    img = load(path)
    batch[:,:,:,1], padding = prepare_image(img, yolomod)

    return img, batch, padding
end
```
"""

# ╔═╡ 63b7fefa-729a-45b0-9ebf-754769d77e08
md"""
## Show me the code

Finalmente, hacemos la detección:

```julia
res = yolomod(batch, detect_thresh=0.5, overlap_thresh=0.8) 
```

La estructura **res** tiene toda la información de la información detectada.
"""

# ╔═╡ 0c1fb2c7-987f-4a2f-8b63-488e6c177de6
md"""
## Show me the code

En este caso ha detectado tres personas en la imagen.
"""

# ╔═╡ 035068e5-3257-4d09-b063-dcfc836d214f
Resource(
	url_imagenes * "familia2_yolo.png",
	:alt => "Ejemplo detección YOLO",
	:width => 700,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 350e7132-8200-4a1d-8d4e-98316373dc53
md"""
El tiempo de inferencia en mi portátil es de 12.3 ms. En este enlace tienes el código de [ejemplo](https://github.com/AprendizajeAutomaticoUJI/Teoria/blob/main/RedesNeuronales/EjemplosRedesNeuronales/ejemploObjectDetector.jl).
"""

# ╔═╡ e9cfe0c6-9a57-49b5-953f-448e3d4ee0db
md"""
# Transferencia de aprendizaje
"""

# ╔═╡ ff881d8c-81f8-4a1f-afae-308d7289b601
md"""
## Transferencia de aprendizaje

Entrenar una red neuronal desde cero es una tarea muy costosa. Estos son algunos de los motivos:

* Se necesita un conjunto de datos etiquetado muy grande (cientos de miles de imágenes).
* Se necesitan recursos hardware que no son comunes.
* Se necesita tiempo de entrenamiento.
* Se necesita energía para alimentar el hardware.

Sin embargo, existe un _atajo_ para poder (re)entrenar redes convolucionales, ya existentes, sobre nuevos conjuntos de datos, para tareas  específicas.

"""

# ╔═╡ 8bbbf78b-7dfd-4bff-a95b-5a95052d6786
md"""
## Transferencia de aprendizaje

El atajo consiste en lo siguiente:

1. Utilizar una arquitectura inicial ya entrenada.
1. Eliminar las capas más cercanas a la salida y sustituirlas por nuevas capas adaptadas a nuestro problema.
1. Durante la fase de entrenamiento:
    1. _Congelar_ los pesos de las capas preexistentes.
    1. Entrenar durante unas épocas hasta que la precisión sobre el conjunto de validación se estabilice.
    1. _Descongelar_ todas las capas y entrenar hasta conseguir una buena precisión.

"""

# ╔═╡ 18b3310a-2571-4a4b-91f1-38a1fa047d56
md"""
## Show me the code

1. Cargar un modelo ya entrenado:

```julia
base_model = ResNet(18; pretrain=true)
```

2. Extraer la parte de la red que se encarga de la extracción de características:

```julia
feature_extractor = base_model.layers[1]
```

3. *Congelar* la parte de extracción de características, es la parte de la red que reaprovechamos:

```julia
Flux.freeze!(feature_extractor)
```
"""

# ╔═╡ 037ea602-4198-46af-a471-89abda06ec7c
md"""
## Show me the code

4. Crear una nueva capa con el número de clases que nos interesa:

```julia
num_new_classes = 10
new_classifier_head = Chain(
    AdaptiveMeanPool((1,1)), 
    Flux.flatten,            
    Dense(512, num_new_classes) # La última capa de extracción de características de ResNet tiene 512 neuronas.
)
```

5. Unir la nueva capa a la capa de extracción de características:

```julia
transfer_model = Chain(feature_extractor, new_classifier_head)
```

6. Entrenar la red con nuestro conjunto de imágenes.
"""

# ╔═╡ 599634bc-09a5-4815-ae58-432211ea053d
md"""
## Show me the code

Exise otra opción, llamada **fine tunning**, en la que se ajusta no sólo la capa de clasificación, sino algunas capas cercanas a la de clasificación. La idea es re-entrenar algunas de las capas de extracción de características para que se ajusten a nuestro nuevo conjunto de datos.

La aplicación de la técnica de **fine tunning** es el siguiente:

1. Cargar un modelo ya entrenado:

```julia
base_model = ResNet(18; pretrain=true)
```

2. Extraer la parte de la red que se encarga de la extracción de características:

```julia
feature_extractor = base_model.layers[1]
```

3. Crear una nueva capa con el número de clases que nos interesa:

```julia
num_new_classes = 10
new_classifier_head = Chain(
    AdaptiveMeanPool((1,1)), 
    Flux.flatten,            
    Dense(512, num_new_classes) # La última capa de extracción de características de ResNet tiene 512 neuronas.
)
```
"""

# ╔═╡ c1b7b04e-eb0e-4893-82f0-837df1912cc1
md"""
## Show me the code

4. Entrenar la red con nuestro conjunto de imágenes para que haga un ajuste inicial de la nueva capa de clasificación.

5. *Descongelar* algunas capas de extracción de características, típicamente la última o los dos últimas antes de la capa de clasificación.

```julia
Flux.unfreeze!(finetune_model.layers[1][end]) # Descongelamos la última capa.
```

6. Entrenar de nuevo con una tasa de aprendizaje pequeña $η = 10^{-4}$ ó $η = 10^{-5}$.
"""

# ╔═╡ 6f25ccbb-039e-421a-88a0-0a5396b7987a
md"""
# Resumen

* Las redes convolucionales introducen dos técnicas para trabajar con imágenes: 
    * Convoluciones.
    * Pooling.
* Hemos revisado algunas de las arquitecturas más interesantes.
* Hemos visto con detalle el trabajo con YOLO.
* La transferencia de aprendizaje nos permite _adaptar_ redes existentes para trabajar en problemas específicos.
"""

# ╔═╡ ff26ee4a-8624-45d9-8bff-4f2b49c10c40
md"""
# Recursos

* [A Comprehensive Guide to Convolutional Neural Networks](https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53).
* [You Only Look Once - YOLO](https://www.ultralytics.com/).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"

[compat]
PlutoTeachingTools = "~0.4.2"
PlutoUI = "~0.7.61"
ShortCodes = "~0.3.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.1"
manifest_format = "2.0"
project_hash = "7a569f1efe044287c9fcf980287e2d7e96f0bb69"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.3.0+1"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.Ghostscript_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Zlib_jll"]
git-tree-sha1 = "38044a04637976140074d0b0621c1edf0eb531fd"
uuid = "61579ee1-b43e-5ca0-a5da-69d92c66a64b"
version = "9.55.1+0"

[[deps.Hyperscript]]
deps = ["Test"]
git-tree-sha1 = "179267cfa5e712760cd43dcae385d7ea90cc25a4"
uuid = "47d2ed2b-36de-50cf-bf87-49c2cf4b8b91"
version = "0.0.5"

[[deps.HypertextLiteral]]
deps = ["Tricks"]
git-tree-sha1 = "7134810b1afce04bbc1045ca1985fbe81ce17653"
uuid = "ac1192a8-f4b3-4bfe-ba22-af5b92cd3ab2"
version = "0.9.5"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "0533e564aae234aff59ab625543145446d8b6ec2"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.1"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JSON3]]
deps = ["Dates", "Mmap", "Parsers", "PrecompileTools", "StructTypes", "UUIDs"]
git-tree-sha1 = "196b41e5a854b387d99e5ede2de3fcb4d0422aae"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.2"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "4255f0032eafd6451d707a51d5f0248b8a165e4d"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.3+0"

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "Ghostscript_jll", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "44f93c47f9cd6c7e431f2f2091fcba8f01cd7e8f"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.10"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"
    TectonicExt = "tectonic_jll"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"
    tectonic_jll = "d7dd28d6-a5e6-559c-9131-7eb760cdacc5"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.11.1+1"

[[deps.LibGit2]]
deps = ["LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.9.0+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "OpenSSL_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.3+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.12.0"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.MIMEs]]
git-tree-sha1 = "c64d943587f7187e751162b3b84445bbbd79f691"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.1.0"

[[deps.MacroTools]]
git-tree-sha1 = "1e0228a030642014fe5cfe68c2c0a818f9e3f522"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.16"

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Mmap]]
uuid = "a63ad114-7e13-5084-954f-fe012c677804"
version = "1.11.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2025.5.20"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.3.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.29+0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.1+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "05868e21324cede2207c6f0f466b4bfef6d5e7ee"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.12.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoTeachingTools]]
deps = ["Downloads", "HypertextLiteral", "Latexify", "Markdown", "PlutoUI"]
git-tree-sha1 = "ce33e4fd343e43905a8416e6148de8c630101909"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.2"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "7e71a55b87222942f0f9337be62e26b1f103d3e4"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.61"

[[deps.PrecompileTools]]
deps = ["Preferences"]
git-tree-sha1 = "5aa36f7049a63a1528fe8f7c3f2113413ffd4e1f"
uuid = "aea7be01-6a6a-4083-8856-8a6e6704d82a"
version = "1.2.1"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "9306f6085165d270f7e3db02af26a400d580f5c6"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.4.3"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "62389eeff14780bfe55195b7204c0d8738436d64"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShortCodes]]
deps = ["Base64", "CodecZlib", "Downloads", "JSON3", "Memoize", "URIs", "UUIDs"]
git-tree-sha1 = "5844ee60d9fd30a891d48bab77ac9e16791a0a57"
uuid = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
version = "0.3.6"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "cbbebadbcc76c5ca1cc4b4f3b0614b3e603b5000"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.2"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.3.1+2"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.15.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.64.0+1"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.5.0+2"
"""

# ╔═╡ Cell order:
# ╟─17163d32-2fd3-11f0-053d-7b01c1dc1e5d
# ╟─49bc4ae1-cc37-4de1-8616-d5bd2cc740ad
# ╟─2537da9f-b15c-448f-805b-8ea2e6bd5996
# ╟─98d86666-f5c0-4389-8b8c-79291db1b57d
# ╟─98925bf9-6cfa-4046-b0c7-de0a0ad286ad
# ╟─f747d2c2-a8ec-46ab-9705-b11b3d48dfe1
# ╟─bc7e1700-5d03-4493-84d7-b96a7da32d1b
# ╟─ab845847-31e8-49e6-bab2-d8aad2bc6e51
# ╟─c2db677d-1029-47ff-ac13-309d586ecc69
# ╟─d8736fb8-a6a9-4e4d-bd28-509b2fdb6336
# ╟─ce478b27-9250-41ff-898b-c887aa166ed3
# ╟─7900f0f7-3ec0-434e-a6a8-2b6f381d3328
# ╟─a0bbbc75-a852-47cf-88d5-982d03eb5c06
# ╟─50a1c924-0a84-4cf0-99b7-476cb4b4afe7
# ╟─8bdb7e0f-dece-4b09-a4e1-f32ee3f7601f
# ╟─0dd08d60-2b9d-41d0-bcc8-927853f7c962
# ╟─799ac9d4-2151-4474-a98e-1476e8bb1033
# ╟─a6aa59f8-f25f-402f-a9e7-9553235f7303
# ╟─ee29cc12-7007-4f16-88a0-1dd4ed894e78
# ╟─4e1c05fb-ebf4-4397-9ceb-838586369223
# ╟─d59746f5-4d77-45cc-b57e-f70c7bf12c2e
# ╟─ba6966b3-72d3-4a53-8a37-371a4ff65eaf
# ╟─9f669593-4b2b-4eab-b33d-5d27aca05f2c
# ╟─57dad7fb-d2cf-4063-a7df-1d66216ec1c6
# ╟─eef3fbbe-99bc-435d-823b-9964c55cbc93
# ╟─0eaa405e-3a79-4c17-b9ea-52b16c1be54b
# ╟─b35d2727-4270-495d-9cc8-b60db5ebce48
# ╟─60734849-05bb-4c6a-a8e3-9bde5ab210b4
# ╟─6d895aff-98da-485a-9a4e-43f9862a5630
# ╟─1b8609a0-11b8-4695-9de2-bc87dbc7ec92
# ╟─fa2e68a3-2921-4694-a203-79d63c744a73
# ╟─96ada7c9-c5d1-48b9-a76f-bcc41cee12a7
# ╟─ab9c62de-7781-4876-a2cc-7a42c501b447
# ╟─b7fb90d0-e601-44fe-9490-fd9fa89e6e31
# ╟─2a12e686-6cb7-40a8-9907-90a970ef175b
# ╟─4f342e4f-8cb2-4888-a238-ae4e024ad881
# ╟─0a008000-3033-4dd6-8bc4-bfcba76bc5d7
# ╟─9d24e406-7115-4879-9541-09db6339fd49
# ╟─34ca8253-22af-480e-8aba-9071bf85fd1c
# ╟─1ed0f5a2-754f-41ee-a362-6884c7ab20cf
# ╟─22d64831-55af-4883-8013-91d7d1aed6f0
# ╟─48d90adc-5dda-47b4-8703-48783f3284c0
# ╟─28b85f5d-ad91-4ac6-9874-552ae80fd5bf
# ╟─47934a59-a752-4b5c-9ac3-ca603aa6a639
# ╟─e77701ee-8989-48b2-9150-b93a230267fb
# ╟─e6c81043-fc99-4367-9999-be72c91d9e5f
# ╟─1ea47d14-3d4e-4414-808a-ca6c0e6a4f89
# ╟─2df53044-e390-43dc-b7f0-c34f7bd523c0
# ╟─80727712-f9a7-44c4-9996-bd6d15f6819c
# ╟─0da704ef-90ea-43ff-ac22-7c84d84c0ed0
# ╟─c19b317f-e5c7-44bc-a802-a76f48f79355
# ╟─ecdc9b78-1779-4341-9d5f-f25761270f9a
# ╟─b8d7a851-babb-4faf-b7a5-a3150b13a516
# ╟─eb7e1c09-f662-4870-8e22-c91124512eaf
# ╟─60c48cb2-29b1-4007-83ee-c8e282e03003
# ╟─5e47956e-2e46-44c0-bf55-74e030139565
# ╟─8aea4bec-4f53-406b-8880-ce00746d08ec
# ╟─574816f0-f6d9-4208-a1fd-12cabf07df40
# ╟─3b3f8538-00ea-49ed-84ec-cff6934e6f27
# ╟─96babbc2-f033-48e4-9349-68710bdfb067
# ╟─a4920e1e-1a26-4580-9f32-17b73e3faf4a
# ╟─410abb9f-56d4-4252-a353-4c1d550f36cd
# ╟─14eabc76-22bc-4b7d-b26d-e0c18095056e
# ╟─af5ae3f6-522c-4a9f-9ff8-9f73161dad9a
# ╟─aa1559b9-0b7f-4973-93a5-bbbdf4ea0ab0
# ╟─8692bcdf-059e-45d6-baab-1af4705a72d4
# ╟─85382e11-cb0f-4e85-bb48-ec21714a230c
# ╟─d01effc2-4200-499f-93d8-a5bb5638449c
# ╟─43fdbfe8-d55c-4474-a358-fa82768e5db1
# ╟─d23085f7-1f1e-4abf-8c96-a64a08988328
# ╟─a072564c-33a9-4141-8df0-1fa926a4702c
# ╟─fda8f26d-087b-406c-9cc1-c63b9d8bd1a5
# ╟─72ec224f-54ca-4acb-a4d1-c3f9881cf299
# ╟─ed9d8902-ddd6-4099-9861-ca8a242e2d4b
# ╟─793035a9-f596-491d-bfd9-ed418c171b7d
# ╟─180d751e-1486-4058-acaa-0e1859d76b36
# ╟─d91c36f5-5aaa-4086-8bfb-384ec9a1a200
# ╟─cca48072-8bd6-4993-aba2-d4e8861cb3a1
# ╟─83ee5b2f-c44f-464b-b3c6-747ed061aad1
# ╟─528c23a5-3a0a-4fcc-99eb-90d9a01ff3df
# ╟─f265ac2e-2014-4827-be72-3422ef6a2f11
# ╟─e35e23ec-fb8a-480b-9904-c0fd95e5fe73
# ╟─348a29c8-c562-4158-a26c-da3b27cb4f84
# ╟─f63ed416-2350-4790-b324-842a69fb429e
# ╟─d2f9757f-68eb-4c4c-8589-4df4bafa830e
# ╟─ffc23bf9-987f-4392-8671-f6a692d12926
# ╟─cdf52389-528b-409e-abf0-2306d71d2303
# ╟─54c73cae-750c-47ec-8ad2-7943177370d5
# ╟─89386d4f-4d89-4868-917d-71a59f124018
# ╟─9b38cde2-1c0a-4871-ab3a-874fca8ac403
# ╟─0c1094af-ddf9-4206-bb79-0a5146f792e1
# ╟─9bbd7f5a-7279-42c4-977f-753cee52be32
# ╟─bea79739-21ec-4ccf-9f55-af442c1dba7f
# ╟─85c037d0-c584-4cfe-a3af-726810a9660e
# ╟─5db3fc14-31c7-40f6-abe3-f2ee3f4f0bf7
# ╟─8f4a4084-ea4c-4111-89e9-4c9cd476d975
# ╟─608ca8ac-4dd9-484c-b2e7-52dceb9c1461
# ╟─6915ef51-2ea5-4705-bc52-42c89205b48a
# ╟─a5882dd9-f4d3-46c9-9096-743671bf13fb
# ╟─63b7fefa-729a-45b0-9ebf-754769d77e08
# ╟─0c1fb2c7-987f-4a2f-8b63-488e6c177de6
# ╟─035068e5-3257-4d09-b063-dcfc836d214f
# ╟─350e7132-8200-4a1d-8d4e-98316373dc53
# ╟─e9cfe0c6-9a57-49b5-953f-448e3d4ee0db
# ╟─ff881d8c-81f8-4a1f-afae-308d7289b601
# ╟─8bbbf78b-7dfd-4bff-a95b-5a95052d6786
# ╟─18b3310a-2571-4a4b-91f1-38a1fa047d56
# ╟─037ea602-4198-46af-a471-89abda06ec7c
# ╟─599634bc-09a5-4815-ae58-432211ea053d
# ╟─c1b7b04e-eb0e-4893-82f0-837df1912cc1
# ╟─6f25ccbb-039e-421a-88a0-0a5396b7987a
# ╟─ff26ee4a-8624-45d9-8bff-4f2b49c10c40
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
