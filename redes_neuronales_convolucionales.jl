### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 49bc4ae1-cc37-4de1-8616-d5bd2cc740ad
using PlutoUI

# ╔═╡ 98d86666-f5c0-4389-8b8c-79291db1b57d
using ShortCodes

# ╔═╡ 17163d32-2fd3-11f0-053d-7b01c1dc1e5d
# html"""
# <link rel="stylesheet" type="text/css" href="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/mi_estilo.css" media="screen" />
# """

# ╔═╡ 98925bf9-6cfa-4046-b0c7-de0a0ad286ad
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ bc7e1700-5d03-4493-84d7-b96a7da32d1b
md"""
# Redes neuronales convolucionales

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)

![](https://ujiapps.uji.es/ade/rest/storage/63c07717-5208-4240-b688-aa6ff558b466?guest=true)
"""

# ╔═╡ c2db677d-1029-47ff-ac13-309d586ecc69
md"""
## Introducción

* Las redes neuronales convolucionales están inspiradas en cómo funciona la visión humana.
* Las redes convolucionales trabajan con imágenes, aunque se pueden utilizar en otros ámbitos.
* Las redes convolucionales son capaces de «reconocer» la relación entreregiones de una imagen, también temporales si lo que se analiza es unasecuencia de vídeo.
"""

# ╔═╡ d8736fb8-a6a9-4e4d-bd28-509b2fdb6336
md"""
## Objetivos de aprendizaje

* Resumir en qué consisten las principales tareas de la visión por computador.
* Definir las bases de las redes neuronales convolucionales: convoluciones y pooling.
* Construir una red neuronal convolucional utilizando Keras y Tensorflow.
* Valorar si existe sobreajuste durante el proceso de entrenamiento.
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

1. [Deep Learning](http://www.deeplearningbook.org), Ian Goodfellow, Joshua Bengio and Aaron Courbille.
1. [Dive into deep learning](https://d2l.ai/index.html), Aston Zhang et al.
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
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/computer_vision.jpg",
	:alt => "Computer Vision Book",
	:width => 600
)

# ╔═╡ 0dd08d60-2b9d-41d0-bcc8-927853f7c962
md"""
## Introducción

Utilizar redes densas en visión por computador es impracticable. El tamaño de 
una imagen, en píxeles, determina el número de neuronas en la capa de entrada 
y posteriores.

El número de parámetros que es necesario entrenar es muy elevado.
"""

# ╔═╡ 799ac9d4-2151-4474-a98e-1476e8bb1033
md"""
## Introducción
Las redes neuronales convolucionales están _inspiradas_ en el funcionamiento 
de la percepción de la visión:
"""

# ╔═╡ a6aa59f8-f25f-402f-a9e7-9553235f7303
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/funcionamiento_vision.png",
	:alt => "Esquema del funcionamiento de la visión.",
	:width => 900
)

# ╔═╡ ee29cc12-7007-4f16-88a0-1dd4ed894e78
md"""
## Introducción

Las redes convolucionales proponen una solución basada en dos elementos:

1. Utilizar convoluciones para **aprender** las características relevantes de 
las imágenes.
1. Utilizar **pooling** para reducir el tamaño de las imágenes.

Además, en muchos casos las últimas capas de una red 
convolucional están formadas por capas densas.

Esto permite utilizar redes neuronales en las tareas típicas de la visión por 
computador.
"""

# ╔═╡ 4e1c05fb-ebf4-4397-9ceb-838586369223
md"""
## Introducción

Las tareas típicas dentro de la visión por computador son:

* Clasificación.
* Ubicación.
* Detección de objetos.
* Seguimiento de objetos.
* Segmentación semántica.
"""

# ╔═╡ d59746f5-4d77-45cc-b57e-f70c7bf12c2e
md"""
## Clasificación

Un modelo de clasificación está entrenado de tal modo que 
es capaz de asignar una clase, o etiqueta, a una nueva muestra.
"""

# ╔═╡ 1c6b168c-5a43-42e4-8691-35b68b31b215
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/gato.jpg" alt="drawing" width="300"/>
Gato

<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/french-bulldog.jpg" alt="drawing" width="300"/>
Perro
"""

# ╔═╡ 57dad7fb-d2cf-4063-a7df-1d66216ec1c6
md"""
## Detección de objetos

Encontrar todos los objetos dentro de una imagen:
"""

# ╔═╡ 7b374ffc-f241-4e6f-ba99-920472d80b1c
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/gato_ubicado.png" alt="drawing" width="300"/>

"""

# ╔═╡ 0eaa405e-3a79-4c17-b9ea-52b16c1be54b
md"""
## Detección de objetos
Encontrar todos los objetos dentro de una imagen:
"""

# ╔═╡ 346be8c7-3fb2-48ed-97bb-d24a4909e12b
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/familia.png" alt="drawing" width="900"/>
"""

# ╔═╡ 60734849-05bb-4c6a-a8e3-9bde5ab210b4
md"""
## Seguimiento de objetos
"""

# ╔═╡ 6d895aff-98da-485a-9a4e-43f9862a5630
YouTube("cHDLvp_NPOk?si=EhYOh0ntyiWEsbJs")

# ╔═╡ 1b8609a0-11b8-4695-9de2-bc87dbc7ec92
md"""
## Segmentación semántica

Asigna cada pixel a uno de los objetos detectados.
"""

# ╔═╡ fa2e68a3-2921-4694-a203-79d63c744a73
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/familia_segmentada.png",
	:alt => "Imagen segmentada.",
	:width => 900
)

# ╔═╡ 96ada7c9-c5d1-48b9-a76f-bcc41cee12a7
md"""
# Fundamentos de la CNN
"""

# ╔═╡ ab9c62de-7781-4876-a2cc-7a42c501b447
md"""
## Convoluciones

Una convolución es una operación entre los píxeles de una imagen.
En estos dos casos, el paso es de 1 pixel. La figura de la derecha muestra 
la opción con márgenes añadidos.

"""

# ╔═╡ 2b191760-6763-4834-a5bc-e11352c10e3a
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/no_padding_no_strides.gif" alt="drawing" width="300"/>

<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/same_padding_no_strides.gif" alt="drawing" width="300"/>
"""

# ╔═╡ 2a12e686-6cb7-40a8-9907-90a970ef175b
md"""
Fuente: Vincent Dumoulin, Francesco Visin - A guide to convolution arithmetic for deep learning
"""

# ╔═╡ 4f342e4f-8cb2-4888-a238-ae4e024ad881
md"""
## Convoluciones

En este otro caso, el paso es de 2 píxeles.
"""

# ╔═╡ a82f3ec5-c8b6-405e-aeef-0a763d55388c
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/no_padding_strides.gif" alt="drawing" width="300"/>

<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/padding_strides.gif" alt="drawing" width="300"/>
"""

# ╔═╡ 9d24e406-7115-4879-9541-09db6339fd49
md"""
## Convoluciones

Las imágenes en color tienen tres canales: rojo, verde y azul. Las convoluciones 
pueden ser distintas para cada uno de los canales.
"""

# ╔═╡ 34ca8253-22af-480e-8aba-9071bf85fd1c
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/convolucion_imagen_color.gif",
	:alt => "Convolución imagen color",
	:width => 900
)

# ╔═╡ 1ed0f5a2-754f-41ee-a362-6884c7ab20cf
md"""
Fuente: https://towardsdatascience.com/a-comprehensive-guide-to-convolutional-neural-networks-the-eli5-way-3bd2b1164a53
"""

# ╔═╡ 68be7ffe-1b21-4ea1-89f6-286217450020
md"""
## Convoluciones

Ejemplo de convolución que emborrona una imagen:

$\begin{equation}
\frac{1}{9}
\begin{bmatrix}
1 & 1 & 1 \\
1 & 1 & 1 \\
1 & 1 & 1 \\
\end{bmatrix}
\end{equation}$

"""

# ╔═╡ 12dfafa3-4c1e-4327-b3ad-ef1e14a5ceb7
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/damero.jpg" alt="drawing" width="300"/>

<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/damero_emborronado.png" alt="drawing" width="300"/>
"""

# ╔═╡ 28b85f5d-ad91-4ac6-9874-552ae80fd5bf
md"""
Fuente: Wikipedia
"""

# ╔═╡ 47934a59-a752-4b5c-9ac3-ca603aa6a639
md"""
## Convoluciones

Ejemplo de convolución que resalta los bordes:

$\begin{equation}
\frac{1}{9}
\begin{bmatrix}
0 & -1 & 0 \\
-1 & 4 & -1 \\
0 & -1 & 0 \\
\end{bmatrix}
\end{equation}$
"""

# ╔═╡ 3276ff69-7257-4162-957f-2d84469f02ec
html"""
<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/damero.jpg" alt="drawing" width="300"/>

<img src="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/damero_resaltado.png" alt="drawing" width="300"/>
"""

# ╔═╡ e6c81043-fc99-4367-9999-be72c91d9e5f
md"""
Fuente: Wikipedia
"""

# ╔═╡ 1ea47d14-3d4e-4414-808a-ca6c0e6a4f89
md"""
## Convoluciones

Lo que va a _aprender_ la red convolucional es a ajustar los valores de las 
matrices de convolución:

$\begin{equation}
\begin{bmatrix}
m_{1,1} & m_{1,2} & m_{1,3} \\
m_{2,1} & m_{2,2} & m_{2,3}\\
m_{3,1} & m_{3,2} & m_{3,3} \\
\end{bmatrix}
\end{equation}$
"""

# ╔═╡ 2df53044-e390-43dc-b7f0-c34f7bd523c0
md"""
## Show me the code

Para instanciar una capa convolucional en Flux

```{julia}
Conv(filter, in => out, σ = identity;
     stride = 1, pad = 0, dilation = 1, groups = 1, [bias, init])

Conv((5,5), 3 => 7, relu; stride=2, pad=SamePad())
```
"""

# ╔═╡ 80727712-f9a7-44c4-9996-bd6d15f6819c
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/capas_convoluciones.png",
	:alt => "Imagen segmentada.",
	:width => 400
)

# ╔═╡ 0da704ef-90ea-43ff-ac22-7c84d84c0ed0
md"""
## Show me the code

**pad**: se añaden píxeles nulos para que la imagen de la salida sea 
$ancho_{entrada}/salto$.

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/padding.png)

Fuente: Hands-on Machine Learning... Aurélien Gèron
"""

# ╔═╡ b8d7a851-babb-4faf-b7a5-a3150b13a516
md"""
## Pooling
El número de parámetros que hay que entrenar crece al aplicar capas de 
convoluciones. Una manera de reducir el número de parámetros es aplicar capas
de pooling:

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/pooling.png)
"""

# ╔═╡ 5e47956e-2e46-44c0-bf55-74e030139565
md"""
## Show me the code

Flux nos proporciona distintas capas de Pooling. Para utilizar polling máximo:

```{julia}
MaxPool((n,m))
```

Para utilizar pooling promedio:

```{julia}
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

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/arquitectura_cnn.png)

Al final hay una estructura de capas totalmente conectadas.

Fuente: Hands-on machine learning, Aurélien Géron
"""

# ╔═╡ 410abb9f-56d4-4252-a353-4c1d550f36cd
md"""
## LeNet5

Esta es la arquitectura original presentada por 
[Yann Le-Cunn et al](https://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf).

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/lenet5.png)

Las **Gaussian connections** del final son la activación **softmax**.

Fuente: Yann LeCunn et al.
"""

# ╔═╡ 8692bcdf-059e-45d6-baab-1af4705a72d4
md"""
## LeNet5

La implementación de la red:

```{julia}
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

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/mnist.png)

Fuente: Wikipedia
"""

# ╔═╡ d23085f7-1f1e-4abf-8c96-a64a08988328
md"""
## LeNet5

Si entrenamos LeNet5 con el conjunto MNIST, utilizando como función de activación la función tanh y como optimizador Stocastic Gradient Descent (SGD), durante 20 épocas, obtenemos una precisión sobre el conjunto de pruebas del 97,01%!!!.
"""

# ╔═╡ 619cde75-694d-4a37-92be-a6793aa9f2d6
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/entrenamiento_lenet5_tanh_sgd.svg",
	:alt => "Entrenamiento LeNet5",
	:width => 600
)

# ╔═╡ 72ec224f-54ca-4acb-a4d1-c3f9881cf299
md"""
## LeNet5

La matriz de confusión:
"""

# ╔═╡ ed9d8902-ddd6-4099-9861-ca8a242e2d4b
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/matriz_confusion_lenet5_tanh_sgd.png",
	:alt => "Matriz de confusión LeNet5 sobre MNIST tanh, sgd, 20 épocas",
	:width => 500
)

# ╔═╡ 793035a9-f596-491d-bfd9-ed418c171b7d
md"""
## LeNet5

Si utilizamos como función de activación 
la función _relu_ y como optimizador Adam, durante 
20 épocas, obtenemos una precisión sobre el conjunto de pruebas de casi el 98,74%. 
"""

# ╔═╡ c93848ac-1597-4212-82d8-e9c3bf935c04
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/entrenamiento_lenet5_relu_adam.svg",
	:alt => "Entrenamiento LeNet5",
	:width => 600
)

# ╔═╡ d91c36f5-5aaa-4086-8bfb-384ec9a1a200
md"""
## LeNet5

La matriz de confusión:
"""

# ╔═╡ cca48072-8bd6-4993-aba2-d4e8861cb3a1
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/matriz_confusion_lenet5_relu_adam.png",
	:alt => "Matriz de confusión LeNet5 sobre MNIST relu, adam, 20 épocas",
	:width => 500
)

# ╔═╡ 528c23a5-3a0a-4fcc-99eb-90d9a01ff3df
YouTube("https://www.youtube.com/watch?v=FwFduRA_L6Q")

# ╔═╡ e35e23ec-fb8a-480b-9904-c0fd95e5fe73
md"""
## ResNet

[ResNet](https://arxiv.org/pdf/1512.03385) introdujo una novedad en su diseño 
para reducir el problema del desvanecimiento del gradiente. La novedad es 
en _saltar_ capas:

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/resnet.png)

$H(x) = F(x) + x \rightarrow F(x) = H(x) - x$

"""

# ╔═╡ d2f9757f-68eb-4c4c-8589-4df4bafa830e
md"""
## ResNet
Lo que aprenden las dos capas intermedias de la figura (representadas por la 
función $F(x)$) es a ajustar la diferencia entre la entrada real a la capa y 
el resultado de la aplicación de la capa.

![](https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/resnet.png)

"""

# ╔═╡ 9b38cde2-1c0a-4871-ab3a-874fca8ac403
md"""
### Metalhead

[Metalhead](https://fluxml.ai/Metalhead.jl/stable/) es una biblioteca en Julia que implementa algunos de los más conocidos algoritmos en visión por computador.

Los modelos de Metalhead se han entrenado con la base de datos [ImageNet](https://www.image-net.org/) que es un conjunto de datos con
1.000 clases , 1.281.167 imágenes de entrenamiento, 50.000
imágenes de validación y 100.000 imágenes de prueba.

"""

# ╔═╡ 0c1094af-ddf9-4206-bb79-0a5146f792e1
md"""
## Show me the code

```{julia}
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
"""

# ╔═╡ 9bbd7f5a-7279-42c4-977f-753cee52be32
md"""
# YOLO - You Only Look Once
"""

# ╔═╡ bea79739-21ec-4ccf-9f55-af442c1dba7f
md"""
## YOLO

You Only Look Once (YOLO) es una red convolucional con un gran desempeño en 
muchas de la tareas típicas en visión por computador.

YOLO es mantenida por la empresa [Ultralytics](https://www.ultralytics.com/es).

Este es el [artículo](https://arxiv.org/abs/1506.02640) donde se presentó esta 
arquitectura de red neuronal.
"""

# ╔═╡ 85c037d0-c584-4cfe-a3af-726810a9660e
md"""
## YOLO
Y esta es la arquitectura tal y como se presenta en el artículo:

![](Imagenes/yolo.png)

"""

# ╔═╡ 5db3fc14-31c7-40f6-abe3-f2ee3f4f0bf7
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/yolo.png",
	:alt => "Arquitectura de YOLO",
	:width => 700
)

# ╔═╡ 6915ef51-2ea5-4705-bc52-42c89205b48a
md"""
## YOLO
Una de la novedades de YOLO es que a la salida propociona la caja envolvente 
de los objetos detectados, así como la probabilidad de su clasificación, y 
también es capaz de realizar segmentación semántica.

YOLO se presenta en varias versiones, basándose en el número de parámetros 
presente en la red (tamaño). Las versiones más pequeñas tienen en desempeño 
más bajo que las versiones grandes, pero, como contrapartida, se pueden 
ejecutar en dispositivos con hardware más modesto.

Todas las versiones están entrenadas con el conjunto de datos 
[COCO](https://cocodataset.org/#home).
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"

[compat]
PlutoUI = "~0.7.61"
ShortCodes = "~0.3.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.5"
manifest_format = "2.0"
project_hash = "c674d145238854ec6334af6c0ca437b0cf3b05f9"

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
version = "1.1.1+0"

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

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.6.0+0"

[[deps.LibGit2]]
deps = ["Base64", "LibGit2_jll", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"
version = "1.11.0"

[[deps.LibGit2_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll"]
uuid = "e37daf67-58a4-590a-8e99-b0245dd2ffc5"
version = "1.7.2+0"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.11.0+1"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"
version = "1.11.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

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
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

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
version = "2023.12.12"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "7d2f8f21da5db6a806faf7b9b292296da42b2810"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.3"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

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
version = "1.2.13+1"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"
"""

# ╔═╡ Cell order:
# ╠═17163d32-2fd3-11f0-053d-7b01c1dc1e5d
# ╠═49bc4ae1-cc37-4de1-8616-d5bd2cc740ad
# ╠═98d86666-f5c0-4389-8b8c-79291db1b57d
# ╠═98925bf9-6cfa-4046-b0c7-de0a0ad286ad
# ╠═bc7e1700-5d03-4493-84d7-b96a7da32d1b
# ╠═c2db677d-1029-47ff-ac13-309d586ecc69
# ╠═d8736fb8-a6a9-4e4d-bd28-509b2fdb6336
# ╠═ce478b27-9250-41ff-898b-c887aa166ed3
# ╠═7900f0f7-3ec0-434e-a6a8-2b6f381d3328
# ╠═a0bbbc75-a852-47cf-88d5-982d03eb5c06
# ╠═50a1c924-0a84-4cf0-99b7-476cb4b4afe7
# ╠═8bdb7e0f-dece-4b09-a4e1-f32ee3f7601f
# ╠═0dd08d60-2b9d-41d0-bcc8-927853f7c962
# ╠═799ac9d4-2151-4474-a98e-1476e8bb1033
# ╠═a6aa59f8-f25f-402f-a9e7-9553235f7303
# ╠═ee29cc12-7007-4f16-88a0-1dd4ed894e78
# ╠═4e1c05fb-ebf4-4397-9ceb-838586369223
# ╠═d59746f5-4d77-45cc-b57e-f70c7bf12c2e
# ╠═1c6b168c-5a43-42e4-8691-35b68b31b215
# ╠═57dad7fb-d2cf-4063-a7df-1d66216ec1c6
# ╠═7b374ffc-f241-4e6f-ba99-920472d80b1c
# ╠═0eaa405e-3a79-4c17-b9ea-52b16c1be54b
# ╠═346be8c7-3fb2-48ed-97bb-d24a4909e12b
# ╠═60734849-05bb-4c6a-a8e3-9bde5ab210b4
# ╠═6d895aff-98da-485a-9a4e-43f9862a5630
# ╠═1b8609a0-11b8-4695-9de2-bc87dbc7ec92
# ╠═fa2e68a3-2921-4694-a203-79d63c744a73
# ╠═96ada7c9-c5d1-48b9-a76f-bcc41cee12a7
# ╠═ab9c62de-7781-4876-a2cc-7a42c501b447
# ╠═2b191760-6763-4834-a5bc-e11352c10e3a
# ╠═2a12e686-6cb7-40a8-9907-90a970ef175b
# ╠═4f342e4f-8cb2-4888-a238-ae4e024ad881
# ╠═a82f3ec5-c8b6-405e-aeef-0a763d55388c
# ╠═9d24e406-7115-4879-9541-09db6339fd49
# ╠═34ca8253-22af-480e-8aba-9071bf85fd1c
# ╠═1ed0f5a2-754f-41ee-a362-6884c7ab20cf
# ╠═68be7ffe-1b21-4ea1-89f6-286217450020
# ╠═12dfafa3-4c1e-4327-b3ad-ef1e14a5ceb7
# ╠═28b85f5d-ad91-4ac6-9874-552ae80fd5bf
# ╠═47934a59-a752-4b5c-9ac3-ca603aa6a639
# ╠═3276ff69-7257-4162-957f-2d84469f02ec
# ╠═e6c81043-fc99-4367-9999-be72c91d9e5f
# ╠═1ea47d14-3d4e-4414-808a-ca6c0e6a4f89
# ╠═2df53044-e390-43dc-b7f0-c34f7bd523c0
# ╠═80727712-f9a7-44c4-9996-bd6d15f6819c
# ╠═0da704ef-90ea-43ff-ac22-7c84d84c0ed0
# ╠═b8d7a851-babb-4faf-b7a5-a3150b13a516
# ╠═5e47956e-2e46-44c0-bf55-74e030139565
# ╠═8aea4bec-4f53-406b-8880-ce00746d08ec
# ╠═574816f0-f6d9-4208-a1fd-12cabf07df40
# ╠═410abb9f-56d4-4252-a353-4c1d550f36cd
# ╠═8692bcdf-059e-45d6-baab-1af4705a72d4
# ╠═85382e11-cb0f-4e85-bb48-ec21714a230c
# ╠═d23085f7-1f1e-4abf-8c96-a64a08988328
# ╠═619cde75-694d-4a37-92be-a6793aa9f2d6
# ╠═72ec224f-54ca-4acb-a4d1-c3f9881cf299
# ╠═ed9d8902-ddd6-4099-9861-ca8a242e2d4b
# ╠═793035a9-f596-491d-bfd9-ed418c171b7d
# ╠═c93848ac-1597-4212-82d8-e9c3bf935c04
# ╠═d91c36f5-5aaa-4086-8bfb-384ec9a1a200
# ╠═cca48072-8bd6-4993-aba2-d4e8861cb3a1
# ╠═528c23a5-3a0a-4fcc-99eb-90d9a01ff3df
# ╠═e35e23ec-fb8a-480b-9904-c0fd95e5fe73
# ╠═d2f9757f-68eb-4c4c-8589-4df4bafa830e
# ╠═9b38cde2-1c0a-4871-ab3a-874fca8ac403
# ╠═0c1094af-ddf9-4206-bb79-0a5146f792e1
# ╠═9bbd7f5a-7279-42c4-977f-753cee52be32
# ╠═bea79739-21ec-4ccf-9f55-af442c1dba7f
# ╠═85c037d0-c584-4cfe-a3af-726810a9660e
# ╠═5db3fc14-31c7-40f6-abe3-f2ee3f4f0bf7
# ╠═6915ef51-2ea5-4705-bc52-42c89205b48a
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
