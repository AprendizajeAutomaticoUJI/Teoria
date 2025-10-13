### A Pluto.jl notebook ###
# v0.20.8

using Markdown
using InteractiveUtils

# ╔═╡ 8da42b7b-7bc8-4bdc-9707-539209d79d1b
using TextAnalysis

# ╔═╡ efc09298-51f8-4013-9677-497b6de474be
using PlutoUI

# ╔═╡ 367d5f39-3928-4709-8de9-2f4b1f046acd
using ShortCodes

# ╔═╡ c2fd34b8-66db-11f0-3efd-21960f239f79
# html"""
# <link rel="stylesheet" type="text/css" href="https://belmonte.uji.es/Docencia/IR2130/Teoria/mi_estilo.css" media="screen" />
# """

# ╔═╡ 2a803c80-dece-4f52-9898-1cc3ba15ebbc
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ e05614d4-5762-4839-b657-1c98b2054de4
imagenes = "https://belmonte.uji.es/Docencia/IR2130/Teoria/RedesNeuronales/Imagenes/"

# ╔═╡ 9e00079f-b241-4c7a-9c24-b229e08428df
md"""
# Procesamiento del lenguaje natural

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 80774b7b-15eb-45a4-b220-c73bdb2da29b
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ b72590de-c5d6-422d-8a7d-572cc3b1c3c3
md"""
## Objetivos de aprendizaje

* Resumir las tareas principales del Procesamiento del Lenguaje Natural.
* Interpretar el fundamento de los Autoencoders.
* Argumentar cómo se construye un Embedding con la técnica de los Autoencoders.
* Construir o adaptar un Embedding.
* Razonar cómo funcionan las redes recurrentes en la tarea de generación de texto.
* Construir o adaptar una red recurrente para la generación de texto.
"""

# ╔═╡ 7128e86b-bdb9-4e7a-84ba-ee8682f8ae4f
md"""
## Objetivos de aprendizaje

* Razonar cómo funciona una red neuronal recurrente para la tarea de clasificación de texto.
* Construir o adaptar una red recurrente para la clasificación de texto.
* Razonar cómo funciona una red neuronal recurrente para la tarea de traducción automática.
* Construir o adaptar una red recurrente para traducción automática.
* Razonar cómo funciona la arquitectura Transformer.

"""

# ╔═╡ 1be76676-2de5-4df0-9e51-e95ce80c4ece
md"""
# Introducción
"""

# ╔═╡ 9476c68a-19fb-4855-9c6e-3d2cf1e9e403
md"""
## Procesamiento del Lenguaje Natural 
El Procesamiento del Lenguaje Natural (PLN, NLP - Natural Language Processing) es un campo de la inteligencia artificial que se ocupa del estudio y desarrollo de algoritmos y modelos capaces de analizar, generar y transformar el lenguaje humano en forma escrita o hablada. 

Su objetivo principal es permitir que las máquinas procesen el lenguaje natural de manera que sea útil y efectiva para interactuar con las personas o extraer conocimiento a partir de textos.
"""

# ╔═╡ 7aa6b29c-3e43-4b2a-9323-655f90835e66
md"""
## Tareas en NLP

Algunas de las tareas básicas en NLP que vamos a ver en esta presentación son:

* Generación de texto.
* Clasificación de texto.
* Traducción automática.

Antes de entrar con NLP, vamos a ver algunas técnicas relacionadas.
"""

# ╔═╡ 0679cc8f-5779-41ad-ac5d-13f1a303b0dc
md"""
## Bibliografía

1. [Deep Learning](http://www.deeplearningbook.org), Ian Goodfellow, Joshua Bengio and Aaron Courbille.
"""

# ╔═╡ 70676122-eb80-4b6d-9c78-e489292c3f5c
md"""
# Autoencoders
"""

# ╔═╡ 2ea8cba5-3286-4d26-8bbe-76b7e1ba868f
md"""
## Introducción

El objetivo de un autoencoder es replicar la entrada en la salida pasando por un espacio de menor dimensión que el espacio de partida:
"""

# ╔═╡ 813ecbd6-74b8-4d26-a92d-c6098044e64c
Resource(
	imagenes * "autoencoder2.png",
	:alt => "Estructuras redes neuronales recurrentes",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 116bf570-3a18-4789-b37c-18755c401f5d
html"""
Fuente: https://lilianweng.github.io/posts/2018-08-12-vae/
"""

# ╔═╡ 15e631ed-dcbd-4a09-a020-474671543c26
md"""
## Introducción

Lo que obtenemos en el espacio intermedio es una representación compacta de la información en el espacio original.

Fíjate en que el entrenamiento es no supervisado, sólo necesitamos muestras de entrenamiento, no es necesario que estén etiquetadas.

Teóricamente, las funciones entre las capas podrían ser tan potentes que conun espacio intermedio de dimensión 1 pudiésemos reconstruir la imagen original. Seria como si el autoencoder asignase un _índice_ que le permitiese recuperarla imagen inicial.

Desafortunadamente, en la práctica, este caso no se da.
"""

# ╔═╡ c0a6ed84-c2e1-4778-9698-3c1a346cc6bf
md"""
## Introducción

El trabajo de un autoencoder parece una tarea sencilla, pero tienen muchas aplicaciones:

* Detectar anomalías.
* Reconstruir imágenes.
* Limpiar imágenes.

"""

# ╔═╡ e2c82807-91c9-422f-8c68-e9550d25f239
md"""
## Detectar anomalías

Entrenamos con un conjunto de muestras sin anomalías. Si al autoencoder llega un dato anómalo la codificación intermedia será incapaz de reconstruir el dato a la salida (gran diferencia entre entrada y salida), y es dato lo podemos considerar anómalo.
"""

# ╔═╡ bfbe0854-0fc8-4f8f-9286-977e734cab12
Resource(
	imagenes * "cern.png",
	:alt => "CERN",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 1c8635e9-5f6e-4005-95d3-a3811498a179
html"""
Fuente: CERN
"""

# ╔═╡ 9c9997ea-e050-4960-b656-3048091b0035
md"""
## Reconstruir imágenes

A partir de una imagen donde falta información, reconstruirla.
"""

# ╔═╡ ac9ef535-5f67-4d49-a56d-fe2bd44993f6
Resource(
	imagenes * "autoencoder_reconstruccion.png",
	:alt => "Reconstruir imágenes",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 4d9883fe-1ead-4d07-ad35-5c8d935412b5
html"""
Fuente: Hands-on machine learning - Aurélien Géron
"""

# ╔═╡ 5c217c15-be06-464e-a1fc-577d6059205c
md"""
## Limpiar imágenes

Limpiar imágenes con ruido.
"""

# ╔═╡ bc49d4cb-ab79-4fa0-809f-fb3ba82e5adc
Resource(
	imagenes * "denoising_autoencoder.png",
	:alt => "Limpiar imágenes",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ d2c96070-c07f-49c4-84f5-bebf7498b314
html"""
Fuente: By neerajkrbansal1996 - https://github.com/neerajkrbansal1996/Denoising-autoencoder, CC0, https://commons.wikimedia.org/w/index.php?curid=128249938
"""

# ╔═╡ a1ff9ae5-a131-40c2-833a-4f17fb765d1b
md"""
# Embeddings
"""

# ╔═╡ 22ec98af-c189-46ab-99a1-5cd8d51a77a2
md"""
## Introducción

La representación de palabras en un ordenador para realizar cálculos no es trivial.

Usualmente, el primer paso suele ser trabajar con un vocabulario restringido en vez de con todas las palabras posible del lenguaje. La codificación de una palabra que no existe en el vocabulario se sustituye por un código **UNKNOWN**. En la frase:

_Puso cara de pasmo al oír la noticia_

si la palabra **pasmo** no figura en el vocabulario que manejamos se sustituye por **UNKNOWN**.

_Puso cara de **UNKNOWN** al oír la noticia_
"""

# ╔═╡ c31ee640-cdba-4469-b755-a6406e101015
md"""
## One-hot encoding

Cada palabra se representa por un vector de dimensión igual al número de palabras del vocabulario. Se activa el bit correspondiente a su índice.
"""

# ╔═╡ 8c5cd066-9fa7-4c62-beed-e858d0f3a311
Resource(
	imagenes * "one_hot_encoding.webp",
	:alt => "One hot encoding",
	:width => 900,
	:style => "display: block; margin: auto;",
	
)

# ╔═╡ 87d0f2f2-eaea-4a7c-bca9-ceabab0da099
md"""
Fuente: Medium
"""

# ╔═╡ 6744bcec-8e4a-420f-aeb9-34de80d0404c
md"""
## Show me the code

En julia puedes utilizar trucos para obtener la matriz one-hot-encoding:
"""

# ╔═╡ 767aaf9a-1984-404a-ae79-a74fc919861d
capitales = ["Houston", "Rome", "Madrid", "London"]

# ╔═╡ cc042ada-03d1-4189-abf0-9337546e6bec
capitales .== permutedims(capitales)

# ╔═╡ 1ebf121c-e8c1-45f8-954f-a7ad10157228
md"""
## Embeddings

Los embeddings son el cojunto de técnicas que se utilizan en NLP para representar texto (palabras, frases, parte de un texto en general) como vectores de números que los algoritmos de aprendizaje automático pueden manejar.

Para conseguir representaciones de texto (embeddings) lo más compactas posibles, se intentar reducir la dimensionalidad del espacio de partida (número de palabras en el vocabulario), a un espacio que mantenga la información y se más fácil de manejar.

El espacio de partida en esta caso en utilizar codificación _one-hot encoding_.
"""

# ╔═╡ f4db6144-ef66-414c-9854-c05d0dad8ef3
md"""
## Embeddings
"""

# ╔═╡ ac928fe1-5926-44d7-912e-be19f5128b84
Resource(
	imagenes * "one_hot_encoding2.png",
	:alt => "One hot encoding",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ bafdc086-0b06-4f33-8fbe-a8933c82ee47
md"""
Cada uno de los vectores de nuestro espacio _one-hot encoding_ es _disperso_, muchas de sus componentes están vacías, no aportan información.

Es deseable reducir la dimensión del espacio sin perder información.
"""

# ╔═╡ 7bd2726c-6c8a-48d9-b1ae-eed8bcc88156
md"""
## Embeddings
"""

# ╔═╡ 4c9944c1-60bf-4e57-a0c9-6219baeb2313
Resource(
	imagenes * "autoencoder.png",
	:alt => "Autoencoder",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 92c5efbe-6148-4813-8e28-ddd73808b6e1
md"""
En **h** vamos a tener una representación _densa_ de la información en el espacio _one-hot encoding_.
"""

# ╔═╡ 1c7fc9ef-5259-498b-80ea-b8a21d4b2fc5
md"""
## Embeddings

En **Julia** existe una capa para crear embeddings:

```julia
Embedding(tam_vocabulario => tam_embedding)
```

Donde **tam_vocabulario** indica el número de palabras en nuestro vocabulario, y **tam_embedding** las dimensiones del embedding.

Esta capa se puede entrenar de manera aislada o puede formar parte de una red neuronal.
"""

# ╔═╡ 7f1ebe1e-7e9c-480a-aad5-23e6d70607cc
md"""
## Word2vec

[Word2vec](https://arxiv.org/pdf/1301.3781) es un embedding muy utilizado, desarrollado por Tomas Mikolov. Se puede entrenar de dos modos:
"""

# ╔═╡ f7129b01-7d93-41f9-a5f2-526ed80311af
Resource(
	imagenes * "cbow.png",
	:alt => "Continuous bag of words",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 732ca76c-97ee-4762-9cd5-cd76752391f5
md"""
Continuous Bag of Words 

Pineapples are **spikey** and yellow

Fuente: https://community.alteryx.com/t5/Data-Science/Word2vec-for-the-Alteryx-Community/ba-p/305285
"""

# ╔═╡ 9c9247a8-fb07-4924-8fa3-445d3f4ade54
md"""
## Word2vec

[Word2vec](https://arxiv.org/pdf/1301.3781) es un embedding muy utilizado, desarrollado por Tomas Mikolov. Se puede entrenar de dos modos:
"""

# ╔═╡ 7f09d56e-77b3-45e5-ac5a-61f80ad82b39
Resource(
	imagenes * "skip_gram.png",
	:alt => "Skip-gram",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ be14b504-bf76-4053-8098-71c791108151
md"""
Skip-gram

Pineapples are **spikey** and yellow

Fuente: https://community.alteryx.com/t5/Data-Science/Word2vec-for-the-Alteryx-Community/ba-p/305285
"""

# ╔═╡ e31a02d0-d16a-4f11-a767-050139079635
md"""
## Word2vec

Una primera característica impresionante de este embedding es que, si lo proyectamos a un espacio de 3 dimensiones veremos que conceptos similares se encuentran cerca unos de otros.

[Visualización de embeddings.](https://projector.tensorflow.org/)
"""

# ╔═╡ bee075f8-65e2-4991-ae10-e749d1119bb2
md"""
## Word2vec

Otra característica no menos impresionante, es que podemos hacer operaciones entre los vectores del espacio de Word2vec:

**king - male +female = queen** (tiene embebida la semántica de realiza)

**Spanish - country + Italy = Italian** (tiene embebida la semántica de lengua)

**dogs - dog + cat = cats** (tiene embebido el sentido de pluralidad)

"""

# ╔═╡ 7a6bf294-4694-43d9-b4b6-5f085dddf29f
md"""
## Show me the code

En Julia podemos trabajar con embeddings utilizando el paquete **Embeddings**, que dispone de embeddings ya entrenados, algunos de ellos en español y catalán.

```{.julia}
using Embeddings

# Para conocer los embeddings que tenemos disponibles
language_files(FastText_Text{:es})
```

```{.shell}
4-element Vector{String}:
 "FastText es CommonCrawl Text/cc.es.300.vec"
 "FastText es Wiki Text/wiki.es.vec"
 "FastText es CommonCrawl Text/cc.es.300.vec"
 "FastText es Wiki Text/wiki.es.vec"
```

```{.julia}
# Para cargar el primero de los embeddings
español = load_embeddings(FastText_Text{:es}, 1)
```
"""

# ╔═╡ 4ca5b39c-e180-4956-bbbc-3ff3b51aa1a4
md"""
## Show me the code

Para obtener el vector de embedding de una palabra en el diccionario puedes utilizar el código de ejemplo que encuentras en la documentación del paquete:

```{.julia}
diccionario = Dict(palabra => indice for (indice, palabra) in enumerate(español.vocab))

function obten_embedding(palabra)
    indice = diccionario[palabra]
    return español.embeddings[:,indice]
end

# Una función de ayuda
function coseno(v1::Vector{Float32}, v2::Vector{Float32})
       return dot(v1, v2) / (sqrt(dot(v1, v1))*sqrt(dot(v2, v2)))
end
```
"""

# ╔═╡ ea93bd5e-d8ca-4d75-807e-bc4b315464ce
md"""
## Show me the code
Vamos a calcular el concepto de reina a partir de los conceptos mujer, rey, hombre, y calcular la distancia coseno entre el vector del embedding para reina y el calculado:

```{.julia}
rey = obten_embedding("rey")
reina = obten_embedding("reina")
hombre = obten_embedding("hombre")
mujer = obten_embedding("mujer")

reina2 = rey - hombre + mujer
coseno(reina, reina2)
rey2 = reina - mujer + hombre
coseno(rey, rey2)
```

Obtenemos:

```{.shell}
0.7006942f0 (45 grados)
0.6734288f0 (47 grados)
```
"""

# ╔═╡ bd0096b0-7059-4444-81c6-ee6daffa0392
md"""
## Show me the code
Vamos a calcular el concepto de gatos a partir de los conceptos perro, perros y gato, y calcular la distancia coseno entre el vector del embedding para gatos y el calculado:

```{.julia}
perro = obten_embedding("perro")
gato = obten_embedding("gato")
perros = obten_embedding("perros")
gatos = obten_embedding("gatos")

gatos2 = perros - perro + gato
coseno(gatos, gatos2)
```

Obtenemos:

```{.shell}
0.88137954f0 (28 grados)
```
"""

# ╔═╡ a94e7a01-d5d1-4b95-8e6c-bf1ad2b0b4e5
md"""
## Show me the code

Finalmente vamos a calcular el concepto de pluralidad a partir de perros y gatos, y compararlos:

```{.julia}
pluralidad_gato = gatos - gato
pluralidad_perro = perros - perro

coseno(pluralidad_gato, pluralidad_perro)
```

Obtenemos:

```{.shell}
0.86211735f0 (30 grados)
```
"""

# ╔═╡ 84f87d3f-d60e-4a95-a48c-e77a25874d0f
md"""
# Tareas en NLP
"""

# ╔═╡ e0917f33-9a77-439a-badf-66a3d04b248a
md"""
## Introducción

Algunas de las tareas típicas en NLP son:

* Generación de texto.
* Clasificación de texto.
* Traducción automática.

Vamos a ver cada una de ellas.
"""

# ╔═╡ 02bce938-3cf8-41fd-b8b5-da36d2bf3782
md"""
## Generación de texto

De manera análoga a la predicción del siguiente valor en secuencias de datos temporales, para generar texto podemos utilizar redes recurrentes (RNN).

Durante el entrenamiento vamos proporcionando a la red secuencias de letras como entrada, y la red debe predecir la siguiente letra, que se proporciona a la salida.

En este caso podemos utilizar una variación de las RNN, las Gated Recurrent Units (GRU).
"""

# ╔═╡ a601d1f6-50e8-460c-8b00-33c67ea535b6
md"""
## Generación de texto

Las celdas (neuronas) GRU (Gated Recurrent Unit) son una mejora de las neuronas LSTM con un rendimiento parecido, pero utilizan menos parámetros, y menos funciones internas para recordar fragmentos de texto.
"""

# ╔═╡ 547eda5f-ba82-41d7-b9d6-0160813fb3f8
Resource(
	imagenes * "gru.png",
	:alt => "Gated recurrent unit",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 92307ff3-745e-41ef-8491-cf6306fc9a64
md"""
## Generación de texto

Crear una red para generar texto con GRU es relativamente sencillo:

```{.python}
modelo = tf.keras.Sequential([
    tf.keras.layers.Embedding(input_dim=letras, output_dim=dimension),
    tf.keras.layers.GRU(128, return_sequences=True),
    tf.keras.layers.Dense(letras, activation="softmax")
])
model.compile(loss="sparse_categorical_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
```

Fíjate en que, en este caso, el embedding se va construyendo en el proceso de entrenamiento de la red.
"""

# ╔═╡ 99437b1f-f38d-40ce-bd43-d071669cef2f
md"""
## Generación de texto

Una vez entrenada la red, si le proporcionamos un texto inicial de partida, la red irá añadiendo carácter a carácter hasta alcanzar el carácter especial **<EOF>** final.

El estilo del texto generado coincidirá con el del texto utilizado para entrenar la red.

Además, podemos hacer *transfer learning*, si tenemos una red entrenada, podemos reentrenarla utilizando la misma técnica que empleamos con redes convolucionales:
"""

# ╔═╡ b167c50a-38dd-4485-aabb-9a5e97ded2b4
md"""
## Generación de texto

1. Utilizar una arquitectura inicial ya entrenada.
1. Eliminar las capas más cercanas a la salida y sustituirlas por nuevas capas adaptadas a nuestro problema.
1. Durante la fase de entrenamiento:
    1. _Congelar_ los pesos de las capas preexistentes.
    1. Entrenar durante unas épocas hasta que la precisión sobre el conjunto de validación se estabilice.
    1. _Descongelar_ todas las capas y entrenar hasta conseguir una buena precisión.
"""

# ╔═╡ 8ef6df12-63ca-4544-8df9-14bb4c944319
md"""
## Clasificación de texto
"""

# ╔═╡ f6bf7355-4d41-4d5b-8034-492871b67b7c
Resource(
	imagenes * "analisis_sentimientos.png",
	:alt => "Análisis de sentimientos",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 4e5b4466-587a-44f5-85c4-561c454e2100
md"""
Supongamos que queremos clasificar textos como positivos, negativos o neutros.

En el conjunto de entrenamiento, o todos los textos tienen el mismo número de palabras.
"""

# ╔═╡ a7b091ab-0dcc-44c9-a93e-118d1610a855
md"""
## Clasificación de texto

Podemos resolverlo insertando una marca especial que indique huecos, donde se esperan palabras.

Pero esto no dará buenos resultados durante el proceso de entrenamiento.

Una mejor solución es indicar a la red que las marcas de _hueco_ no sean tenidas en cuenta durante el proceso de entrenamiento.

```{.python}
tf.keras.layers.Embedding(tam_vacobulario, tam_embedding, mask_zero=True)
```

De igual modo, los _huecos_ tampoco se tendrán en cuenta en el proceso de clasificación.
"""

# ╔═╡ dc52bfd9-2bfd-4882-88cd-f869a135945c
md"""
## Clasificación de texto

Para construir la red que analiza sentimientos, primero construimos la capa de vectorización, ahora estamos trabajando con palabras, e inicialmente no sabemos cuantas palabras tiene nuestro corpus:

```{.python}
text_vec_layer = tf.keras.layers.TextVectorization(max_tokens=tam_maximo)
text_vec_layer.adapt([textos, etiquetas])
```

Y ahora construimos la red:

```{.python}
modelo = tf.keras.Sequential([
    text_vec_layer,
    tf.keras.layers.Embedding(tam_vocabulario, tam_embedding),
    tf.keras.layers.GRU(128),
    tf.keras.layers.Dense(1, activation="sigmoid")
])
model.compile(loss="binary_crossentropy", optimizer="nadam",
              metrics=["accuracy"])
```
"""

# ╔═╡ 0b8d86b8-ab46-466f-91a4-70b689eb6b21
md"""
## Traducción automática

Las redes recurrentes almacenan _memoria_ que es útil para tareas de traducción de textos, ya que es importante guardar el contexto de las palabras en la traducción.
"""

# ╔═╡ 96221fff-1ac1-497c-97da-62c71571100c
md"""
## Traducción automática

Durante la fase de entrenamiento.
"""

# ╔═╡ 7c25120d-4043-4d81-8961-06b5192cf526
Resource(
	imagenes * "nlp_rnn.png",
	:alt => "Traducción automática",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 81b23354-0862-4226-a2e9-26a9ceb700e2
md"""
Fuente: Hands-on machine learning - Aurélien Géron
"""

# ╔═╡ 55b122db-fe73-4ccf-ad85-2ecb282de7e3
md"""
## Traducción automática

Durante la fase de traducción.
"""

# ╔═╡ 4c98dcf0-2a02-474e-a4c6-7f9605ecc208
Resource(
	imagenes * "nlp_rnn_decoder.png",
	:alt => "Fase de traducción",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ cb4da7e6-eb30-4369-9c6d-60e21bf7256c
md"""
Fuente: Hands-on machine learning - Aurélien Géron
"""

# ╔═╡ e44c1ec9-4d42-4dc7-9d4c-368d886686aa
md"""
## Traducción automática

Durante la fase de traducción.
"""

# ╔═╡ e217765a-ad91-43d0-ac80-e4d7f1e5f38a
Resource(
	imagenes * "rnn_traduccion.png",
	:alt => "Fase de traducción",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 140b7a93-4bd6-44a3-baaf-562d364b916d
md"""
Fuente: A Survey of Deep Learning Techniques for Neural Machine Translation, S. Yan et al., 2020.
"""

# ╔═╡ d943e918-6d72-45d9-b118-b88dcaacea65
md"""
## Traducción automática

Una mejora de la arquitectura anterior son las redes bidireccionales:
"""

# ╔═╡ bb095489-b75b-47c1-b785-e59c317691f6
Resource(
	imagenes * "rnn_bidireccional.png",
	:alt => "Fase de traducción",
	:width => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ fbf000ab-aca2-4fb7-8b3f-ef6791ad7cc1
md"""
En este caso, la información del estado oculto se transmite en dos direcciones, con lo que el contexto se amplia hacia adelante en la secuencia, y también hacia atrás.

Fuente: A Survey of Deep Learning Techniques for Neural Machine Translation, S. Yan et al., 2020.
"""

# ╔═╡ 9f58a7f5-ced0-4788-b5b3-4a57a0063b31
md"""
## Transformers

Los transformers son una vuelta de tuerca al concepto de mantener la información entre partes de la frase cuando se analiza.

En las redes recurrentes bidireccionales, las posibles relaciones fluyen tanto hacia adelante en la frase como hacia atrás, en el proceso de entrenamiento.

Los transformers utilizan el concepto de atención, que enmascara las partes de la frase que son de interés para la palabra que se está traduciendo en cada  momento.
"""

# ╔═╡ 6c7acc38-6290-4cb5-893d-133404ac3c86
md"""
## Transformers

Este es el mecanismo de atención tal y como se presenta en el [artículo](https://arxiv.org/pdf/1706.03762) original

Los dos componentes interesante son los bloques de atención (Attention).
"""

# ╔═╡ 0ae76bbd-5c4c-4ff5-8ec1-14eb89453076
Resource(
	imagenes * "transformer.png",
	:alt => "Transformers",
	:height => 500,
	:style => "display: block; margin: auto;",
)

# ╔═╡ b5add997-923c-4827-b1a6-0e513d981480
md"""
Fuente: Attention is all you need - A. Vaswani et al.
"""

# ╔═╡ d13bc23b-bf3a-4385-add2-4c2c268fc4f1
md"""
## Transformers
"""

# ╔═╡ 53e70c90-c0a5-4c9c-ae0a-3437f21adeb7
Resource(
	imagenes * "transformer2.png",
	:alt => "Transformers",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 4f4e0350-1e9e-4d30-b428-4336f6cb4f7a
md"""
La parte novedosa es utilizar máscaras para que la red encuentre relación entre las partes del texto aunque estas se encuentren alejadas entre sí dentro de la misma frase.

Fuente: Attention is all you need - A. Vashvani et al.
"""

# ╔═╡ e9c837ad-f0a7-41f8-a932-e4699d58c277
md"""
## Transformers

Los Transformers son una de las tecnologías más avanzada para tareas de NLP.

Si te interesa este tema, la página web de referencia es 
[Hugging Face](https://huggingface.co/), donde podrás encontrar una buena 
cantidad de modelos pre-entrenados y mucha información sobre NLP.

Otra web muy útil es la de 
[Somos NLP](https://somosnlp.org/) que une a la comunidad de NLP en español.
"""

# ╔═╡ 0050209d-57cd-4dcf-aac0-b753172a552c
md"""
## Hugging Face

Hugging Face, entre otras muchas cosas, es un repositorio de LLM listos para usar o adaptar en tus aplicaciones.

Su uso básico es muy sencillo.

Primero tendremos que instalar la versión de keras con la que trabaja los transformers de Hugging Face.

```{.bash}
micromamba install tf-keras
```

Y cargamos la biblioteca:

```{.python}
from transformers import pipeline
```
"""

# ╔═╡ 3c6c6b3b-2ff6-49ae-b534-ddfccaca12fb
md"""
## Hugging Face

Si lo que queremos hacer es **clasificación de textos**, lo primero es encontrar 
un modelo entrenado para ello y en español.

Luego, creamos un pipeline indicando la tarea y el modelo.

```{.python}
nombre_modelo = "UMUTeam/roberta-spanish-sentiment-analysis"
clasificador = pipeline("sentiment-analysis", model=nombre_modelo)
clasificacion = clasificador("No me gusta comer fuera de casa")
print(clasificacion)
```

El resultado es:

```{.bash}
[{'label': 'negative', 'score': 0.9891014695167542}]
```
"""

# ╔═╡ f1c88b97-179c-41b8-860a-5fc7b61780c6
md"""
## Hugging Face

Otro ejemplo de clasificación de sentimientos:

```{.bash}
clasificacion = clasificador("Ayer fui al cine a ver una peli y pasé una tarde muy agradable")
print(clasificacion)
```

El resultado es:

```{.bash}
[{'label': 'positive', 'score': 0.9974034428596497}]
```

Como puedes ver, en ambos casos el resultado es bastante bueno.
"""

# ╔═╡ 8afabde3-ce96-4590-aebb-ab61f6ad1572
md"""
## Hugging Face

Si la tarea que queremos hacer es generar texto:

```{.python}
nombre_modelo = "Kukedlc/Llama-7b-spanish"
generador = pipeline("text-generation", model=nombre_modelo)

texto = generador("Hoy voy a comer a casa", max_length=30)
print(texto)
```

El resultado es:

```{.bash}
Hoy voy a comer a casa de mis padres.
```
"""

# ╔═╡ 0788fccf-cfe2-4b12-9ba8-7c2a1def2218
md"""
## Hugging Face

Alguna tarea más compleja como el reconocimiento de entidades:

```{.python}
ner = pipeline("ner", grouped_entities=True)
ner("Me llamo Óscar y trabajo como profesor en la UJI de Castellón.")
```

El resultado es:

```{.bash}
[{'entity_group': 'PER',
  'score': 0.6795335,
  'word': 'Óscar',
  'start': 9,
  'end': 14},
 {'entity_group': 'ORG',
  'score': 0.99412817,
  'word': 'UJI',
  'start': 45,
  'end': 48},
 {'entity_group': 'LOC',
  'score': 0.94170475,
  'word': 'Castellón',
  'start': 52,
  'end': 61}]
```
"""

# ╔═╡ 6cff35d1-55f3-492f-8026-aa0d13700e69
md"""
## Hugging Face

Finalmente, hacer resúmenes de textos:

```{.python}
resumidor = pipeline("summarization")
resumen = resumidor( "En una economía tan estacional como la española, a no
ser que haya eventos disruptivos como una pandemia o un colapso como el de
2008, el comportamiento de la afiliación a la Seguridad Social y del paro
registrado riman cada mes. Y noviembre, con la temporada turística veraniega
completamente agotada y a la espera de la Navidad, no suele ser un buen mes
para el mercado laboral. Así, España perdió en noviembre 30.050 empleos, hasta
dejar la afiliación media en 21.302.463 trabajadores, el mayor bajón en el
undécimo mes desde 2019. El retroceso se centra en la hostelería, un sector en
el que se perdieron 120.000 empleos respecto a octubre, lo que también se nota
en la desagregación por territorios: la ocupación solo cae con fuerza en
Baleares, una región de monocultivo turístico. Los datos son algo mejores en
paro registrado, con un retroceso de 16.036 personas. Pero es una caída leve,
peor que la de los tres últimos años, de la mano de un dato positivo: el total
de parados en noviembre es el menor desde 2007, antes de la Gran Recesión. En
la misma línea, la cifra de ocupados es la más alta que se haya registrado
nunca en un mes de noviembre.")
```

"""

# ╔═╡ 320611a2-9366-4aa6-86ba-ab7fc247ab84
md"""
## Hugging Face

Y este es el resultado del resumen:

```{.bash} 
[{'summary_text': ' España perdió en noviembre 30.050 empleos, hasta
dejar la afiliación media en 21.302.463 trabajadores . El retroceso se centra
en la hostelería, un sector en el que se perdieron 120,000 empleo a octubre
.'}]
```

Como puedes ver, bastante bueno.

"""

# ╔═╡ 11b6f7ff-f6d5-4f0d-8a9f-aa1eb2604b55
md"""
## Otras tareas en NLP

Hay otras muchas tareas dentro del campo de NLP:

* Respuestas a preguntas.
* Resumidores de texto.
* Texto a voz y voz a texto.
* Etiquetado de partes del discurso.

Sólo por mencionar algunas.
"""

# ╔═╡ 61de2811-9a8d-40d3-9248-22c147dc2aee
md"""
# Resumen

* El procesamiento de lenguaje natural es una disciplina muy amplia y con gran tradición dentro de la inteligencia artificial.
* Hemos visto algunas de las piezas básicas para trabajar con NLP, como los embeddings.
* Hemos visto como las redes recurrentes son muy buenas para algunas tareas típicas dentro de NLP, como la generación de textos, la clasificación y la traducción.
* Actualmente, los transformers son la arquitectura con un mejor rendimiento en tareas de NLP.
* Hugging Face ofrece una gran cantidad de modelo entrenados listos para ser utilizados, o como base para crear nuestros propios modelos.
"""

# ╔═╡ 9d9bd2f8-71d8-4d52-8437-805e6de7c391
md"""
# Referencias

1. [Visualización de embeddings.](https://projector.tensorflow.org/)
1. [Hugging face.](https://huggingface.co/)
1. [Somos NLP.](https://somosnlp.org)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
TextAnalysis = "a2db99b7-8b79-58f8-94bf-bbc811eef33d"

[compat]
PlutoUI = "~0.7.68"
ShortCodes = "~0.3.6"
TextAnalysis = "~0.8.2"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.7"
manifest_format = "2.0"
project_hash = "afc6e5481b635b74bcf8a71383dcb1a8dd6b5eec"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "962834c22b66e32aa10f7611c08c8ca4e20749a9"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.8"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

    [deps.ColorTypes.weakdeps]
    StyledStrings = "f489334b-da3d-4c2e-b8f0-e476e12c162b"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "0037835448781bb46feb39866934e243886d756a"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.18.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "d9d26935a0bcffc87d2613ce14c527c99fc543fd"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.5.0"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataDeps]]
deps = ["HTTP", "Libdl", "Reexport", "SHA", "Scratch", "p7zip_jll"]
git-tree-sha1 = "8ae085b71c462c2cb1cfedcb10c3c877ec6cf03f"
uuid = "124859b0-ceae-595e-8997-d05f6a7a8dfe"
version = "0.7.13"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "4e1fe97fdaed23e9dc21d4d664bea76b65fc50a0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.22"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"
version = "1.11.0"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.DocStringExtensions]]
git-tree-sha1 = "7442a5dfe1ebb773c29cc2962a8980f47221d76c"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.5"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.HTML_Entities]]
deps = ["RelocatableFolders", "StrTables"]
git-tree-sha1 = "781d638b8892cd1a1e867f51863d8881c0a62834"
uuid = "7693890a-d069-55fe-a829-b4a6d304f0ee"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "ed5e9c58612c4e081aecdb6e1a479e18462e041e"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.17"

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

[[deps.IrrationalConstants]]
git-tree-sha1 = "e2222959fbc6c19554dc15174c81bf7bf3aa691c"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

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
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

    [deps.JSON3.extensions]
    JSON3ArrowExt = ["ArrowTypes"]

    [deps.JSON3.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"

[[deps.Languages]]
deps = ["InteractiveUtils", "JSON", "RelocatableFolders"]
git-tree-sha1 = "0cf92ba8402f94c9f4db0ec156888ee8d299fcb8"
uuid = "8ef0a80b-9436-5d2c-a485-80b904378c43"
version = "0.4.6"

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

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "13ca9e2586b89836fd20cccf56e57e2b9ae7f38f"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.29"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"
version = "1.11.0"

[[deps.LoggingExtras]]
deps = ["Dates", "Logging"]
git-tree-sha1 = "f02b56007b064fbfddb4c9cd60161b6dd0f40df3"
uuid = "e6f89c97-d47a-5376-807f-9c37f3926c36"
version = "1.1.0"

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

[[deps.MbedTLS]]
deps = ["Dates", "MbedTLS_jll", "MozillaCACerts_jll", "NetworkOptions", "Random", "Sockets"]
git-tree-sha1 = "c067a280ddc25f196b5e7df3877c6b226d390aaf"
uuid = "739be429-bea8-5141-9913-cc70e7f3736d"
version = "1.1.9"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

[[deps.Memoize]]
deps = ["MacroTools"]
git-tree-sha1 = "2b1dfcba103de714d31c033b5dacc2e4a12c7caa"
uuid = "c03570c3-d221-55d1-a50c-7939bbd78826"
version = "0.4.4"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "ec4f7fbeab05d7747bdf98eb74d130a2a2ed298d"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.2.0"

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

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "f1a7e086c677df53e064e0fdd2c9d0b0833e3f6e"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.5.0"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2ae7d4ddec2e13ad3bddf5c0796f7547cf682391"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.5.2+0"

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
version = "1.11.0"

    [deps.Pkg.extensions]
    REPLExt = "REPL"

    [deps.Pkg.weakdeps]
    REPL = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "Downloads", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "ec9e63bd098c50e4ad28e7cb95ca7a4860603298"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.68"

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

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "fbb92c6c56b34e1a2c4c36058f68f332bec840e7"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.11.0"

[[deps.PtrArrays]]
git-tree-sha1 = "1d36ef11a9aaf1e8b74dacc6a731dd1de8fd493d"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.3.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "9b81b8393e50b7d4e6d0a9f14e192294d3b7c109"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.3.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.ShortCodes]]
deps = ["Base64", "CodecZlib", "Downloads", "JSON3", "Memoize", "URIs", "UUIDs"]
git-tree-sha1 = "5844ee60d9fd30a891d48bab77ac9e16791a0a57"
uuid = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"
version = "0.3.6"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.Snowball]]
deps = ["Languages", "Snowball_jll", "WordTokenizers"]
git-tree-sha1 = "8b466b16804ab8687f8d3a1b5312a0aa1b7d8b64"
uuid = "fb8f903a-0164-4e73-9ffe-431110250c3b"
version = "0.1.1"

[[deps.Snowball_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "6ff3a185a583dca7265cbfcaae1da16aa3b6a962"
uuid = "88f46535-a3c0-54f4-998e-4320a1339f51"
version = "2.2.0+0"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "64d974c2e6fdf07f8155b5b2ca2ffa9069b608d9"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.2"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"
weakdeps = ["SparseArrays"]

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "9d72a13a3f4dd3795a195ac5a44d7d6ff5f552ff"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.1"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "2c962245732371acd51700dbb268af311bddd719"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.6"

[[deps.StrTables]]
deps = ["Dates"]
git-tree-sha1 = "5998faae8c6308acc25c25896562a1e66a3bb038"
uuid = "9700d1a9-a7c8-5760-9816-a99fda30bb8f"
version = "1.0.1"

[[deps.StructTypes]]
deps = ["Dates", "UUIDs"]
git-tree-sha1 = "159331b30e94d7b11379037feeb9b690950cace8"
uuid = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"
version = "1.11.0"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableTraits]]
deps = ["IteratorInterfaceExtensions"]
git-tree-sha1 = "c06b2f539df1c6efa794486abfb6ed2022561a39"
uuid = "3783bdb8-4a98-5b6b-af9a-565f29a5fe9c"
version = "1.0.1"

[[deps.Tables]]
deps = ["DataAPI", "DataValueInterfaces", "IteratorInterfaceExtensions", "OrderedCollections", "TableTraits"]
git-tree-sha1 = "f2c1efbc8f3a609aadf318094f8fc5204bdaf344"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.1"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TextAnalysis]]
deps = ["DataStructures", "DelimitedFiles", "DocStringExtensions", "InteractiveUtils", "JSON", "Languages", "LinearAlgebra", "Printf", "ProgressMeter", "Random", "Serialization", "Snowball", "SparseArrays", "Statistics", "StatsBase", "Tables", "WordTokenizers"]
git-tree-sha1 = "b2da9be079f3b4882bfe939a8c97c51a9cd03c59"
uuid = "a2db99b7-8b79-58f8-94bf-bbc811eef33d"
version = "0.8.2"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "bef26fb046d031353ef97a82e3fdb6afe7f21b1a"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.6.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.WordTokenizers]]
deps = ["DataDeps", "HTML_Entities", "StrTables", "Unicode"]
git-tree-sha1 = "01dd4068c638da2431269f49a5964bf42ff6c9d2"
uuid = "796a5d58-b03d-544a-977e-18100b691f6e"
version = "0.5.6"

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
# ╠═c2fd34b8-66db-11f0-3efd-21960f239f79
# ╠═8da42b7b-7bc8-4bdc-9707-539209d79d1b
# ╠═efc09298-51f8-4013-9677-497b6de474be
# ╠═367d5f39-3928-4709-8de9-2f4b1f046acd
# ╠═2a803c80-dece-4f52-9898-1cc3ba15ebbc
# ╠═e05614d4-5762-4839-b657-1c98b2054de4
# ╠═9e00079f-b241-4c7a-9c24-b229e08428df
# ╠═80774b7b-15eb-45a4-b220-c73bdb2da29b
# ╠═b72590de-c5d6-422d-8a7d-572cc3b1c3c3
# ╠═7128e86b-bdb9-4e7a-84ba-ee8682f8ae4f
# ╠═1be76676-2de5-4df0-9e51-e95ce80c4ece
# ╠═9476c68a-19fb-4855-9c6e-3d2cf1e9e403
# ╠═7aa6b29c-3e43-4b2a-9323-655f90835e66
# ╠═0679cc8f-5779-41ad-ac5d-13f1a303b0dc
# ╠═70676122-eb80-4b6d-9c78-e489292c3f5c
# ╠═2ea8cba5-3286-4d26-8bbe-76b7e1ba868f
# ╠═813ecbd6-74b8-4d26-a92d-c6098044e64c
# ╠═116bf570-3a18-4789-b37c-18755c401f5d
# ╠═15e631ed-dcbd-4a09-a020-474671543c26
# ╠═c0a6ed84-c2e1-4778-9698-3c1a346cc6bf
# ╠═e2c82807-91c9-422f-8c68-e9550d25f239
# ╠═bfbe0854-0fc8-4f8f-9286-977e734cab12
# ╠═1c8635e9-5f6e-4005-95d3-a3811498a179
# ╠═9c9997ea-e050-4960-b656-3048091b0035
# ╠═ac9ef535-5f67-4d49-a56d-fe2bd44993f6
# ╠═4d9883fe-1ead-4d07-ad35-5c8d935412b5
# ╠═5c217c15-be06-464e-a1fc-577d6059205c
# ╠═bc49d4cb-ab79-4fa0-809f-fb3ba82e5adc
# ╠═d2c96070-c07f-49c4-84f5-bebf7498b314
# ╠═a1ff9ae5-a131-40c2-833a-4f17fb765d1b
# ╠═22ec98af-c189-46ab-99a1-5cd8d51a77a2
# ╠═c31ee640-cdba-4469-b755-a6406e101015
# ╠═8c5cd066-9fa7-4c62-beed-e858d0f3a311
# ╠═87d0f2f2-eaea-4a7c-bca9-ceabab0da099
# ╠═6744bcec-8e4a-420f-aeb9-34de80d0404c
# ╠═767aaf9a-1984-404a-ae79-a74fc919861d
# ╠═cc042ada-03d1-4189-abf0-9337546e6bec
# ╠═1ebf121c-e8c1-45f8-954f-a7ad10157228
# ╠═f4db6144-ef66-414c-9854-c05d0dad8ef3
# ╠═ac928fe1-5926-44d7-912e-be19f5128b84
# ╠═bafdc086-0b06-4f33-8fbe-a8933c82ee47
# ╠═7bd2726c-6c8a-48d9-b1ae-eed8bcc88156
# ╠═4c9944c1-60bf-4e57-a0c9-6219baeb2313
# ╠═92c5efbe-6148-4813-8e28-ddd73808b6e1
# ╠═1c7fc9ef-5259-498b-80ea-b8a21d4b2fc5
# ╠═7f1ebe1e-7e9c-480a-aad5-23e6d70607cc
# ╠═f7129b01-7d93-41f9-a5f2-526ed80311af
# ╠═732ca76c-97ee-4762-9cd5-cd76752391f5
# ╠═9c9247a8-fb07-4924-8fa3-445d3f4ade54
# ╠═7f09d56e-77b3-45e5-ac5a-61f80ad82b39
# ╠═be14b504-bf76-4053-8098-71c791108151
# ╠═e31a02d0-d16a-4f11-a767-050139079635
# ╠═bee075f8-65e2-4991-ae10-e749d1119bb2
# ╠═7a6bf294-4694-43d9-b4b6-5f085dddf29f
# ╠═4ca5b39c-e180-4956-bbbc-3ff3b51aa1a4
# ╠═ea93bd5e-d8ca-4d75-807e-bc4b315464ce
# ╠═bd0096b0-7059-4444-81c6-ee6daffa0392
# ╠═a94e7a01-d5d1-4b95-8e6c-bf1ad2b0b4e5
# ╠═84f87d3f-d60e-4a95-a48c-e77a25874d0f
# ╠═e0917f33-9a77-439a-badf-66a3d04b248a
# ╠═02bce938-3cf8-41fd-b8b5-da36d2bf3782
# ╠═a601d1f6-50e8-460c-8b00-33c67ea535b6
# ╠═547eda5f-ba82-41d7-b9d6-0160813fb3f8
# ╠═92307ff3-745e-41ef-8491-cf6306fc9a64
# ╠═99437b1f-f38d-40ce-bd43-d071669cef2f
# ╠═b167c50a-38dd-4485-aabb-9a5e97ded2b4
# ╠═8ef6df12-63ca-4544-8df9-14bb4c944319
# ╠═f6bf7355-4d41-4d5b-8034-492871b67b7c
# ╠═4e5b4466-587a-44f5-85c4-561c454e2100
# ╠═a7b091ab-0dcc-44c9-a93e-118d1610a855
# ╠═dc52bfd9-2bfd-4882-88cd-f869a135945c
# ╠═0b8d86b8-ab46-466f-91a4-70b689eb6b21
# ╠═96221fff-1ac1-497c-97da-62c71571100c
# ╠═7c25120d-4043-4d81-8961-06b5192cf526
# ╠═81b23354-0862-4226-a2e9-26a9ceb700e2
# ╠═55b122db-fe73-4ccf-ad85-2ecb282de7e3
# ╠═4c98dcf0-2a02-474e-a4c6-7f9605ecc208
# ╠═cb4da7e6-eb30-4369-9c6d-60e21bf7256c
# ╠═e44c1ec9-4d42-4dc7-9d4c-368d886686aa
# ╠═e217765a-ad91-43d0-ac80-e4d7f1e5f38a
# ╠═140b7a93-4bd6-44a3-baaf-562d364b916d
# ╠═d943e918-6d72-45d9-b118-b88dcaacea65
# ╠═bb095489-b75b-47c1-b785-e59c317691f6
# ╠═fbf000ab-aca2-4fb7-8b3f-ef6791ad7cc1
# ╠═9f58a7f5-ced0-4788-b5b3-4a57a0063b31
# ╠═6c7acc38-6290-4cb5-893d-133404ac3c86
# ╠═0ae76bbd-5c4c-4ff5-8ec1-14eb89453076
# ╠═b5add997-923c-4827-b1a6-0e513d981480
# ╠═d13bc23b-bf3a-4385-add2-4c2c268fc4f1
# ╠═53e70c90-c0a5-4c9c-ae0a-3437f21adeb7
# ╠═4f4e0350-1e9e-4d30-b428-4336f6cb4f7a
# ╠═e9c837ad-f0a7-41f8-a932-e4699d58c277
# ╠═0050209d-57cd-4dcf-aac0-b753172a552c
# ╠═3c6c6b3b-2ff6-49ae-b534-ddfccaca12fb
# ╠═f1c88b97-179c-41b8-860a-5fc7b61780c6
# ╠═8afabde3-ce96-4590-aebb-ab61f6ad1572
# ╠═0788fccf-cfe2-4b12-9ba8-7c2a1def2218
# ╠═6cff35d1-55f3-492f-8026-aa0d13700e69
# ╠═320611a2-9366-4aa6-86ba-ab7fc247ab84
# ╠═11b6f7ff-f6d5-4f0d-8a9f-aa1eb2604b55
# ╠═61de2811-9a8d-40d3-9248-22c147dc2aee
# ╠═9d9bd2f8-71d8-4d52-8437-805e6de7c391
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
