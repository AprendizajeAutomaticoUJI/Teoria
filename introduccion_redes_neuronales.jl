### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# ╔═╡ aa6d92eb-7a97-4084-bfc7-8474b34a2c71
using DataFrames

# ╔═╡ ee78d551-2259-4955-b934-c60d3453706c
using PlutoUI

# ╔═╡ d5fd2304-0353-11f0-29d8-3158c4dbe8dd
# html"""
# <link rel="stylesheet" type="text/css" href="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/mi_estilo.css" media="screen" />
# """

# ╔═╡ 785309b6-ddfc-4e99-9699-63f60e73a787
TableOfContents(title = "Contenidos", depth=1)

# ╔═╡ 0b5df837-5a80-43c6-b3ce-a1554d550776
url_imagenes = "https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/"

# ╔═╡ 2c54bdd7-0d73-496c-9fd1-031dcdf2c861
md"""
# Regresión logística

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)

![](https://ujiapps.uji.es/ade/rest/storage/63c07717-5208-4240-b688-aa6ff558b466?guest=true)
"""

# ╔═╡ 0ebbbc1b-066d-4c2b-8ad0-2165a145d5f1
md"""
## Introducción

Las redes neuronales son modelos de aprendizaje automático muy potentes, pero 
complicados de entrenar.

Su entrenamiento se basa en el algoritmo de retro propagación, que es un una 
combinación del algoritmo de descenso de 
gradiente y de la regla de la cadena.

Las redes neuronales pueden resolver tanto problemas supervisados como no 
supervisados. Además se pueden utilizar tanto en tareas de clasificación como 
de regresión.
"""

# ╔═╡ 5b797826-439b-48ce-a293-03534db1f2a9
md"""
## Introducción

Las redes neuronales son modelos de aprendizaje automático muy potentes, pero 
complicados de entrenar.

Su entrenamiento se basa en el algoritmo de retro propagación, que es un una 
combinación del algoritmo de descenso de 
gradiente y de la regla de la cadena.

Las redes neuronales pueden resolver tanto problemas supervisados como no 
supervisados. Además se pueden utilizar tanto en tareas de clasificación como 
de regresión.
"""

# ╔═╡ f65234a5-1af3-48de-be23-a6c3050b79a6
md"""
## Objetivos de aprendizaje

1. Esquematizar la estructura de una neurona y un red neuronal.
1. Resumir los pasos para crear y entrenar una red neuronal.
1. Construir una red neuronal sencilla para problemas de regresión.
1. Construir una red neuronal sencilla para problemas de clasificación.
1. Interpretar la _historia_ del proceso de entrenamiento.
"""

# ╔═╡ ce3edc15-c454-4419-8619-5d800aed2d5c
md"""
## Referencias

1. [Deep Learning](http://www.deeplearningbook.org), Ian Goodfellow, Joshua Bengio and Aaron Courbille.
1. [Dive into deep learning](https://d2l.ai/index.html), Aston Zhang et al.
1. Pattern recognition and machine learning. Christopher M. Bishop.
"""

# ╔═╡ 0a89e6b8-46c1-4dfc-b11e-6373dc743ec4
md"""
# Bases biológicas de las NN
## Bases biológicas de las NN

Neuronas vista la microscopio dibujadas por Santiago Ramón y Cajal.
"""

# ╔═╡ 770f2494-be6f-4c6b-aa9b-687241c0f62a
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/neuronas_Ramon_Cajal.jpg",
	:alt => "Neuronas vistas al microcopio por Santiago Ramón y Cajal",
	:width => 600
)

# ╔═╡ 20b5ac11-a251-48e7-bfe2-252d8a4b1fe9
md"""
Neuronas vista la microscopio dibujadas por Santiago Ramón y Cajal.
"""

# ╔═╡ fd15e146-5c83-406b-a056-fc1fba51ffc2
md"""
## Bases biológicas de las NN

Una neurona recibe señal de otras neuronas a través de la dendrita, y envía 
señal a otras neuronas a través del axón.

"""

# ╔═╡ 3e361edb-a6fc-4c26-8c91-f44eec0a6a6c
Resource(
	"https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/RedesNeuronales/neurona_real.jpg",
	:alt => "Neurona",
	:width => 600
)

# ╔═╡ 0f6e55dd-d1e4-4833-8c3f-a6390daff2a5
md"""
Diseñado por Freepik (https://www.freepik.es)
"""

# ╔═╡ 8796c632-7302-4213-be40-67e7b11f5a35
md"""
## Bases biológicas de las NN
"""

# ╔═╡ b64fa92a-dfc8-4049-a069-9a1c4f61b65c
Resource(
	url_imagenes * "sinapsis.png",
	:alt => "Neurona",
	:width => 600
)

# ╔═╡ 135e5c27-0f1a-46a5-9997-254c6547ff1e
html"""
<font size=2>
By The original uploader was Nrets at English Wikipedia. - Transferred from
en.wikipedia to Commons., CC BY-SA 3.0,
https://commons.wikimedia.org/w/index.php?curid=2006912
</font>
"""

# ╔═╡ 4022783b-c5a4-4df3-b342-51641fc60fdc
md"""
!!! info "Título"
	Aquí el texto.

	Y más texto
"""

# ╔═╡ 5e36d267-f394-4b53-a29e-b68a297113a1
md"""
# Estructura de las NN
## Estructura de una neurona artificial

Una neurona artificial:
"""

# ╔═╡ 4d082d8e-7b85-4e94-91b4-0e550ccd9458
Resource(
	url_imagenes * "neurona.png",
	:width => 600
)

# ╔═╡ cabcaec7-3fca-4ea9-929f-0d3f3e6843a2
md"""
## Estructura de una neurona artificial
Activación de la neurona artificial. Cada seña de entrada se multiplica por 
un peso y se añade un _bias_.
"""

# ╔═╡ df53f204-76e8-43f7-858f-48c5f8f78eeb
Resource(
	url_imagenes * "activacion.png",
	:width => 600
)

# ╔═╡ c3685572-264d-48d9-aee6-142ae90576b3
md"""
## Estructura de una neurona artificial
Aunque _apilemos_ varias capas el resultado es siempre lineal con respecto de 
los valores de entrada (_Demostración_).
"""

# ╔═╡ bbbc0b92-185b-4126-b858-e94af0211a6c
Resource(
	url_imagenes * "sin_funcion_activacion.png",
	:width => 600
)

# ╔═╡ d0e46bda-cd81-4717-87b1-e5ef283f6e7b
md"""
Es necesario introducir la _no linearidad_ de otro modo.
"""

# ╔═╡ 9d41b30b-afab-4f83-ae60-9fc50dea90b9
md"""
## Estructura de una neurona artificial

La función de activación es el ingrediente que introduce la no-linearidad en 
las redes neuronales.
"""

# ╔═╡ 3063a028-22e6-467f-82fb-139492ed4e6b
Resource(
	url_imagenes * "funcion_activacion.png",
	:width => 600
)

# ╔═╡ 359e990e-41bc-43bc-9504-d520476ea035
md"""
Función de activación sigmoide.
$\sigma(\omega x) = \frac{1}{1+e^{-\omega x}}$
"""

# ╔═╡ 26ffe43c-fcda-4c52-be27-ce945f1084d9
md"""
## Estructura de una NN

El perceptron (Rosenblatt, 1958) fue la primera propuesta de algoritmo 
inspirado en el funcionamiento de las neuronas biológicas.
"""

# ╔═╡ aba74d40-c561-4492-a7a0-f49a69f6455e
Resource(
	url_imagenes * "perceptron.png",
	:fig_align => :center,
	:height => 250
)

# ╔═╡ a46f42c1-64a2-4a07-9ee7-e8517eb56dd8
html"""
<font size=2>
Fuente: https://www.deep-mind.org/2023/03/26/the-universal-approximation-theorem/
</font>
"""

# ╔═╡ aab30f28-b671-45f8-bea1-b637351ad432
md"""
Aunque esta arquitectura es muy sencilla, permite hacer tareas simples de 
clasificación.
"""

# ╔═╡ 089c0310-259c-4158-b49f-10b25969beac
md"""
## Estructura de una NN

El siguiente paso fue añadir sucesivas capas para mejorar los resultados 
de las redes. Esta arquitectura se llama _Multi Layer Perceptron_ (MLP).
"""

# ╔═╡ b6e330c6-8285-4f5b-a933-7c5a35f1fbab
Resource(
	url_imagenes * "estructura_red_neuronal.png"
)

# ╔═╡ 8e144c0b-8ae4-4865-931f-a52afa55965d
html"""
<font size=2>
Fuente: https://www.deep-mind.org/2023/03/26/the-universal-approximation-theorem/
</font>
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
DataFrames = "~1.7.0"
PlutoUI = "~0.7.61"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.4"
manifest_format = "2.0"
project_hash = "0439387b86f9705c95eae9cd8f82d816f5af82ad"

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

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.Compat]]
deps = ["TOML", "UUIDs"]
git-tree-sha1 = "8ae8d32e09f0dcf42a36b90d4e17f5dd2e4c4215"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.16.0"
weakdeps = ["Dates", "LinearAlgebra"]

    [deps.Compat.extensions]
    CompatLinearAlgebraExt = "LinearAlgebra"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.1.1+0"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "abe83f3a2f1b857aac70ef8b269080af17764bbe"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.16.0"

[[deps.DataFrames]]
deps = ["Compat", "DataAPI", "DataStructures", "Future", "InlineStrings", "InvertedIndices", "IteratorInterfaceExtensions", "LinearAlgebra", "Markdown", "Missings", "PooledArrays", "PrecompileTools", "PrettyTables", "Printf", "Random", "Reexport", "SentinelArrays", "SortingAlgorithms", "Statistics", "TableTraits", "Tables", "Unicode"]
git-tree-sha1 = "fb61b4812c49343d7ef0b533ba982c46021938a6"
uuid = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
version = "1.7.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "1d0a14036acb104d9e89698bd408f63ab58cdc82"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.20"

[[deps.DataValueInterfaces]]
git-tree-sha1 = "bfc1187b79289637fa0ef6d4436ebdfe6905cbd6"
uuid = "e2d170a0-9d28-54be-80f0-106bbe20a464"
version = "1.0.0"

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

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

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

[[deps.InlineStrings]]
git-tree-sha1 = "6a9fde685a7ac1eb3495f8e812c5a7c3711c2d5e"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.3"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

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
git-tree-sha1 = "1833212fd6f580c20d4291da9c1b4e8a655b128e"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "1.0.0"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

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

[[deps.OrderedCollections]]
git-tree-sha1 = "cc4054e898b852042d7b503313f7ad03de99c3dd"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.8.0"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

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

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

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

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

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

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "725421ae8e530ec29bcbdddbe91ff8053421d023"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.1"

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
git-tree-sha1 = "598cd7c1f68d1e205689b1c2fe65a9f85846f297"
uuid = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
version = "1.12.0"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.Tricks]]
git-tree-sha1 = "6cae795a5a9313bbb4f60683f7263318fc7d1505"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.10"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

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
# ╠═d5fd2304-0353-11f0-29d8-3158c4dbe8dd
# ╠═aa6d92eb-7a97-4084-bfc7-8474b34a2c71
# ╠═ee78d551-2259-4955-b934-c60d3453706c
# ╠═785309b6-ddfc-4e99-9699-63f60e73a787
# ╠═0b5df837-5a80-43c6-b3ce-a1554d550776
# ╠═2c54bdd7-0d73-496c-9fd1-031dcdf2c861
# ╠═0ebbbc1b-066d-4c2b-8ad0-2165a145d5f1
# ╠═5b797826-439b-48ce-a293-03534db1f2a9
# ╠═f65234a5-1af3-48de-be23-a6c3050b79a6
# ╠═ce3edc15-c454-4419-8619-5d800aed2d5c
# ╠═0a89e6b8-46c1-4dfc-b11e-6373dc743ec4
# ╠═770f2494-be6f-4c6b-aa9b-687241c0f62a
# ╠═20b5ac11-a251-48e7-bfe2-252d8a4b1fe9
# ╠═fd15e146-5c83-406b-a056-fc1fba51ffc2
# ╠═3e361edb-a6fc-4c26-8c91-f44eec0a6a6c
# ╠═0f6e55dd-d1e4-4833-8c3f-a6390daff2a5
# ╠═8796c632-7302-4213-be40-67e7b11f5a35
# ╠═b64fa92a-dfc8-4049-a069-9a1c4f61b65c
# ╠═135e5c27-0f1a-46a5-9997-254c6547ff1e
# ╠═4022783b-c5a4-4df3-b342-51641fc60fdc
# ╠═5e36d267-f394-4b53-a29e-b68a297113a1
# ╠═4d082d8e-7b85-4e94-91b4-0e550ccd9458
# ╠═cabcaec7-3fca-4ea9-929f-0d3f3e6843a2
# ╠═df53f204-76e8-43f7-858f-48c5f8f78eeb
# ╠═c3685572-264d-48d9-aee6-142ae90576b3
# ╠═bbbc0b92-185b-4126-b858-e94af0211a6c
# ╠═d0e46bda-cd81-4717-87b1-e5ef283f6e7b
# ╠═9d41b30b-afab-4f83-ae60-9fc50dea90b9
# ╠═3063a028-22e6-467f-82fb-139492ed4e6b
# ╠═359e990e-41bc-43bc-9504-d520476ea035
# ╠═26ffe43c-fcda-4c52-be27-ce945f1084d9
# ╠═aba74d40-c561-4492-a7a0-f49a69f6455e
# ╠═a46f42c1-64a2-4a07-9ee7-e8517eb56dd8
# ╠═aab30f28-b671-45f8-bea1-b637351ad432
# ╠═089c0310-259c-4158-b49f-10b25969beac
# ╠═b6e330c6-8285-4f5b-a933-7c5a35f1fbab
# ╠═8e144c0b-8ae4-4865-931f-a52afa55965d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
