### A Pluto.jl notebook ###
# v0.20.13

using Markdown
using InteractiveUtils

# ╔═╡ 0a611a39-2f34-4190-84ea-36791341ed75
using PlutoUI

# ╔═╡ 1db7081a-2bdc-11f0-0c01-4b497af0a469
# html"""
# <link rel="stylesheet" type="text/css" href="https://www3.uji.es/~belfern/Docencia/IR2130_imagenes/mi_estilo.css" media="screen" />
# """

# ╔═╡ 8211d6e7-2da7-42dc-a03e-f7e8827d0bd4
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 2ba8182e-7607-4516-8c65-7f2b8362672f
md"""
# Proyectos de Aprendizaje Automático

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)

![](https://ujiapps.uji.es/ade/rest/storage/63c07717-5208-4240-b688-aa6ff558b466?guest=true)
"""

# ╔═╡ 39a5d438-5034-49d6-9256-9fbe43fc9e01
md"""
## Introducción

Un proyecto de aprendizaje automático necesita de cierta organización previa antes de su inicio. 
En particular, el uso de los datos debe ser claramente establecido.

El desarrollo de proyectos aprendizaje automático es una fase continua de prueba y error, siempre hay espacio suficiente para probar nuevas idea.

Como en todo proyecto, es muy importante ir documentando todo el trabajo, así como las conclusiones que se van obteniendo.
"""

# ╔═╡ 276f4250-bd8c-408c-9cb0-0493f5f7b3fe
md"""
## Objetivos de aprendizaje

* Organizar las tareas necesarias para llevar a cabo un proyecto de AA.
* Construir un modelo de AA y valorar lo consecución de los objetivos.
* Planear posibles mejoras a partir de los resultados obtenidos.
"""

# ╔═╡ 67b9cdd1-32b3-431b-9fe4-88fc39057eb0
md"""
## Referencias

* Hands-on machine learning with Scikit-learn, Keras and TensorFlow. Capítulo 2 y Apéndice A.
"""

# ╔═╡ 27f48568-a11b-43a0-bc3a-a2306d3c9963
md"""
# Definir el problema y tener una imagen del conjunto
"""

# ╔═╡ 1eff8fdd-86c1-4ffc-ad78-cce43ee6a786
md"""
## Definir el problema y tener una imagen del conjunto

1. Definir los objetivos de la solución que se plantea.
1. Decidir qué tipo de aproximación es la más adecuada: aprendizaje supervisado, no supervisado, por refuerzo, tarea de clasificación o regresión, etc.
1. Decidir la métrica de evaluación que se va a emplear para validar el modelo.
1. Contrastar con soluciones ya existentes.
1. ¿Se puede contar con la colaboración de expertos?
"""

# ╔═╡ f8771a01-7018-4ae0-a6e6-1d21c994e9a9
md"""
# Obtener los datos
"""

# ╔═╡ 72a9bc7c-bb05-4814-97d6-90fb17bb37bb
md"""
## Obtener los datos

1. ¿Existen datos o hay que adquirirlos?
1. Si los datos existen, ¿hay documentación sobre ellos? Por ejemplo, descripción y tipo de los campos, cómo se han obtenido, posibles errores en las mediciones, etc.
1. Comprobar los temas legales sobre los datos.
1. Obtener los datos y gestionar el acceso.
"""

# ╔═╡ 856c2092-93ac-4c77-8455-d838fd74ab62
md"""
# Explorar los datos para conocerlos mejor
"""

# ╔═╡ de706874-4566-477a-bcfa-f4145ea20cdb
md"""
## Explorar los datos para conocerlos mejor

1. Si el volumen de los datos es muy grande, tomar una muestra de ellos para su inspección.
1. Realizar un análisis estadístico descriptivo de los datos.
1. Visualizar los datos.
1. Analizar las correlaciones entre las características.
1. Registrar todas las ideas que vayan surgiendo.
"""

# ╔═╡ ecf46d82-359d-4e54-ac55-e5748a9ef97e
md"""
# Preparar lo datos para que muestren los patrones
"""

# ╔═╡ 27d4223e-66cb-4136-9d37-bb9cc704a0da
md"""
## Preparar lo datos para que muestren los patrones

1. Limpiar los datos si es necesario.
1. Seleccionar/eliminar las características interesantes.
1. Escalar los datos si es necesario.

Automatizar todos los proceso para que sea sencillo repetirlos si fuese 
necesario.
"""

# ╔═╡ 7bef6ea7-48f6-4491-9b16-e08a5597c101
md"""
# Identificar los algoritmos prometedores
"""

# ╔═╡ 7cdff43a-666f-4704-9ece-dbc81d0ab0c2
md"""
## Identificar los algoritmos prometedores

1. Seleccionar los algoritmos más interesantes para el problema en cuestión.
1. Hacer pruebas rápidas para cada uno de los algoritmos seleccionados.
1. Comprobar si se cumplen las condiciones *ideales* para los algoritmos seleccionados.
1. Aplicar la/las métricas seleccionadas y registrar los resultados.
"""

# ╔═╡ eeb73f9a-c310-491c-8d11-3e4954302d59
md"""
## Identificar los algoritmos prometedores

5. Añadir/eliminar características y repetir las pruebas.
1. Seleccionar los algoritmos que dan mejores resultados para la siguiente fase.

De nuevo, automatizar todos los proceso para que sea sencillo repetirlos si 
fuese necesario.
"""

# ╔═╡ d119b883-aaa4-48c4-ab72-1c40913a04ce
md"""
# Ajustar los modelos y combinarlos para obtener una solución
"""

# ╔═╡ bb0fa793-9662-4926-9eef-077d22963c15
md"""
## Ajustar los modelos y combinarlos para obtener una solución

1. Intentar ajustar cada uno de los modelos seleccionados.
1. Probar un ensamblado con todos los modelos seleccionados.
1. Registrar todas las conclusiones obtenidas.
"""

# ╔═╡ 5842c918-9ebf-4903-be1c-7f5377b66002
md"""
# Presentar la solución
"""

# ╔═╡ 2754be9e-f58b-4921-a3dc-d43eed844362
md"""
## Presentar la solución

1. Recopilar todas las conclusiones, ideas y detalles que se han ido registrando durante el desarrollo del proyecto.
1. Hacer una presentación de los resultados obtenidos.
1. Realizar una crítica del trabajo realizado.
1. Plantear posibles mejoras.
"""

# ╔═╡ 9c99a746-9534-477c-8f73-8a5cf4f94a78
md"""
# Poner en marcha el proyecto y mejorarlo continuamente
"""

# ╔═╡ a1f64f7d-eedb-4ac3-9ceb-e8b751a561c9
md"""
## Poner en marcha el proyecto y mejorarlo continuamente

1. Lanzar el proyecto y monitorizarlo a lo largo del tiempo.
1. Revisar los datos de monitorización.
1. Detectar posibles fallos del modelo en producción.
1. Documentar posibles mejoras para una nueva iteración del proyecto.
"""

# ╔═╡ 2532887c-9182-4167-8ee8-6d3d5cf486e5
md"""
# Resumen
"""

# ╔═╡ 1160e27a-611a-4868-8cd3-77fe30fa4edd
md"""
## Resumen

- Tener una estrategia para el desarrollo de un proyecto de AA es fundamental para alcanzar el éxito en su desarrollo.
- Es muy importante *familiarizarse* con los datos, así como con el contexto del problema.
- Hacer una exploración inicial de los datos permite *familiarizarse* con ellos.
- Si existe la posibilidad, hablar con los especialistas en la materia para aclara cualquier duda.
"""

# ╔═╡ 8eb3ddb5-8ccd-446e-9e1d-bc1150e3f8d9
md"""
## Resumen

- Probar varios modelos antes de decidirse por uno.
- No realizar una ajuste de parámetros temprano porque puede sesgar la selección del modelo.
- Si es posible, combinar más de una modelo en la solución final.
- Anotar todas la ideas que vayan surgiendo.
- Cuidar la presentación de los resultados.
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.61"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.6"
manifest_format = "2.0"
project_hash = "6d1b77f27e79835fc27b2d7e99ab8fcaf37aa976"

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

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.6+0"

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

[[deps.Statistics]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ae3bb1eb3bba077cd276bc5cfc337cc65c3075c0"
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.11.1"

    [deps.Statistics.extensions]
    SparseArraysExt = ["SparseArrays"]

    [deps.Statistics.weakdeps]
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

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
# ╠═1db7081a-2bdc-11f0-0c01-4b497af0a469
# ╠═0a611a39-2f34-4190-84ea-36791341ed75
# ╠═8211d6e7-2da7-42dc-a03e-f7e8827d0bd4
# ╠═2ba8182e-7607-4516-8c65-7f2b8362672f
# ╠═39a5d438-5034-49d6-9256-9fbe43fc9e01
# ╠═276f4250-bd8c-408c-9cb0-0493f5f7b3fe
# ╠═67b9cdd1-32b3-431b-9fe4-88fc39057eb0
# ╠═27f48568-a11b-43a0-bc3a-a2306d3c9963
# ╠═1eff8fdd-86c1-4ffc-ad78-cce43ee6a786
# ╠═f8771a01-7018-4ae0-a6e6-1d21c994e9a9
# ╠═72a9bc7c-bb05-4814-97d6-90fb17bb37bb
# ╠═856c2092-93ac-4c77-8455-d838fd74ab62
# ╠═de706874-4566-477a-bcfa-f4145ea20cdb
# ╠═ecf46d82-359d-4e54-ac55-e5748a9ef97e
# ╠═27d4223e-66cb-4136-9d37-bb9cc704a0da
# ╠═7bef6ea7-48f6-4491-9b16-e08a5597c101
# ╠═7cdff43a-666f-4704-9ece-dbc81d0ab0c2
# ╠═eeb73f9a-c310-491c-8d11-3e4954302d59
# ╠═d119b883-aaa4-48c4-ab72-1c40913a04ce
# ╠═bb0fa793-9662-4926-9eef-077d22963c15
# ╠═5842c918-9ebf-4903-be1c-7f5377b66002
# ╠═2754be9e-f58b-4921-a3dc-d43eed844362
# ╠═9c99a746-9534-477c-8f73-8a5cf4f94a78
# ╠═a1f64f7d-eedb-4ac3-9ceb-e8b751a561c9
# ╠═2532887c-9182-4167-8ee8-6d3d5cf486e5
# ╠═1160e27a-611a-4868-8cd3-77fe30fa4edd
# ╠═8eb3ddb5-8ccd-446e-9e1d-bc1150e3f8d9
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
