### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ d9c87ebb-d033-4d6c-82dd-fe806c9f40d4
using PlutoUI

# ╔═╡ a8613fc2-a430-11f0-1cbc-5752a791d4b4
# html"""
# <link rel="stylesheet" type="text/css" href="https://belmonte.uji.es/Docencia/IR2130/Teoria/mi_estilo.css" media="screen" />
# """

# ╔═╡ ab6a53f4-0755-43e6-9980-4197976755ef
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 76ebc4c8-2cc7-43be-9cf0-c53d7d93f09d
imagenes = "https://belmonte.uji.es/Docencia/IR2130/Teoria/AprendizajePorRefuerzo/Imagenes/"

# ╔═╡ 99dfd58b-0076-4282-bc26-04be14b48eaf
md"""
# Aprendizaje por Refuerzo

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 45263a1d-0ade-43e7-b167-07d73edc6e4d
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 98a7f0ab-937f-43e3-a769-59ae9df7afc3
md"""
## Objetivos de aprendizaje

- Argumentar por qué no se puede aplicar la técnica Q-learning en algunos problemas con alta dimensionalidad en el espacio de estados.
- Argumentar cómo se reformula el aprendizaje por refuerzo para aplicar redes neuronales profundas.
- Construir una solución sencilla para ejemplos académicos.
"""

# ╔═╡ 2ba99e7c-f076-449e-87a0-8377162f2938
md"""
## Bibliografía

1. [Grokking Deep Reinforcement Learning](https://cataleg.uji.es/permalink/34CVA_UJI/1nbr95r/alma991004764852306336). Antonio Morales. Maning Publications. 2020.
1. Hands-on machine learning, Aurélien Géron.
"""

# ╔═╡ c4d433f0-912e-481d-8a60-d627f84e16c9
md"""
# Introducción
"""

# ╔═╡ 4a7b4cd1-df95-4deb-bac7-61b6bcda655a
md"""
## Introducción

Imagina un juego por ordenador, clásico, por simplificar, como el breakout, el número de estados posibles es enorme. Intentar aplicar el algoritmo Q es estos caso es inviable.

La idea es estos casos es reducir el número de parámetros (estados en el caso del algoritmo Q) que el algoritmo debe aprender, y utilizar una buena representación del problema.
"""

# ╔═╡ ed2e8efc-8554-4318-8338-9e5e91e7321d
md"""
## Introducción

Podemos utilizar una red neuronal para trabajar las ideas anteriores:

1. Pasamos del espacio de estados al espacio de parámetros de la red.
1. Utilizamos una representación dependiente del problema.
"""

# ╔═╡ 765c63ae-9565-4ad0-8c88-31ced0c455f3
md"""
# Aprendizaje por refuerzo profundo
"""

# ╔═╡ 550dfa63-0625-494f-8c21-eb89b753d865
md"""
## Deep Q Networks

Una de las primeras aproximaciones de aplicación de redes neuronales profundas a soluciones de aprendizaje por refuerzo apareció publicada como: [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602)

Recordemos la expresión de la actualización de Q:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ \boxed{R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a)} - Q(S_t,A_t) ]
```

Lo que queremos es que nuestra red neuronal _aprenda_ (aproxime) la parte encuadrada de la función $Q$.
"""

# ╔═╡ f151d343-9c01-4f13-95a2-d6cd823307fb
md"""
## Deep Q Networks

La idea es _relativamente_ sencilla: como no contamos con un conjunto de entrenamiento para entrenar la red, las muestras de entrenamiento los tomo del propio juego y con ellas calculo:

```math
y_t = R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) 
```

Que es la parte de actualización si el *learning rate* lo tomo como $1$.
"""

# ╔═╡ 5643b8cb-97fa-4c5f-83c1-5b91aa500257
md"""
## Deep Q Networks

El entrenamiento procede del siguiente modo:

Defino la función de pérdidas de la red como:

```math
\mathcal{L}(\theta) = \left[ y_i - Q(\theta)\right]^2
```

1. Tomo un estado en el juego, y veo que efecto tiene realizar todas las acciones posibles desde ese estado.
1. Calculo todas las $y_t$, ya que todas me sirven para entrenar la red.
1. Hago descenso de gradiente para minimizar la anterior función de pérdidas.
"""

# ╔═╡ a22eefbf-7c27-4da2-a245-2d8e24bad2ed
md"""
## Show me the code

Afortunadamente, estas técnicas están implementadas en paquetes de Python. Uno de estos paquetes es [stable-baseline3](https://stable-baselines3.readthedocs.io/en/master/#)

Instalamos el paquete:

```bash
micromamba install stable-baseline3
```

Importamos los paquetes necesarios:

```python
import gymnasium as gym
from stable_baselines3 import DQN
```
"""

# ╔═╡ 104c2496-af24-41b9-a445-5b962463dedc
md"""
## Show me the code

Y este es el código que crea el entorno y entrena una DQN:

```python
env = gym.make("LunarLander-v2", enable_wind=False, gravity=-10.0,
               wind_power=15.0, turbulence_power=1.5)

env.reset()
model = DQN("MlpPolicy", env, verbose=1, 
            gamma=0.99, learning_rate=5e-4, batch_size=128, buffer_size=50_000, 
            target_update_interval=250, train_freq=4, gradient_steps=-1,
            learning_starts=0, exploration_fraction=0.12, exploration_final_eps=0.1, 
            policy_kwargs={'net_arch': [256, 256]})

model.learn(total_timesteps=200_000)

env.close()
```

Fíjate en el elevado número de iteraciones que tenemos que hacer para que el resultado sea bueno.
"""

# ╔═╡ 60d7b109-c6d4-41ae-8330-a24804242111
md"""
## Show me the code

Y este es el resultado
"""

# ╔═╡ b3829d4a-88df-40c8-a90e-80b949f76581
Resource(
	imagenes * "lunar_lander.gif",
	:alt => "Lunar lander",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 017af455-ec55-497a-ac07-3fbd6fbc1bed
md"""
Aterrizaje conseguido!!!
"""

# ╔═╡ eec0f571-889b-4344-96c8-2e84bb1ffa3b
md"""
# Resumen
"""

# ╔═╡ db6ad5c6-87d7-4b7b-becf-9f232365f0f1
md"""
## Resumen
1. En problemas con espacios de estados de gran dimensionalidad no se pueden aplicar directamente las técnicas de aprendizaje por refuerzo.
1. Aplicar redes neuronales en aprendizaje por refuerzo implica: redefinir el espacio de estados.
1. La red neuronal _aprenderá_ a ajustar los pesos a partir de la información que le proporciona el entorno y de las posibles acciones.
"""

# ╔═╡ 4fecad69-0852-4de4-b015-69591344836d
md"""
## Referencias

1. [Stable Baseline 3.](https://stable-baselines3.readthedocs.io/en/master/#)
1. [Deep Reinforcement Learning for Robotics: A Survey of Real-World
   Successes.](https://arxiv.org/pdf/2408.03539)
1. [A friendly introduction to deep reinforcement learning, Q-networks and policy gradients](https://www.youtube.com/watch?v=SgC6AZss478&t=23s)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"

[compat]
PlutoUI = "~0.7.68"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "701fb2bf464f8aa820d1ec11f0c1b4d98e4b75fe"

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
git-tree-sha1 = "67e11ee83a43eb71ddc950302c53bf33f0690dfe"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.12.1"
weakdeps = ["StyledStrings"]

    [deps.ColorTypes.extensions]
    StyledStringsExt = "StyledStrings"

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
version = "1.7.0"

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

[[deps.JuliaSyntaxHighlighting]]
deps = ["StyledStrings"]
uuid = "ac6e5ff7-fb65-4e79-a425-ec3bc9c03011"
version = "1.12.0"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.4"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "OpenSSL_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "8.15.0+0"

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

[[deps.Markdown]]
deps = ["Base64", "JuliaSyntaxHighlighting", "StyledStrings"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"
version = "1.11.0"

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
version = "3.5.4+0"

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
git-tree-sha1 = "0f27480397253da18fe2c12a4ba4eb9eb208bf3d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.5.0"

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

[[deps.Tricks]]
git-tree-sha1 = "372b90fe551c019541fafc6ff034199dc19c8436"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.12"

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
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.7.0+0"
"""

# ╔═╡ Cell order:
# ╠═a8613fc2-a430-11f0-1cbc-5752a791d4b4
# ╠═d9c87ebb-d033-4d6c-82dd-fe806c9f40d4
# ╠═ab6a53f4-0755-43e6-9980-4197976755ef
# ╠═76ebc4c8-2cc7-43be-9cf0-c53d7d93f09d
# ╠═99dfd58b-0076-4282-bc26-04be14b48eaf
# ╠═45263a1d-0ade-43e7-b167-07d73edc6e4d
# ╠═98a7f0ab-937f-43e3-a769-59ae9df7afc3
# ╠═2ba99e7c-f076-449e-87a0-8377162f2938
# ╠═c4d433f0-912e-481d-8a60-d627f84e16c9
# ╠═4a7b4cd1-df95-4deb-bac7-61b6bcda655a
# ╠═ed2e8efc-8554-4318-8338-9e5e91e7321d
# ╠═765c63ae-9565-4ad0-8c88-31ced0c455f3
# ╠═550dfa63-0625-494f-8c21-eb89b753d865
# ╠═f151d343-9c01-4f13-95a2-d6cd823307fb
# ╠═5643b8cb-97fa-4c5f-83c1-5b91aa500257
# ╠═a22eefbf-7c27-4da2-a245-2d8e24bad2ed
# ╠═104c2496-af24-41b9-a445-5b962463dedc
# ╠═60d7b109-c6d4-41ae-8330-a24804242111
# ╠═b3829d4a-88df-40c8-a90e-80b949f76581
# ╠═017af455-ec55-497a-ac07-3fbd6fbc1bed
# ╠═eec0f571-889b-4344-96c8-2e84bb1ffa3b
# ╠═db6ad5c6-87d7-4b7b-becf-9f232365f0f1
# ╠═4fecad69-0852-4de4-b015-69591344836d
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
