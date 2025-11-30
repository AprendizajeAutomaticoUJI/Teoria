### A Pluto.jl notebook ###
# v0.20.21

using Markdown
using InteractiveUtils

# ╔═╡ 646115db-9b69-482e-b0bf-fb287ed023c4
using PlutoUI

# ╔═╡ 80589f75-0e75-4171-9f8e-d08ecc8cdf67
using PlutoTeachingTools

# ╔═╡ 963761bd-4b3c-40cb-96fd-d004e9e1e94d
using ShortCodes

# ╔═╡ f722bfd0-66eb-11f0-01ea-7b434238558a
# html"""
# <link rel="stylesheet" type="text/css" href="https://belmonte.uji.es/Docencia/IR2130/Teoria/mi_estilo.css" media="screen" />
# """

# ╔═╡ 913974f6-4c99-4f2e-be95-9a66c8fce004
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ 25e87a36-ca03-4497-9571-165f9f9ce5e0
imagenes = "https://belmonte.uji.es/Docencia/IR2130/Teoria/AprendizajePorRefuerzo/Imagenes/"

# ╔═╡ 270a45a5-bfb3-47a4-b5fb-981cb9343af1
md"""
# Aprendizaje por Refuerzo

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)
"""

# ╔═╡ 49dde659-7259-44f8-9352-7b9a09a8a324
Resource(
	"https://belmonte.uji.es/imgs/uji.jpg",
	:alt => "Logo UJI",
	:width => 400,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 9344959d-aab8-4c21-96bd-7b0a16d6f068
md"""
## Objetivos de aprendizaje

- Interpretar cuales son las características del aprendizaje por refuerzo.
- Resumir los conceptos de agente, estado, acción y recompensa.
- Conectar cada uno de los conceptos anteriores con el proceso de aprendizaje.
- Construir una solución utilizando el algoritmo Q-learning.
"""

# ╔═╡ 9bb76759-af68-4f85-8276-b99cabae0857
md"""
## Objetivos de aprendizaje

- Argumentar la utilidad del descuento.
- Interpretar el dilema explotación frente a exploración.
- Argumentar la utilidad del parámetro $\epsilon$.
"""

# ╔═╡ 19069594-b31c-47fa-9e58-ba183fae3409
md"""
## Bibliografía

1. [Reinforcement learning: An introduction](http://incompleteideas.net/book/the-book.html). Disponible on-line.
1. [Grokking Deep Reinforcement Learning](https://cataleg.uji.es/permalink/34CVA_UJI/1nbr95r/alma991004764852306336). Antonio Morales. Maning Publications. 2020.
"""

# ╔═╡ e65a18ad-9f34-4afc-a3c7-3f9f1bd346a1
md"""
# Introducción
"""

# ╔═╡ f62853c5-1bb6-4ec4-b256-695121d29029
md"""
## Introducción

Durante el curso, hemos visto algoritmos de aprendizaje automático que se basan en un conjunto de datos de entrenamiento para construir un modelo.

En el caso del **aprendizaje por refuerzo** no existe ningún conjunto de datos con el que construir el modelo, en vez de ello, operamos en un entorno que nos devuelve una recompensa dependiendo de la acción que realicemos y el estado en el que nos encontremos.

Son necesarios nuevos algoritmos para encontrar las soluciones dentro de este nuevo paradigma.
"""

# ╔═╡ 322740c2-ad9e-4335-bb12-bf71006de7a1
md"""
## Introducción

En el **aprendizaje supervisado** construimos un modelo con datos y valores de salida. Nuestro modelo nos dará un nueva salida para cada nuevo dato.

En **aprendizaje no supervisado** construimos un modelo sólo con datos, y el modelo aprende a identificar las similitudes entre los datos.
"""

# ╔═╡ bc67e92a-4c09-4104-882f-632db5972747
md"""
## Introducción

En **aprendizaje por refuerzo** tenemos **estados** (dentro de un **entorno**), **acciones** (ejecutadas por un **agente**) y **recompensas**, y el objetivo es maximizar la recompensa global tomando las mejores acciones en cada uno de los estados por los que vamos pasando.
"""

# ╔═╡ 0d30181c-687b-472a-92a1-4d60016596c6
Resource(
	imagenes * "rl.png",
	:alt => "Aprendizaje por refuerzo",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ a24c2f05-b7e1-4a27-b770-a66022306bdb
html"""
Fuente: Reinforcement learning: An introduction
"""

# ╔═╡ 3d047e1f-d2ee-464a-bb30-9ff08acaf3f4
md"""
## Introducción

Estrategia en juegos:
"""

# ╔═╡ b0e55979-4466-47d6-8a75-39a6ca1a7192
Columns(
	md"Estrategia en juegos",
	Resource(
		imagenes * "alpha_go.png",
		:alt => "Logo de Alpha Go",
		:width => 400,
		:style => "display: block; margin: auto;",
	);
	widths = [30, 70]
)

# ╔═╡ 539dcba5-102d-4c11-a400-94a6ca077b5b
YouTube("WXuK6gekU1Y")

# ╔═╡ 46d7f494-efff-40d2-a1bc-bca8ffd0cd9f
md"""
## Introducción
"""

# ╔═╡ 38115240-6cf5-47ab-bb91-94f8c5f37b4a
Columns(
	md"Plafinicación en robótica",
	Resource(
		imagenes * "robotics_rl.jpg",
		:alt => "Planificación en robótica",
		:width => 400,
		:style => "display: block; margin: auto;",
	);
	widths = [30, 70]
)

# ╔═╡ 0d625f98-49c6-4cf3-baff-9c973145e484
md"""
## Introducción
"""

# ╔═╡ dfba5a6e-826f-49d4-ae8f-fa1b9b86158f
Columns(
	md"Medicina y salud",
	Resource(
		imagenes * "alpha_fold.png",
		:alt => "Medicina y salud",
		:width => 400,
		:style => "display: block; margin: auto;",
	);
	widths = [30, 70]
)

# ╔═╡ b3ec2529-5127-4f4c-a221-07b1f665081b
md"""
## Introducción
"""

# ╔═╡ 39adf584-77e4-4109-9cf7-303615c2e638
Columns(
	md"Conducción autónoma",
	Resource(
		imagenes * "conduccion_autonoma.jpg",
		:alt => "Conducción autónoma",
		:width => 400,
		:style => "display: block; margin: auto;",
	);
	widths = [30, 70]
)

# ╔═╡ 3551e3f8-8e24-413c-9947-0babb6910cfb
md"""
# Conceptos clave
"""

# ╔═╡ e785e43e-c7c4-4603-b3f2-241efabc4697
md"""
## Entorno, Agente, Acción, Estado y Recompensa

El **entorno** es todo aquello que el agente puede observar y con lo que puede interaccionar. El entorno puede reaccionar a las acciones realizadas por el agente, y devolverle recompensas.
"""

# ╔═╡ 447bb4af-4da1-40f0-98a3-d68e2b34b5b1
Resource(
	imagenes * "rl.png",
	:alt => "Aprendizaje por refuerzo",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 4b07e8e5-524d-4e68-8a0d-f5cd7c47a894
md"""
## Entorno, Agente, Acción, Estado y Recompensa

El **agente** es quien decide qué acción tomar.
"""

# ╔═╡ 3a92bd3f-4d2a-41a9-9f77-70edf3fbf10c
Columns(
	Resource(
		imagenes * "rl.png",
		:alt => "Aprendizaje por refuerzo",
		:width => 900,
		:style => "display: block; margin: auto;",
	),
	md"""
	El **agente** debe escoger la siguiente **acción** a llevar a cabo dependiendodel **estado** actual del **entorno** con el objetivo de maximizar la **recompensa**.
	"""
)

# ╔═╡ b29b0b8c-ed15-49ea-95f5-6634a6d74178
md"""
$A = \{a_1, a_2,...,a_n\}$

Representa el conjunto de posibles acciones.
"""

# ╔═╡ 7464261c-999c-4397-b411-bd07025a3beb
md"""
## Entorno, Agente, Acción, Estado y Recompensa

La secuencia de pasos durante el proceso de aprendizaje es:
"""

# ╔═╡ 10d98301-fc2d-4125-94a2-2ea7f3b42307
Columns(
	Resource(
		imagenes * "rl.png",
		:alt => "Aprendizaje por refuerzo",
		:width => 900,
		:style => "display: block; margin: auto;",
	),
	md"""
	1. Se observa el estado del entorno.
	1. Se realiza una acción.
	1. Se obtiene una recompensa.
	"""
)

# ╔═╡ 3d832db2-de0c-41e1-bf1e-cc0c152f4f98
md"""
$\{S_1, A_1, R_1, S_2, A_2, R_2,...,S_t,A_t,R_t\}$
"""

# ╔═╡ 0e3c3c29-76a2-4404-8fdb-8b442ce1eb9f
md"""
## Entorno, Agente, Acción, Estado y Recompensa
"""

# ╔═╡ 8cbb1134-49b5-472c-aadb-1754230742ea
Resource(
	imagenes * "rl.png",
	:alt => "Aprendizaje por refuerzo",
	:width => 900,
	:style => "display: block; margin: auto;",
)

# ╔═╡ 57e7e52c-5cef-463d-87ac-b0bb606ddfe9
md"""
Un detalle importante: observa que el entorno puede cambiar, si el agente es un jugador virtual de ajedrez, a cada movimiento del agente le sigue un movimientodel contrario.
"""

# ╔═╡ d17c48c8-259a-4ca9-97e7-c844370c52fe
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Si el conjunto de estados, acciones y recompensas es finito, se conoce el 
estado actual y la acción elegida, entonces, se puede calcular la probabilidad
del siguiente estado y la recompensa obtenida.

```math
p(s',r|s,a) = p(S_t=s', R_t=r | S_{t-1} = s, A_{t-1} = a)
```

Con la condición: ``\sum\limits_{s' \in S} \sum\limits_{r \in R}^{} p(s',r|s,a) = 1``

Fíjate en que la igualdad anterior es simplemente una definición.
"""

# ╔═╡ 1e0cad5b-10b0-4839-b112-25f5b15f7798
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Si sumamos sobre todas las recompensas que el agente puede obtener después de realizar la acción ``a``, obtenemos las probabilidades de transición a los siguienetes estados:

```math
p(s'|s, a) = p(S_t = s' | S_{t-1} = s, A_{t-1} = a) = \sum\limits_{r \in R}^{} p(s',r|s,a)
```

Fíjate en que, de nuevo, esta nueva expresión es símplemente una manipulación con la definición de probabilidad condicionada.
"""

# ╔═╡ 7ef876f4-b64d-4a20-9e69-95c08075275c
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Por otro lado, si calculamos el valor esperado de las recompensas que el agente puede obtener después de realizar la acción ``a`` desde el estado ``s``, obtenemos la recompensa esperada para al realizar la acción ``a`` en el estado ``s``:

```math
r(s, a) = E[R_t | S_{t-1} = s, A_{t-1} = a) = \sum\limits_{r \in R}^{} r \sum\limits_{s' \in S} p(s',r|s,a)
```

Que vuelve a ser una manipulación de las expresiones de probabilidad anteriores.
"""

# ╔═╡ 2ead2629-7a7e-4ad4-86bf-50698e12f13d
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Finalmente, también podemos calcular el valor esperado de la recompensa al llegar al estado ``s'`` desde el estado ``s`` al realizar la acción ``a``:

```math
r(s, a, s') = E[R_t | S_{t-1} = s, A_{t-1} = a, S_t = s') = \sum\limits_{r \in R}^{} r \frac{p(s',r|s,a)}{p(s'|s,a)}
```

Donde hemos dividido por ``p(s'|s,a)`` para que todas las probabilidades ``\frac{p(s',r|s,a)}{p(s'|s,a)} = 1``.
"""

# ╔═╡ 13c68676-b38b-4fe2-91e8-f9a808ebef96
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Tal y como lo hemos definido, este marco de trabajo se conoce con el nombre de Procesos de Decisión de Markov (*Markov Decision Process, MDP*).
"""

# ╔═╡ 5d78edeb-d092-459c-ad88-6345fb838350
md"""
## Entorno, Agente, Acción, Estado y Recompensa

Ejemplos de agente pueden ser:

* Un coche autónomo.
* Un robot en un entorno industrial.
* Un dron.
* Un jugador virtual de go o ajedrez.
* Un personaje en un video juego.
"""

# ╔═╡ 7a1582ac-c323-48de-ba53-fad1ce12ea26
Columns(
	Resource(
		imagenes * "drone.jpg",
		:alt => "Drone",
		:width => 400,
		:style => "display: block; margin: auto;",
	),
	Resource(
		imagenes * "robot_industrial.jpg",
		:alt => "Robot industrial",
		:width => 400,
		:style => "display: block; margin: auto;",
	)
)

# ╔═╡ 43e5877d-1cb7-4a6f-9331-e1ec7bf4ef8b
html"""
Fuente: Pexels.
"""

# ╔═╡ 79784c4f-b1c8-4ad7-ac1e-e9b2331f3663
md"""
## Episodios y Ganancia

En casos como los de un coche autónomo, o un jugador virtual existe uno, o varios, estados finales. Un coche lleva a los pasajeros del punto de recogida al punto de entrega; un jugador virtual gana o pierde una partida, a este concepto lo llamaremos **episodio**.

En esta caso resulta sencillo definir el concepto de **Ganancia** a futuro como la suma de las recompensas que obtiene el agente durante el episodio a partir de un cierto instante y hacia adelante:

$G_t = R_{t+1} + R_{t+2} + ... + R_T$

Donde $T$ es el tiempo final del episodio.
"""

# ╔═╡ 4e5ad759-e840-43c0-a342-ebf413a5fb2b
md"""
## Episodios y Ganancia

Hay otros casos, como el de un robot en un ambiente industrial, donde no podemos identificar episodios, el agente está realizando de modo continuo acciones y la secuencia de acciones nunca acaba.

En este caso, para acotar la ganancia para que no crezca hacia el infinito, se introduce el concepto de **factor de descuento**, que es una especie de decaimiento exponencial de la recompensa a futuro. 

$G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... = 
\sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$

Una analogía monetaria, 100 euros de hoy no tendrán el mismo valor dentro de 
10 años, se habrán devaluado.
"""

# ╔═╡ 77ab5a72-0022-42cb-bd72-24f6ea51c201
md"""
## Episodios y Ganancia

La ganancia se puede expresar de una manera más conveniente del siguiente modo:

```math
G_t = R_{t+1} + \gamma G_{t+1}
```

Expresión recursiva en la que se relaciona la ganancia en un instante con la ganancia en el siguiente instante posterior.
"""

# ╔═╡ cf4d406f-9570-4bf8-a266-9f6d7cac24ed
md"""
## Episodios y Ganancia

El **factor de descuento** $\gamma$ también se utiliza cuando el espacio de estados es finito.

El dominio suele ser $0 \leq \gamma \leq 1$, siendo un valor típico $\gamma =
0.99$.
"""

# ╔═╡ 3d815877-88a7-44b6-b9df-8d329d1a7e3b
md"""
## Política

La acción se escoge según cierta política $\pi$:

$\pi: S \rightarrow A$

La política $\pi$ puede ser:

* **Determinista**, $a = \pi(s)$ la acción a escoger depende del estado actual.
* **No determinista** $a = p(a|s)$, se escoge la acción según cierta probabilidad asignada a cada una de ellas.
"""

# ╔═╡ 2e8856b0-a21a-46dc-9601-f928b31245c1
md"""
## Función valor del estado

La **Función valor** mide la recompensa total esperada que se obtiene desde el estado ``s`` hasta el final del juego (o episodio), siguiendo una política concreta ``\pi``.

Un ejemplo para entender el objetivo de la función valor es el juego del ajedrez. En el ajedrez es importante controlar las posiciones del centro del tablero porque son estas las más importantes, ya que si realizo movimientos desde ellas la probabilidad de ganar la partida aumenta.

La función valor asigna un valor a cada uno de los posibles estados en los que se puede encontrar el agente, es como calcular un valor para cada uno de los escaques del juego del ajedrez.

En términos de valor esperado, la función valor ``v_\pi`` para un determinado estado ``s`` es el valor esperado para la ganancia si se sigue la política ``\pi`` desde ese estado ``s``:

```math
v_\pi(s) = E_\pi[G_t | S_t = s] = E_\pi\left[ \sum_{k=0}^{\infty} \gamma^kR_{t+k+1} | S_t = s \right]
```
"""

# ╔═╡ 26c0db62-9f96-43cc-a18b-f64290293576
md"""
## Función valor del estado

La función valor de estado se puede expresar de forma recursiva en función de los siguiente estados posible al actual:

$$\begin{align}
v_\pi(s) &= E_\pi[G_t | S_t = s] \\
&= E_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s] \\
&= \sum_a \pi(a|s) \sum_{s'} \sum_r p(s',r | r,a) \left[r + \gamma E_\pi[G_{t+1} | S_{t+1} = s']\right] \\
&= \sum_a \pi(a|s) \sum_{s',r'} p(s',r | r,a) \left[r + \gamma v_\pi(s')]\right]
\end{align}$$

A esta expresión se la conoce como **Ecuación de Bellman**.
"""

# ╔═╡ fc948717-9f3b-403c-8f65-d05d7bb92b85
md"""
## Función valor de la acción/estado

De modo análogo, la ganancia esperada (valor esperado de la ganancia) al realizar una acción ``a`` en un estado ``s`` al seguir la política ``\pi`` es:

```math
q_\pi(s,a) = E_\pi(G_t|S_t = s, A_t = a) = E_\pi \left[ \sum_{k=0}^{\infty} \gamma^kR_{t+k+1} | S_t = s, A_t = a \right]
```

Que también la podemos expresar de manera recursiva como:

```math
\begin{align}
q_\pi(s,a) &= E_\pi[G_t | S_t = s, A_t = a] \\
&= E_\pi[R_{t+1} + \gamma G_{t+1} | S_t = s, A_t = a] \\
&= \sum_{a'} \sum_{s'} \sum_r p(s',r | r,a) \left[r + \gamma E_\pi[G_{t+1} | S_{t+1} = s', A_{t+1} = a']\right] \\
&= \sum_{a',s',r'} p(s',r | r,a) \left[r + \gamma q_\pi(s',a')]\right]
\end{align}
```
"""

# ╔═╡ 68fb378b-e910-4344-b0aa-832ebe930ba1
md"""
## Función valor de la acción/estado

Fíjate en que políticas diferentes van a dar lugar a funciones de valor estado y valor acción estado distintas; en ajedrez la política «apropiarse del centro» da diferentes valores de para los estados y las acciones estado que la política «proteger al rey».

Podemos establecer un orden entre dos políticas del siguiente modo: una política ``\pi`` es mejor que otra política ``\pi'`` si ``v_\pi(s) \ge v_{\pi'}(s)`` para todos los posibles estados.

Con esta relación de orden, podemos encontrar al menos una política ``\pi^*``, quizás más de una, para la que sus valores de ``v_{\pi^*}(s)`` sean mayores que para cualquier otra política. O dicho de otro modo, buscamos, la política cuyas acciones para cada estado maximizan la ganancia.
"""

# ╔═╡ 1e7a508a-0745-46ad-bd0e-a53fe94d4ec7
md"""
## Función valor de la acción/estado

Lo que buscamos es encontrar una política que maximice la ganancia esperada a futuro, es decir, buscamos las política ``\pi_*`` para la que la función valor estado:

```math
v_{\pi^*}(s) = \max_a \sum_{s',r'} p(s',r | r,a) \left[r + \gamma v_{\pi^*}(s')]\right]
```

sea la máxima para cada estado, y también lo sea la función acción estado:

```math
q_{\pi^*}(s,a) = \sum_{s',r'} p(s',r | r,a) \left[r + \gamma \max_a' q_{\pi^*}(s',a')]\right]
```

"""

# ╔═╡ c45243d4-456d-494b-b27b-1ea79d2e0b48
md"""
# El algoritmo Q-learning
"""

# ╔═╡ 63e07476-a0bb-4dc4-9f3d-5f315e9880de
md"""
## Función Q

Una aproximación al óptimo de la **función acción-valor** es el algoritmo **Q-learning**:

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) - Q(S_t,A_t) ]$

Donde $\alpha$ es la tasa de aprendizaje y $\gamma$ es el factor de descuento.

La función $Q(s,a)$ es una medida de lo bueno que es estar en el estado $s$, y realizar la acción $a$; mide la _calidad_ (Q-uality) de la pareja estado-acción.
"""

# ╔═╡ f9c33887-db0e-4757-90b9-d6f91ccabf86
md"""
## Función Q

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) - Q(S_t,A_t) ]$

$\gamma \max_{a \in A} Q(S_{t+1},a) \rightarrow$ mejor estado que se puede conseguir al realizar una acción $a \in A$ (con descuento).

$R_{t+1} \rightarrow$ recompensa inmediata al realizar la acción $a \in A$.
"""

# ╔═╡ 7c118104-3aec-478c-a6f7-5bdee9defe76
Columns(
	Resource(
		imagenes * "frozen_lake_estado_14.png",
		:alt => "Lago helado",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	**Ejemplo**
	
	En el estado mostrado (14), la mejor acción que puede tomar el elfo es moverse
	hacia la derecha, por lo que la calidad (Q) de la pareja (estado=14,
	acción=derecha) se incrementa.
	""";
	widths = [30, 70]
)

# ╔═╡ fb18dbd7-5223-45c7-bd20-d6c648c01c2b
md"""
## Función Q

$Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) - Q(S_t,A_t) ]$

Fíjate en que la función $Q(S_t,A_t)$ se actualiza al elegir la mejor acción 
$a$ en cada paso del aprendizaje.

Un detalle muy interesante de esta función es que converge hacia el óptimo con 
independencia de la política que utilicemos.
"""

# ╔═╡ dbd20ea5-091a-4f08-84e4-7639cf9339b5
md"""
## Función Q en espacios finitos

Para ver cómo funciona el algoritmo **Q-learning** vamos a utilizar, inicialmente, un problema con un número de estados finito.
"""

# ╔═╡ a18f901f-9be6-49f0-aed5-5f0089fdf165
Columns(
	Resource(
		imagenes * "frozen_lake.png",
		:alt => "Lago helado",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	**Lago Helado**:
	
	El agente debe llegar desde la posición de inicio hasta el regalo, si se alcanza 
	el regalo, la recompensa es 1.
	
	Las posibles **acciones** son moverse a la izquierda, abajo, derecha, o arriba 
	desde la posición actual.
	
	El juego acaba si se cae en un agujero, o si el número de acciones antes de 
	alcanzar el regalo alcanza un límite, la recompensa es 0.
	""";
	widths = [30, 70]
)

# ╔═╡ d4dd8e39-baa8-4077-8f21-69d9de1c1b33
md"""
## Función Q en espacios finitos
"""

# ╔═╡ fb43650f-53e1-4af6-8f30-09389bd024dd
Columns(
	Resource(
		imagenes * "frozen_lake.png",
		:alt => "Lago helado",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	El problema tiene 16 estado posibles (uno por cada casilla), y las acciones que el agente puede tomar son 4.
	
	|       | Izquierda   | Abajo     | Derecha  | Arriba  | 
	|-------|:-----------:|:---------:|:--------:|:-------:|
	|**1**  |     0       |     0     |     0    |    0    |
	|**2**  |     0       |     0     |     0    |    0    |
	|**...**|     0       |     0     |     0    |    0    |
	|**15** |     0       |     0     |     0    |    0    |
	|**16** |     0       |     0     |     0    |    0    |
	""";
	widths = [30, 70]
)

# ╔═╡ 5a45ca41-dcf4-4b38-9af7-a047378aeeff
md"""
Inicialmente la matriz está llena de ceros.
"""

# ╔═╡ d72fa66d-21df-402e-bfc4-0c4d3fddf646
md"""
## Función Q en espacios finitos
"""

# ╔═╡ 08c1a771-382a-445a-b430-19f28d58dfcf
Columns(
	Resource(
		imagenes * "frozen_lake.png",
		:alt => "Lago helado",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	Nuestra tabla Q tiene 16 x 4 = 64 elementos.
	
	Si aplicamos el algoritmo:
	```python
	while not terminado:  
	    accion = np.argmax(Q[estado,:])
	    nuevo_estado, recompensa, terminado,_,_ = lago.step(accion)
	    Q[estado, accion] += alfa * (recompensa + 
	        gamma*np.max(Q[nuevo_estado,:]) - Q[estado, accion])
	    estado = nuevo_estado
	```
	""";
	widths = [25, 75]
)

# ╔═╡ 52c34bd3-9ac5-49dd-94ce-0d46e968d25c
md"""
```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) - Q(S_t,A_t) ]
```

Veremos que, simplemente, el algoritmo no aprende. ¿Por qué?
"""

# ╔═╡ 0fed9fc3-8914-4570-971b-f53397e98070
md"""
## Función Q en espacios finitos

El problema en esta expresión:

```math
Q(S_t,A_t) \leftarrow Q(S_t,A_t) + \alpha[ R_{t+1} + \gamma \max_{a \in A} Q(S_{t+1},a) - Q(S_t,A_t) ]
```

Está en esta parte:

```math
Q(S_t,A_t) \leftarrow  ... \max_{a \in A} Q(S_{t+1},a) ...
```

Como todas las posiciones de la tabla son **0**, nunca se actualiza la tabla, dicho de otro modo, el algoritmo no _aprende_.
"""

# ╔═╡ 46198f73-8966-43ff-9878-13de4cca96d9
md"""
## Función Q en espacios finitos

Este problema se conoce con el nombre de *Dilema de exploración frente a explotación*.

Si siempre queremos explotar (maximizar) el resultado, puede que nunca exploremos nueva soluciones que nos pueden acercar al óptimo.

¿Cómo lo solucionamos?
"""

# ╔═╡ 0e7bcdd9-eb1e-4099-bd54-fad2c3992b4b
md"""
## Función Q en espacios finitos

Introduciendo un factor de exploración en el algoritmo.

```python
def selecciona_accion(estado):
    if rng.random() < epsilon: 
        accion = lago.action_space.sample()
    else:
        accion = np.argmax(Q[estado, :])

    return accion
```

Y la introducimos en el algoritmo Q:

```python
while not terminado:  
    accion = selecciona_accion(estado)
    nuevo_estado, recompensa, terminado, _, _ = lago.step(accion)
    Q[estado, accion] += alfa * (recompensa + gamma*np.max(Q[nuevo_estado,:]) - 
        Q[estado, accion])
    estado = nuevo_estado

epsilon = max(epsilon - epsilon_decay, 0)
```
"""

# ╔═╡ 5ffc2df3-9fc7-4e54-aca3-1145f5f09b72
md"""
## Función Q en espacios finitos

A esta técnica se le llama **Q-learning $\epsilon$ - voraz**

El valor inicial de $\epsilon$ puede ser $1$ y decaer en cada episodio de aprendizaje con alguna tasa $\epsilon$-decay $=0.00001$.

Al principio la política es de exploración, y cuando la tabla **Q** ya contiene algunos valores, $\epsilon$ ha decaído de manera que se pasa a la fase de explotación.
"""

# ╔═╡ 34a29918-675d-4d18-bf2c-728166acc9aa
md"""
## Función Q en espacios finitos

Con esta mejora, el algoritmo ya es capaz de aprender, y encontrar la solución:
"""

# ╔═╡ fb24292b-4737-4f9f-a5b6-0c4a0d85bf53
Columns(
	Resource(
		imagenes * "lago_helado.gif",
		:alt => "Lago helado",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	Fíjate en que no hay una única solución posible ya que la fase de exploración es estocástica.
	""";
	widths = [30, 70]
)

# ╔═╡ 1ea8f988-ef7b-485c-a901-d172f0f88471
md"""
Veamos ahora cómo podemos tratar el caso de entornos con espacios de estados continuo.
"""

# ╔═╡ a5afa66c-7447-454d-b0a8-03dba7fb1bbd
md"""
## Función Q en espacios continuos

Hay casos en los que el espacio de estados no es continuo.
"""

# ╔═╡ 9ba9044e-3f77-4a48-a9ac-df196bd77f96
Columns(
	Resource(
		imagenes * "cartpole.gif",
		:alt => "Péndulo invertido",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	**Cartpole**: Hay que mantener el bastón en la vertical, sin que se caiga.

	El espacio de estado está formado por la posición de la base, su velocidad, el ángulo que forma el bastón con la vertical, y la velocidad angular.

	Todas las variables de este espacio son continuas.
	""";
	widths = [35, 65]
)

# ╔═╡ 6c6dbb72-61be-4e6f-93b1-73a15eae1646
md"""
¿Cómo procedemos?
"""

# ╔═╡ 29d0e749-6072-4888-8fc8-eae42a6b7f06
md"""
## Función Q en espacios continuos

Discretizamos el espacio, convertimos las variables continuas en discreta sobre un número suficiente de valores posibles (densidad).
"""

# ╔═╡ a235da3d-a99f-41c4-a9f6-47a8e975fbb5
Columns(
	Resource(
		imagenes * "cartpole.gif",
		:alt => "Péndulo invertido",
		# :width => 400,
		:style => "display: block; margin: auto;",
	),
	md"""
	```python
	espacio_posiciones = np.linspace(-2.4, 2.4, 10)
	espacio_velocidades = np.linspace(-4, 4, 10)
	espacio_angulos = np.linspace(-.2095, .2095, 10)
	espacio_velocidad_angular = np.linspace(-4, 4, 10)
	```
	Y, a partir de este momento, procedemos de igual forma que lo hemos hecho en el 
	caso discreto.
	""";
	widths = [35, 65]
)

# ╔═╡ 0cffafb0-a1d3-453f-a366-5cbf5b5f12da
md"""
Detalle importante, la matriz Q en este caso tiene 10 x 10 x 10 x 10 = 10.000 posiciones.
"""

# ╔═╡ fc6453cd-0a0e-4e4a-9601-962dc62b58cb
md"""
## Función Q en espacios continuos

Cuando el espacio de estados es muy grande, la aproximación con tablas Q no se puede aplicar.

Pero, podemos aplicar redes neuronales para ayudar en la resolución.
"""

# ╔═╡ 43bb1241-f83c-4566-8a26-cd990807471f
md"""
# Resumen
"""

# ╔═╡ 3831b6c1-d605-4d6b-b3ab-4dc8def26432
md"""
## Resumen

1. El aprendizaje por refuerzo no parte de un conjunto de datos de entrenamiento.
1. Los conceptos clave son el agente, los estados, las acciones y las recompensas.
1. El objetivo de los algoritmos de aprendizaje automático es maximizar la recompensa a futuro.
"""

# ╔═╡ e4463d7c-68d4-4f4f-9ba5-694c81054d20
md"""
## Resumen

4. Un algoritmo que se aproxima al óptimo del máximo de recompensa a futuro es el algoritmo **Q-learning**.
1. **Q-learning** tiene una aplicación directa en el caso de entornos discretos, pero si los queremos aplicar a entornos continuos, debemos discretizar los espacios de estados.
"""

# ╔═╡ fffa4aac-a902-4f8b-9e92-1cc52e49b017
md"""
## Referencias

1. [Reinforcement learning at MIT.](https://www.youtube.com/watch?v=8JVRbHAVCws)
1. [Gymnasium.](https://gymnasium.farama.org/)
"""

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
PlutoTeachingTools = "661c6b06-c737-4d37-b85c-46df65de6f69"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
ShortCodes = "f62ebe17-55c5-4640-972f-b59c0dd11ccf"

[compat]
PlutoTeachingTools = "~0.4.6"
PlutoUI = "~0.7.68"
ShortCodes = "~0.3.6"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.12.2"
manifest_format = "2.0"
project_hash = "70629a17fedc4b6b66c91c2e3f7dce41300725cb"

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
git-tree-sha1 = "411eccfe8aba0814ffa0fdf4860913ed09c34975"
uuid = "0f8b85d8-7281-11e9-16c2-39a750bddbf1"
version = "1.14.3"

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
version = "3.5.4+0"

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
git-tree-sha1 = "dacc8be63916b078b592806acd13bb5e5137d7e9"
uuid = "661c6b06-c737-4d37-b85c-46df65de6f69"
version = "0.4.6"

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
# ╠═f722bfd0-66eb-11f0-01ea-7b434238558a
# ╠═646115db-9b69-482e-b0bf-fb287ed023c4
# ╠═80589f75-0e75-4171-9f8e-d08ecc8cdf67
# ╠═963761bd-4b3c-40cb-96fd-d004e9e1e94d
# ╠═913974f6-4c99-4f2e-be95-9a66c8fce004
# ╠═25e87a36-ca03-4497-9571-165f9f9ce5e0
# ╠═270a45a5-bfb3-47a4-b5fb-981cb9343af1
# ╠═49dde659-7259-44f8-9352-7b9a09a8a324
# ╠═9344959d-aab8-4c21-96bd-7b0a16d6f068
# ╠═9bb76759-af68-4f85-8276-b99cabae0857
# ╠═19069594-b31c-47fa-9e58-ba183fae3409
# ╠═e65a18ad-9f34-4afc-a3c7-3f9f1bd346a1
# ╠═f62853c5-1bb6-4ec4-b256-695121d29029
# ╠═322740c2-ad9e-4335-bb12-bf71006de7a1
# ╠═bc67e92a-4c09-4104-882f-632db5972747
# ╠═0d30181c-687b-472a-92a1-4d60016596c6
# ╠═a24c2f05-b7e1-4a27-b770-a66022306bdb
# ╟─3d047e1f-d2ee-464a-bb30-9ff08acaf3f4
# ╠═b0e55979-4466-47d6-8a75-39a6ca1a7192
# ╠═539dcba5-102d-4c11-a400-94a6ca077b5b
# ╠═46d7f494-efff-40d2-a1bc-bca8ffd0cd9f
# ╠═38115240-6cf5-47ab-bb91-94f8c5f37b4a
# ╠═0d625f98-49c6-4cf3-baff-9c973145e484
# ╠═dfba5a6e-826f-49d4-ae8f-fa1b9b86158f
# ╠═b3ec2529-5127-4f4c-a221-07b1f665081b
# ╠═39adf584-77e4-4109-9cf7-303615c2e638
# ╠═3551e3f8-8e24-413c-9947-0babb6910cfb
# ╠═e785e43e-c7c4-4603-b3f2-241efabc4697
# ╠═447bb4af-4da1-40f0-98a3-d68e2b34b5b1
# ╠═4b07e8e5-524d-4e68-8a0d-f5cd7c47a894
# ╠═3a92bd3f-4d2a-41a9-9f77-70edf3fbf10c
# ╠═b29b0b8c-ed15-49ea-95f5-6634a6d74178
# ╠═7464261c-999c-4397-b411-bd07025a3beb
# ╠═10d98301-fc2d-4125-94a2-2ea7f3b42307
# ╠═3d832db2-de0c-41e1-bf1e-cc0c152f4f98
# ╠═0e3c3c29-76a2-4404-8fdb-8b442ce1eb9f
# ╠═8cbb1134-49b5-472c-aadb-1754230742ea
# ╠═57e7e52c-5cef-463d-87ac-b0bb606ddfe9
# ╠═d17c48c8-259a-4ca9-97e7-c844370c52fe
# ╠═1e0cad5b-10b0-4839-b112-25f5b15f7798
# ╠═7ef876f4-b64d-4a20-9e69-95c08075275c
# ╠═2ead2629-7a7e-4ad4-86bf-50698e12f13d
# ╠═13c68676-b38b-4fe2-91e8-f9a808ebef96
# ╠═5d78edeb-d092-459c-ad88-6345fb838350
# ╠═7a1582ac-c323-48de-ba53-fad1ce12ea26
# ╠═43e5877d-1cb7-4a6f-9331-e1ec7bf4ef8b
# ╠═79784c4f-b1c8-4ad7-ac1e-e9b2331f3663
# ╠═4e5ad759-e840-43c0-a342-ebf413a5fb2b
# ╠═77ab5a72-0022-42cb-bd72-24f6ea51c201
# ╠═cf4d406f-9570-4bf8-a266-9f6d7cac24ed
# ╠═3d815877-88a7-44b6-b9df-8d329d1a7e3b
# ╠═2e8856b0-a21a-46dc-9601-f928b31245c1
# ╠═26c0db62-9f96-43cc-a18b-f64290293576
# ╠═fc948717-9f3b-403c-8f65-d05d7bb92b85
# ╠═68fb378b-e910-4344-b0aa-832ebe930ba1
# ╠═1e7a508a-0745-46ad-bd0e-a53fe94d4ec7
# ╠═c45243d4-456d-494b-b27b-1ea79d2e0b48
# ╠═63e07476-a0bb-4dc4-9f3d-5f315e9880de
# ╠═f9c33887-db0e-4757-90b9-d6f91ccabf86
# ╠═7c118104-3aec-478c-a6f7-5bdee9defe76
# ╠═fb18dbd7-5223-45c7-bd20-d6c648c01c2b
# ╠═dbd20ea5-091a-4f08-84e4-7639cf9339b5
# ╠═a18f901f-9be6-49f0-aed5-5f0089fdf165
# ╠═d4dd8e39-baa8-4077-8f21-69d9de1c1b33
# ╠═fb43650f-53e1-4af6-8f30-09389bd024dd
# ╠═5a45ca41-dcf4-4b38-9af7-a047378aeeff
# ╠═d72fa66d-21df-402e-bfc4-0c4d3fddf646
# ╠═08c1a771-382a-445a-b430-19f28d58dfcf
# ╠═52c34bd3-9ac5-49dd-94ce-0d46e968d25c
# ╠═0fed9fc3-8914-4570-971b-f53397e98070
# ╠═46198f73-8966-43ff-9878-13de4cca96d9
# ╠═0e7bcdd9-eb1e-4099-bd54-fad2c3992b4b
# ╠═5ffc2df3-9fc7-4e54-aca3-1145f5f09b72
# ╠═34a29918-675d-4d18-bf2c-728166acc9aa
# ╠═fb24292b-4737-4f9f-a5b6-0c4a0d85bf53
# ╠═1ea8f988-ef7b-485c-a901-d172f0f88471
# ╠═a5afa66c-7447-454d-b0a8-03dba7fb1bbd
# ╠═9ba9044e-3f77-4a48-a9ac-df196bd77f96
# ╠═6c6dbb72-61be-4e6f-93b1-73a15eae1646
# ╠═29d0e749-6072-4888-8fc8-eae42a6b7f06
# ╠═a235da3d-a99f-41c4-a9f6-47a8e975fbb5
# ╠═0cffafb0-a1d3-453f-a366-5cbf5b5f12da
# ╠═fc6453cd-0a0e-4e4a-9601-962dc62b58cb
# ╠═43bb1241-f83c-4566-8a26-cd990807471f
# ╠═3831b6c1-d605-4d6b-b3ab-4dc8def26432
# ╠═e4463d7c-68d4-4f4f-9ba5-694c81054d20
# ╠═fffa4aac-a902-4f8b-9e92-1cc52e49b017
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
