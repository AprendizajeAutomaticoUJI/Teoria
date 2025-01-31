### A Pluto.jl notebook ###
# v0.20.3

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    #! format: off
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
    #! format: on
end

# ╔═╡ 6f6339a7-9b4d-4272-94f7-9234d3d3be41
begin
	using CSV
	using DataFrames
	using HTTP
	using Plots
	import PlotlyBase, PlotlyKaleido
	using PlutoUI
	using Distributions
	using StatsPlots
	using HypothesisTests
	using GLM
	using MLJ
	import MLJLinearModels
	using Polynomials
	
	plotly()
end

# ╔═╡ df4e377a-895e-483d-8894-75629bb2533f
html"""
# # Aumentar el zoom hasta 170
<style>
	p {
		font-size: 27px
	}

	li {
		font-size: 20px;
	}
	
	body h1 {
		font-size: 50px;
		font-family: sans-serif;
	}
	
	body h2 {
		font-size: 40px;
		font-family: sans-serif;
		padding-top: 10px;
	}
	
	main {
		# max-width: 950px !important;
		max-width: 90% !important;
		margin-right: 80px !important; # Debe quedar comentada para editar
	}
</style>
"""

# ╔═╡ 266e632b-30a5-4ae4-981f-8e2ab61e3232
TableOfContents(title="Contenidos", depth=1)

# ╔═╡ d0fe37ee-bbc1-11ef-2f0c-4b6bc41d2c3a
md"""
# Regresión lineal

Óscar Belmonte Fernández - IR2130 Aprendizaje Automático

Grado en Inteligencia Robótica - Universitat Jaume I (UJI)

![](https://ujiapps.uji.es/ade/rest/storage/63c07717-5208-4240-b688-aa6ff558b466?guest=true)
"""

# ╔═╡ cc81afb1-c73d-4e75-b572-21a70608dd9d
md"""
## Introducción

La regresión lineal es uno de los algoritmos para problemas de regresión más antiguos y utilizados.

No obstante hay que conocer en qué problemas se puede utilizar y en qué otros no se puede utilizar.

La regresión lineal simple se puede extender, de manera muy sencilla, a problemas de regresión con múltiples variables y regresión con polinomios.
"""

# ╔═╡ e709b44b-57b0-482d-bc11-93b91451d790
md"""
## Introducción

Aunque existe una formula exacta para resolver problemas de regresión, estudiaremos la técnica del descenso de gradiente para encontrar la solución de la regresión lineal.

Finalmente veremos qué es la regularización y qué problema nos ayuda a resolver.
"""

# ╔═╡ 515ef8bd-82e3-4f94-8fbc-0626ed34e5b7
md"""
## Objetivos de Aprendizaje

- Decidir cuando en un problema se puede emplear la regresión lineal.
- Estimar la bondad de un ajuste con regresión lineal.
- Demostrar el fundamente del descenso de gradiente.
- Razonar si la regularización es apropiada para un determinado problema.
- Construir una solución de regresión (múltiple, polinómica).
"""

# ╔═╡ cda801a3-6b0c-49f0-afbd-798850b354ca
md"""
## Referencias
"""

# ╔═╡ eb52ead3-2e23-4884-96c6-4ccdbf529be2
md"""
# Fórmula exacta de la regresión lineal
"""

# ╔═╡ 9cebf899-82f9-4a69-9ef1-5c98bb8b20fc
md"""
## Objetivos de la regresión lineal

La hipótesis de partida es que existe una relación lineal entre una variable que se utiliza como predictor $x$, y la salida $y$:

$h_{\theta}(x) = y = \theta_0 + \theta_1 x + \epsilon$

Donde $\epsilon \sim N(0, \sigma^2)$ sigue una distribución normal, con media $0$ y varianza $\sigma^2$.
"""

# ╔═╡ b3fe5074-6461-4d1e-bea3-f005533210a0
md"""
## Objetivo de la regresión lineal

Es decir, tenemos un conjunto de $N$ datos para los cuales:


$h_{\theta}(x_1) = y_1 = \theta_0 + \theta_1 x_1 + \epsilon$
$h_{\theta}(x_2) = y_2 = \theta_0 + \theta_1 x_2 + \epsilon$
$...$
$h_{\theta}(x_N) = y_N = \theta_0 + \theta_1 x_N + \epsilon$

Que podemos expresar de manera matricial como:

$h_{\theta}(\mathbf{X}) = \mathbf{y} = \mathbf{X \theta} + \mathbf{\epsilon}$
"""

# ╔═╡ 4309c399-3035-4a0a-8f88-2184efd415b9
md"""
## Objetivo de la regresión lineal

$h_{\theta}(\mathbf{X}) = \mathbf{y} = \mathbf{X \theta} + \mathbf{\epsilon}$

Donde:

```math
\begin{bmatrix}
y_1 \\
y_2\\
...\\
y_N \\
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
...\\
1 & x_N \\
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\end{bmatrix}
+
\begin{bmatrix}
\epsilon \\
\epsilon \\
... \\
\epsilon \\
\end{bmatrix}

```

En aprendizaje automático se utilizan vectores columna.
"""

# ╔═╡ 8783e92c-3ff2-49e0-a85f-264a0ae77afc
md"""
## Objetivo de la regresión lineal

El objetivo de la regresión lineal es, encontrar unos estimadores de $\theta$ que vamos a escribir como $\hat \theta$, de tal modo que los $\hat y$ que obtengo al utilizar esos estimadores:

```math
\begin{bmatrix}
\hat y_1 \\
\hat y_2\\
...\\
\hat y_N \\
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 \\
1 & x_2 \\
...\\
1 & x_N \\
\end{bmatrix}
\begin{bmatrix}
\hat \theta_0 \\
\hat \theta_1 \\
\end{bmatrix}
```

estén a la *menor distancia posible* de los datos $y$.
"""

# ╔═╡ 410f5c00-91f0-4f00-924f-6691227bf1cd
md"""
## Objetivo de la regresión lineal

Veamos un caso real, la relación entre el peso y la altura de los nativos adultos de la etnia !Kung San.

$height = h_{\theta}(weight) = \theta_0 + \theta_1 weight + \epsilon$

A partir del predictor ($weight$) queremos obtener el valor de la altura ($height$) en cm.
"""

# ╔═╡ a079b8eb-f179-41af-9712-de65be3d7a64
md"""
## Objetivo de la regresión lineal
"""

# ╔═╡ 48d3993b-fcae-4d91-a53a-a936dcaea321
begin
	path = "https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Howell1.csv"
	data = CSV.File(HTTP.get(path).body) |> DataFrame
	data[!, "bias"] = ones(length(data.weight))
	adultos = data[data.age .>= 18, :]
end

# ╔═╡ 2e638e63-6f50-4dd0-921a-0e5d33e31fe1
md"""
## Objetivo de la regresión lineal
"""

# ╔═╡ bf9d5f2e-a1ae-4b9a-88ac-a7fcef6945bb
scatter(adultos.weight, adultos.height, xlabel="weight", ylabel="height", label=false, size=(900,400))

# ╔═╡ a7712a98-dbef-4f62-a83c-546ebfb3a40e
md"""
## Función de pérdidas $\mathcal{L}(h_\mathbf{\theta})$

El objetivo es encontrar los parámetros $\theta$ que minimizan la distancia entre los datos reales $y_i$ y los valores calculados $\hat y_i = h_\theta(x_i)$.

$\mathcal{L}(h_\mathbf{\theta}) = \frac{1}{N} \sum_{i=1}^N \lvert y_i - \hat y_i \rvert ^2$

Que se puede expresar en forma matricial como:

$\mathcal{L_{\theta}} = \frac{1}{N} \lVert \mathbf{y} - \mathbf{X \theta} \rVert^2$
"""

# ╔═╡ 6ebfa722-b8f5-4837-871b-addba1c23586
md"""
## Minimizar la función de pérdidas

Tenemos que encontrar el mínimo de la función de pérdidas.

Tomamos las derivadas parciales respecto a los parámetros $\theta$:

$\vec\nabla_{\mathbf{\theta}}\mathcal{L} = 
\vec\nabla_{\mathbf{\theta}} \lVert \mathbf{y} - \mathbf{X \theta} \rVert^2 = 
0$

y llegamos a la expresón matricial:

$\mathbf{\theta} = (\mathbf{X^T X})^{-1}\mathbf{X^T y}$

*Pregunta: ¿La matriz $\mathbf{X^T X}$ es siempre invertible?

"""

# ╔═╡ cefd6efd-8687-4ab4-93e6-6ae33432d774
md"""
## Minimizar la función de pérdidas

Aplicando el cálculo a nuestros datos, obtenemos la recta de regresión:
"""

# ╔═╡ 1b7baa00-78ea-48f6-af34-ccc0795d66b8
begin
	X = Matrix(adultos[:, [:bias, :weight]])
	y = adultos[:, :height]
	θ = (X'X)\(X'y)
	println("θ = ", θ )
end

# ╔═╡ 433bea47-fd81-4e3f-8fd0-d0fe76d6f6ec
md"""
Si dibujamos el conjunto de datos y la recta de regresión lineal.
"""

# ╔═╡ 86ad6318-1a86-469f-870c-591e9e306d74
begin
	scatter(adultos.weight, adultos.height, xlabel="weight", ylabel="height", label="datos")
	extremos_matriz = [1 minimum(X[:,2]); 1 maximum(X[:,2])]
	plot!(extremos_matriz[:,2], extremos_matriz*θ, linewidth=3, label="ajuste", size=(900,400))
end

# ╔═╡ bdc38a15-ff58-4431-8f3b-d1e310044af5
md"""
## Residuos

Los residuos son la parte no lineal de los datos, hemos supuesto que siguen una distribución normal:

$r_i = y_i - h_{\theta}(x_i)$
"""

# ╔═╡ 8af8b802-49d8-474f-b3a3-1278a69093f9
residuos = y - X*θ;

# ╔═╡ 1c428df8-1517-42b3-aa47-215ea2d9bb6f
md"""
## Residuos

La distribución de los residuos en nuestro caso es:
"""

# ╔═╡ 1f0385e8-a071-4473-8136-7cbcf6f924b8
scatter(residuos, title="Residuos", legend=false, xlabel="muestra", ylabel="residuos", size=(900,400))

# ╔═╡ 23f13774-25c3-42de-800f-2437856ad9fc
md"""
## Residuos

Si representamos el histograma de los residuos podemos observar que se asemeja a una distribución normal:
"""

# ╔═╡ 090c1c8b-35c8-43bc-80a3-b8b75304ed31
h_residuos = histogram(residuos, title="Distribución de los residuos", xlabel="residuos", ylabel="Número de muestras", legend=false, size=(900,400))

# ╔═╡ b19afa83-a5b2-4dc7-b266-76f60e27f2ff
md"""
## Varianza de los residuos
Resulta interesante calcular el error del estimador.

La varianza de los residuos es la contribución de la parte no linela del modelo $\epsilon \sim N(0,\sigma^2)$.

Un estimador para la varianza de los residuos es:

${\hat \sigma}^2 = \frac{1}{N} \sum_{i=1}^N (y_i - h_{\theta}(x_i))^2$

que es un estimador con sesgo.
"""

# ╔═╡ 4c05c9d4-e724-4956-9866-7037025150ee
md"""
## Error estándar de $\hat \theta$

Un estimador sin sesgo (*unbiased*) para $\sigma^2$ lo 
podemos calcular como:

$MSE = \frac{1}{N-2} \sum_{i=1}^N (y_i - h_{\theta}(x_i))^2$

ya que tenemos sólo $N-2$ grados de libertad.
"""

# ╔═╡ 499b6c91-9032-43fa-a54c-94027676125a
md"""
## Error estándar de $\mathbf{\hat \theta}$

El error estándar del parámetro estimado $\hat \theta_0$ se calcula como:

$SE(\hat \theta_0) = \sqrt{MSE \left(\frac{1}{n} + \frac{\bar x^2}{S_{xx}}\right)}$


donde $S_{xx} = \sum_{i=1}^N (x_i - \bar x)^2$ es la varianza de los datos.

El error estándar del parámetro estimado $\hat \theta_1$ se calcula como:

$SE(\hat \theta_1) = \sqrt{\frac{MSE}{S_{xx}}}$
"""

# ╔═╡ 48398a06-6e89-43ac-876f-eeb513e69b92
md"""
## Error estándar de $\mathbf{\hat \theta}$

En el ejemplo que estamos tratando, al aplicar las fórmulas, obtenemos 
$SE(\hat \theta_0) = 1.9111$ y $SE(\hat \theta_1) = 0.0420$.

Finalmente:

$\hat \theta_0 = 113.9 \pm 1.9$

$\hat \theta_1 = 0.90 \pm 0.04$
"""

# ╔═╡ 2f86b3e0-66e9-4cca-86fd-a83f6d1dbf5b
md"""
## Normalidad de los residuos

Ahora debemos comprobar la hipótesis de normalidad de los residuos, sin ella 
la regresión lineal que acabamos de hacer no tiene sentido.

Empecemos visualizando el histograma de los residuos y el ajuste a una normal.
"""

# ╔═╡ 7a3f806a-5d26-40f1-b665-7bf686728fde
ajuste_residuos = Distributions.fit(Normal, residuos)

# ╔═╡ cf291a6a-18ba-476a-a9c9-ddcd1f75c514
begin
	histogram(residuos, normed=true, label="histograma", size=(900,400))
	plot!(ajuste_residuos, width=2, label="ajuste a normal")
end

# ╔═╡ 287328fd-2389-4e7e-84f9-4cd932db899a
md"""
## Normalidad de los residuos
Para calcular la bondad del ajuste a una normal podemos utilizar el test de Shapiro-Wilk, que nos da un p-valor de $p = 0.2499$.

En ambos caso del p-valor es mayor que $0.05$, luego no podemos descartar 
que los residuos sigan una distribución normal.

"""

# ╔═╡ 5443ad0a-deb3-463f-8975-12fb969df1cc
ShapiroWilkTest(residuos)

# ╔═╡ 965d8844-8fdd-4e88-a6bd-01281a5fb642
md"""
## Normalidad de los residuos

Podemos usar pruebas más generales, como la de Anderson-Darling.
"""

# ╔═╡ f5a3b766-4781-47de-8097-af3bd8bc4ecd
OneSampleADTest(residuos, ajuste_residuos)

# ╔═╡ ac831749-b0aa-49f5-ad1b-92e45599551d
md"""
## Normalidad de los residuos

O la de Kolmogorov-Smirnov:
"""

# ╔═╡ 7eb49193-addb-4833-9261-0a0ceab55e7e
ApproximateOneSampleKSTest(residuos, ajuste_residuos)

# ╔═╡ f4c5fdba-5493-4240-93dd-86345c9adad7
md"""
En todos los casos el $p-valor > 0.05$ y por lo tanto no podemos rechazar que la distribución de los residuos no sea Normal.
"""

# ╔═╡ 90a3a2c0-4d45-40d9-9bdc-a966c628ba90
md"""
## Normalidad de los residuos

Otra prueba gráfica que podemos utilizar son los qqplot o gráficos cuantil-cuantil.
"""

# ╔═╡ ad16e07e-f7c2-4bce-9f71-0d0334459ef6
qqnorm(residuos, xlabel="Cuantiles teóricos", ylabel="Cuantiles de los datos", title="Gráfico cuantil-cualtil de los residuos", size=(900,400))

# ╔═╡ 08794500-ec04-4834-8852-c79680f0d136
md"""
Si podemos apreciar que el gráfico qqplot es una recta, no podemos rechazar que nuestros datos sigan una distribución normal.
"""

# ╔═╡ 3ffe9740-d133-410a-a3ff-77c2ba64217f
md"""
## ¿Cuál es la linealidad de nuestros datos?

El coeficiente de determinación ($R^2$) nos da una idea de la varianza de los 
datos que puede explicar el modelo.

Suma de los cuadrados de los residuos: $SS_{res} = \sum_{i=1}^N(y_i - h_\theta(i))^2$

Suma total de los cuadrados: $SS_{tot} = \sum_{i=1}^N(y_i - \bar{y})^2$

$$R^2 = 1 - \frac{SS_{res}}{SS_{tot}}$$

La fracción $\frac{SS_{res}}{SS_{tot}}$ es la fracción de la varianza no 
explicada por los datos.

En nuestros caso $R^2 = 0.5696$ y la varianza no explicada es $0.4303$.
"""

# ╔═╡ caba4624-8388-43b9-a9bf-b0bb5ed27213
md"""
## ¿Cuál es la linealidad de nuestros datos?

Otra medida de la linealidad de los datos, es el coeficiente de correlación de 
Pearson, que no es más que la raíz cuadrada de $R^2$:

$$\rho = \sqrt{R^2}$$

En nuestro caso obtenemos $\rho = 0.7547$. 

Como criterio general, se considera 
que el ajuste es bueno cuando este valor está por encima de 0.7.
"""

# ╔═╡ 900f41a8-59a2-4301-88a5-9f0b4c25d4ac
md"""
## En resumen
1. Hemos supuesto que existe una relación lineal entre el predictor (peso) 
   y la variable predicha (altura).
1. Hemos supuesto que los residuos de nuestros datos siguen una distribución 
   normal con media $\mu = 0$ y varianza $\sigma^2$
1. Hemos calculado los valores estimados para $\theta_0,\theta_1,\sigma$.
1. Hemos calculado sus errores estándar.
1. Hemos ajustado los residuos a una normal.
1. Hemos hecho pruebas para comprobar la bondad del ajuste.
1. Hemos utilizado un gráfico cuantil-cuantil.
"""

# ╔═╡ 6518ba79-f3cc-4ccb-9935-fe037f256d4e
md"""
## Paquetes en Julia
Todos estos cálculos los hemos hecho con un poco de álgebra. Existen paquetes en Julia que nos facililtan estos cálculos, como por ejemplo, el paquete **GLM** (Generalised Linear Models)
"""

# ╔═╡ e3278b0e-ff00-4bc9-b06e-d2757a08f997
regresion_glm = lm(@formula(height ~ weight), adultos)

# ╔═╡ 3a665608-f044-41a9-a619-7ed799457277
histogram(residuals(regresion_glm))

# ╔═╡ ea80c78a-6ec8-44f9-9876-6449c2399ee3
stderror(regresion_glm)

# ╔═╡ c105b7ce-98bd-4c34-86b4-818309495dc5
coef(regresion_glm)

# ╔═╡ 747f66f6-1c86-450f-824f-7a94bdd7ec6a
r2(regresion_glm)

# ╔═╡ ed4f53d4-4d73-4f02-abf8-2cb68447525f
tmp = Distributions.fit(Normal, residuals(regresion_glm))

# ╔═╡ 23b1b815-1048-4ebb-9d12-eba8dfa8e60c
md"""
## Paquetes en Julia
Vamos a ver el resultado de la regresión en una gráfica.

Primera calculamos los extremos de la variable **weight**:
"""

# ╔═╡ 58e0b3a9-9f2c-438f-ac4a-db490a3f6685
extremos = collect(extrema(adultos.weight)) # Usamos collect para convertir la tupla en un vector.

# ╔═╡ 7da174c6-91a4-4508-818f-6d35720df915
md"""
Construímos un DataFrame (**atención a la etiqueta de la columna**)
"""

# ╔═╡ 1da2e2e0-b1bd-4e86-b397-54f293a90c06
extremos_df = DataFrame(weight = extremos);

# ╔═╡ 7c3f884f-7159-4594-aef9-a975767e750d
md"""
Calculamos el valor del regresor en los extremos:
"""

# ╔═╡ 6f07b059-a4b3-4a32-b62e-fbadb15d71a3
prediccion_glm = GLM.predict(regresion_glm, extremos_df)

# ╔═╡ 203a5b66-2c6d-4707-ae13-ed2563a2f305
begin
	scatter(adultos.weight, adultos.height, xlabel="weight", ylabel="height", label="datos", size=(900,400))
	plot!(extremos_df.weight, prediccion_glm, width=3, label="regresión")
end

# ╔═╡ 23dfc277-7289-4a69-8ec5-75fbd05b43e2
md"""
Nota: sobre los modelos de GLM se pueden aplicar funciones como r2(modelo), que calcula r2, y aic(modelo) que calcula AIC. También bic(modelo) para calcular BIC. Este último es interesante si incluyo AIC como una medida para seleccionar el grado de un polinomio.
"""

# ╔═╡ f437681b-464d-449d-befc-75d60a5c4974
md"""
## Paquetes en Julia

También podemos utilizar el paquete **MLJ** (Machine Learning for Julia)

Primero cargamos el modelo que nos interesa
"""

# ╔═╡ 660be5ac-4791-48c3-b1f4-824f53c80215
LinearRegressor = @load LinearRegressor pkg=MLJLinearModels verbosity=0

# ╔═╡ 9087d54f-5682-4778-812a-15bd28431fb7
md"""
Lo instanciamos
"""

# ╔═╡ 95e17cad-7135-4b7f-93f2-5b7c72023e27
regresor = LinearRegressor()

# ╔═╡ 19905308-9b52-4f49-a4e8-7212bc08fb0a
md"""
**MLJ** tiene una interfaz uniforme, trabajamos con todos los modelos de aprendizaje automático con los mismos pasos, creando una máquina que contine, el modelo, y los datos.
"""

# ╔═╡ 57a30f20-7dc2-4c02-8621-32f03c5d5f71
modelo = machine(regresor, adultos[:, [:weight]], adultos.height)

# ╔═╡ 83788e2c-84f8-4d0c-bdb0-930680eafa4c
md"""
## Paquetes en Julia

Y ahora lo entrenamos:
"""

# ╔═╡ c76bc69f-eeaf-4861-b735-f2d4a99cd4d3
fit!(modelo)

# ╔═╡ 8cab10cf-e7d4-42a9-b9b9-c6243ea80ec4
md"""
Depués de entrenar el modelo, podemos ver el valor de los parámetros:
"""

# ╔═╡ cb016b85-517d-4e28-ab73-60adab8007b5
fitted_params(modelo)

# ╔═╡ 3566a81e-5a17-4c34-87b4-3b7dfbe2fc1d
md"""
Puedes comprobar que estos parámetros coinciden con los proporcionados al utilizar el paquete GLM, de hecho el paquete JML **recubre** algoritmos proporcionados por otros paquetes.

Recuerda que el objetivo de MLJ es ofrecer una interfaz uniforme al programador.
"""

# ╔═╡ e38b43eb-832d-4ac2-894f-d5d287b6709b
md"""
## Paquetes en Julia

El resultado, evidentemente, es el mismo:
"""

# ╔═╡ db09d078-cbf5-442a-b819-a049f93cee06
prediccion_mlj = MLJ.predict(modelo, extremos_df)

# ╔═╡ b9bfee39-4092-4c69-816c-ca830b94c935
begin
	scatter(adultos.weight, adultos.height, xlabel="weight", ylabel="height", label="datos", size=(900,400))
	plot!(extremos_df.weight, prediccion_mlj, width=3, label="regresión")
end

# ╔═╡ 4c52220e-3b6e-46fe-b7d7-87bfc5facf99
md"""
## Paquetes de Julia

En el caso anterior, estamos utilizando el mismo conjunto de datos para 
entrenar el modelo, que para calcular el error del modelo sobre los datos.

Usualmente se utilizan dos conjuntos de datos, con uno de ellos entrenamos el 
modelo y el otro conjunto de datos lo utilizamos para evaluar el modelo.
"""

# ╔═╡ bfdf0bd6-42ab-4d7b-b45a-21035fefcb34
md"""
## Conjunto de entrenamiento y conjunto de pruebas

Al conjunto que utilizamos para entrenar el modelo lo llamas *conjunto de 
entrenamiento*, y al conjunto que utilizamos para evaluar el modelo *conjunto 
de prueba*.
"""

# ╔═╡ b972c073-20c7-41e0-bc72-61838652ea79
(Xtrain, Xtest), (ytrain, ytest) = partition((adultos.weight, adultos.height), 0.8, multi=true, rng=69)

# ╔═╡ 563f0e01-3117-40f0-9d4f-0f4926bdc336
md"""
## Evaluación cruzada

La evaluación cruzada consiste en dividir el conjunto de datos inicial en un conjunto de entranamiento y otro de pruebas de manera aleatoria, repetidas veces. En cada repetición se crear un modelo con los datos de entrenameinto y se prueba con los datos de prueba, el resultados final es el promedio de todas las repeticiones.
"""

# ╔═╡ 2da21951-935d-4ba8-9a3b-c2628a71a31e
begin
	cv = CV(nfolds = 10)
	validacion_cruzada = evaluate(regresor, select(adultos, :weight), adultos.height, resampling=cv, measure=rms)
end

# ╔═╡ 215eb03b-4c21-4321-bb3b-70bb68aace81
md"""
## Estimación de parámetros por máxima verosimilitud

Para terminar con esta sección vamos a ver cómo la expresión de la función 
de pérdidas: 
$\mathcal{L}(h_\mathbf{\theta}) = \frac{1}{N} \sum_{i=1}^N \lvert y_i - x_i\theta \rvert ^2$

aparece de modo natural al asumir que los residuos están normalmente 
distribuídos.
"""

# ╔═╡ 878d4d09-9d5d-4d2b-9670-c7ae116ee5e9
md"""
## Estimación de parámetros por máxima verosimilitud
Hemos supuesto que los residuos siguen un distribución normal
independientemente del punto donde calculamos la regresión, lo que significa
suponer que:

$p(y_i|\theta,\sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(y_i-x_i\theta)^2}{2\sigma^2}}$
"""

# ╔═╡ 317be3e6-1e2c-4565-866e-be8d03330c56
md"""
## Estimación de parámetros por máxima verosimilitud
Un estimador máximo verosímil es aquel que maximiza la función de verosimilitud, 
que no es más que el producto de las probabilidades de cada una de las 
muestras. Si suponemos que las 
variables de la muestra son independientes y siguen la misma distribución de 
probabilidad (iid: independientes e idénticamente distribuidas):

$p(\mathbf{y}|\theta,\sigma^2) = \prod_{i=1}^N \frac{1}{\sqrt{2\pi\sigma^2}}
e^{-\frac{(y_i-x_i\theta)^2}{2\sigma^2}}$

donde hemos extendido el productorio a todas las muestas.
"""

# ╔═╡ 4b0609e8-cc0b-44e9-94e9-01c34301dfe7
md"""
## Estimación de parámetros por máxima verosimilitud
Si tomamos logaritmos de la función de verosimilitud:

$ln(p(\mathbf{y}|\theta,\sigma^2)) = {-\frac{1}{2}}\sum_{i=1}^N ln(2\pi\sigma^2) - 
\sum_{i=1}^N \frac{(y_i-x_i\theta)^2}{2\sigma^2}$

Como la función logaritmo es monótonamente creciente maximizar la función de 
verosimilitud es lo mismo que minimizar la misma función cambiada de signo.
"""

# ╔═╡ 4a71d082-f1ca-4582-8722-0d6ec6b67de4
md"""
## Estimación de parámetros por máxima verosimilitud

$-ln(p(\mathbf{y}|\theta,\sigma^2)) = {\frac{1}{2}}\sum_{n=1}^N ln(2\pi\sigma^2) + 
\boxed{\sum_{i=1}^N \frac{(y_i-x_i\theta)^2}{2\sigma^2}}$

Si fijamos $\sigma^2$ minizar la expresión anterior significa minimizar el 
sumatorio que depende de $\theta$, que es proporcional a la función de pérdidas.

"""

# ╔═╡ 94a22149-789c-4a25-b91f-23fc24e702fc
md"""
## Estimación de parámetros por máxima verosimilitud
Por otro lado, podemos calcular el estimador máximo verosímil derivando la 
expresión anterior con respecto a $\sigma^2$ e igualando a cero, con lo que 
obtenemos:

$\hat \sigma^2 = \frac{1}{N} \sum_{i=1}^N (y_i - x_i\theta)^2$

Que es exactamente el estimador con sesgo que ya habíamos calculado 
(*Demostración*).
"""

# ╔═╡ 6e8cdc3d-aa37-4b66-84b2-6b1db9947d62
md"""
# Regresión lineal múltiple
"""

# ╔═╡ 07c535e5-c9bb-4ecb-bcb4-391f257a91c3
md"""
## Extensión de la regresión lineal

Hasta ahora, sólo hemos utilizado una característica como predictor en 
nuestro modelo.

Pero, podemos añadir más características como predictores de nuestro modelo.

$h_{\theta}(x) = y = \theta_0 + \theta_1 x_1 + \theta_2 x_2 +...+ \epsilon$
"""

# ╔═╡ b97dac76-fdb8-4da0-8295-0f8a9dc20cf3
md"""
## Extensión de la regresión lineal

Recordemos la estructura de nuestro conjunto de datos:
"""

# ╔═╡ e9069ed2-3a71-4206-b569-4f73bb97eb19
first(data[:, Not(:bias)], 4)

# ╔═╡ 05ad2c30-e21b-40e2-85a9-f85176a5c075
md"""
Ahora, vamos a incluir en la regresión todas las características:

$height = \theta_0 + \theta_1 weight + \theta_2 age + \theta_3 male + \epsilon$

"""

# ╔═╡ 04aca69e-71a6-4b66-b4a0-fe8aaabd5fe9
md"""
## Extensión de la regresión lineal 
Fíjate en que lo que estamos haciendo es añadir más columnas a la matriz de 
los predictores:

```math
\begin{bmatrix}
y_1 \\
y_2\\
y_3\\
...\\
y_N \\
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1^1 & x_1^2 & ... & x_1^m\\
1 & x_2^1 & x_2^2 & ... & x_2^m\\
1 & x_3^1 & x_3^2 & ... & x_3^m\\
... & ... & ... & ... & ... \\
1 & x_n^1 & x_n^2 & ... & x_N^m\\
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
...\\
\theta_m \\
\end{bmatrix}
+
\begin{bmatrix}
\epsilon \\
\epsilon \\
\epsilon \\
... \\
\epsilon \\
\end{bmatrix}
```

"""

# ╔═╡ 6aa1e98d-13df-4732-bc3a-b907bcb8e4bd
md"""
## Extensión de la regresión lineal 
Que podemos resolver de este modo si incluimos todas las características:
"""

# ╔═╡ dff726bd-66dc-49d0-8dfd-bd762cad734b
begin
	X_todas = Matrix(adultos[:, [:bias, :weight, :age, :male]])
	y_todas = adultos[:, :height]
	θ_todas = (X_todas'X_todas)\(X_todas'y_todas)
	println("θ = ", θ_todas)
end

# ╔═╡ ab5b96a3-97d9-45fe-a8fe-88b5094bca8e
md"""
## Extensión de la regresión lineal
El paquete **GLM** también nos permite hacer regresión multivariada:
"""

# ╔═╡ e4e29ff3-fc14-4966-8cfb-e8296a4a4bd1
regresion_multivariada_glm = lm(@formula(height ~ weight + age + male), adultos)

# ╔═╡ 14e10278-99fc-4932-aa52-6d02cf6ac7db
md"""
Los valores estimados de los parámetros son los mismos, pero obtenemos mucha más información.
"""

# ╔═╡ 158f021b-5fa8-49be-a8df-6863a272e4e8
md"""
## Extensión de la regresión lineal

Si usamos el paquete **MLJ**:
"""

# ╔═╡ 55211ee4-a8da-46b2-96f1-83fc170677ee
begin
	X_multiple = coerce(adultos[:, [:weight, :age, :male]], MLJ.Count => MLJ.Continuous)
	maquina_multiple = machine(regresor, X_multiple, adultos.height)
	fit!(maquina_multiple)
end

# ╔═╡ 1661096e-713c-435a-b47d-278f454ad76b
evaluate!(maquina_multiple, resampling=MLJ.InSample(), measure=rms)

# ╔═╡ c3976756-cf99-4092-86d2-994b34894bb7
md"""
Si empleamos validación cruzada:
"""

# ╔═╡ 70802a0d-d500-457c-9823-0bf99f23509e
evaluate!(maquina_multiple, resampling=MLJ.CV(nfolds=10, rng=69), measure=rms)

# ╔═╡ 354706f0-c7a2-49b6-be7d-5fe9d333aa30
md"""
## Normalidad de los residuos
"""

# ╔═╡ b4ffd1ca-dc72-45fe-a6a4-b0f9569cec64
residuos_multiple = y - MLJ.predict(maquina_multiple, X_multiple)

# ╔═╡ 1e051daa-f2db-447b-93b3-19c594fc0d66
ajuste_residuos_multiple = Distributions.fit(Normal, residuos_multiple)

# ╔═╡ 4b2dfb5f-9319-4cd9-9e6c-69efab579681
ShapiroWilkTest(residuos_multiple)

# ╔═╡ 0f85c1b1-ce14-4870-8943-efb72e9ec7d9
md"""
La prueba de Shapiro-Wilk no pasa. Veamos qué ocurre con las otras dos pruebas:
"""

# ╔═╡ eefd48fc-ca9a-42fb-a321-53be6bf98381
OneSampleADTest(residuos_multiple, ajuste_residuos_multiple)

# ╔═╡ ef313c53-798c-4e84-b6da-4d90ed49db51
ApproximateOneSampleKSTest(residuos_multiple, ajuste_residuos_multiple)

# ╔═╡ 14a01c6a-9c06-434d-b1b9-87be793c071a
md"""
La normalidad de los residuos sí que pasa las dos últimas pruebas.
"""

# ╔═╡ 613788e4-0430-41da-ad36-55c24de472a2
md"""
## Normalidad de los residuos

Utilicemos una representación cuantil-cuantil:
"""

# ╔═╡ 2ecb47e6-58e1-4065-afa4-1c0eea75a6ee
qqnorm(residuos_multiple, xlabel="Cuantil teórico", ylabel="Cuantil de los datos", title="Gráfico cuantil-cuantil de los residuos", size=(900,400))

# ╔═╡ 7705d15c-ff72-488d-9733-d535f8c0d33d
md"""
Vemos que tenemos un dato anómalo, abajo a la izquierda en el gráfico. Si eliminamos el dato anómalo y volvemos a hacer la prueba:
"""

# ╔═╡ 9df11318-64a8-48cc-803a-2c19a0796fe8
ShapiroWilkTest(filter(x -> x > -15, residuos_multiple))

# ╔═╡ 90df1425-d688-4195-9aea-fc1494000448
md"""
La prueba de normalidad ahora pasa, y si representamos el gráfico cuantil-cuantil, vemos que ha mejorado
"""

# ╔═╡ 0ba2bf47-2dab-4ef3-af7e-bd97d1f0ebc4
qqnorm(filter(x -> x > -15, residuos_multiple), size=(900,400))

# ╔═╡ d4bfb41a-7f9d-4ae6-aa1c-a3e1bac8025d
md"""
## Resumen

Todo lo que hemos aprendido en el caso de un único predictor y una única variable predicha lo podemos extender al caso de varios predictores y una única variable predicha.
"""

# ╔═╡ 47e52680-e6ea-4de9-b621-9294c13c99ee
md"""
# Regresión polinomial
"""

# ╔═╡ 4ca27647-e6ed-42f5-aac4-bf9dd959d9c5
md"""
## Extensión de la regresión lineal

Visualicemos todos los datos de nuestro conjunto, no sólo los datos de las personas adultas:
"""

# ╔═╡ e3c90bb0-22fb-4853-ba20-c77dd87d0096
scatter(data.weight, data.height, xlabel="weight", ylabel="height", title="Altura frente a peso en el cojunto Howell", legend=false, size=(900,400))

# ╔═╡ f9f038a8-a3dd-4253-af34-0903d4b13216
md"""
A simple vista parece que su comportamiento no es lineal.

¿Podemos utilizar un polinomio como modelo de los datos?
"""

# ╔═╡ 735b0a1f-8568-4ce2-8761-c321366f5a35
md"""
## Extensión de la regresión lineal
En el caso de ajuste de un polinomio tenemos:

$h_{\theta}(x) = y = \theta_0 + \theta_1 x + \theta_2 x^2 +...+ \theta_n x^n + \epsilon$

Fíjate en que los parámetros que buscamos $\mathbf{\theta}$ siguen siendo 
lineales, no hay ninguna potencia de los parámetros, la potencia está en los 
datos.
"""

# ╔═╡ be0d88a4-5520-43bd-8914-9b566a343acc
md"""
## Extensión de la regresión lineal
Que lo podemos expresar de modo matricial como:

```math
\begin{bmatrix}
y_1 \\
y_2\\
y_3\\
...\\
y_N \\
\end{bmatrix}
=
\begin{bmatrix}
1 & x_1 & (x_1)^2 & ... & (x_1)^m\\
1 & x_2 & (x_2)^2 & ... & (x_2)^m\\
1 & x_3 & (x_3)^2 & ... & (x_3)^m\\
... & ... & ... & ... & ... \\
1 & x_N & (x_N)^2 & ... & (x_N)^m\\
\end{bmatrix}
\begin{bmatrix}
\theta_0 \\
\theta_1 \\
\theta_2 \\
...\\
\theta_m \\
\end{bmatrix}
+
\begin{bmatrix}
\epsilon \\
\epsilon \\
\epsilon \\
... \\
\epsilon \\
\end{bmatrix}
```
"""

# ╔═╡ 690417e7-50a6-401c-9ad9-1d42abc384a3
md"""
## Extensión de la regresión lineal

Probemos primero con un polinomio de grado 2:
"""

# ╔═╡ e982d0a1-3469-48d5-a286-28fc2f88f714
md"""
Grado del polinomio: $(@bind grado NumberField(1:12, default=2))
"""

# ╔═╡ b117c48e-2e1b-4908-8747-58d080a829f0
function ajuste_polinomial(grado::Int)
	fit = Polynomials.fit(data[:,:weight], data[:,:height], grado)
	# print(fit)
	rmse = rms(fit.(data[:, :weight]), data[:, :height])
	fit, rmse
end;

# ╔═╡ 4f59d4d0-8c87-41d6-b313-1cdc368a5120
function dibuja_ajuste(grado::Int)
	fit, rmse = ajuste_polinomial(grado)
	scatter(data[:, :weight], data[:, :height], 
		ylim=(50, 200), 
		xlabel="weight", ylabel="height", 
		label="Datos", legend=false, 
		title="Grado: " * string(grado) * ", RMSE: " * string(rmse))
	plot!(fit, extrema(data[:, :weight])..., width=3)
end;

# ╔═╡ 0ad863f8-3479-4da5-92b2-7bdc2f4044d5
dibuja_ajuste(grado)

# ╔═╡ 53bdf7bb-95f7-4413-aec7-1c88375fcf22
md"""
## Encontrar el mejor grado del polinomio

La siguiente gráfica muestra el MSE frente al grado del polinomio:
"""

# ╔═╡ 1c7e1986-b12f-4d4a-9be9-12befefb98bd
begin
	datos = [ajuste_polinomial(x)[2] for x in 1:12]
	plot(datos, xlim=(0,12), size=(900,400), xticks=(1:12))
	scatter!(datos)
end

# ╔═╡ f8967d16-3b32-47af-93eb-13eee06fc199
md"""
Parece que desde grado 3 hasta 8 son buenas elecciones.
"""

# ╔═╡ 5390815a-39a1-4aee-bb0c-d93edb60c9ec
md"""
## Encontrar el mejor grado de un polinomio

Comprobemos la hipótesis de normalidad de los residuos para un polinomio de grado 3:
"""

# ╔═╡ 1d3429b4-30d3-407c-aa96-40b91823870e
regresion_grado3_glm = lm(@formula(height ~ weight + weight^2 + weight^3), adultos)

# ╔═╡ d9e5abac-a4a9-41f9-96c7-cd53727c4186
ajuste_residuos_grado3_glm = Distributions.fit(Normal, residuals(regresion_grado3_glm))

# ╔═╡ e9ae8022-0ec7-4206-b006-d070322f10c5
begin
	histogram(residuals(regresion_grado3_glm), normed=true, label="Histograma")
	plot!(ajuste_residuos_grado3_glm, width=2, label="Ajuste polinomio cúbico")
end

# ╔═╡ 7f60da0c-5876-47c0-9cf6-1bce1891d4f6
md"""
Las distintas medidas de bondad del ajuste no nos premiten rechazar la hipótesis nula: los residuos siguen una distribución normal.
"""

# ╔═╡ e0d5372d-d876-4e56-a090-bbe7158ae845
ShapiroWilkTest(residuals(regresion_grado3_glm))

# ╔═╡ 0db3efc7-01e6-4d0b-bb52-4626faac179c
OneSampleADTest(residuals(regresion_grado3_glm), Distributions.fit(Normal, residuals(regresion_grado3_glm)))

# ╔═╡ 1d383b64-a596-42ca-9eed-6ec35c7d61ea
ApproximateOneSampleKSTest(residuals(regresion_grado3_glm), Distributions.fit(Normal, residuals(regresion_grado3_glm)))

# ╔═╡ dbbca94b-d3eb-40f2-b303-f40f537dbbd9
md"""
El gráfico quantil-quantil corrobora la bondad del ajuste.
"""

# ╔═╡ 5cd0a735-6ab1-4467-aed0-591f876658a7
qqnorm(residuals(regresion_grado3_glm))

# ╔═╡ 78f74061-afc3-44af-9753-3feeab77b889
md"""
Otras métricas que podemos utilizar para seleccionar entre varios modelos son:

* [AIC (Akaike Information Criterion)](https://en.wikipedia.org/wiki/Akaike_information_criterion) y 
* [BIC (Bayessian Information Criterion](https://en.wikipedia.org/wiki/Bayesian_information_criterion).

Estas métricas tienen en cuenta tanto el grado de ajuste del modelo a los datos como la complejidad del modelo.
"""

# ╔═╡ fa1025bf-df2c-4bcc-83d5-8f0d88a56e57
print("AIC: " * string(GLM.aic(regresion_grado3_glm)))

# ╔═╡ 0984464e-7c1d-418b-b956-436744c84704
print("BIC: " * string(GLM.bic(regresion_grado3_glm)))

# ╔═╡ 8b6c1f57-a7a6-45e1-81f3-a3b06cbd25cc
md"""
# Descenso de gradiente
"""

# ╔═╡ aad3bc5b-c389-409d-acb5-895788e3ce42
md"""
# Regularización
"""

# ╔═╡ 5c0b5a16-fd1d-4b8c-aa70-d746332b4d28
md"""
# Resumen

- Hemos analizado con detalle la regresión lineal.
- Hemos ampliado la regresión lineal a múltiple.
- Hemos extendido la regresión lineal a modelos polinómicos.
- Hemos estudiado qué es la técnica de descenso del gradiente.
- Hemos introducido la regularización para mejorar el rendimiento.
- Es importante que compruebes que se cumplen las condiciones para aplicar regresión lineal: residuos distribuido según una gaussiana centrada en el 0.
"""

# ╔═╡ 31f685d8-b2b3-49d3-88a0-e45f24e69bd8
# ╠═╡ disabled = true
#=╠═╡
v = [1,2,3]
  ╠═╡ =#

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
CSV = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
Distributions = "31c24e10-a181-5473-b8eb-7969acd0382f"
GLM = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
HypothesisTests = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
MLJ = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
MLJLinearModels = "6ee0df7b-362f-4a72-a706-9e79364fb692"
PlotlyBase = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
PlotlyKaleido = "f2990250-8cf9-495f-b13a-cce12b45703c"
Plots = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
PlutoUI = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
Polynomials = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
StatsPlots = "f3b207a7-027a-5e70-b257-86293d7955fd"

[compat]
CSV = "~0.10.15"
DataFrames = "~1.7.0"
Distributions = "~0.25.113"
GLM = "~1.8.3"
HTTP = "~1.10.14"
HypothesisTests = "~0.11.3"
MLJ = "~0.20.2"
MLJLinearModels = "~0.10.0"
PlotlyBase = "~0.8.19"
PlotlyKaleido = "~2.2.5"
Plots = "~1.40.9"
PlutoUI = "~0.7.60"
Polynomials = "~4.0.12"
StatsPlots = "~0.15.7"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.11.3"
manifest_format = "2.0"
project_hash = "154956d92134fbeaad3b3a9012c644efadc63458"

[[deps.ARFFFiles]]
deps = ["CategoricalArrays", "Dates", "Parsers", "Tables"]
git-tree-sha1 = "678eb18590a8bc6674363da4d5faa4ac09c40a18"
uuid = "da404889-ca92-49ff-9e8b-0aa6b4d38dc8"
version = "1.5.0"

[[deps.AbstractFFTs]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "d92ad398961a3ed262d8bf04a1a2b8340f915fef"
uuid = "621f4979-c628-5d54-868e-fcf4e3e8185c"
version = "1.5.0"
weakdeps = ["ChainRulesCore", "Test"]

    [deps.AbstractFFTs.extensions]
    AbstractFFTsChainRulesCoreExt = "ChainRulesCore"
    AbstractFFTsTestExt = "Test"

[[deps.AbstractPlutoDingetjes]]
deps = ["Pkg"]
git-tree-sha1 = "6e1d2a35f2f90a4bc7c2ed98079b2ba09c35b83a"
uuid = "6e696c72-6542-2067-7265-42206c756150"
version = "1.3.2"

[[deps.Accessors]]
deps = ["CompositionsBase", "ConstructionBase", "InverseFunctions", "LinearAlgebra", "MacroTools", "Markdown"]
git-tree-sha1 = "96bed9b1b57cf750cca50c311a197e306816a1cc"
uuid = "7d9f7c33-5ae7-4f3b-8dc6-eff91059b697"
version = "0.1.39"

    [deps.Accessors.extensions]
    AccessorsAxisKeysExt = "AxisKeys"
    AccessorsDatesExt = "Dates"
    AccessorsIntervalSetsExt = "IntervalSets"
    AccessorsStaticArraysExt = "StaticArrays"
    AccessorsStructArraysExt = "StructArrays"
    AccessorsTestExt = "Test"
    AccessorsUnitfulExt = "Unitful"

    [deps.Accessors.weakdeps]
    AxisKeys = "94b1ba4f-4ee9-5380-92f1-94cde586c3c5"
    Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    Requires = "ae029012-a4dd-5104-9daa-d747884805df"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Adapt]]
deps = ["LinearAlgebra", "Requires"]
git-tree-sha1 = "50c3c56a52972d78e8be9fd135bfb91c9574c140"
uuid = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
version = "4.1.1"
weakdeps = ["StaticArrays"]

    [deps.Adapt.extensions]
    AdaptStaticArraysExt = "StaticArrays"

[[deps.AliasTables]]
deps = ["PtrArrays", "Random"]
git-tree-sha1 = "9876e1e164b144ca45e9e3198d0b689cadfed9ff"
uuid = "66dad0bd-aa9a-41b7-9441-69ab47430ed8"
version = "1.1.3"

[[deps.ArgCheck]]
git-tree-sha1 = "680b3b8759bd4c54052ada14e52355ab69e07876"
uuid = "dce04be8-c92d-5529-be00-80e4d2c0e197"
version = "2.4.0"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.2"

[[deps.Arpack]]
deps = ["Arpack_jll", "Libdl", "LinearAlgebra", "Logging"]
git-tree-sha1 = "9b9b347613394885fd1c8c7729bfc60528faa436"
uuid = "7d9fca2a-8960-54d3-9f78-7d1dccf2cb97"
version = "0.5.4"

[[deps.Arpack_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "OpenBLAS_jll", "Pkg"]
git-tree-sha1 = "5ba6c757e8feccf03a1554dfaf3e26b3cfc7fd5e"
uuid = "68821587-b530-5797-8361-c406ea357684"
version = "3.5.1+1"

[[deps.ArrayInterface]]
deps = ["Adapt", "LinearAlgebra"]
git-tree-sha1 = "017fcb757f8e921fb44ee063a7aafe5f89b86dd1"
uuid = "4fba245c-0d91-5ea0-9b3e-6abc04ee57a9"
version = "7.18.0"

    [deps.ArrayInterface.extensions]
    ArrayInterfaceBandedMatricesExt = "BandedMatrices"
    ArrayInterfaceBlockBandedMatricesExt = "BlockBandedMatrices"
    ArrayInterfaceCUDAExt = "CUDA"
    ArrayInterfaceCUDSSExt = "CUDSS"
    ArrayInterfaceChainRulesCoreExt = "ChainRulesCore"
    ArrayInterfaceChainRulesExt = "ChainRules"
    ArrayInterfaceGPUArraysCoreExt = "GPUArraysCore"
    ArrayInterfaceReverseDiffExt = "ReverseDiff"
    ArrayInterfaceSparseArraysExt = "SparseArrays"
    ArrayInterfaceStaticArraysCoreExt = "StaticArraysCore"
    ArrayInterfaceTrackerExt = "Tracker"

    [deps.ArrayInterface.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    CUDSS = "45b445bb-4962-46a0-9369-b4df9d0f772e"
    ChainRules = "082447d4-558c-5d27-93f4-14fc19e9eca2"
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    GPUArraysCore = "46192b85-c4d5-4398-a991-12ede77f4527"
    ReverseDiff = "37e2e3b7-166d-5795-8a7a-e32c996b4267"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArraysCore = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
    Tracker = "9f7883ad-71c0-57eb-9f7f-b5c9e6d3789c"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"
version = "1.11.0"

[[deps.Atomix]]
deps = ["UnsafeAtomics"]
git-tree-sha1 = "c3b238aa28c1bebd4b5ea4988bebf27e9a01b72b"
uuid = "a9b6321e-bd34-4604-b9c9-b65b8de01458"
version = "1.0.1"

    [deps.Atomix.extensions]
    AtomixCUDAExt = "CUDA"
    AtomixMetalExt = "Metal"
    AtomixoneAPIExt = "oneAPI"

    [deps.Atomix.weakdeps]
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    Metal = "dde4c033-4e86-420c-a63e-0dd931031962"
    oneAPI = "8f75cd03-7ff8-4ecb-9b8f-daf728133b1b"

[[deps.AxisAlgorithms]]
deps = ["LinearAlgebra", "Random", "SparseArrays", "WoodburyMatrices"]
git-tree-sha1 = "01b8ccb13d68535d73d2b0c23e39bd23155fb712"
uuid = "13072b0f-2c55-5437-9ae7-d433b7a33950"
version = "1.1.0"

[[deps.BangBang]]
deps = ["Accessors", "ConstructionBase", "InitialValues", "LinearAlgebra", "Requires"]
git-tree-sha1 = "e2144b631226d9eeab2d746ca8880b7ccff504ae"
uuid = "198e06fe-97b7-11e9-32a5-e1d131e6ad66"
version = "0.4.3"

    [deps.BangBang.extensions]
    BangBangChainRulesCoreExt = "ChainRulesCore"
    BangBangDataFramesExt = "DataFrames"
    BangBangStaticArraysExt = "StaticArrays"
    BangBangStructArraysExt = "StructArrays"
    BangBangTablesExt = "Tables"
    BangBangTypedTablesExt = "TypedTables"

    [deps.BangBang.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"
    StructArrays = "09ab397b-f2b6-538f-b94a-2f83cf4a842a"
    Tables = "bd369af6-aec1-5ad0-b16a-f7cc5008161c"
    TypedTables = "9d95f2ec-7b3d-5a63-8d20-e2491e220bb9"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"
version = "1.11.0"

[[deps.Baselet]]
git-tree-sha1 = "aebf55e6d7795e02ca500a689d326ac979aaf89e"
uuid = "9718e550-a3fa-408a-8086-8db961cd8217"
version = "0.1.1"

[[deps.BitFlags]]
git-tree-sha1 = "0691e34b3bb8be9307330f88d1a3c3f25466c24d"
uuid = "d1d4a3ce-64b1-5f1a-9ba4-7e7e69966f35"
version = "0.1.9"

[[deps.Bzip2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "35abeca13bc0425cff9e59e229d971f5231323bf"
uuid = "6e34b625-4abd-537c-b88f-471c36dfa7a0"
version = "1.0.8+3"

[[deps.CSV]]
deps = ["CodecZlib", "Dates", "FilePathsBase", "InlineStrings", "Mmap", "Parsers", "PooledArrays", "PrecompileTools", "SentinelArrays", "Tables", "Unicode", "WeakRefStrings", "WorkerUtilities"]
git-tree-sha1 = "deddd8725e5e1cc49ee205a1964256043720a6c3"
uuid = "336ed68f-0bac-5ca0-87d4-7b16caf5d00b"
version = "0.10.15"

[[deps.Cairo_jll]]
deps = ["Artifacts", "Bzip2_jll", "CompilerSupportLibraries_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "JLLWrappers", "LZO_jll", "Libdl", "Pixman_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "009060c9a6168704143100f36ab08f06c2af4642"
uuid = "83423d85-b0ee-5818-9007-b63ccbeb887a"
version = "1.18.2+1"

[[deps.CategoricalArrays]]
deps = ["DataAPI", "Future", "Missings", "Printf", "Requires", "Statistics", "Unicode"]
git-tree-sha1 = "1568b28f91293458345dabba6a5ea3f183250a61"
uuid = "324d7699-5711-5eae-9e2f-1d82baa6b597"
version = "0.10.8"

    [deps.CategoricalArrays.extensions]
    CategoricalArraysJSONExt = "JSON"
    CategoricalArraysRecipesBaseExt = "RecipesBase"
    CategoricalArraysSentinelArraysExt = "SentinelArrays"
    CategoricalArraysStructTypesExt = "StructTypes"

    [deps.CategoricalArrays.weakdeps]
    JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
    RecipesBase = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
    SentinelArrays = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
    StructTypes = "856f2bd8-1eba-4b0a-8007-ebc267875bd4"

[[deps.CategoricalDistributions]]
deps = ["CategoricalArrays", "Distributions", "Missings", "OrderedCollections", "Random", "ScientificTypes"]
git-tree-sha1 = "926862f549a82d6c3a7145bc7f1adff2a91a39f0"
uuid = "af321ab8-2d2e-40a6-b165-3d674595d28e"
version = "0.1.15"

    [deps.CategoricalDistributions.extensions]
    UnivariateFiniteDisplayExt = "UnicodePlots"

    [deps.CategoricalDistributions.weakdeps]
    UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[[deps.ChainRulesCore]]
deps = ["Compat", "LinearAlgebra"]
git-tree-sha1 = "3e4b134270b372f2ed4d4d0e936aabaefc1802bc"
uuid = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
version = "1.25.0"
weakdeps = ["SparseArrays"]

    [deps.ChainRulesCore.extensions]
    ChainRulesCoreSparseArraysExt = "SparseArrays"

[[deps.Clustering]]
deps = ["Distances", "LinearAlgebra", "NearestNeighbors", "Printf", "Random", "SparseArrays", "Statistics", "StatsBase"]
git-tree-sha1 = "9ebb045901e9bbf58767a9f34ff89831ed711aae"
uuid = "aaaa29a8-35af-508c-8bc3-b662a17a0fe5"
version = "0.15.7"

[[deps.CodecZlib]]
deps = ["TranscodingStreams", "Zlib_jll"]
git-tree-sha1 = "bce6804e5e6044c6daab27bb533d1295e4a2e759"
uuid = "944b1d66-785c-5afd-91f1-9de20f533193"
version = "0.7.6"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "PrecompileTools", "Random"]
git-tree-sha1 = "c785dfb1b3bfddd1da557e861b919819b82bbe5b"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.27.1"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "b10d0b65641d57b8b4d5e234446582de5047050d"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.5"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "Requires", "Statistics", "TensorCore"]
git-tree-sha1 = "a1f44953f2382ebb937d60dafbe2deea4bd23249"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.10.0"
weakdeps = ["SpecialFunctions"]

    [deps.ColorVectorSpace.extensions]
    SpecialFunctionsExt = "SpecialFunctions"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "64e15186f0aa277e174aa81798f7eb8598e0157e"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.13.0"

[[deps.Combinatorics]]
git-tree-sha1 = "08c8b6831dc00bfea825826be0bc8336fc369860"
uuid = "861a8166-3701-5b0c-9a16-15d98fcdc6aa"
version = "1.0.2"

[[deps.CommonSolve]]
git-tree-sha1 = "0eee5eb66b1cf62cd6ad1b460238e60e4b09400c"
uuid = "38540f10-b2f7-11e9-35d8-d573e4eb0ff2"
version = "0.2.4"

[[deps.CommonSubexpressions]]
deps = ["MacroTools"]
git-tree-sha1 = "cda2cfaebb4be89c9084adaca7dd7333369715c5"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.1"

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

[[deps.CompositionsBase]]
git-tree-sha1 = "802bb88cd69dfd1509f6670416bd4434015693ad"
uuid = "a33af91c-f02d-484b-be07-31d278c5ca2b"
version = "0.1.2"
weakdeps = ["InverseFunctions"]

    [deps.CompositionsBase.extensions]
    CompositionsBaseInverseFunctionsExt = "InverseFunctions"

[[deps.ComputationalResources]]
git-tree-sha1 = "52cb3ec90e8a8bea0e62e275ba577ad0f74821f7"
uuid = "ed09eef8-17a6-5b46-8889-db040fac31e3"
version = "0.3.2"

[[deps.ConcurrentUtilities]]
deps = ["Serialization", "Sockets"]
git-tree-sha1 = "f36e5e8fdffcb5646ea5da81495a5a7566005127"
uuid = "f0e56b4a-5159-44fe-b623-3e5288b988bb"
version = "2.4.3"

[[deps.ConstructionBase]]
git-tree-sha1 = "76219f1ed5771adbb096743bff43fb5fdd4c1157"
uuid = "187b0558-2788-49d3-abe0-74a17ed4e7c9"
version = "1.5.8"

    [deps.ConstructionBase.extensions]
    ConstructionBaseIntervalSetsExt = "IntervalSets"
    ConstructionBaseLinearAlgebraExt = "LinearAlgebra"
    ConstructionBaseStaticArraysExt = "StaticArrays"

    [deps.ConstructionBase.weakdeps]
    IntervalSets = "8197267c-284f-5f27-9208-e0e47529a953"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.ContextVariablesX]]
deps = ["Compat", "Logging", "UUIDs"]
git-tree-sha1 = "25cc3803f1030ab855e383129dcd3dc294e322cc"
uuid = "6add18c4-b38d-439d-96f6-d6bc489c04c5"
version = "0.1.3"

[[deps.Contour]]
git-tree-sha1 = "439e35b0b36e2e5881738abc8857bd92ad6ff9a8"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.3"

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

[[deps.Dbus_jll]]
deps = ["Artifacts", "Expat_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fc173b380865f70627d7dd1190dc2fce6cc105af"
uuid = "ee1fde0b-3d02-5ea6-8484-8dfef6360eab"
version = "1.14.10+0"

[[deps.DefineSingletons]]
git-tree-sha1 = "0fba8b706d0178b4dc7fd44a96a92382c9065c2c"
uuid = "244e2a9f-e319-4986-a169-4d1fe445cd52"
version = "0.1.2"

[[deps.DelimitedFiles]]
deps = ["Mmap"]
git-tree-sha1 = "9e2f36d3c96a820c678f2f1f1782582fcf685bae"
uuid = "8bb1440f-4735-579b-a4ab-409b98df4dab"
version = "1.9.1"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "23163d55f885173722d1e4cf0f6110cdbaf7e272"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.15.1"

[[deps.Distances]]
deps = ["LinearAlgebra", "Statistics", "StatsAPI"]
git-tree-sha1 = "c7e3a542b999843086e2f29dac96a618c105be1d"
uuid = "b4f34e82-e78d-54a5-968a-f98e89d6e8f7"
version = "0.10.12"
weakdeps = ["ChainRulesCore", "SparseArrays"]

    [deps.Distances.extensions]
    DistancesChainRulesCoreExt = "ChainRulesCore"
    DistancesSparseArraysExt = "SparseArrays"

[[deps.Distributed]]
deps = ["Random", "Serialization", "Sockets"]
uuid = "8ba89e20-285c-5b6f-9357-94700520ee1b"
version = "1.11.0"

[[deps.Distributions]]
deps = ["AliasTables", "FillArrays", "LinearAlgebra", "PDMats", "Printf", "QuadGK", "Random", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns"]
git-tree-sha1 = "4b138e4643b577ccf355377c2bc70fa975af25de"
uuid = "31c24e10-a181-5473-b8eb-7969acd0382f"
version = "0.25.115"

    [deps.Distributions.extensions]
    DistributionsChainRulesCoreExt = "ChainRulesCore"
    DistributionsDensityInterfaceExt = "DensityInterface"
    DistributionsTestExt = "Test"

    [deps.Distributions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    DensityInterface = "b429d917-457f-4dbc-8f4c-0cc954292b1d"
    Test = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.EarlyStopping]]
deps = ["Dates", "Statistics"]
git-tree-sha1 = "98fdf08b707aaf69f524a6cd0a67858cefe0cfb6"
uuid = "792122b4-ca99-40de-a6bc-6742525f08b6"
version = "0.3.0"

[[deps.EpollShim_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a4be429317c42cfae6a7fc03c31bad1970c310d"
uuid = "2702e6a9-849d-5ed8-8c21-79e8b8f9ee43"
version = "0.0.20230411+1"

[[deps.ExceptionUnwrapping]]
deps = ["Test"]
git-tree-sha1 = "d36f682e590a83d63d1c7dbd287573764682d12a"
uuid = "460bff9d-24e4-43bc-9d9f-a8973cb893f4"
version = "0.1.11"

[[deps.Expat_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f42a5b1e20e009a43c3646635ed81a9fcaccb287"
uuid = "2e619515-83b5-522b-bb60-26c02a35a201"
version = "2.6.4+2"

[[deps.FFMPEG]]
deps = ["FFMPEG_jll"]
git-tree-sha1 = "53ebe7511fa11d33bec688a9178fac4e49eeee00"
uuid = "c87230d0-a227-11e9-1b43-d7ebe4e7570a"
version = "0.4.2"

[[deps.FFMPEG_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "JLLWrappers", "LAME_jll", "Libdl", "Ogg_jll", "OpenSSL_jll", "Opus_jll", "PCRE2_jll", "Zlib_jll", "libaom_jll", "libass_jll", "libfdk_aac_jll", "libvorbis_jll", "x264_jll", "x265_jll"]
git-tree-sha1 = "466d45dc38e15794ec7d5d63ec03d776a9aff36e"
uuid = "b22a6f82-2f65-5046-a5b2-351ab43fb4e5"
version = "4.4.4+1"

[[deps.FFTW]]
deps = ["AbstractFFTs", "FFTW_jll", "LinearAlgebra", "MKL_jll", "Preferences", "Reexport"]
git-tree-sha1 = "4820348781ae578893311153d69049a93d05f39d"
uuid = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
version = "1.8.0"

[[deps.FFTW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5cf2433259aa3985046792e2afc01fcec076b549"
uuid = "f5851436-0d7a-5f13-b9de-f02708fd171a"
version = "3.3.10+2"

[[deps.FLoops]]
deps = ["BangBang", "Compat", "FLoopsBase", "InitialValues", "JuliaVariables", "MLStyle", "Serialization", "Setfield", "Transducers"]
git-tree-sha1 = "0a2e5873e9a5f54abb06418d57a8df689336a660"
uuid = "cc61a311-1640-44b5-9fba-1b764f453329"
version = "0.2.2"

[[deps.FLoopsBase]]
deps = ["ContextVariablesX"]
git-tree-sha1 = "656f7a6859be8673bf1f35da5670246b923964f7"
uuid = "b9860ae5-e623-471e-878b-f6a53c775ea6"
version = "0.1.1"

[[deps.FeatureSelection]]
deps = ["MLJModelInterface", "ScientificTypesBase", "Tables"]
git-tree-sha1 = "d78c565b6296e161193eb0f053bbcb3f1a82091d"
uuid = "33837fe5-dbff-4c9e-8c2f-c5612fe2b8b6"
version = "0.2.2"

[[deps.FilePathsBase]]
deps = ["Compat", "Dates"]
git-tree-sha1 = "7878ff7172a8e6beedd1dea14bd27c3c6340d361"
uuid = "48062228-2e41-5def-b9a4-89aafe57970f"
version = "0.9.22"
weakdeps = ["Mmap", "Test"]

    [deps.FilePathsBase.extensions]
    FilePathsBaseMmapExt = "Mmap"
    FilePathsBaseTestExt = "Test"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"
version = "1.11.0"

[[deps.FillArrays]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "6a70198746448456524cb442b8af316927ff3e1a"
uuid = "1a297f60-69ca-5386-bcde-b61e274b549b"
version = "1.13.0"
weakdeps = ["PDMats", "SparseArrays", "Statistics"]

    [deps.FillArrays.extensions]
    FillArraysPDMatsExt = "PDMats"
    FillArraysSparseArraysExt = "SparseArrays"
    FillArraysStatisticsExt = "Statistics"

[[deps.FiniteDiff]]
deps = ["ArrayInterface", "LinearAlgebra", "Setfield"]
git-tree-sha1 = "84e3a47db33be7248daa6274b287507dd6ff84e8"
uuid = "6a86dc24-6348-571c-b903-95158fe2bd41"
version = "2.26.2"

    [deps.FiniteDiff.extensions]
    FiniteDiffBandedMatricesExt = "BandedMatrices"
    FiniteDiffBlockBandedMatricesExt = "BlockBandedMatrices"
    FiniteDiffSparseArraysExt = "SparseArrays"
    FiniteDiffStaticArraysExt = "StaticArrays"

    [deps.FiniteDiff.weakdeps]
    BandedMatrices = "aae01518-5342-5314-be14-df237901396f"
    BlockBandedMatrices = "ffab5731-97b5-5995-9138-79e8c1846df0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    StaticArrays = "90137ffa-7385-5640-81b9-e52037218182"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "05882d6995ae5c12bb5f36dd2ed3f61c98cbb172"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.5"

[[deps.Fontconfig_jll]]
deps = ["Artifacts", "Bzip2_jll", "Expat_jll", "FreeType2_jll", "JLLWrappers", "Libdl", "Libuuid_jll", "Zlib_jll"]
git-tree-sha1 = "21fac3c77d7b5a9fc03b0ec503aa1a6392c34d2b"
uuid = "a3f928ae-7b40-5064-980b-68af3947d34b"
version = "2.15.0+0"

[[deps.Format]]
git-tree-sha1 = "9c68794ef81b08086aeb32eeaf33531668d5f5fc"
uuid = "1fa38f19-a742-5d3f-a2b9-30dd87b9d5f8"
version = "1.3.7"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "a2df1b776752e3f344e5116c06d75a10436ab853"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.38"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.FreeType2_jll]]
deps = ["Artifacts", "Bzip2_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "786e968a8d2fb167f2e4880baba62e0e26bd8e4e"
uuid = "d7e528f0-a631-5988-bf34-fe36492bcfd7"
version = "2.13.3+1"

[[deps.FriBidi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "846f7026a9decf3679419122b49f8a1fdb48d2d5"
uuid = "559328eb-81f9-559d-9380-de523a88c83c"
version = "1.0.16+0"

[[deps.Future]]
deps = ["Random"]
uuid = "9fa8497b-333b-5362-9e8d-4d0656e87820"
version = "1.11.0"

[[deps.GLFW_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libglvnd_jll", "Xorg_libXcursor_jll", "Xorg_libXi_jll", "Xorg_libXinerama_jll", "Xorg_libXrandr_jll", "libdecor_jll", "xkbcommon_jll"]
git-tree-sha1 = "fcb0584ff34e25155876418979d4c8971243bb89"
uuid = "0656b61e-2033-5cc2-a64a-77c0f6c09b89"
version = "3.4.0+2"

[[deps.GLM]]
deps = ["Distributions", "LinearAlgebra", "Printf", "Reexport", "SparseArrays", "SpecialFunctions", "Statistics", "StatsAPI", "StatsBase", "StatsFuns", "StatsModels"]
git-tree-sha1 = "97829cfda0df99ddaeaafb5b370d6cab87b7013e"
uuid = "38e38edf-8417-5370-95a0-9cbb8c7f171a"
version = "1.8.3"

[[deps.GPUArraysCore]]
deps = ["Adapt"]
git-tree-sha1 = "83cf05ab16a73219e5f6bd1bdfa9848fa24ac627"
uuid = "46192b85-c4d5-4398-a991-12ede77f4527"
version = "0.2.0"

[[deps.GR]]
deps = ["Artifacts", "Base64", "DelimitedFiles", "Downloads", "GR_jll", "HTTP", "JSON", "Libdl", "LinearAlgebra", "Preferences", "Printf", "Qt6Wayland_jll", "Random", "Serialization", "Sockets", "TOML", "Tar", "Test", "p7zip_jll"]
git-tree-sha1 = "424c8f76017e39fdfcdbb5935a8e6742244959e8"
uuid = "28b8d3ca-fb5f-59d9-8090-bfdbd6d07a71"
version = "0.73.10"

[[deps.GR_jll]]
deps = ["Artifacts", "Bzip2_jll", "Cairo_jll", "FFMPEG_jll", "Fontconfig_jll", "FreeType2_jll", "GLFW_jll", "JLLWrappers", "JpegTurbo_jll", "Libdl", "Libtiff_jll", "Pixman_jll", "Qt6Base_jll", "Zlib_jll", "libpng_jll"]
git-tree-sha1 = "b90934c8cb33920a8dc66736471dc3961b42ec9f"
uuid = "d2c73de3-f751-5644-a686-071e5b155ba9"
version = "0.73.10+0"

[[deps.Gettext_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Libiconv_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "9b02998aba7bf074d14de89f9d37ca24a1a0b046"
uuid = "78b55507-aeef-58d4-861c-77aaff3498b1"
version = "0.21.0+0"

[[deps.Glib_jll]]
deps = ["Artifacts", "Gettext_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Libiconv_jll", "Libmount_jll", "PCRE2_jll", "Zlib_jll"]
git-tree-sha1 = "b0036b392358c80d2d2124746c2bf3d48d457938"
uuid = "7746bdde-850d-59dc-9ae8-88ece973131d"
version = "2.82.4+0"

[[deps.Graphite2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "01979f9b37367603e2848ea225918a3b3861b606"
uuid = "3b182d85-2403-5c21-9c21-1e1f0cc25472"
version = "1.3.14+1"

[[deps.Grisu]]
git-tree-sha1 = "53bb909d1151e57e2484c3d1b53e19552b887fb2"
uuid = "42e2da0e-8278-4e71-bc24-59509adca0fe"
version = "1.0.2"

[[deps.HTTP]]
deps = ["Base64", "CodecZlib", "ConcurrentUtilities", "Dates", "ExceptionUnwrapping", "Logging", "LoggingExtras", "MbedTLS", "NetworkOptions", "OpenSSL", "PrecompileTools", "Random", "SimpleBufferStream", "Sockets", "URIs", "UUIDs"]
git-tree-sha1 = "c67b33b085f6e2faf8bf79a61962e7339a81129c"
uuid = "cd3eb016-35fb-5094-929b-558a96fad6f3"
version = "1.10.15"

[[deps.HarfBuzz_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "Glib_jll", "Graphite2_jll", "JLLWrappers", "Libdl", "Libffi_jll"]
git-tree-sha1 = "55c53be97790242c29031e5cd45e8ac296dadda3"
uuid = "2e76f6c2-a576-52d4-95c1-20adfe4de566"
version = "8.5.0+0"

[[deps.HypergeometricFunctions]]
deps = ["LinearAlgebra", "OpenLibm_jll", "SpecialFunctions"]
git-tree-sha1 = "b1c2585431c382e3fe5805874bda6aea90a95de9"
uuid = "34004b35-14d8-5ef3-9330-4cdb6864b03a"
version = "0.3.25"

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

[[deps.HypothesisTests]]
deps = ["Combinatorics", "Distributions", "LinearAlgebra", "Printf", "Random", "Rmath", "Roots", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "6c3ce99fdbaf680aa6716f4b919c19e902d67c9c"
uuid = "09f84164-cd44-5f33-b23f-e6b0d136a0d5"
version = "0.11.3"

[[deps.IOCapture]]
deps = ["Logging", "Random"]
git-tree-sha1 = "b6d6bfdd7ce25b0f9b2f6b3dd56b2673a66c8770"
uuid = "b5f81e59-6552-4d32-b1f0-c071b021bf89"
version = "0.2.5"

[[deps.InitialValues]]
git-tree-sha1 = "4da0f88e9a39111c2fa3add390ab15f3a44f3ca3"
uuid = "22cec73e-a1b8-11e9-2c92-598750a2cf9c"
version = "0.3.1"

[[deps.InlineStrings]]
git-tree-sha1 = "45521d31238e87ee9f9732561bfee12d4eebd52d"
uuid = "842dd82b-1e85-43dc-bf29-5d0ee9dffc48"
version = "1.4.2"

    [deps.InlineStrings.extensions]
    ArrowTypesExt = "ArrowTypes"
    ParsersExt = "Parsers"

    [deps.InlineStrings.weakdeps]
    ArrowTypes = "31f734f8-188a-4ce0-8406-c8a06bd891cd"
    Parsers = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"

[[deps.IntelOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "LazyArtifacts", "Libdl"]
git-tree-sha1 = "10bd689145d2c3b2a9844005d01087cc1194e79e"
uuid = "1d5cc7b8-4909-519e-a0f8-d0f5ad9712d0"
version = "2024.2.1+0"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"
version = "1.11.0"

[[deps.Interpolations]]
deps = ["Adapt", "AxisAlgorithms", "ChainRulesCore", "LinearAlgebra", "OffsetArrays", "Random", "Ratios", "Requires", "SharedArrays", "SparseArrays", "StaticArrays", "WoodburyMatrices"]
git-tree-sha1 = "88a101217d7cb38a7b481ccd50d21876e1d1b0e0"
uuid = "a98d9a8b-a2ab-59e6-89dd-64a1c18fca59"
version = "0.15.1"
weakdeps = ["Unitful"]

    [deps.Interpolations.extensions]
    InterpolationsUnitfulExt = "Unitful"

[[deps.InverseFunctions]]
git-tree-sha1 = "a779299d77cd080bf77b97535acecd73e1c5e5cb"
uuid = "3587e190-3f89-42d0-90ee-14403ec27112"
version = "0.1.17"
weakdeps = ["Dates", "Test"]

    [deps.InverseFunctions.extensions]
    InverseFunctionsDatesExt = "Dates"
    InverseFunctionsTestExt = "Test"

[[deps.InvertedIndices]]
git-tree-sha1 = "6da3c4316095de0f5ee2ebd875df8721e7e0bdbe"
uuid = "41ab1584-1d38-5bbf-9106-f11c6c58b48f"
version = "1.3.1"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.IterationControl]]
deps = ["EarlyStopping", "InteractiveUtils"]
git-tree-sha1 = "e663925ebc3d93c1150a7570d114f9ea2f664726"
uuid = "b3c1a2ee-3fec-4384-bf48-272ea71de57c"
version = "0.5.4"

[[deps.IterativeSolvers]]
deps = ["LinearAlgebra", "Printf", "Random", "RecipesBase", "SparseArrays"]
git-tree-sha1 = "59545b0a2b27208b0650df0a46b8e3019f85055b"
uuid = "42fd0dbc-a981-5370-80f2-aaf504508153"
version = "0.9.4"

[[deps.IteratorInterfaceExtensions]]
git-tree-sha1 = "a3f24677c21f5bbe9d2a714f95dcd58337fb2856"
uuid = "82899510-4779-5014-852e-03e436cf321d"
version = "1.0.0"

[[deps.JLFzf]]
deps = ["Pipe", "REPL", "Random", "fzf_jll"]
git-tree-sha1 = "71b48d857e86bf7a1838c4736545699974ce79a2"
uuid = "1019f520-868f-41f5-a6de-eb00f4b6a39c"
version = "0.1.9"

[[deps.JLLWrappers]]
deps = ["Artifacts", "Preferences"]
git-tree-sha1 = "a007feb38b422fbdab534406aeca1b86823cb4d6"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.7.0"

[[deps.JSON]]
deps = ["Dates", "Mmap", "Parsers", "Unicode"]
git-tree-sha1 = "31e996f0a15c7b280ba9f76636b3ff9e2ae58c9a"
uuid = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
version = "0.21.4"

[[deps.JpegTurbo_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "3447a92280ecaad1bd93d3fce3d408b6cfff8913"
uuid = "aacddb02-875f-59d6-b918-886e6ef4fbf8"
version = "3.1.0+1"

[[deps.JuliaVariables]]
deps = ["MLStyle", "NameResolution"]
git-tree-sha1 = "49fb3cb53362ddadb4415e9b73926d6b40709e70"
uuid = "b14d175d-62b4-44ba-8fb7-3064adc8c3ec"
version = "0.2.4"

[[deps.Kaleido_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "43032da5832754f58d14a91ffbe86d5f176acda9"
uuid = "f7e6163d-2fa5-5f23-b69c-1db539e41963"
version = "0.2.1+0"

[[deps.KernelAbstractions]]
deps = ["Adapt", "Atomix", "InteractiveUtils", "MacroTools", "PrecompileTools", "Requires", "StaticArrays", "UUIDs"]
git-tree-sha1 = "b9a838cd3028785ac23822cded5126b3da394d1a"
uuid = "63c18a36-062a-441e-b654-da1e3ab1ce7c"
version = "0.9.31"

    [deps.KernelAbstractions.extensions]
    EnzymeExt = "EnzymeCore"
    LinearAlgebraExt = "LinearAlgebra"
    SparseArraysExt = "SparseArrays"

    [deps.KernelAbstractions.weakdeps]
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.KernelDensity]]
deps = ["Distributions", "DocStringExtensions", "FFTW", "Interpolations", "StatsBase"]
git-tree-sha1 = "7d703202e65efa1369de1279c162b915e245eed1"
uuid = "5ab0869b-81aa-558d-bb23-cbf5423bbe9b"
version = "0.6.9"

[[deps.LAME_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "170b660facf5df5de098d866564877e119141cbd"
uuid = "c1c5ebd0-6772-5130-a774-d5fcae4a789d"
version = "3.100.2+0"

[[deps.LERC_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78e0f4b5270c4ae09c7c5f78e77b904199038945"
uuid = "88015f11-f218-50d7-93a8-a6af411a945d"
version = "4.0.0+2"

[[deps.LLVMOpenMP_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "78211fb6cbc872f77cad3fc0b6cf647d923f4929"
uuid = "1d63c593-3942-5779-bab2-d838dc0a180e"
version = "18.1.7+0"

[[deps.LZO_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "16e6ec700154e8004dba90b4aec376f68905d104"
uuid = "dd4b983a-f0e5-5f8d-a1b7-129d4a5fb1ac"
version = "2.10.2+2"

[[deps.LaTeXStrings]]
git-tree-sha1 = "dda21b8cbd6a6c40d9d02a73230f9d70fed6918c"
uuid = "b964fa9f-0449-5b57-a5c2-d3ea65f4040f"
version = "1.4.0"

[[deps.Latexify]]
deps = ["Format", "InteractiveUtils", "LaTeXStrings", "MacroTools", "Markdown", "OrderedCollections", "Requires"]
git-tree-sha1 = "ce5f5621cac23a86011836badfedf664a612cee4"
uuid = "23fbe1c1-3f47-55db-b15f-69d7ec21a316"
version = "0.16.5"

    [deps.Latexify.extensions]
    DataFramesExt = "DataFrames"
    SparseArraysExt = "SparseArrays"
    SymEngineExt = "SymEngine"

    [deps.Latexify.weakdeps]
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    SparseArrays = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
    SymEngine = "123dc426-2d89-5057-bbad-38513e3affd8"

[[deps.LatinHypercubeSampling]]
deps = ["Random", "StableRNGs", "StatsBase", "Test"]
git-tree-sha1 = "825289d43c753c7f1bf9bed334c253e9913997f8"
uuid = "a5e1c1ea-c99a-51d3-a14d-a9a37257b02d"
version = "1.9.0"

[[deps.LazyArtifacts]]
deps = ["Artifacts", "Pkg"]
uuid = "4af54fe1-eca0-43a8-85a7-787d91b784e3"
version = "1.11.0"

[[deps.LearnAPI]]
deps = ["InteractiveUtils", "Statistics"]
git-tree-sha1 = "ec695822c1faaaa64cee32d0b21505e1977b4809"
uuid = "92ad9a40-7767-427a-9ee6-6e577f1266cb"
version = "0.1.0"

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

[[deps.Libffi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "27ecae93dd25ee0909666e6835051dd684cc035e"
uuid = "e9f186c6-92d2-5b65-8a66-fee21dc1b490"
version = "3.2.2+2"

[[deps.Libgcrypt_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgpg_error_jll"]
git-tree-sha1 = "8be878062e0ffa2c3f67bb58a595375eda5de80b"
uuid = "d4300ac3-e22c-5743-9152-c294e39db1e4"
version = "1.11.0+0"

[[deps.Libglvnd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll", "Xorg_libXext_jll"]
git-tree-sha1 = "ff3b4b9d35de638936a525ecd36e86a8bb919d11"
uuid = "7e76a0d4-f3c7-5321-8279-8d96eeed0f29"
version = "1.7.0+0"

[[deps.Libgpg_error_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "a7f43994b47130e4f491c3b2dbe78fe9e2aed2b3"
uuid = "7add5ba3-2f88-524e-9cd5-f83b8a55f7b8"
version = "1.51.0+0"

[[deps.Libiconv_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "61dfdba58e585066d8bce214c5a51eaa0539f269"
uuid = "94ce4f54-9a6c-5748-9c1c-f9c7231a4531"
version = "1.17.0+1"

[[deps.Libmount_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "d841749621f4dcf0ddc26a27d1f6484dfc37659a"
uuid = "4b2f31a3-9ecc-558c-b454-b3730dcb73e9"
version = "2.40.2+1"

[[deps.Libtiff_jll]]
deps = ["Artifacts", "JLLWrappers", "JpegTurbo_jll", "LERC_jll", "Libdl", "XZ_jll", "Zlib_jll", "Zstd_jll"]
git-tree-sha1 = "b404131d06f7886402758c9ce2214b636eb4d54a"
uuid = "89763e89-9b03-5906-acba-b20f662cd828"
version = "4.7.0+0"

[[deps.Libuuid_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "9d630b7fb0be32eeb5e8da515f5e8a26deb457fe"
uuid = "38a345b3-de98-5d2b-a5d3-14cd9215e700"
version = "2.40.2+1"

[[deps.LineSearches]]
deps = ["LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "Printf"]
git-tree-sha1 = "e4c3be53733db1051cc15ecf573b1042b3a712a1"
uuid = "d3d80556-e9d4-5f37-9878-2ab0fcc64255"
version = "7.3.0"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
version = "1.11.0"

[[deps.LinearMaps]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "ee79c3208e55786de58f8dcccca098ced79f743f"
uuid = "7a12625a-238d-50fd-b39a-03d52299707e"
version = "3.11.3"
weakdeps = ["ChainRulesCore", "SparseArrays", "Statistics"]

    [deps.LinearMaps.extensions]
    LinearMapsChainRulesCoreExt = "ChainRulesCore"
    LinearMapsSparseArraysExt = "SparseArrays"
    LinearMapsStatisticsExt = "Statistics"

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
git-tree-sha1 = "65f28ad4b594aebe22157d6fac869786a255b7eb"
uuid = "6c6e2e6c-3030-632d-7369-2d6c69616d65"
version = "0.1.4"

[[deps.MKL_jll]]
deps = ["Artifacts", "IntelOpenMP_jll", "JLLWrappers", "LazyArtifacts", "Libdl", "oneTBB_jll"]
git-tree-sha1 = "f046ccd0c6db2832a9f639e2c669c6fe867e5f4f"
uuid = "856f044c-d86e-5d09-b602-aeab76dc8ba7"
version = "2024.2.0+0"

[[deps.MLFlowClient]]
deps = ["Dates", "FilePathsBase", "HTTP", "JSON", "ShowCases", "URIs", "UUIDs"]
git-tree-sha1 = "9abb12b62debc27261c008daa13627255bf79967"
uuid = "64a0f543-368b-4a9a-827a-e71edb2a0b83"
version = "0.5.1"

[[deps.MLJ]]
deps = ["CategoricalArrays", "ComputationalResources", "Distributed", "Distributions", "FeatureSelection", "LinearAlgebra", "MLJBalancing", "MLJBase", "MLJEnsembles", "MLJFlow", "MLJIteration", "MLJModels", "MLJTuning", "OpenML", "Pkg", "ProgressMeter", "Random", "Reexport", "ScientificTypes", "StatisticalMeasures", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "521eec7a22417d54fdc66f5dc0b7dc9628931c54"
uuid = "add582a8-e3ab-11e8-2d5e-e98b27df1bc7"
version = "0.20.7"

[[deps.MLJBalancing]]
deps = ["MLJBase", "MLJModelInterface", "MLUtils", "OrderedCollections", "Random", "StatsBase"]
git-tree-sha1 = "f707a01a92d664479522313907c07afa5d81df19"
uuid = "45f359ea-796d-4f51-95a5-deb1a414c586"
version = "0.1.5"

[[deps.MLJBase]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Dates", "DelimitedFiles", "Distributed", "Distributions", "InteractiveUtils", "InvertedIndices", "LearnAPI", "LinearAlgebra", "MLJModelInterface", "Missings", "OrderedCollections", "Parameters", "PrettyTables", "ProgressMeter", "Random", "RecipesBase", "Reexport", "ScientificTypes", "Serialization", "StatisticalMeasuresBase", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "6f45e12073bc2f2e73ed0473391db38c31e879c9"
uuid = "a7f614a8-145f-11e9-1d2a-a57a1082229d"
version = "1.7.0"
weakdeps = ["StatisticalMeasures"]

    [deps.MLJBase.extensions]
    DefaultMeasuresExt = "StatisticalMeasures"

[[deps.MLJEnsembles]]
deps = ["CategoricalArrays", "CategoricalDistributions", "ComputationalResources", "Distributed", "Distributions", "MLJModelInterface", "ProgressMeter", "Random", "ScientificTypesBase", "StatisticalMeasuresBase", "StatsBase"]
git-tree-sha1 = "84a5be55a364bb6b6dc7780bbd64317ebdd3ad1e"
uuid = "50ed68f4-41fd-4504-931a-ed422449fee0"
version = "0.4.3"

[[deps.MLJFlow]]
deps = ["MLFlowClient", "MLJBase", "MLJModelInterface"]
git-tree-sha1 = "508bff8071d7d1902d6f1b9d1e868d58821f1cfe"
uuid = "7b7b8358-b45c-48ea-a8ef-7ca328ad328f"
version = "0.5.0"

[[deps.MLJIteration]]
deps = ["IterationControl", "MLJBase", "Random", "Serialization"]
git-tree-sha1 = "ad16cfd261e28204847f509d1221a581286448ae"
uuid = "614be32b-d00c-4edb-bd02-1eb411ab5e55"
version = "0.6.3"

[[deps.MLJLinearModels]]
deps = ["DocStringExtensions", "IterativeSolvers", "LinearAlgebra", "LinearMaps", "MLJModelInterface", "Optim", "Parameters"]
git-tree-sha1 = "7f517fd840ca433a8fae673edb31678ff55d969c"
uuid = "6ee0df7b-362f-4a72-a706-9e79364fb692"
version = "0.10.0"

[[deps.MLJModelInterface]]
deps = ["Random", "ScientificTypesBase", "StatisticalTraits"]
git-tree-sha1 = "ceaff6618408d0e412619321ae43b33b40c1a733"
uuid = "e80e1ace-859a-464e-9ed9-23947d8ae3ea"
version = "1.11.0"

[[deps.MLJModels]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Combinatorics", "Dates", "Distances", "Distributions", "InteractiveUtils", "LinearAlgebra", "MLJModelInterface", "Markdown", "OrderedCollections", "Parameters", "Pkg", "PrettyPrinting", "REPL", "Random", "RelocatableFolders", "ScientificTypes", "StatisticalTraits", "Statistics", "StatsBase", "Tables"]
git-tree-sha1 = "9dbbe1e9a8ba1fd60fefb5e39dd9a070bbda9c79"
uuid = "d491faf4-2d78-11e9-2867-c94bc002c0b7"
version = "0.17.6"

[[deps.MLJTuning]]
deps = ["ComputationalResources", "Distributed", "Distributions", "LatinHypercubeSampling", "MLJBase", "ProgressMeter", "Random", "RecipesBase", "StatisticalMeasuresBase"]
git-tree-sha1 = "38aab60b1274ce7d6da784808e3be69e585dbbf6"
uuid = "03970b2e-30c4-11ea-3135-d1576263f10f"
version = "0.8.8"

[[deps.MLStyle]]
git-tree-sha1 = "bc38dff0548128765760c79eb7388a4b37fae2c8"
uuid = "d8e11817-5142-5d16-987a-aa16d5891078"
version = "0.4.17"

[[deps.MLUtils]]
deps = ["ChainRulesCore", "Compat", "DataAPI", "DelimitedFiles", "FLoops", "NNlib", "Random", "ShowCases", "SimpleTraits", "Statistics", "StatsBase", "Tables", "Transducers"]
git-tree-sha1 = "b45738c2e3d0d402dffa32b2c1654759a2ac35a4"
uuid = "f1d291b0-491e-4a28-83b9-f70985020b54"
version = "0.4.4"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "2fa9ee3e63fd3a4f7a9a4f4744a52f4856de82df"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.13"

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

[[deps.Measures]]
git-tree-sha1 = "c13304c81eec1ed3af7fc20e75fb6b26092a1102"
uuid = "442fdcdd-2543-5da2-b0f3-8c86c306513e"
version = "0.3.2"

[[deps.MicroCollections]]
deps = ["Accessors", "BangBang", "InitialValues"]
git-tree-sha1 = "44d32db644e84c75dab479f1bc15ee76a1a3618f"
uuid = "128add7d-3638-4c79-886c-908ea0c25c34"
version = "0.2.0"

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

[[deps.MultivariateStats]]
deps = ["Arpack", "Distributions", "LinearAlgebra", "SparseArrays", "Statistics", "StatsAPI", "StatsBase"]
git-tree-sha1 = "816620e3aac93e5b5359e4fdaf23ca4525b00ddf"
uuid = "6f286f6a-111f-5878-ab1e-185364afe411"
version = "0.10.3"

[[deps.NLSolversBase]]
deps = ["DiffResults", "Distributed", "FiniteDiff", "ForwardDiff"]
git-tree-sha1 = "a0b464d183da839699f4c79e7606d9d186ec172c"
uuid = "d41bc354-129a-5804-8e4c-c37616107c6c"
version = "7.8.3"

[[deps.NNlib]]
deps = ["Adapt", "Atomix", "ChainRulesCore", "GPUArraysCore", "KernelAbstractions", "LinearAlgebra", "Random", "Statistics"]
git-tree-sha1 = "1177f161cda2083543b9967d7ca2a3e24e721e13"
uuid = "872c559c-99b0-510c-b3b7-b6c96a88d5cd"
version = "0.9.26"

    [deps.NNlib.extensions]
    NNlibAMDGPUExt = "AMDGPU"
    NNlibCUDACUDNNExt = ["CUDA", "cuDNN"]
    NNlibCUDAExt = "CUDA"
    NNlibEnzymeCoreExt = "EnzymeCore"
    NNlibFFTWExt = "FFTW"
    NNlibForwardDiffExt = "ForwardDiff"

    [deps.NNlib.weakdeps]
    AMDGPU = "21141c5a-9bdb-4563-92ae-f87d6854732e"
    CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba"
    EnzymeCore = "f151be2c-9106-41f4-ab19-57ee4f262869"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    cuDNN = "02a925ec-e4fe-4b08-9a7e-0d78e3d38ccd"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NameResolution]]
deps = ["PrettyPrint"]
git-tree-sha1 = "1a0fa0e9613f46c9b8c11eee38ebb4f590013c5e"
uuid = "71a1bf82-56d0-4bbc-8a3c-48b961074391"
version = "0.1.5"

[[deps.NearestNeighbors]]
deps = ["Distances", "StaticArrays"]
git-tree-sha1 = "8a3271d8309285f4db73b4f662b1b290c715e85e"
uuid = "b8a86587-4115-5ab1-83bc-aa920d37bbce"
version = "0.4.21"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.Observables]]
git-tree-sha1 = "7438a59546cf62428fc9d1bc94729146d37a7225"
uuid = "510215fc-4207-5dde-b226-833fc4488ee2"
version = "0.5.5"

[[deps.OffsetArrays]]
git-tree-sha1 = "5e1897147d1ff8d98883cda2be2187dcf57d8f0c"
uuid = "6fe1bfb0-de20-5000-8ca7-80f57d26f881"
version = "1.15.0"
weakdeps = ["Adapt"]

    [deps.OffsetArrays.extensions]
    OffsetArraysAdaptExt = "Adapt"

[[deps.Ogg_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "887579a3eb005446d514ab7aeac5d1d027658b8f"
uuid = "e7412a2a-1a6e-54c0-be00-318e2571c051"
version = "1.3.5+1"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.27+1"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+2"

[[deps.OpenML]]
deps = ["ARFFFiles", "HTTP", "JSON", "Markdown", "Pkg", "Scratch"]
git-tree-sha1 = "63603b2b367107e87dbceda4e33c67aed17e50e0"
uuid = "8b6db2d4-7670-4922-a472-f9537c81ab66"
version = "0.3.2"

[[deps.OpenSSL]]
deps = ["BitFlags", "Dates", "MozillaCACerts_jll", "OpenSSL_jll", "Sockets"]
git-tree-sha1 = "38cb508d080d21dc1128f7fb04f20387ed4c0af4"
uuid = "4d8831e6-92b7-49fb-bdf8-b643e874388c"
version = "1.4.3"

[[deps.OpenSSL_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "f58782a883ecbf9fb48dcd363f9ccd65f36c23a8"
uuid = "458c3c95-2e84-50aa-8efc-19380b2a3a95"
version = "3.0.15+2"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "418e63d434f5ca12b188bbb287dfbe10a5af1da4"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+1"

[[deps.Optim]]
deps = ["Compat", "FillArrays", "ForwardDiff", "LineSearches", "LinearAlgebra", "NLSolversBase", "NaNMath", "Parameters", "PositiveFactorizations", "Printf", "SparseArrays", "StatsBase"]
git-tree-sha1 = "ab7edad78cdef22099f43c54ef77ac63c2c9cc64"
uuid = "429524aa-4258-5aef-a3af-852621145aeb"
version = "1.10.0"

    [deps.Optim.extensions]
    OptimMOIExt = "MathOptInterface"

    [deps.Optim.weakdeps]
    MathOptInterface = "b8f27783-ece8-5eb3-8dc8-9495eed66fee"

[[deps.Opus_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6703a85cb3781bd5909d48730a67205f3f31a575"
uuid = "91d4177d-7536-5919-b921-800302f37372"
version = "1.3.3+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "12f1439c4f986bb868acda6ea33ebc78e19b95ad"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.7.0"

[[deps.PCRE2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "efcefdf7-47ab-520b-bdef-62a2eaa19f15"
version = "10.42.0+1"

[[deps.PDMats]]
deps = ["LinearAlgebra", "SparseArrays", "SuiteSparse"]
git-tree-sha1 = "949347156c25054de2db3b166c52ac4728cbad65"
uuid = "90014a1f-27ba-587c-ab20-58faa44d9150"
version = "0.11.31"

[[deps.Pango_jll]]
deps = ["Artifacts", "Cairo_jll", "Fontconfig_jll", "FreeType2_jll", "FriBidi_jll", "Glib_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ed6834e95bd326c52d5675b4181386dfbe885afb"
uuid = "36c8627f-9965-5494-a995-c6b170f724f3"
version = "1.55.5+0"

[[deps.Parameters]]
deps = ["OrderedCollections", "UnPack"]
git-tree-sha1 = "34c0e9ad262e5f7fc75b10a9952ca7692cfc5fbe"
uuid = "d96e819e-fc66-5662-9728-84c9c7592b0a"
version = "0.12.3"

[[deps.Parsers]]
deps = ["Dates", "PrecompileTools", "UUIDs"]
git-tree-sha1 = "8489905bcdbcfac64d1daa51ca07c0d8f0283821"
uuid = "69de0a69-1ddd-5017-9359-2bf0b02dc9f0"
version = "2.8.1"

[[deps.Pipe]]
git-tree-sha1 = "6842804e7867b115ca9de748a0cf6b364523c16d"
uuid = "b98c9c47-44ae-5843-9183-064241ee97a0"
version = "1.3.0"

[[deps.Pixman_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "LLVMOpenMP_jll", "Libdl"]
git-tree-sha1 = "35621f10a7531bc8fa58f74610b1bfb70a3cfc6b"
uuid = "30392449-352a-5448-841d-b1acce4e97dc"
version = "0.43.4+0"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "Random", "SHA", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.11.0"
weakdeps = ["REPL"]

    [deps.Pkg.extensions]
    REPLExt = "REPL"

[[deps.PlotThemes]]
deps = ["PlotUtils", "Statistics"]
git-tree-sha1 = "41031ef3a1be6f5bbbf3e8073f210556daeae5ca"
uuid = "ccf2f8ad-2431-5c83-bf29-c5338b663b6a"
version = "3.3.0"

[[deps.PlotUtils]]
deps = ["ColorSchemes", "Colors", "Dates", "PrecompileTools", "Printf", "Random", "Reexport", "StableRNGs", "Statistics"]
git-tree-sha1 = "3ca9a356cd2e113c420f2c13bea19f8d3fb1cb18"
uuid = "995b91a9-d308-5afd-9ec6-746e21dbc043"
version = "1.4.3"

[[deps.PlotlyBase]]
deps = ["ColorSchemes", "Dates", "DelimitedFiles", "DocStringExtensions", "JSON", "LaTeXStrings", "Logging", "Parameters", "Pkg", "REPL", "Requires", "Statistics", "UUIDs"]
git-tree-sha1 = "56baf69781fc5e61607c3e46227ab17f7040ffa2"
uuid = "a03496cd-edff-5a9b-9e67-9cda94a718b5"
version = "0.8.19"

[[deps.PlotlyKaleido]]
deps = ["Base64", "JSON", "Kaleido_jll"]
git-tree-sha1 = "3210de4d88af7ca5de9e26305758a59aabc48aac"
uuid = "f2990250-8cf9-495f-b13a-cce12b45703c"
version = "2.2.5"

[[deps.Plots]]
deps = ["Base64", "Contour", "Dates", "Downloads", "FFMPEG", "FixedPointNumbers", "GR", "JLFzf", "JSON", "LaTeXStrings", "Latexify", "LinearAlgebra", "Measures", "NaNMath", "Pkg", "PlotThemes", "PlotUtils", "PrecompileTools", "Printf", "REPL", "Random", "RecipesBase", "RecipesPipeline", "Reexport", "RelocatableFolders", "Requires", "Scratch", "Showoff", "SparseArrays", "Statistics", "StatsBase", "TOML", "UUIDs", "UnicodeFun", "UnitfulLatexify", "Unzip"]
git-tree-sha1 = "dae01f8c2e069a683d3a6e17bbae5070ab94786f"
uuid = "91a5bcdd-55d7-5caf-9e0b-520d859cae80"
version = "1.40.9"

    [deps.Plots.extensions]
    FileIOExt = "FileIO"
    GeometryBasicsExt = "GeometryBasics"
    IJuliaExt = "IJulia"
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.Plots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    GeometryBasics = "5c1252a2-5f33-56bf-86c9-59e7332b4326"
    IJulia = "7073ff75-c697-5162-941a-fcdaad2a7d2a"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.PlutoUI]]
deps = ["AbstractPlutoDingetjes", "Base64", "ColorTypes", "Dates", "FixedPointNumbers", "Hyperscript", "HypertextLiteral", "IOCapture", "InteractiveUtils", "JSON", "Logging", "MIMEs", "Markdown", "Random", "Reexport", "URIs", "UUIDs"]
git-tree-sha1 = "eba4810d5e6a01f612b948c9fa94f905b49087b0"
uuid = "7f904dfe-b85e-4ff6-b463-dae2292396a8"
version = "0.7.60"

[[deps.Polynomials]]
deps = ["LinearAlgebra", "OrderedCollections", "RecipesBase", "Requires", "Setfield", "SparseArrays"]
git-tree-sha1 = "adc25dbd4d13f148f3256b6d4743fe7e63a71c4a"
uuid = "f27b6e38-b328-58d1-80ce-0feddd5e7a45"
version = "4.0.12"

    [deps.Polynomials.extensions]
    PolynomialsChainRulesCoreExt = "ChainRulesCore"
    PolynomialsFFTWExt = "FFTW"
    PolynomialsMakieCoreExt = "MakieCore"
    PolynomialsMutableArithmeticsExt = "MutableArithmetics"

    [deps.Polynomials.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    FFTW = "7a1cc6ca-52ef-59f5-83cd-3a7055c09341"
    MakieCore = "20f20a25-4f0e-4fdf-b5d1-57303727442b"
    MutableArithmetics = "d8a4904e-b15c-11e9-3269-09a3773c0cb0"

[[deps.PooledArrays]]
deps = ["DataAPI", "Future"]
git-tree-sha1 = "36d8b4b899628fb92c2749eb488d884a926614d3"
uuid = "2dfb63ee-cc39-5dd5-95bd-886bf059d720"
version = "1.4.3"

[[deps.PositiveFactorizations]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "17275485f373e6673f7e7f97051f703ed5b15b20"
uuid = "85a6dd25-e78a-55b7-8502-1745935b8125"
version = "0.2.4"

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

[[deps.PrettyPrint]]
git-tree-sha1 = "632eb4abab3449ab30c5e1afaa874f0b98b586e4"
uuid = "8162dcfd-2161-5ef2-ae6c-7681170c5f98"
version = "0.2.0"

[[deps.PrettyPrinting]]
git-tree-sha1 = "142ee93724a9c5d04d78df7006670a93ed1b244e"
uuid = "54e16d92-306c-5ea0-a30b-337be88ac337"
version = "0.4.2"

[[deps.PrettyTables]]
deps = ["Crayons", "LaTeXStrings", "Markdown", "PrecompileTools", "Printf", "Reexport", "StringManipulation", "Tables"]
git-tree-sha1 = "1101cd475833706e4d0e7b122218257178f48f34"
uuid = "08abe8d2-0d0c-5749-adfa-8a2ac140af0d"
version = "2.4.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"
version = "1.11.0"

[[deps.ProgressMeter]]
deps = ["Distributed", "Printf"]
git-tree-sha1 = "8f6bc219586aef8baf0ff9a5fe16ee9c70cb65e4"
uuid = "92933f4c-e287-5a05-a399-4b506db050ca"
version = "1.10.2"

[[deps.PtrArrays]]
git-tree-sha1 = "77a42d78b6a92df47ab37e177b2deac405e1c88f"
uuid = "43287f4e-b6f4-7ad1-bb20-aadabca52c3d"
version = "1.2.1"

[[deps.Qt6Base_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Fontconfig_jll", "Glib_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "OpenSSL_jll", "Vulkan_Loader_jll", "Xorg_libSM_jll", "Xorg_libXext_jll", "Xorg_libXrender_jll", "Xorg_libxcb_jll", "Xorg_xcb_util_cursor_jll", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_keysyms_jll", "Xorg_xcb_util_renderutil_jll", "Xorg_xcb_util_wm_jll", "Zlib_jll", "libinput_jll", "xkbcommon_jll"]
git-tree-sha1 = "492601870742dcd38f233b23c3ec629628c1d724"
uuid = "c0090381-4147-56d7-9ebc-da0b1113ec56"
version = "6.7.1+1"

[[deps.Qt6Declarative_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6ShaderTools_jll"]
git-tree-sha1 = "e5dd466bf2569fe08c91a2cc29c1003f4797ac3b"
uuid = "629bc702-f1f5-5709-abd5-49b8460ea067"
version = "6.7.1+2"

[[deps.Qt6ShaderTools_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll"]
git-tree-sha1 = "1a180aeced866700d4bebc3120ea1451201f16bc"
uuid = "ce943373-25bb-56aa-8eca-768745ed7b5a"
version = "6.7.1+1"

[[deps.Qt6Wayland_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Qt6Base_jll", "Qt6Declarative_jll"]
git-tree-sha1 = "729927532d48cf79f49070341e1d918a65aba6b0"
uuid = "e99dba38-086e-5de3-a5b1-6e4c66e897c3"
version = "6.7.1+1"

[[deps.QuadGK]]
deps = ["DataStructures", "LinearAlgebra"]
git-tree-sha1 = "cda3b045cf9ef07a08ad46731f5a3165e56cf3da"
uuid = "1fd47b50-473d-5c70-9696-f719f8f3bcdc"
version = "2.11.1"

    [deps.QuadGK.extensions]
    QuadGKEnzymeExt = "Enzyme"

    [deps.QuadGK.weakdeps]
    Enzyme = "7da242da-08ed-463a-9acd-ee780be4f1d9"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "StyledStrings", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"
version = "1.11.0"

[[deps.Random]]
deps = ["SHA"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
version = "1.11.0"

[[deps.Ratios]]
deps = ["Requires"]
git-tree-sha1 = "1342a47bf3260ee108163042310d26f2be5ec90b"
uuid = "c84ed2f1-dad5-54f0-aa8e-dbefe2724439"
version = "0.4.5"
weakdeps = ["FixedPointNumbers"]

    [deps.Ratios.extensions]
    RatiosFixedPointNumbersExt = "FixedPointNumbers"

[[deps.RecipesBase]]
deps = ["PrecompileTools"]
git-tree-sha1 = "5c3d09cc4f31f5fc6af001c250bf1278733100ff"
uuid = "3cdcf5f2-1ef4-517c-9805-6587b60abb01"
version = "1.3.4"

[[deps.RecipesPipeline]]
deps = ["Dates", "NaNMath", "PlotUtils", "PrecompileTools", "RecipesBase"]
git-tree-sha1 = "45cf9fd0ca5839d06ef333c8201714e888486342"
uuid = "01d81517-befc-4cb6-b9ec-a95719d0359c"
version = "0.6.12"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.RelocatableFolders]]
deps = ["SHA", "Scratch"]
git-tree-sha1 = "ffdaf70d81cf6ff22c2b6e733c900c3321cab864"
uuid = "05181044-ff0b-4ac5-8273-598c1e38db00"
version = "1.0.1"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.Rmath]]
deps = ["Random", "Rmath_jll"]
git-tree-sha1 = "852bd0f55565a9e973fcfee83a84413270224dc4"
uuid = "79098fc4-a85e-5d69-aa6a-4863f24498fa"
version = "0.8.0"

[[deps.Rmath_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "58cdd8fb2201a6267e1db87ff148dd6c1dbd8ad8"
uuid = "f50d1b31-88e8-58de-be2c-1cc44531875f"
version = "0.5.1+0"

[[deps.Roots]]
deps = ["Accessors", "CommonSolve", "Printf"]
git-tree-sha1 = "8e3694d669323cdfb560e344dc872b984de23b71"
uuid = "f2b01f46-fcfa-551c-844a-d8ac1e96c665"
version = "2.2.2"

    [deps.Roots.extensions]
    RootsChainRulesCoreExt = "ChainRulesCore"
    RootsForwardDiffExt = "ForwardDiff"
    RootsIntervalRootFindingExt = "IntervalRootFinding"
    RootsSymPyExt = "SymPy"
    RootsSymPyPythonCallExt = "SymPyPythonCall"

    [deps.Roots.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
    IntervalRootFinding = "d2bf35a9-74e0-55ec-b149-d360ff49b807"
    SymPy = "24249f21-da20-56a4-8eb1-6a02cf4ae2e6"
    SymPyPythonCall = "bc8888f7-b21e-4b7c-a06a-5d9c9496438c"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.ScientificTypes]]
deps = ["CategoricalArrays", "ColorTypes", "Dates", "Distributions", "PrettyTables", "Reexport", "ScientificTypesBase", "StatisticalTraits", "Tables"]
git-tree-sha1 = "75ccd10ca65b939dab03b812994e571bf1e3e1da"
uuid = "321657f4-b219-11e9-178b-2701a2544e81"
version = "3.0.2"

[[deps.ScientificTypesBase]]
git-tree-sha1 = "a8e18eb383b5ecf1b5e6fc237eb39255044fd92b"
uuid = "30f210dd-8aff-4c5f-94ba-8e64358c1161"
version = "3.0.0"

[[deps.Scratch]]
deps = ["Dates"]
git-tree-sha1 = "3bac05bc7e74a75fd9cba4295cde4045d9fe2386"
uuid = "6c6a2e73-6563-6170-7368-637461726353"
version = "1.2.1"

[[deps.SentinelArrays]]
deps = ["Dates", "Random"]
git-tree-sha1 = "712fb0231ee6f9120e005ccd56297abbc053e7e0"
uuid = "91c51154-3ec4-41a3-a24f-3f23e20d615c"
version = "1.4.8"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"
version = "1.11.0"

[[deps.Setfield]]
deps = ["ConstructionBase", "Future", "MacroTools", "StaticArraysCore"]
git-tree-sha1 = "e2cc6d8c88613c05e1defb55170bf5ff211fbeac"
uuid = "efcf1570-3423-57d1-acb7-fd33fddbac46"
version = "1.1.1"

[[deps.SharedArrays]]
deps = ["Distributed", "Mmap", "Random", "Serialization"]
uuid = "1a1011a3-84de-559e-8e89-a11a2f7dc383"
version = "1.11.0"

[[deps.ShiftedArrays]]
git-tree-sha1 = "503688b59397b3307443af35cd953a13e8005c16"
uuid = "1277b4bf-5013-50f5-be3d-901d8477a67a"
version = "2.0.0"

[[deps.ShowCases]]
git-tree-sha1 = "7f534ad62ab2bd48591bdeac81994ea8c445e4a5"
uuid = "605ecd9f-84a6-4c9e-81e2-4798472b76a3"
version = "0.1.0"

[[deps.Showoff]]
deps = ["Dates", "Grisu"]
git-tree-sha1 = "91eddf657aca81df9ae6ceb20b959ae5653ad1de"
uuid = "992d4aef-0814-514b-bc4d-f2e9a6c4116f"
version = "1.0.3"

[[deps.SimpleBufferStream]]
git-tree-sha1 = "f305871d2f381d21527c770d4788c06c097c9bc1"
uuid = "777ac1f9-54b0-4bf8-805c-2214025038e7"
version = "1.2.0"

[[deps.SimpleTraits]]
deps = ["InteractiveUtils", "MacroTools"]
git-tree-sha1 = "5d7e3f4e11935503d3ecaf7186eac40602e7d231"
uuid = "699a6c99-e7fa-54fc-8d76-47d257e15c1d"
version = "0.9.4"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"
version = "1.11.0"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "66e0a8e672a0bdfca2c3f5937efb8538b9ddc085"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.2.1"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"
version = "1.11.0"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "64cca0c26b4f31ba18f13f6c12af7c85f478cfde"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.5.0"
weakdeps = ["ChainRulesCore"]

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

[[deps.SplittablesBase]]
deps = ["Setfield", "Test"]
git-tree-sha1 = "e08a62abc517eb79667d0a29dc08a3b589516bb5"
uuid = "171d559e-b47b-412a-8079-5efa626c420e"
version = "0.1.15"

[[deps.StableRNGs]]
deps = ["Random"]
git-tree-sha1 = "83e6cce8324d49dfaf9ef059227f91ed4441a8e5"
uuid = "860ef19b-820b-49d6-a774-d7a799459cd3"
version = "1.0.2"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "PrecompileTools", "Random", "StaticArraysCore"]
git-tree-sha1 = "47091a0340a675c738b1304b58161f3b0839d454"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.9.10"
weakdeps = ["ChainRulesCore", "Statistics"]

    [deps.StaticArrays.extensions]
    StaticArraysChainRulesCoreExt = "ChainRulesCore"
    StaticArraysStatisticsExt = "Statistics"

[[deps.StaticArraysCore]]
git-tree-sha1 = "192954ef1208c7019899fbf8049e717f92959682"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.3"

[[deps.StatisticalMeasures]]
deps = ["CategoricalArrays", "CategoricalDistributions", "Distributions", "LearnAPI", "LinearAlgebra", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "StatisticalMeasuresBase", "Statistics", "StatsBase"]
git-tree-sha1 = "c1d4318fa41056b839dfbb3ee841f011fa6e8518"
uuid = "a19d573c-0a75-4610-95b3-7071388c7541"
version = "0.1.7"

    [deps.StatisticalMeasures.extensions]
    LossFunctionsExt = "LossFunctions"
    ScientificTypesExt = "ScientificTypes"

    [deps.StatisticalMeasures.weakdeps]
    LossFunctions = "30fc2ffe-d236-52d8-8643-a9d8f7c094a7"
    ScientificTypes = "321657f4-b219-11e9-178b-2701a2544e81"

[[deps.StatisticalMeasuresBase]]
deps = ["CategoricalArrays", "InteractiveUtils", "MLUtils", "MacroTools", "OrderedCollections", "PrecompileTools", "ScientificTypesBase", "Statistics"]
git-tree-sha1 = "17dfb22e2e4ccc9cd59b487dce52883e0151b4d3"
uuid = "c062fc1d-0d66-479b-b6ac-8b44719de4cc"
version = "0.1.1"

[[deps.StatisticalTraits]]
deps = ["ScientificTypesBase"]
git-tree-sha1 = "542d979f6e756f13f862aa00b224f04f9e445f11"
uuid = "64bff920-2084-43da-a3e6-9bb72801c0c9"
version = "3.4.0"

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
git-tree-sha1 = "1ff449ad350c9c4cbc756624d6f8a8c3ef56d3ed"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.7.0"

[[deps.StatsBase]]
deps = ["AliasTables", "DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "29321314c920c26684834965ec2ce0dacc9cf8e5"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.34.4"

[[deps.StatsFuns]]
deps = ["HypergeometricFunctions", "IrrationalConstants", "LogExpFunctions", "Reexport", "Rmath", "SpecialFunctions"]
git-tree-sha1 = "b423576adc27097764a90e163157bcfc9acf0f46"
uuid = "4c63d2b9-4356-54db-8cca-17b64c39e42c"
version = "1.3.2"
weakdeps = ["ChainRulesCore", "InverseFunctions"]

    [deps.StatsFuns.extensions]
    StatsFunsChainRulesCoreExt = "ChainRulesCore"
    StatsFunsInverseFunctionsExt = "InverseFunctions"

[[deps.StatsModels]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "Printf", "REPL", "ShiftedArrays", "SparseArrays", "StatsAPI", "StatsBase", "StatsFuns", "Tables"]
git-tree-sha1 = "9022bcaa2fc1d484f1326eaa4db8db543ca8c66d"
uuid = "3eaba693-59b7-5ba5-a881-562e759f1c8d"
version = "0.7.4"

[[deps.StatsPlots]]
deps = ["AbstractFFTs", "Clustering", "DataStructures", "Distributions", "Interpolations", "KernelDensity", "LinearAlgebra", "MultivariateStats", "NaNMath", "Observables", "Plots", "RecipesBase", "RecipesPipeline", "Reexport", "StatsBase", "TableOperations", "Tables", "Widgets"]
git-tree-sha1 = "3b1dcbf62e469a67f6733ae493401e53d92ff543"
uuid = "f3b207a7-027a-5e70-b257-86293d7955fd"
version = "0.15.7"

[[deps.StringManipulation]]
deps = ["PrecompileTools"]
git-tree-sha1 = "a6b1675a536c5ad1a60e5a5153e1fee12eb146e3"
uuid = "892a3eda-7b42-436c-8928-eab12a02cf0e"
version = "0.4.0"

[[deps.StyledStrings]]
uuid = "f489334b-da3d-4c2e-b8f0-e476e12c162b"
version = "1.11.0"

[[deps.SuiteSparse]]
deps = ["Libdl", "LinearAlgebra", "Serialization", "SparseArrays"]
uuid = "4607b0f0-06f3-5cda-b6b1-a6196a1729e9"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "7.7.0+0"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.TableOperations]]
deps = ["SentinelArrays", "Tables", "Test"]
git-tree-sha1 = "e383c87cf2a1dc41fa30c093b2a19877c83e1bc1"
uuid = "ab02a1b2-a7df-11e8-156e-fb1833f50b87"
version = "1.2.0"

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

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"
version = "1.11.0"

[[deps.TranscodingStreams]]
git-tree-sha1 = "0c45878dcfdcfa8480052b6ab162cdd138781742"
uuid = "3bb67fe8-82b1-5028-8e26-92a6c54297fa"
version = "0.11.3"

[[deps.Transducers]]
deps = ["Accessors", "ArgCheck", "BangBang", "Baselet", "CompositionsBase", "ConstructionBase", "DefineSingletons", "Distributed", "InitialValues", "Logging", "Markdown", "MicroCollections", "Requires", "SplittablesBase", "Tables"]
git-tree-sha1 = "7deeab4ff96b85c5f72c824cae53a1398da3d1cb"
uuid = "28d57a85-8fef-5791-bfe6-a80928e7c999"
version = "0.4.84"

    [deps.Transducers.extensions]
    TransducersAdaptExt = "Adapt"
    TransducersBlockArraysExt = "BlockArrays"
    TransducersDataFramesExt = "DataFrames"
    TransducersLazyArraysExt = "LazyArrays"
    TransducersOnlineStatsBaseExt = "OnlineStatsBase"
    TransducersReferenceablesExt = "Referenceables"

    [deps.Transducers.weakdeps]
    Adapt = "79e6a3ab-5dfb-504d-930d-738a2a938a0e"
    BlockArrays = "8e7c35d0-a365-5155-bbbb-fb81a777f24e"
    DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
    LazyArrays = "5078a376-72f3-5289-bfd5-ec5146d43c02"
    OnlineStatsBase = "925886fa-5bf2-5e8e-b522-a9147a512338"
    Referenceables = "42d2dcc6-99eb-4e98-b66c-637b7d73030e"

[[deps.Tricks]]
git-tree-sha1 = "7822b97e99a1672bfb1b49b668a6d46d58d8cbcb"
uuid = "410a4b4d-49e4-4fbc-ab6d-cb71b17b3775"
version = "0.1.9"

[[deps.URIs]]
git-tree-sha1 = "67db6cc7b3821e19ebe75791a9dd19c9b1188f2b"
uuid = "5c2747f8-b7ea-4ff2-ba2e-563bfd36b1d4"
version = "1.5.1"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"
version = "1.11.0"

[[deps.UnPack]]
git-tree-sha1 = "387c1f73762231e86e0c9c5443ce3b4a0a9a0c2b"
uuid = "3a884ed6-31ef-47d7-9d2a-63182c4928ed"
version = "1.0.2"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"
version = "1.11.0"

[[deps.UnicodeFun]]
deps = ["REPL"]
git-tree-sha1 = "53915e50200959667e78a92a418594b428dffddf"
uuid = "1cfade01-22cf-5700-b092-accc4b62d6e1"
version = "0.4.1"

[[deps.Unitful]]
deps = ["Dates", "LinearAlgebra", "Random"]
git-tree-sha1 = "01915bfcd62be15329c9a07235447a89d588327c"
uuid = "1986cc42-f94f-5a68-af5c-568840ba703d"
version = "1.21.1"
weakdeps = ["ConstructionBase", "InverseFunctions"]

    [deps.Unitful.extensions]
    ConstructionBaseUnitfulExt = "ConstructionBase"
    InverseFunctionsUnitfulExt = "InverseFunctions"

[[deps.UnitfulLatexify]]
deps = ["LaTeXStrings", "Latexify", "Unitful"]
git-tree-sha1 = "975c354fcd5f7e1ddcc1f1a23e6e091d99e99bc8"
uuid = "45397f5d-5981-4c77-b2b3-fc36d6e9b728"
version = "1.6.4"

[[deps.UnsafeAtomics]]
git-tree-sha1 = "b13c4edda90890e5b04ba24e20a310fbe6f249ff"
uuid = "013be700-e6cd-48c3-b4a1-df204f14c38f"
version = "0.3.0"

    [deps.UnsafeAtomics.extensions]
    UnsafeAtomicsLLVM = ["LLVM"]

    [deps.UnsafeAtomics.weakdeps]
    LLVM = "929cbde3-209d-540e-8aea-75f648917ca0"

[[deps.Unzip]]
git-tree-sha1 = "ca0969166a028236229f63514992fc073799bb78"
uuid = "41fe7b60-77ed-43a1-b4f0-825fd5a5650d"
version = "0.2.0"

[[deps.Vulkan_Loader_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Wayland_jll", "Xorg_libX11_jll", "Xorg_libXrandr_jll", "xkbcommon_jll"]
git-tree-sha1 = "2f0486047a07670caad3a81a075d2e518acc5c59"
uuid = "a44049a8-05dd-5a78-86c9-5fde0876e88c"
version = "1.3.243+0"

[[deps.Wayland_jll]]
deps = ["Artifacts", "EpollShim_jll", "Expat_jll", "JLLWrappers", "Libdl", "Libffi_jll", "Pkg", "XML2_jll"]
git-tree-sha1 = "85c7811eddec9e7f22615371c3cc81a504c508ee"
uuid = "a2964d1f-97da-50d4-b82a-358c7fce9d89"
version = "1.21.0+2"

[[deps.Wayland_protocols_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "5db3e9d307d32baba7067b13fc7b5aa6edd4a19a"
uuid = "2381bf8a-dfd0-557d-9999-79630e7b1b91"
version = "1.36.0+0"

[[deps.WeakRefStrings]]
deps = ["DataAPI", "InlineStrings", "Parsers"]
git-tree-sha1 = "b1be2855ed9ed8eac54e5caff2afcdb442d52c23"
uuid = "ea10d353-3f73-51f8-a26c-33c1cb351aa5"
version = "1.4.2"

[[deps.Widgets]]
deps = ["Colors", "Dates", "Observables", "OrderedCollections"]
git-tree-sha1 = "e9aeb174f95385de31e70bd15fa066a505ea82b9"
uuid = "cc8bc4a8-27d6-5769-a93b-9d913e69aa62"
version = "0.6.7"

[[deps.WoodburyMatrices]]
deps = ["LinearAlgebra", "SparseArrays"]
git-tree-sha1 = "c1a7aa6219628fcd757dede0ca95e245c5cd9511"
uuid = "efce3f68-66dc-5838-9240-27a6d6f5f9b6"
version = "1.0.0"

[[deps.WorkerUtilities]]
git-tree-sha1 = "cd1659ba0d57b71a464a29e64dbc67cfe83d54e7"
uuid = "76eceee3-57b5-4d4a-8e66-0e911cebbf60"
version = "1.6.1"

[[deps.XML2_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libiconv_jll", "Zlib_jll"]
git-tree-sha1 = "a2fccc6559132927d4c5dc183e3e01048c6dcbd6"
uuid = "02c8fc9c-b97f-50b9-bbe4-9be30ff0a78a"
version = "2.13.5+0"

[[deps.XSLT_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Libgcrypt_jll", "Libgpg_error_jll", "Libiconv_jll", "XML2_jll", "Zlib_jll"]
git-tree-sha1 = "7d1671acbe47ac88e981868a078bd6b4e27c5191"
uuid = "aed1982a-8fda-507f-9586-7b0439959a61"
version = "1.1.42+0"

[[deps.XZ_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "ecda72ccaf6a67c190c9adf27034ee569bccbc3a"
uuid = "ffd25f8a-64ca-5728-b0f7-c24cf3aae800"
version = "5.6.3+1"

[[deps.Xorg_libICE_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "326b4fea307b0b39892b3e85fa451692eda8d46c"
uuid = "f67eecfb-183a-506d-b269-f58e52b52d7c"
version = "1.1.1+0"

[[deps.Xorg_libSM_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libICE_jll"]
git-tree-sha1 = "3796722887072218eabafb494a13c963209754ce"
uuid = "c834827a-8449-5923-a945-d239c165b7dd"
version = "1.2.4+0"

[[deps.Xorg_libX11_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxcb_jll", "Xorg_xtrans_jll"]
git-tree-sha1 = "9dafcee1d24c4f024e7edc92603cedba72118283"
uuid = "4f6342f7-b3d2-589e-9d20-edeb45f2b2bc"
version = "1.8.6+1"

[[deps.Xorg_libXau_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "2b0e27d52ec9d8d483e2ca0b72b3cb1a8df5c27a"
uuid = "0c0b7dd1-d40b-584c-a123-a41640f87eec"
version = "1.0.11+1"

[[deps.Xorg_libXcursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXfixes_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "807c226eaf3651e7b2c468f687ac788291f9a89b"
uuid = "935fb764-8cf2-53bf-bb30-45bb1f8bf724"
version = "1.2.3+0"

[[deps.Xorg_libXdmcp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "02054ee01980c90297412e4c809c8694d7323af3"
uuid = "a3789734-cfe1-5b06-b2d0-1dd0d9d62d05"
version = "1.1.4+1"

[[deps.Xorg_libXext_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "d7155fea91a4123ef59f42c4afb5ab3b4ca95058"
uuid = "1082639a-0dae-5f34-9b06-72781eeb8cb3"
version = "1.3.6+1"

[[deps.Xorg_libXfixes_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "6fcc21d5aea1a0b7cce6cab3e62246abd1949b86"
uuid = "d091e8ba-531a-589c-9de9-94069b037ed8"
version = "6.0.0+0"

[[deps.Xorg_libXi_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXfixes_jll"]
git-tree-sha1 = "984b313b049c89739075b8e2a94407076de17449"
uuid = "a51aa0fd-4e3c-5386-b890-e753decda492"
version = "1.8.2+0"

[[deps.Xorg_libXinerama_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll"]
git-tree-sha1 = "a1a7eaf6c3b5b05cb903e35e8372049b107ac729"
uuid = "d1454406-59df-5ea1-beac-c340f2130bc3"
version = "1.1.5+0"

[[deps.Xorg_libXrandr_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libXext_jll", "Xorg_libXrender_jll"]
git-tree-sha1 = "b6f664b7b2f6a39689d822a6300b14df4668f0f4"
uuid = "ec84b674-ba8e-5d96-8ba1-2a689ba10484"
version = "1.5.4+0"

[[deps.Xorg_libXrender_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "a490c6212a0e90d2d55111ac956f7c4fa9c277a6"
uuid = "ea2f1a96-1ddc-540d-b46f-429655e07cfa"
version = "0.9.11+1"

[[deps.Xorg_libpthread_stubs_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "fee57a273563e273f0f53275101cd41a8153517a"
uuid = "14d82f49-176c-5ed1-bb49-ad3f5cbd8c74"
version = "0.1.1+1"

[[deps.Xorg_libxcb_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "XSLT_jll", "Xorg_libXau_jll", "Xorg_libXdmcp_jll", "Xorg_libpthread_stubs_jll"]
git-tree-sha1 = "1a74296303b6524a0472a8cb12d3d87a78eb3612"
uuid = "c7cfdc94-dc32-55de-ac96-5a1b8d977c5b"
version = "1.17.0+1"

[[deps.Xorg_libxkbfile_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libX11_jll"]
git-tree-sha1 = "dbc53e4cf7701c6c7047c51e17d6e64df55dca94"
uuid = "cc61e674-0454-545c-8b26-ed2c68acab7a"
version = "1.1.2+1"

[[deps.Xorg_xcb_util_cursor_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xcb_util_image_jll", "Xorg_xcb_util_jll", "Xorg_xcb_util_renderutil_jll"]
git-tree-sha1 = "04341cb870f29dcd5e39055f895c39d016e18ccd"
uuid = "e920d4aa-a673-5f3a-b3d7-f755a4d47c43"
version = "0.1.4+0"

[[deps.Xorg_xcb_util_image_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "0fab0a40349ba1cba2c1da699243396ff8e94b97"
uuid = "12413925-8142-5f55-bb0e-6d7ca50bb09b"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_libxcb_jll"]
git-tree-sha1 = "e7fd7b2881fa2eaa72717420894d3938177862d1"
uuid = "2def613f-5ad1-5310-b15b-b15d46f528f5"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_keysyms_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "d1151e2c45a544f32441a567d1690e701ec89b00"
uuid = "975044d2-76e6-5fbe-bf08-97ce7c6574c7"
version = "0.4.0+1"

[[deps.Xorg_xcb_util_renderutil_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "dfd7a8f38d4613b6a575253b3174dd991ca6183e"
uuid = "0d47668e-0667-5a69-a72c-f761630bfb7e"
version = "0.3.9+1"

[[deps.Xorg_xcb_util_wm_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Xorg_xcb_util_jll"]
git-tree-sha1 = "e78d10aab01a4a154142c5006ed44fd9e8e31b67"
uuid = "c22f9ab0-d5fe-5066-847c-f4bb1cd4e361"
version = "0.4.1+1"

[[deps.Xorg_xkbcomp_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_libxkbfile_jll"]
git-tree-sha1 = "ab2221d309eda71020cdda67a973aa582aa85d69"
uuid = "35661453-b289-5fab-8a00-3d9160c6a3a4"
version = "1.4.6+1"

[[deps.Xorg_xkeyboard_config_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Xorg_xkbcomp_jll"]
git-tree-sha1 = "691634e5453ad362044e2ad653e79f3ee3bb98c3"
uuid = "33bec58e-1273-512f-9401-5d533626f822"
version = "2.39.0+0"

[[deps.Xorg_xtrans_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "b9ead2d2bdb27330545eb14234a2e300da61232e"
uuid = "c5fb5394-a638-5e4d-96e5-b29de1b5cf10"
version = "1.5.0+1"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+1"

[[deps.Zstd_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7dc5adc3f9bfb9b091b7952f4f6048b7e37acafc"
uuid = "3161d3a3-bdf6-5164-811a-617609db77b4"
version = "1.5.6+2"

[[deps.eudev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "gperf_jll"]
git-tree-sha1 = "431b678a28ebb559d224c0b6b6d01afce87c51ba"
uuid = "35ca27e7-8b34-5b7f-bca9-bdc33f59eb06"
version = "3.2.9+0"

[[deps.fzf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "6e50f145003024df4f5cb96c7fce79466741d601"
uuid = "214eeab7-80f7-51ab-84ad-2988db7cef09"
version = "0.56.3+0"

[[deps.gperf_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "0ba42241cb6809f1a278d0bcb976e0483c3f1f2d"
uuid = "1a1c6b14-54f6-533d-8383-74cd7377aa70"
version = "3.1.1+1"

[[deps.libaom_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "1827acba325fdcdf1d2647fc8d5301dd9ba43a9d"
uuid = "a4ae2306-e953-59d6-aa16-d00cac43593b"
version = "3.9.0+0"

[[deps.libass_jll]]
deps = ["Artifacts", "Bzip2_jll", "FreeType2_jll", "FriBidi_jll", "HarfBuzz_jll", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "e17c115d55c5fbb7e52ebedb427a0dca79d4484e"
uuid = "0ac62f75-1d6f-5e53-bd7c-93b484bb37c0"
version = "0.15.2+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.11.0+0"

[[deps.libdecor_jll]]
deps = ["Artifacts", "Dbus_jll", "JLLWrappers", "Libdl", "Libglvnd_jll", "Pango_jll", "Wayland_jll", "xkbcommon_jll"]
git-tree-sha1 = "9bf7903af251d2050b467f76bdbe57ce541f7f4f"
uuid = "1183f4f0-6f2a-5f1a-908b-139f9cdfea6f"
version = "0.2.2+0"

[[deps.libevdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "141fe65dc3efabb0b1d5ba74e91f6ad26f84cc22"
uuid = "2db6ffa8-e38f-5e21-84af-90c45d0032cc"
version = "1.11.0+0"

[[deps.libfdk_aac_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "8a22cf860a7d27e4f3498a0fe0811a7957badb38"
uuid = "f638f0a6-7fb0-5443-88ba-1cc74229b280"
version = "2.0.3+0"

[[deps.libinput_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "eudev_jll", "libevdev_jll", "mtdev_jll"]
git-tree-sha1 = "ad50e5b90f222cfe78aa3d5183a20a12de1322ce"
uuid = "36db933b-70db-51c0-b978-0f229ee0e533"
version = "1.18.0+0"

[[deps.libpng_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Zlib_jll"]
git-tree-sha1 = "9c42636e3205e555e5785e902387be0061e7efc1"
uuid = "b53b4c65-9356-5827-b1ea-8c7a1a84506f"
version = "1.6.44+1"

[[deps.libvorbis_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Ogg_jll", "Pkg"]
git-tree-sha1 = "490376214c4721cdaca654041f635213c6165cb3"
uuid = "f27f6e37-5d2b-51aa-960f-b287f2bc3b7a"
version = "1.3.7+2"

[[deps.mtdev_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "814e154bdb7be91d78b6802843f76b6ece642f11"
uuid = "009596ad-96f7-51b1-9f1b-5ce2d5e8a71e"
version = "1.1.6+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.59.0+0"

[[deps.oneTBB_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl"]
git-tree-sha1 = "7d0ea0f4895ef2f5cb83645fa689e52cb55cf493"
uuid = "1317d2d5-d96f-522e-a858-c73665f53c3e"
version = "2021.12.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+2"

[[deps.x264_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "4fea590b89e6ec504593146bf8b988b2c00922b2"
uuid = "1270edf5-f2f9-52d2-97e9-ab00b5d0237a"
version = "2021.5.5+0"

[[deps.x265_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "ee567a171cce03570d77ad3a43e90218e38937a9"
uuid = "dfaa095f-4041-5dcd-9319-2fabd8486b76"
version = "3.5.0+0"

[[deps.xkbcommon_jll]]
deps = ["Artifacts", "JLLWrappers", "Libdl", "Pkg", "Wayland_jll", "Wayland_protocols_jll", "Xorg_libxcb_jll", "Xorg_xkeyboard_config_jll"]
git-tree-sha1 = "63406453ed9b33a0df95d570816d5366c92b7809"
uuid = "d8fb68d0-12a3-5cfd-a85a-d49703b185fd"
version = "1.4.1+2"
"""

# ╔═╡ Cell order:
# ╠═df4e377a-895e-483d-8894-75629bb2533f
# ╠═6f6339a7-9b4d-4272-94f7-9234d3d3be41
# ╠═266e632b-30a5-4ae4-981f-8e2ab61e3232
# ╠═d0fe37ee-bbc1-11ef-2f0c-4b6bc41d2c3a
# ╠═cc81afb1-c73d-4e75-b572-21a70608dd9d
# ╠═e709b44b-57b0-482d-bc11-93b91451d790
# ╠═515ef8bd-82e3-4f94-8fbc-0626ed34e5b7
# ╠═cda801a3-6b0c-49f0-afbd-798850b354ca
# ╠═eb52ead3-2e23-4884-96c6-4ccdbf529be2
# ╠═9cebf899-82f9-4a69-9ef1-5c98bb8b20fc
# ╠═b3fe5074-6461-4d1e-bea3-f005533210a0
# ╠═4309c399-3035-4a0a-8f88-2184efd415b9
# ╠═8783e92c-3ff2-49e0-a85f-264a0ae77afc
# ╠═410f5c00-91f0-4f00-924f-6691227bf1cd
# ╠═a079b8eb-f179-41af-9712-de65be3d7a64
# ╠═48d3993b-fcae-4d91-a53a-a936dcaea321
# ╠═2e638e63-6f50-4dd0-921a-0e5d33e31fe1
# ╠═bf9d5f2e-a1ae-4b9a-88ac-a7fcef6945bb
# ╠═a7712a98-dbef-4f62-a83c-546ebfb3a40e
# ╠═6ebfa722-b8f5-4837-871b-addba1c23586
# ╠═cefd6efd-8687-4ab4-93e6-6ae33432d774
# ╠═1b7baa00-78ea-48f6-af34-ccc0795d66b8
# ╠═433bea47-fd81-4e3f-8fd0-d0fe76d6f6ec
# ╠═86ad6318-1a86-469f-870c-591e9e306d74
# ╠═bdc38a15-ff58-4431-8f3b-d1e310044af5
# ╠═8af8b802-49d8-474f-b3a3-1278a69093f9
# ╠═1c428df8-1517-42b3-aa47-215ea2d9bb6f
# ╠═1f0385e8-a071-4473-8136-7cbcf6f924b8
# ╠═23f13774-25c3-42de-800f-2437856ad9fc
# ╠═090c1c8b-35c8-43bc-80a3-b8b75304ed31
# ╠═b19afa83-a5b2-4dc7-b266-76f60e27f2ff
# ╠═4c05c9d4-e724-4956-9866-7037025150ee
# ╠═499b6c91-9032-43fa-a54c-94027676125a
# ╠═48398a06-6e89-43ac-876f-eeb513e69b92
# ╠═2f86b3e0-66e9-4cca-86fd-a83f6d1dbf5b
# ╠═7a3f806a-5d26-40f1-b665-7bf686728fde
# ╠═cf291a6a-18ba-476a-a9c9-ddcd1f75c514
# ╠═287328fd-2389-4e7e-84f9-4cd932db899a
# ╠═5443ad0a-deb3-463f-8975-12fb969df1cc
# ╠═965d8844-8fdd-4e88-a6bd-01281a5fb642
# ╠═f5a3b766-4781-47de-8097-af3bd8bc4ecd
# ╠═ac831749-b0aa-49f5-ad1b-92e45599551d
# ╠═7eb49193-addb-4833-9261-0a0ceab55e7e
# ╠═f4c5fdba-5493-4240-93dd-86345c9adad7
# ╠═90a3a2c0-4d45-40d9-9bdc-a966c628ba90
# ╠═ad16e07e-f7c2-4bce-9f71-0d0334459ef6
# ╠═08794500-ec04-4834-8852-c79680f0d136
# ╠═3ffe9740-d133-410a-a3ff-77c2ba64217f
# ╠═caba4624-8388-43b9-a9bf-b0bb5ed27213
# ╠═900f41a8-59a2-4301-88a5-9f0b4c25d4ac
# ╠═6518ba79-f3cc-4ccb-9935-fe037f256d4e
# ╠═e3278b0e-ff00-4bc9-b06e-d2757a08f997
# ╠═3a665608-f044-41a9-a619-7ed799457277
# ╠═ea80c78a-6ec8-44f9-9876-6449c2399ee3
# ╠═c105b7ce-98bd-4c34-86b4-818309495dc5
# ╠═747f66f6-1c86-450f-824f-7a94bdd7ec6a
# ╠═ed4f53d4-4d73-4f02-abf8-2cb68447525f
# ╠═23b1b815-1048-4ebb-9d12-eba8dfa8e60c
# ╠═58e0b3a9-9f2c-438f-ac4a-db490a3f6685
# ╠═7da174c6-91a4-4508-818f-6d35720df915
# ╠═1da2e2e0-b1bd-4e86-b397-54f293a90c06
# ╠═7c3f884f-7159-4594-aef9-a975767e750d
# ╠═6f07b059-a4b3-4a32-b62e-fbadb15d71a3
# ╠═203a5b66-2c6d-4707-ae13-ed2563a2f305
# ╠═23dfc277-7289-4a69-8ec5-75fbd05b43e2
# ╠═f437681b-464d-449d-befc-75d60a5c4974
# ╠═660be5ac-4791-48c3-b1f4-824f53c80215
# ╠═9087d54f-5682-4778-812a-15bd28431fb7
# ╠═95e17cad-7135-4b7f-93f2-5b7c72023e27
# ╠═19905308-9b52-4f49-a4e8-7212bc08fb0a
# ╠═57a30f20-7dc2-4c02-8621-32f03c5d5f71
# ╠═83788e2c-84f8-4d0c-bdb0-930680eafa4c
# ╠═c76bc69f-eeaf-4861-b735-f2d4a99cd4d3
# ╠═8cab10cf-e7d4-42a9-b9b9-c6243ea80ec4
# ╠═cb016b85-517d-4e28-ab73-60adab8007b5
# ╠═3566a81e-5a17-4c34-87b4-3b7dfbe2fc1d
# ╠═e38b43eb-832d-4ac2-894f-d5d287b6709b
# ╠═db09d078-cbf5-442a-b819-a049f93cee06
# ╠═b9bfee39-4092-4c69-816c-ca830b94c935
# ╠═4c52220e-3b6e-46fe-b7d7-87bfc5facf99
# ╠═bfdf0bd6-42ab-4d7b-b45a-21035fefcb34
# ╠═b972c073-20c7-41e0-bc72-61838652ea79
# ╠═563f0e01-3117-40f0-9d4f-0f4926bdc336
# ╠═2da21951-935d-4ba8-9a3b-c2628a71a31e
# ╠═215eb03b-4c21-4321-bb3b-70bb68aace81
# ╠═878d4d09-9d5d-4d2b-9670-c7ae116ee5e9
# ╠═317be3e6-1e2c-4565-866e-be8d03330c56
# ╠═4b0609e8-cc0b-44e9-94e9-01c34301dfe7
# ╠═4a71d082-f1ca-4582-8722-0d6ec6b67de4
# ╠═94a22149-789c-4a25-b91f-23fc24e702fc
# ╠═6e8cdc3d-aa37-4b66-84b2-6b1db9947d62
# ╠═07c535e5-c9bb-4ecb-bcb4-391f257a91c3
# ╠═b97dac76-fdb8-4da0-8295-0f8a9dc20cf3
# ╠═e9069ed2-3a71-4206-b569-4f73bb97eb19
# ╠═05ad2c30-e21b-40e2-85a9-f85176a5c075
# ╠═04aca69e-71a6-4b66-b4a0-fe8aaabd5fe9
# ╠═6aa1e98d-13df-4732-bc3a-b907bcb8e4bd
# ╠═dff726bd-66dc-49d0-8dfd-bd762cad734b
# ╠═ab5b96a3-97d9-45fe-a8fe-88b5094bca8e
# ╠═e4e29ff3-fc14-4966-8cfb-e8296a4a4bd1
# ╠═14e10278-99fc-4932-aa52-6d02cf6ac7db
# ╠═158f021b-5fa8-49be-a8df-6863a272e4e8
# ╠═55211ee4-a8da-46b2-96f1-83fc170677ee
# ╠═1661096e-713c-435a-b47d-278f454ad76b
# ╠═c3976756-cf99-4092-86d2-994b34894bb7
# ╠═70802a0d-d500-457c-9823-0bf99f23509e
# ╠═354706f0-c7a2-49b6-be7d-5fe9d333aa30
# ╠═b4ffd1ca-dc72-45fe-a6a4-b0f9569cec64
# ╠═1e051daa-f2db-447b-93b3-19c594fc0d66
# ╠═4b2dfb5f-9319-4cd9-9e6c-69efab579681
# ╠═0f85c1b1-ce14-4870-8943-efb72e9ec7d9
# ╠═eefd48fc-ca9a-42fb-a321-53be6bf98381
# ╠═ef313c53-798c-4e84-b6da-4d90ed49db51
# ╠═14a01c6a-9c06-434d-b1b9-87be793c071a
# ╠═613788e4-0430-41da-ad36-55c24de472a2
# ╠═2ecb47e6-58e1-4065-afa4-1c0eea75a6ee
# ╠═7705d15c-ff72-488d-9733-d535f8c0d33d
# ╠═9df11318-64a8-48cc-803a-2c19a0796fe8
# ╠═90df1425-d688-4195-9aea-fc1494000448
# ╠═0ba2bf47-2dab-4ef3-af7e-bd97d1f0ebc4
# ╠═d4bfb41a-7f9d-4ae6-aa1c-a3e1bac8025d
# ╠═47e52680-e6ea-4de9-b621-9294c13c99ee
# ╠═4ca27647-e6ed-42f5-aac4-bf9dd959d9c5
# ╠═e3c90bb0-22fb-4853-ba20-c77dd87d0096
# ╠═f9f038a8-a3dd-4253-af34-0903d4b13216
# ╠═735b0a1f-8568-4ce2-8761-c321366f5a35
# ╠═be0d88a4-5520-43bd-8914-9b566a343acc
# ╠═690417e7-50a6-401c-9ad9-1d42abc384a3
# ╠═e982d0a1-3469-48d5-a286-28fc2f88f714
# ╠═b117c48e-2e1b-4908-8747-58d080a829f0
# ╠═4f59d4d0-8c87-41d6-b313-1cdc368a5120
# ╠═0ad863f8-3479-4da5-92b2-7bdc2f4044d5
# ╠═53bdf7bb-95f7-4413-aec7-1c88375fcf22
# ╠═1c7e1986-b12f-4d4a-9be9-12befefb98bd
# ╠═f8967d16-3b32-47af-93eb-13eee06fc199
# ╠═5390815a-39a1-4aee-bb0c-d93edb60c9ec
# ╠═1d3429b4-30d3-407c-aa96-40b91823870e
# ╠═d9e5abac-a4a9-41f9-96c7-cd53727c4186
# ╠═e9ae8022-0ec7-4206-b006-d070322f10c5
# ╠═7f60da0c-5876-47c0-9cf6-1bce1891d4f6
# ╠═e0d5372d-d876-4e56-a090-bbe7158ae845
# ╠═0db3efc7-01e6-4d0b-bb52-4626faac179c
# ╠═1d383b64-a596-42ca-9eed-6ec35c7d61ea
# ╠═dbbca94b-d3eb-40f2-b303-f40f537dbbd9
# ╠═5cd0a735-6ab1-4467-aed0-591f876658a7
# ╠═78f74061-afc3-44af-9753-3feeab77b889
# ╠═fa1025bf-df2c-4bcc-83d5-8f0d88a56e57
# ╠═0984464e-7c1d-418b-b956-436744c84704
# ╠═8b6c1f57-a7a6-45e1-81f3-a3b06cbd25cc
# ╠═aad3bc5b-c389-409d-acb5-895788e3ce42
# ╠═5c0b5a16-fd1d-4b8c-aa70-d746332b4d28
# ╠═31f685d8-b2b3-49d3-88a0-e45f24e69bd8
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
