import Pkg
Pkg.add("HTTP")
Pkg.add("CSV")
Pkg.add("DataFrames")
import HTTP
import CSV
import DataFrames

function carga_datos()
	url = "https://raw.githubusercontent.com/AprendizajeAutomaticoUJI/DataSets/refs/heads/master/Howell1.csv"
	data = CSV.File(HTTP.get(url).body) |> DataFrame
	adultos = data[data.age .> 17, :]
	hombres = select(data[data.male .== 1, :], Not(:male))
	hombres_adultos = hombres[hombres.age .> 17, :]
	mujeres = select(data[data.male .== 0, :], Not(:male))
	mujeres_adultas = mujeres[mujeres.age .> 17, :]
	adultos, hombres, hombres_adultos, mujeres, mujeres_adultas
end
