# Brenda Estefanía Estrada López
# Tecnologías para Análisis de Datos Masivos


# Librerias e instalación
import pandas as pd
from palmerpenguins import load_penguins


# 1. Vamos a cargar el conjunto de datos. Muestra por pantalla el número de observaciones y sus características. 
# Mira el tipo de datos de cada una de sus columnas.

penguins = load_penguins()
print(penguins)

totales = penguins.count()
print(totales)


# 2. Ya sabemos que este conjunto de datos tiene observaciones NA. Vamos a eliminarlas y a verificar que efectivamente
# No queda ninguno:
penguins = penguins.dropna()
print(penguins)
#Elimino aquellas filas donde se encuentre un NaN y lo guardo dentro de la misma variable para trabajar con ella
print(penguins.isna().any()) 
#para verificar si existe algún nan utilizo la funcion isna() la cual demuestra que no hay datos NANs


# 3. ¿Cuántos individuos hay de cada sexo? Puedes obtener la longitud media del pico según el sexo:
individuos_por_sexo = penguins["sex"].value_counts()
print(individuos_por_sexo)
# El conteo de la variable sex me muestra que existe un total de 168 machos y 165 hembras en mi muestra

# Importo nueva libreria
import numpy as np
media_picos = penguins.groupby(['sex',"species"])["bill_length_mm"].mean()
print(media_picos)


# 4. Vamos a añadir una columna, vamos a realizar una estimación (muy grosera) del área del pico de los pingüinos 
# (bill) tal como si esta fuese un rectángulo. Esta nueva columnas se llama bill_area y debe encontrarse en la última 
# posición. Verifica que es correcto.

# El área de un rectangulo se calcula multiplicando la base x altura, en este caso bill_length x bill_depth
# Creo una nueva columna con el valor solicitado
bill_area = penguins["bill_length_mm"]*penguins["bill_depth_mm"]
# Agrego la columna a mi dataframe
penguins["bill_area"] = bill_area
print(penguins)


# Para verificar que el resultado de mi nueva columna es correcto, lo verifico haciendo el calculo manual del primer dato
# Y verifico que corresponda con mi primer resultado.
comprobacion = penguins["bill_length_mm"][0]*penguins["bill_depth_mm"][0]
# El valor calculado manualmente es correcto ya que corresponde con el primer
# valor de la columna bill_area
print(penguins["bill_area"][0] == comprobacion)


# 5. Hagamos algo un poco más elaborado, vamos a realizar una agrupación en función del sexo y de la especie de cada 
# Observación. Queremos obtener solamente la información referente al sexo Femenino.

# Para sacar la longitud media del pico por sexo primero aplico la funcion groupby y me quedo con los datos de "female"
# Agrupo
por_sexo = penguins.groupby('sex')
# Filtro datos "female"
females = penguins.loc[por_sexo.groups['female'].values]
# Por ultimo vuelvo a agrupar por especie con la
# Funcion group by, la cual me devolverá la longitud media del pico de cada especie de hembras.
mean_bill_fem = females.groupby(["species"]).aggregate({"bill_length_mm":np.mean})
print(mean_bill_fem)


# 6. Como ya sabemos, la variable peso, se encuentra en gramos, la pasaremos a kg. Para ello crearemos una nueva 
# columna llamada body_mass_kg y eliminaremos body_mass_g.

# Primero realizo la conversion de gr a kg dividiendo los gr en 1000
body_mass_kg = penguins["body_mass_g"] / 1000
# Agrego a la nueva columna llamada 'body_mass_kr'
penguins["body_mass_kg "] = body_mass_kg
# Finalmente elimino la golumna 'body_mass_g'
tabla_final = penguins.drop(columns=["body_mass_g"])
print(tabla_final)

