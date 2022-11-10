

#get_ipython().system('pip install palmerpenguins')



import pandas as pd
from palmerpenguins import load_penguins


# 1. Vamos a cargar el conjunto de datos. Muestra por pantalla el número de observaciones y sus características. 
# Mira el tipo de datos de cada una de sus columnas.

penguins = load_penguins()
penguins.head()

totales = penguins.count()
print(totales)


# 2. Ya sabemos que este conjunto de datos tiene observaciones NA. Vamos a eliminarlas y a verificar que efectivamente
# no queda ninguno:
penguins = penguins.dropna()
print(penguins)
#Elimino aquellas filas donde se encuentre un NaN y lo guardo dentro de la misma variable para trabajar con ella
print(penguins.isna().any()) 
#para verificar si existe algún nan utilizo la funcion isna() la cual demuestra que no hay datos NANs


# 3. ¿Cuántos individuos hay de cada sexo? Puedes obtener la longitud media del pico según el sexo:
penguins["sex"].value_counts()
#El conteo de la variable sex me muestra que existe un total de 168 machos y 165 hembras en mi muestra

import numpy as np #Importo nueva libreria
media_picos = penguins.groupby(['sex',"species"])["bill_length_mm"].mean()
media_picos


# 4. Vamos a añadir una columna, vamos a realizar una estimación (muy grosera) del área del pico de los pingüinos 
# (bill) tal como si esta fuese un rectángulo. Esta nueva columnas se llama bill_area y debe encontrarse en la última 
# posición. Verifica que es correcto.

bill_area = penguins["bill_length_mm"]*penguins["bill_depth_mm"] #creo una nueva columna con el valor solicitado
penguins["bill_area"] = bill_area #Agrego la columna a mi dataframe
penguins
#El área de un rectangulo se calcula multiplicando la base x altura, en este caso bill_length x bill_depth

#Para verificar que el resultado de mi nueva columna es correcto, lo verifico haciendo el calculo manual del primer dato
#y verifico que corresponda con mi primer resultado.
comprobacion = penguins["bill_length_mm"][0]*penguins["bill_depth_mm"][0]
penguins["bill_area"][0] == comprobacion #El valor calculado manualmente es correcto ya que corresponde con el primer 
#valor de la columna bill_area 


# 5. Hagamos algo un poco más elaborado, vamos a realizar una agrupación en función del sexo y de la especie de cada 
# observación. Queremos obtener solamente la información referente al sexo Femenino.

#para sacar la longitud media del pico por sexo primero aplico la funcion groupby y me quedo con los datos de "female"
por_sexo = penguins.groupby('sex') #Agrupo
females = penguins.loc[por_sexo.groups['female'].values] #filtro datos "female"
mean_bill_fem = females.groupby(["species"]).aggregate({"bill_length_mm":np.mean}) #por ultimo vuelvo a agrupar por especie con la 
#funcion group by, la cual me devolverá la longitud media del pico de cada especie de hembras.
mean_bill_fem


# 6. Como ya sabemos, la variable peso, se encuentra en gramos, la pasaremos a kg. Para ello crearemos una nueva 
# columna llamada body_mass_kg y eliminaremos body_mass_g.

body_mass_kg = penguins["body_mass_g"] / 1000
penguins["body_mass_kg "] = body_mass_kg 
tabla_final = penguins.drop(columns=["body_mass_g"])
tabla_final

