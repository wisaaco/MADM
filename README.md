### Brenda Estefanía Estrada López
### Tecnologías para Análisis de Datos Masivos 


#### Librerias e instalación

```python
pip install palmerpenguins
import pandas as pd
from palmerpenguins import load_penguins
import numpy as np
```

1. Vamos a cargar el conjunto de datos. Muestra por pantalla el número de observaciones y sus características. 
Mira el tipo de datos de cada una de sus columnas.

```python
penguins = load_penguins()
penguins.head()
       species     island  bill_length_mm  ...  body_mass_g     sex  year
0       Adelie  Torgersen            39.1  ...       3750.0    male  2007
1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
2       Adelie  Torgersen            40.3  ...       3250.0  female  2007
3       Adelie  Torgersen             NaN  ...          NaN     NaN  2007
4       Adelie  Torgersen            36.7  ...       3450.0  female  2007
..         ...        ...             ...  ...          ...     ...   ...
339  Chinstrap      Dream            55.8  ...       4000.0    male  2009
340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
341  Chinstrap      Dream            49.6  ...       3775.0    male  2009
342  Chinstrap      Dream            50.8  ...       4100.0    male  2009
343  Chinstrap      Dream            50.2  ...       3775.0  female  2009

[344 rows x 8 columns]
```

```python
totales = penguins.count()
totales

species              344
island               344
bill_length_mm       342
bill_depth_mm        342
flipper_length_mm    342
body_mass_g          342
sex                  333
year                 344
dtype: int64
```
2. Ya sabemos que este conjunto de datos tiene observaciones NA. Vamos a eliminarlas y a verificar que efectivamente
no queda ninguno:

```python
penguins = penguins.dropna()
print(penguins)

#Elimino aquellas filas donde se encuentre un NaN y lo guardo dentro de la misma variable para trabajar con ella

       species     island  bill_length_mm  ...  body_mass_g     sex  year
0       Adelie  Torgersen            39.1  ...       3750.0    male  2007
1       Adelie  Torgersen            39.5  ...       3800.0  female  2007
2       Adelie  Torgersen            40.3  ...       3250.0  female  2007
4       Adelie  Torgersen            36.7  ...       3450.0  female  2007
5       Adelie  Torgersen            39.3  ...       3650.0    male  2007
..         ...        ...             ...  ...          ...     ...   ...
339  Chinstrap      Dream            55.8  ...       4000.0    male  2009
340  Chinstrap      Dream            43.5  ...       3400.0  female  2009
341  Chinstrap      Dream            49.6  ...       3775.0    male  2009
342  Chinstrap      Dream            50.8  ...       4100.0    male  2009
343  Chinstrap      Dream            50.2  ...       3775.0  female  2009

[333 rows x 8 columns]
```


```python
print(penguins.isna().any()) 
#para verificar si existe algún nan utilizo la funcion isna() la cual demuestra que no hay datos NANs

species              False
island               False
bill_length_mm       False
bill_depth_mm        False
flipper_length_mm    False
body_mass_g          False
sex                  False
year                 False
dtype: bool
```
3. ¿Cuántos individuos hay de cada sexo? Puedes obtener la longitud media del pico según el sexo:

```python
individuos_por_sexo = penguins["sex"].value_counts()
print(individuos_por_sexo)
#El conteo de la variable sex me muestra que existe un total de 168 machos y 165 hembras en mi muestra

male      168
female    165
Name: sex, dtype: int64
```

```python
media_picos = penguins.groupby(['sex',"species"])["bill_length_mm"].mean()
print(media_picos)

sex     species  
female  Adelie       37.257534
        Chinstrap    46.573529
        Gentoo       45.563793
male    Adelie       40.390411
        Chinstrap    51.094118
        Gentoo       49.473770
Name: bill_length_mm, dtype: float64
```
4. Vamos a añadir una columna, vamos a realizar una estimación (muy grosera) del área del pico de los pingüinos 
(bill) tal como si esta fuese un rectángulo. Esta nueva columnas se llama bill_area y debe encontrarse en la última 
posición. Verifica que es correcto.

```python
bill_area = penguins["bill_length_mm"]*penguins["bill_depth_mm"] #creo una nueva columna con el valor solicitado
penguins["bill_area"] = bill_area #Agrego la columna a mi dataframe
print(penguins)
#El área de un rectangulo se calcula multiplicando la base x altura, en este caso bill_length x bill_depth

       species     island  bill_length_mm  ...     sex  year  bill_area
0       Adelie  Torgersen            39.1  ...    male  2007     731.17
1       Adelie  Torgersen            39.5  ...  female  2007     687.30
2       Adelie  Torgersen            40.3  ...  female  2007     725.40
4       Adelie  Torgersen            36.7  ...  female  2007     708.31
5       Adelie  Torgersen            39.3  ...    male  2007     809.58
..         ...        ...             ...  ...     ...   ...        ...
339  Chinstrap      Dream            55.8  ...    male  2009    1104.84
340  Chinstrap      Dream            43.5  ...  female  2009     787.35
341  Chinstrap      Dream            49.6  ...    male  2009     902.72
342  Chinstrap      Dream            50.8  ...    male  2009     965.20
343  Chinstrap      Dream            50.2  ...  female  2009     938.74

[333 rows x 9 columns]
```

```python
#Para verificar que el resultado de mi nueva columna es correcto, lo verifico haciendo el calculo manual del primer dato
#y verifico que corresponda con mi primer resultado.
comprobacion = penguins["bill_length_mm"][0]*penguins["bill_depth_mm"][0]
print(penguins["bill_area"][0] == comprobacion) #El valor calculado manualmente es correcto ya que corresponde con el primer
#valor de la columna bill_area 

True
```
5. Hagamos algo un poco más elaborado, vamos a realizar una agrupación en función del sexo y de la especie de cada 
observación. Queremos obtener solamente la información referente al sexo Femenino.

```python
#para sacar la longitud media del pico por sexo primero aplico la funcion groupby y me quedo con los datos de "female"
por_sexo = penguins.groupby('sex') #Agrupo
females = penguins.loc[por_sexo.groups['female'].values] #filtro datos "female"
mean_bill_fem = females.groupby(["species"]).aggregate({"bill_length_mm":np.mean}) #por ultimo vuelvo a agrupar por especie con la 
#funcion group by, la cual me devolverá la longitud media del pico de cada especie de hembras.
print(mean_bill_fem)

           bill_length_mm
species                  
Adelie          37.257534
Chinstrap       46.573529
Gentoo          45.563793
```
6. Como ya sabemos, la variable peso, se encuentra en gramos, la pasaremos a kg. Para ello crearemos una nueva 
columna llamada body_mass_kg y eliminaremos body_mass_g.

```python
#primero realizo la conversion de gr a kg dividiendo los gr en 1000
body_mass_kg = penguins["body_mass_g"] / 1000
penguins["body_mass_kg "] = body_mass_kg #agrego a la nueva columna llamada 'body_mass_kr'
tabla_final = penguins.drop(columns=["body_mass_g"]) #finalmente elimino la golumna 'body_mass_g'
print(tabla_final)

       species     island  bill_length_mm  ...  year  bill_area body_mass_kg 
0       Adelie  Torgersen            39.1  ...  2007     731.17         3.750
1       Adelie  Torgersen            39.5  ...  2007     687.30         3.800
2       Adelie  Torgersen            40.3  ...  2007     725.40         3.250
4       Adelie  Torgersen            36.7  ...  2007     708.31         3.450
5       Adelie  Torgersen            39.3  ...  2007     809.58         3.650
..         ...        ...             ...  ...   ...        ...           ...
339  Chinstrap      Dream            55.8  ...  2009    1104.84         4.000
340  Chinstrap      Dream            43.5  ...  2009     787.35         3.400
341  Chinstrap      Dream            49.6  ...  2009     902.72         3.775
342  Chinstrap      Dream            50.8  ...  2009     965.20         4.100
343  Chinstrap      Dream            50.2  ...  2009     938.74         3.775

[333 rows x 9 columns]

Process finished with exit code 0
```


