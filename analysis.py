import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math
from scipy import stats as st
from math import factorial
from scipy.stats import binom
from scipy.stats import norm
from scipy.stats import ttest_ind

df_games = pd.read_csv('/content/games.csv')

print(df_games.describe())

df_games.info()

#transformo nombres de las columnas a minuscula
column_low = []

for columna in df_games:
    columna = columna.lower()
    column_low.append(columna)

df_games.columns = column_low

print(df_games.head())

# busco nulos y duplicados

print('cantidad de nulos:')
print(df_games.isna().sum())
print('cantidad de duplicados:', df_games.duplicated().sum())

#Busco los dos nulos en la columna 'name'
name_null = df_games[df_games['name'].isnull()]
print(name_null)

#Busco los dos nulos en la columna 'genre'
genre_null = df_games[df_games['genre'].isnull()]
print(genre_null)

df_games.drop(df_games[df_games['name'].isnull()].index, inplace=True)
#reseteo del index:
df_games.reset_index(drop=True, inplace=True)


#compruebo los datos fueron eliminados

#print(df_games.isna().sum())

#Busco si algun año en particular es el que esta como nan en la columna
print(df_games['year_of_release'].sort_values().unique())

#checkeo como se observa el grafico de años.
df_games['year_of_release'].plot(kind='hist',bins=27)
plt.show()

#Observo cómo se ven los datos que en 'year of release' estan nulos
df_games[df_games['year_of_release'].isnull()]

df_games['year_of_release'] = df_games['year_of_release'].fillna(0).astype('int64')

#compruebo los datos fueron reemplazados y pasados a int

#print(df_games['year_of_release'].info())


#observo como se ven los datos de critic score
print(df_games['critic_score'].sort_values().unique())
df_games['critic_score'].plot(kind='hist',bins=98)
plt.show()

#como queda si utilizo la media para rellenar esos datos faltantes
df_games_critic_mean = df_games['critic_score'].fillna(df_games['critic_score'].mean())
df_games_critic_mean.plot(kind='hist', bins=98)
plt.show()

#como quedarian los datos si utilizo la mediana
df_games_critic = df_games['critic_score'].fillna(df_games['critic_score'].median())
df_games_critic.plot(kind='hist', bins=98)
plt.show()

#pasar los datos eliminados a un nuevo df
#df_games_critic_drop = df_games.drop(df_games[df_games['critic_score'].notna()].index)
#df_games.reset_index(drop=True, inplace=True)
#print(len(df_games_critic_drop) / len(df_games + df_games_critic_drop))
#df_games.drop(df_games[df_games['critic_score'].isna()].index, inplace=True)

df_games['critic_score'] = df_games['critic_score'].fillna(999)

#convierto datos a int
df_games['critic_score'] = df_games['critic_score'].astype('int64')
print(df_games['critic_score'].info())


#observo como se ven los datos de user score
print(df_games['user_score'].sort_values().unique())

#checkeo cuantos datos tiene 'tbd' y si tienen algo más en comun
print(df_games[df_games['user_score']=='tbd'])

#checkeo cuantos datos son nulos y si tienen algo más en comun

print(df_games[df_games['user_score'].isnull()])

#relleno nulos con 99.9 y reemplazo 'tbd' con 88.8
df_games['user_score'] = df_games['user_score'].replace('tbd', 88.8)
df_games['user_score'] = df_games['user_score'].fillna(99.9)
#convierto de object a float
df_games['user_score'] = df_games['user_score'].astype('float')
#multiplico por 10 todos los datos de la fila
df_games['user_score'] = df_games['user_score']*10

#convierto de float a int
df_games['user_score'] = df_games['user_score'].astype('int64')
df_games.info()

#observo los score que tienen datos reales
df_games['user_score'].plot(kind='hist', bins=100, xlim=[0,110], ylim=[0,2500])
plt.show()


#como se ven los datos en rating
df_games['rating'].sort_values().unique()

df_games['rating'] = df_games['rating'].fillna('Unknown')


#Creo columna de ventas totales
df_games['total_sales'] = df_games['eu_sales'] + df_games['jp_sales'] + df_games['na_sales'] + df_games['other_sales']

#compruebo que todo el DF está óptima para trabajar
df_games.info()


#cuento cúantos juegos fueron lanzados cada año
year_pivot = df_games.pivot_table(index='year_of_release', values='name', aggfunc='count')
year_pivot.plot(kind='bar', figsize=(15,5), yticks=range(0, 1600, 100))
print(year_pivot)
plt.show()


#total de ventas por plataforma

ventas_totales = df_games.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
ventas_totales = ventas_totales[ventas_totales['year_of_release'] > 1980]
totales_pivot = ventas_totales.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')
totales_pivot.plot(figsize=(25,7))
plt.show()
#print(totales_pivot.fillna(0))


#Total de ventas por plataforma
ventas_pivot = df_games.pivot_table(index='platform', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)
ventas_pivot.plot(kind='bar', figsize=(15,5))
#print(ventas_pivot)
plt.show()

#Se van a tomar los primeros 10 plataformas que tienen mayor cantidad de ganancias por ventas para ver su distribucion con los años.

top_10 = ventas_pivot.head(10).reset_index()
top_10 = df_games[df_games['platform'].isin(top_10['platform'])].reset_index(drop=True)
top_10 = top_10.groupby(['platform','year_of_release'])['total_sales'].sum().reset_index()
top_10 = top_10[top_10['year_of_release'] > 1980]
top_10_pivot = top_10.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')
top_10_pivot.plot(kind='line', figsize=(25,7))
plt.show()

#para mejor visualizacion
top_10_pivot.plot.bar(stacked=True,figsize=(25,7))
plt.show()


#Se van a observar cómo se ven las paltaformas y sus ventas en los ultimos 10 años.
ventas_totales_2007 = df_games.groupby(['platform', 'year_of_release'])['total_sales'].sum().reset_index()
ventas_totales_2007 = ventas_totales_2007[ventas_totales_2007['year_of_release'] >= 2007]
totales_pivot_2007 = ventas_totales_2007.pivot_table(index='year_of_release', columns='platform', values='total_sales', aggfunc='sum')
totales_pivot_2007.plot.bar(figsize=(25,7))
plt.show()


#Se filta el DF para crear uno nuevo donde se observen las ventas globales desde el 2007 por plataforma.
platform_sales = df_games[df_games['platform'].isin(ventas_totales_2007['platform'])].pivot_table(index='name', columns='platform', values='total_sales', aggfunc='sum')
print(platform_sales.describe())
#print(platform_sales[platform_sales['PS4'] > 0]['PS4'])

platform_sales.plot(kind='box', ylim=[0, 2], figsize=(25,7))
plt.show()


ps4_venta_contra_user_reviews = df_games[df_games['user_score'] <= 100]
ps4_venta_contra_user_reviews = ps4_venta_contra_user_reviews[ps4_venta_contra_user_reviews['platform'] == 'PS4']
print(ps4_venta_contra_user_reviews)
ps4_venta_contra_user_reviews.plot(kind='line', x='total_sales', y='user_score', figsize=(25,5))
ps4_venta_contra_user_reviews.plot(kind='scatter', x='total_sales', y='user_score', figsize=(25,5))

plt.show()

corr_user_venta = ps4_venta_contra_user_reviews['user_score'].corr(ps4_venta_contra_user_reviews['total_sales'])
print(corr_user_venta)


ps4_venta_contra_critic_reviews = df_games[df_games['critic_score'] <= 100]
ps4_venta_contra_critic_reviews = ps4_venta_contra_critic_reviews[ps4_venta_contra_critic_reviews['platform'] == 'PS4']
ps4_venta_contra_critic_reviews.plot(kind='line', x='total_sales', y='critic_score', figsize=(15,5))
ps4_venta_contra_critic_reviews.plot(kind='scatter', x='total_sales', y='critic_score', figsize=(15,5))

plt.show()

corr_critic_venta = ps4_venta_contra_critic_reviews['critic_score'].corr(ps4_venta_contra_critic_reviews['total_sales'])
print(corr_critic_venta)



venta_por_genero = df_games.pivot_table(index='genre', values='total_sales', aggfunc='sum').sort_values(by='total_sales', ascending=False)
venta_por_genero.plot(kind='bar', figsize=(15,5))
plt.show()


# df_jp = df_games.groupby(['platform', 'genre', 'rating'])['jp_sales'].sum().reset_index()
# df_na = df_games.groupby(['platform', 'genre', 'rating'])['na_sales'].sum().reset_index()
# df_eu = df_games.groupby(['platform', 'genre', 'rating'])['eu_sales'].sum().reset_index()
# print('Europa sales:')
# print(df_eu.sort_values(by='eu_sales', ascending=False).head(5))
# print('America del Norte sales:')
# print(df_na.sort_values(by='na_sales', ascending=False).head(5))
# print('Japon sales:')
# print(df_jp.sort_values(by='jp_sales', ascending=False).head(5))

#Se van a observar cómo se ven las plataformas en las diferentes regiones y sus ventas en los ultimos 10 años.
eu_2007 = df_games.groupby(['platform', 'year_of_release'])[['eu_sales']].sum().reset_index()
eu_2007 = eu_2007[eu_2007['year_of_release'] >= 2007]

#checkeo las plataformas más populares
pop_plat_eu = eu_2007.groupby('platform')['eu_sales'].sum().sort_values(ascending=False).head(5)

#filtro el DF por las 5 plataformas más populares de la region en los ultimos 10 años
eu_2007 = eu_2007[eu_2007['platform'].isin(pop_plat_eu.index)]
#print(eu_2007.set_index('platform'))
eu_pivot_2007 = eu_2007.pivot_table(index='year_of_release', columns='platform', values=['eu_sales'])
eu_pivot_2007.plot.bar(figsize=(25,7))
plt.show()


#Se van a observar cómo se ven las plataformas en las diferentes regiones y sus ventas en los ultimos 10 años.
na_2007 = df_games.groupby(['platform', 'year_of_release'])[['na_sales']].sum().reset_index()
na_2007 = na_2007[na_2007['year_of_release'] >= 2007]
pop_plat_na = na_2007.groupby('platform')['na_sales'].sum().sort_values(ascending=False).head(5)
#print(pop_plat_na)
na_2007 = na_2007[na_2007['platform'].isin(pop_plat_na.index)]
#print(na_2007.set_index('platform'))
na_pivot_2007 = na_2007.pivot_table(index='year_of_release', columns='platform', values=['na_sales'])
na_pivot_2007.plot.bar(figsize=(25,7))
plt.show()



#Se van a observar cómo se ven las plataformas en las diferentes regiones y sus ventas en los ultimos 10 años.
jp_2007 = df_games.groupby(['platform', 'year_of_release'])[['jp_sales']].sum().reset_index()
jp_2007 = jp_2007[jp_2007['year_of_release'] >= 2007]
pop_plat_jp = jp_2007.groupby('platform')['jp_sales'].sum().sort_values(ascending=False).head(5)
#print(pop_plat_jp)
jp_2007 = jp_2007[jp_2007['platform'].isin(pop_plat_jp.index)]
#print(jp_2007.set_index('platform'))
jp_pivot_2007 = jp_2007.pivot_table(index='year_of_release', columns='platform', values=['jp_sales'])
jp_pivot_2007.plot.bar(figsize=(25,7))
plt.show()



#Se van a observar cómo se ven las plataformas en las diferentes regiones y sus ventas en los ultimos 10 años.
others_2007 = df_games.groupby(['platform', 'year_of_release'])[['other_sales']].sum().reset_index()
others_2007 = others_2007[others_2007['year_of_release'] >= 2007]
pop_plat_others = others_2007.groupby('platform')['other_sales'].sum().sort_values(ascending=False).head(5)
#print(pop_plat_others)
others_2007 = others_2007[others_2007['platform'].isin(pop_plat_others.index)]
#print(others_2007.set_index('platform'))
others_pivot_2007 = others_2007.pivot_table(index='year_of_release', columns='platform', values=['other_sales'])
others_pivot_2007.plot.bar(figsize=(25,7))
plt.show()


genre_na = df_games.pivot_table(index='genre', values='na_sales', aggfunc='sum').sort_values(by='na_sales', ascending=False).head(5)
genre_eu = df_games.pivot_table(index='genre', values='eu_sales', aggfunc='sum').sort_values(by='eu_sales', ascending=False).head(5)
genre_jp = df_games.pivot_table(index='genre', values='jp_sales', aggfunc='sum').sort_values(by='jp_sales', ascending=False).head(5)
genre_na_pivot = genre_na.plot(kind='bar', figsize=(15,5), color='orange')
genre_eu_pivot = genre_eu.plot(kind='bar', figsize=(15,5), color='purple')
genre_jp_pivot = genre_jp.plot(kind='bar', figsize=(15,5), color='red')
plt.show()



#cuales son los generos
print(df_games['rating'].unique())

#agrupo por region y rating para observar las ventas

rating_jp = df_games.pivot_table(index='rating', values='jp_sales', aggfunc='sum').sort_values(by='jp_sales', ascending=False)
rating_eu = df_games.pivot_table(index='rating',values='eu_sales', aggfunc='sum').sort_values(by='eu_sales', ascending=False)
rating_na = df_games.pivot_table(index='rating',values='na_sales', aggfunc='sum').sort_values(by='na_sales', ascending=False)
rating_na_pivot = rating_na.plot(kind='bar', figsize=(25,7), color='orange', rot=45)
rating_eu_pivot = rating_eu.plot(kind='bar', figsize=(25,7), color='purple', rot=45)
rating_jp_pivot = rating_jp.plot(kind='bar', figsize=(25,7), color='red', rot=45)
plt.show()


df_games['platform'].unique()

#HP0: las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son las mismas
#HP1: las calificaciones promedio de los usuarios para las plataformas Xbox One y PC son diferentes

#Filtro el DF por los 'user_score' reales.
df_games_filtered = df_games[df_games['user_score']<= 88]
alfa = 0.05

xone_user_score = df_games_filtered[df_games_filtered['platform'] == 'XOne']['user_score'].mean()

pc_user_score = df_games_filtered[df_games_filtered['platform'] == 'PC']['user_score'].mean()

resultado = st.ttest_ind(xone_user_score, pc_user_score, equal_var=False)


if resultado.pvalue < alfa:
    print('Rechazamos la hipótesis nula')
else:
    print('No rechazamos la hipótesis nula')


#HP0: las calificaciones promedio de los usuarios para los generos de accion y deportes son iguales.
#HP1: las calificaciones promedio de los usuarios para los generos de accion y deportes son diferentes.

#Filtro el DF por los 'user_score' reales.
df_games_filtered = df_games[df_games['user_score']<= 88]

alfa = 0.05
action_user_score = df_games_filtered[df_games_filtered['genre'] == 'action']['user_score'].mean()
sport_user_score = df_games_filtered[df_games_filtered['genre'] == 'sport']['user_score'].mean()

resultado = st.ttest_ind(action_user_score, sport_user_score, equal_var=False)


if resultado.pvalue < alfa:
    print('Rechazamos la hipótesis nula')
else:
    print('No rechazamos la hipótesis nula')

"""Conclusión general

Luego del análisis realizado sobre los datos con los que hemos trabajado se ha determinado que:

Hay mayor cantidad de plataformas a partir de los 2000.
en los ultimos 10 años las ventas tenido su pico mas alto entre 2007-2011, pero luego se ve un significativo descenso en la venta de juegos.
* Los generos de juego mas populares son de accion, deporte, shooter (disparos/armas), y juego de roles.
* La esperanza de vida de una plataforma ronda un promedio de 6 a 10 años.
* Las plataformas que al 2016 generan ventas en las tres regiones son PS4, PS3, X360, 3DS, Wii.
* Al checkear cantidad de ventas globales observamos que en orden descendiente las mas populares son: PS4, XOne, 3DS, PC y WiiU.

Ahora bien, como conclusion general: Generos en los cuales hacer foco: accion, deporte y de armas. La plataforma PS4 puede generar ganancias en el 2017, sin lugar a duda por la popularidad que tiene en las regiones. Ademas, teniendo en cuenta el promedio de vida, solo tiene 4 años y es la plataforma que mayor ganancias genero en 2016. La segunda plataforma, seria XOne que genero grandes ganancias este 2016. Por ultimo, no es la que mayor ganancias genero este año pero si se mantiene estable hace años, es PC.
"""