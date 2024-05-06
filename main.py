import pandas as pd
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import TfidfVectorizer
import uvicorn
import os

# Crear la aplicación FastAPI
app = FastAPI()


'''
_____________________________________________________________________________________________________________
'''
## Define la dirección relativa
relative_path = 'venv/code'
# Obtener la ruta absoluta al directorio actual
current_directory = os.path.dirname(os.path.realpath(__file__))
# Combinar la dirección relativa con la ruta actual para obtener la ruta absoluta
template_path = os.path.join(current_directory, relative_path)

# Crear una instancia de Jinja2Templates para renderizar templates HTML
templates = Jinja2Templates(directory=os.path.join(current_directory, "templates"))

@app.get("/", response_class=HTMLResponse, tags=['Homepage'])
async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
'''
_____________________________________________________________________________________________________________
'''
# Cargar los DataFrames desde los archivos parquet
df_games =pd.read_parquet('venv/data/steam_games_etl_comprimido.parquet')
df_items = pd.read_parquet('venv/data/users_items_etl_comprimido.parquet')

# 'PlayTimeGenre' año mas con horas jugadas por Género
@app.get('/consulta1', tags= ['PI1-MLOps-queries'])
async def PlayTimeGenre(genre:str) -> Dict[str, int]:
    
    # Filtrar por género
    genre_games = df_games[df_games['genres'].str.contains(genre, case=False)]
    
    # Obtener los IDs de los videojuegos del género especificado
    genre_game_ids = genre_games['id_game'].unique()
    
    # Filtrar los items por los IDs de los videojuegos del género
    genre_items = df_items[df_items['item_id'].isin(genre_game_ids)]
    
    # Sumar el tiempo de juego en horas para cada videojuego
    genre_items.loc[:, 'playtime_hours'] = genre_items['playtime_forever'] / 60
    
    # Unir los DataFrames para obtener los años de lanzamiento de los videojuegos del género
    merged_df = pd.merge(genre_items, df_games[['id_game', 'year_release']], left_on='item_id', right_on='id_game')
    
    # Agrupar por año de lanzamiento y sumar las horas jugadas
    year_playtime = merged_df.groupby('year_release')['playtime_hours'].sum()
    
    # Encontrar el año con más horas jugadas
    max_playtime_year = year_playtime.idxmax()
    
    # Construir el resultado final
    result = {f"Año de lanzamiento con más horas jugadas para Género {genre}": max_playtime_year}
    
    return result

'''
_____________________________________________________________________________________________________________
'''
# Cargar los DataFrames desde los archivos parquet
df_games =pd.read_parquet('venv/data/steam_games_etl_comprimido.parquet')
df_items = pd.read_parquet('venv/data/users_items_etl_comprimido.parquet')

# 'UserForGenre' usuario con mas horas jugadas por Género
@app.get('/consulta2', tags= ['PI1-MLOps-queries'])
async def UserForGenre(genre:str) :
    
    # Filtrar por género
    genre_games = df_games[df_games['genres'].str.contains(genre, case=False)]
    
    # Obtener los IDs de los videojuegos del género especificado
    genre_game_ids = genre_games['id_game'].unique()
    
    # Filtrar los items por los IDs de los videojuegos del género
    genre_items = df_items[df_items['item_id'].isin(genre_game_ids)]
    
    # Sumar el tiempo de juego en horas para cada videojuego y usuario
    genre_items['playtime_hours'] = genre_items['playtime_forever'] / 60
    
    # Agrupar por usuario y sumar las horas jugadas para cada uno
    user_playtime = genre_items.groupby('user_id').agg({'playtime_hours': 'sum'})
    
    # Encontrar al usuario con más horas jugadas
    user_with_most_playtime = user_playtime.idxmax()[0]
    
    # Obtener los juegos donde el usuario con más horas jugadas jugó
    games_played_by_user = genre_items[genre_items['user_id'] == user_with_most_playtime]['item_name'].unique()
    
    # Volver al DataFrame df_games para cruzar con las horas jugadas por año
    merged_df = pd.merge(df_games, genre_items, left_on='app_name', right_on='item_name')
    
    # Agrupar por año y sumar las horas jugadas para el usuario y género especificados
    user_genre_year_playtime = merged_df.groupby(['user_id', 'year_release']).agg({'playtime_hours': 'sum'})
    
    # Obtener las horas jugadas por año para el usuario con más horas jugadas
    if user_with_most_playtime in user_genre_year_playtime.index:
        user_most_playtime_year = user_genre_year_playtime.loc[user_with_most_playtime].reset_index()
    else:
        user_most_playtime_year = pd.DataFrame(columns=['year_release', 'playtime_hours'])
    
    # Construir el resultado final
    result = {
    f"Usuario con más horas jugadas para Género {genre}": str(user_with_most_playtime),
    "Horas jugadas": [
        {"Año": int(row['year_release']), "Horas": float(row['playtime_hours'])} for index, row in user_most_playtime_year.iterrows()
    ]
}
    
    return result

'''
_____________________________________________________________________________________________________________
'''
# Cargar los DataFrames desde los archivos parquet
df_reviews = pd.read_parquet('venv/data/users_reviews_etl_comprimido.parquet')
df_items = pd.read_parquet('venv/data/users_items_etl_comprimido.parquet')

# 'UsersRecommend' top 3 de juegos MAS recomendados por año
@app.get('/consulta3', tags= ['PI1-MLOps-queries'])
async def UsersRecommend(year:int)-> List[Dict[str, Any]]:
    # Filtrar por año y recommend=True
    filtered_reviews = df_reviews[(df_reviews['date'] == int(year)) & (df_reviews['recommend'] == True)]
    
    # Obtener los tres primeros item_id con el mayor número de sentiment_analysis igual a 1 o 2
    top_items = filtered_reviews.groupby('item_id').sum().sort_values(by='sentiment_analysis', ascending=False).head(3)
    
    # Obtener los nombres de los juegos correspondientes a los item_id encontrados
    recommendations = []
    for idx, row in top_items.iterrows():
        # Verificar si el item_id existe en df_items
        if idx in df_items['item_id'].values:
            item_name = df_items.loc[df_items['item_id'] == idx, 'item_name'].iloc[0]
            recommendations.append({"Puesto " + str(len(recommendations) + 1): item_name})
            '''
            En caso que item_id que viene de df_reviews, no existe en item_id de df_items, 
            la API devolvera item_id como valor del diccionario esto motivado a que por tamaño del df_items 
            se hizo necesario hacer un sample a los datos y se redujo considerablemente
            '''
        else:            
            recommendations.append({"Puesto " + str(len(recommendations) + 1): idx})
    
    return recommendations

'''
_____________________________________________________________________________________________________________
'''
# Cargar los DataFrames desde los archivos parquet
df_reviews = pd.read_parquet('venv/data/users_reviews_etl_comprimido.parquet')
df_items = pd.read_parquet('venv/data/users_items_etl_comprimido.parquet')

# 'UsersNotRecommend' top 3 de juegos MENOS recomendados por año
@app.get('/consulta4', tags= ['PI1-MLOps-queries'])
async def UsersNotRecommend(year:int) -> List[Dict[str, Any]]:
    # Filtrar por año y recommend=False
    filtered_reviews = df_reviews[(df_reviews['date'] == int(year)) & (df_reviews['recommend'] == False)]
    
    # Obtener los tres primeros item_id con el mayor número de sentiment_analysis igual a 0
    top_items = filtered_reviews.groupby('item_id').sum().sort_values(by='sentiment_analysis', ascending=False).head(3)
    
    # Obtener los nombres de los juegos correspondientes a los item_id encontrados
    recommendations = []
    for idx, row in top_items.iterrows():
        # Verificar si el item_id existe en df_items
        if idx in df_items['item_id'].values:
            item_name = df_items.loc[df_items['item_id'] == idx, 'item_name'].iloc[0]
            recommendations.append({"Puesto " + str(len(recommendations) + 1): item_name})
        else:
            recommendations.append({"Puesto " + str(len(recommendations) + 1): idx})
    
    return recommendations

'''
_____________________________________________________________________________________________________________
''' 
# 'sentiment_analysis' cantidad de reseñas por año de lanzamiento
@app.get('/consulta5', tags= ['PI1-MLOps-queries'])
async def sentiment_analysis(year:int) -> Dict[str, int]:
    # Cargar el DataFrame desde el archivo parquet
    df = pd.read_parquet('venv/data/users_reviews_etl_comprimido.parquet')

    # Filtrar las filas del DataFrame para el año proporcionado
    df_year = df[df['date'] == int(year)]

    # Contar los valores en la columna 'sentiment_analysis'
    count_negative = (df_year['sentiment_analysis'] == 0).sum()
    count_neutral = (df_year['sentiment_analysis'] == 1).sum()
    count_positive = (df_year['sentiment_analysis'] == 2).sum()

    # Crear un diccionario con los resultados
    result_mapped = {
        'Negative': count_negative,
        'Neutral': count_neutral,
        'Positive': count_positive
    }

    return result_mapped

'''
_____________________________________________________________________________________________________________
_____________________________________________________________________________________________________________
''' 
# sistema de recomendación user-item
# Carga de datos


'''
_____________________________________________________________________________________________________________
''' 
