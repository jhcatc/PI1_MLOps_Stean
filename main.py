import pandas as pd
import numpy as np
from typing import List, Dict, Any
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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
@app.get('/consulta1', tags= ['PI1-MLOps-Consultas'])
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
@app.get('/consulta2', tags= ['PI1-MLOps-Consultas'])
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
@app.get('/consulta3', tags= ['PI1-MLOps-Consultas'])
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
@app.get('/consulta4', tags= ['PI1-MLOps-Consultas'])
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
@app.get('/consulta5', tags= ['PI1-MLOps-Consultas'])
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
# sistema de recomendación item-item
def recomendacion_juego(id_game: int, df_games_path: str = 'venv/data/steam_games_etl_comprimido.parquet', df_reviews_path: str = 'venv/data/users_reviews_etl_comprimido.parquet') -> dict:
    # Cargar DataFrames
    df_games = pd.read_parquet(df_games_path)
    df_reviews = pd.read_parquet(df_reviews_path)

    # Obtener el género del juego ingresado
    genres = df_games.loc[df_games['id_game'] == id_game, 'genres'].iloc[0]

    # Filtrar juegos por géneros relacionados
    related_games = df_games[df_games['genres'].apply(lambda x: any(genre in x for genre in genres))]

    # Obtener los IDs de los juegos relacionados
    related_game_ids = related_games['id_game'].tolist()

    # Filtrar reviews por los juegos relacionados y que han sido recomendados
    recommended_reviews = df_reviews[(df_reviews['item_id'].isin(related_game_ids)) & (df_reviews['recommend'] == True)]

    # Obtener IDs únicos de los juegos recomendados
    recommended_game_ids = recommended_reviews['item_id'].unique()

    # Filtrar los juegos relacionados por los juegos recomendados
    recommended_games = related_games[related_games['id_game'].isin(recommended_game_ids)]

    # Crear un CountVectorizer para los géneros
    vectorizer = CountVectorizer(tokenizer=lambda x: x.split(','))

    # Obtener la matriz de géneros
    genre_matrix = vectorizer.fit_transform(related_games['genres'])

    # Calcular la similitud del coseno entre los juegos
    similarity_matrix = cosine_similarity(genre_matrix, genre_matrix)

    # Obtener el índice del juego ingresado
    index = related_games.index[related_games['id_game'] == id_game].tolist()[0]

    # Obtener las similitudes del juego ingresado con otros juegos
    sim_scores = list(enumerate(similarity_matrix[index]))

    # Ordenar los juegos por similitud
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Obtener los 5 juegos más similares excluyendo el juego ingresado
    top_similar_games = sim_scores[1:6]

    # Obtener los nombres de los juegos recomendados
    recommended_game_names = recommended_games.loc[recommended_games['id_game'].isin([related_games.iloc[i[0]]['id_game'] for i in top_similar_games]), 'app_name'].tolist()

    return {"Juegos recomendados": recommended_game_names}

# Endpoint para la recomendación de juegos
@app.post('/item-item/', tags= ['PI1-MLOps-Recomendaciones'])
def obtener_recomendacion(id_game: int):
    recomendaciones = recomendacion_juego(id_game)
    return recomendaciones

'''
_____________________________________________________________________________________________________________
''' 
# sistema de recomendación user-item
def recomendacion_usuario(user_id: str) -> Dict:
    # Cargar los DataFrames
    df_games = pd.read_parquet('venv/data/steam_games_etl_comprimido.parquet')
    df_reviews = pd.read_parquet('venv/data/users_reviews_etl_comprimido.parquet')

    # Filtrar las revisiones del usuario y encontrar los juegos recomendados
    user_reviews = df_reviews[df_reviews['user_id'] == user_id]
    recommended_games = user_reviews[user_reviews['recommend'] == True]['item_id']

    recommended_games_names = []

    # Verificar si hay juegos recomendados por el usuario
    if len(recommended_games) == 0:
        return {"Juegos recomendados": []}

    # Recorrer los juegos recomendados por el usuario
    for game_id in recommended_games:
        # Verificar si el ID del juego recomendado está en df_games
        if game_id not in df_games['id_game'].values:
            continue
        
        # Encontrar el juego en df_games
        game = df_games[df_games['id_game'] == game_id].iloc[0]
        
        # Calcular la similitud del coseno entre los géneros del juego recomendado y todos los juegos en df_games
        count_vectorizer = CountVectorizer()
        genre_matrix = count_vectorizer.fit_transform([game['genres']] + list(df_games['genres']))
        cosine_similarities = cosine_similarity(genre_matrix[0:1], genre_matrix[1:]).flatten()

        # Obtener los índices de los juegos más similares
        similar_games_indices = cosine_similarities.argsort()[:-6:-1][1:]

        # Obtener los nombres de los juegos más similares
        similar_games_names = df_games.iloc[similar_games_indices]['app_name'].tolist()
        recommended_games_names.extend(similar_games_names)

    return {"Juegos recomendados": recommended_games_names}  

@app.post('/user-item/', tags= ['PI1-MLOps-Recomendaciones'])
async def optener_recomendacion_ususario(user_id: str):
    recomendaciones = recomendacion_usuario(user_id)
    return recomendaciones


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)