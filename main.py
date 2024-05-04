import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
import uvicorn
from sklearn.metrics import jaccard_score
import os

# Crear la aplicación FastAPI
app = FastAPI()


'''
_____________________________________________________________________________________________________________
'''
## Define la dirección relativa
relative_path = 'venv\\code'
# Obtener la ruta absoluta al directorio actual
current_directory = os.path.dirname(os.path.realpath(__file__))
# Combinar la dirección relativa con la ruta actual para obtener la ruta absoluta
template_path = os.path.join(current_directory, relative_path)

# Crear una instancia de Jinja2Templates para renderizar templates HTML
templates = Jinja2Templates(directory=os.path.join(current_directory, "templates"))

@app.get("/", response_class=HTMLResponse, tags=['Página Principal'])

async def main(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})
'''
_____________________________________________________________________________________________________________
'''
# 'PlayTimeGenre' año mas con horas jugadas por Género
@app.get('/consulta1', tags= ['PI1-MLOps'])

async def PlayTimeGenre(genero:str):
    return()


'''
_____________________________________________________________________________________________________________
'''
# 'UserForGenre' usuario con mas horas jugadas por Género
@app.get('/consulta2', tags= ['PI1-MLOps'])

async def UserForGenre(genero:str):
    return()


'''
_____________________________________________________________________________________________________________
'''
# 'UsersRecommend' top 3 de juegos MAS recomendados por año
@app.get('/consulta3', tags= ['PI1-MLOps'])

async def UsersRecommend(año:int):
    return()


'''
_____________________________________________________________________________________________________________
'''
# 'UsersNotRecommend' top 3 de juegos MENOS recomendados por año
@app.get('/consulta4', tags= ['PI1-MLOps'])

async def UsersNotRecommend(año:int):
    return()


'''
_____________________________________________________________________________________________________________
''' 
# 'sentiment_analysis' cantidad de reseñas por año de lanzamiento
@app.get('/consulta5', tags= ['PI1-MLOps'])

async def sentiment_analysis(año: int) -> dict:
    # Cargar el archivo .parquet
    df_reviews = pd.read_parquet('venv/data/users_reviews_etl_comprimido.parquet')
    
    # Convertir la columna 'date' a tipo datetime
    df_reviews['date'] = pd.to_datetime(df_reviews['date'])
    
    # Filtrar los datos por el año consultado
    df_año = df_reviews[df_reviews['date'].dt.year == año]
    
    # Contar los valores únicos en la columna de análisis sentimental 'sentiment_analysis'
    counts = df_año['sentiment_analysis'].value_counts().to_dict()
    
    # Crear un nuevo diccionario para mapear valores numéricos a etiquetas
    mapeo_sentimientos = {0: 'Negative', 1: 'Neutral', 2: 'Positive'}
    
    # Crear un nuevo diccionario con las etiquetas correspondientes
    nuevo_counts = {mapeo_sentimientos[key]: value for key, value in counts.items()}
    
    # Si algún tipo de análisis no está presente en el año dado, se establecera su conteo en 0
    for sentimiento in ['Negative', 'Neutral', 'Positive']:
        if sentimiento not in nuevo_counts:
            nuevo_counts[sentimiento] = 0
    
    return nuevo_counts

'''
_____________________________________________________________________________________________________________
''' 
# sistema de recomendación item-item
# Cargar el DataFrame
df_sistRec = pd.read_parquet('venv/data/steam_games_etl_comprimido.parquet')

class SistemaRecomendacion:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def calcular_similitud(self, juego_referencia):
        # Verificar si el ID del juego existe en el DataFrame
        if juego_referencia not in self.dataframe['id'].values:
            return None  # Retorna None si el ID no existe

        # Obtener el género del juego de referencia y convertirlo en lista
        genero_referencia = self.dataframe.loc[self.dataframe['id'] == juego_referencia, 'genres'].iloc[0].split(', ')
        
        # Calcular la similitud de Jaccard entre el género del juego de referencia y todos los demás juegos
        similitudes = {}
        for index, row in self.dataframe.iterrows():
            if row['id'] != juego_referencia:
                genero_otro_juego = row['genres'].split(', ')
                similitud = jaccard_score(genero_referencia, genero_otro_juego, average='macro')
                similitudes[row['app_name']] = similitud

        # Ordenar los juegos por similitud y devolver los 5 más similares
        juegos_similares = sorted(similitudes.items(), key=lambda x: x[1], reverse=True)[:5]
        return juegos_similares

sistema_recomendacion = SistemaRecomendacion(df_sistRec)

@app.post('/item-item', tags= ['Recomendaciones'])
async def obtener_recomendaciones(id_producto: int):
    recomendaciones = sistema_recomendacion.calcular_similitud(id_producto)
    if recomendaciones is None:
        return {"mensaje": "El ID del juego no existe en el sistema"}
    else:
        return {"recomendaciones": recomendaciones}
'''
_____________________________________________________________________________________________________________
''' 
# Punto de entrada de la aplicación
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

