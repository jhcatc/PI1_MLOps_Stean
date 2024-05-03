import pandas as pd
import numpy as np
from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, HTMLResponse
import os


app = FastAPI()

'''
_____________________________________________________________________________________________________________
'''
# Obteniendo la ruta absoluta del directorio actual
current_directory = os.path.dirname(os.path.realpath(__file__))

# Monta la carpeta "static" como archivos estáticos en la ruta "/static"
app.mount("/static", StaticFiles(directory=os.path.join(current_directory, "venv", "code", "app", "static")), name="static")

# Crea una instancia de Jinja2Templates para renderizar templates HTML
templates = Jinja2Templates(directory=os.path.join(current_directory, "venv", "code", "app"))

# Ruta para servir el archivo index.html en la ruta principal "/"
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
    df_reviews = pd.read_parquet('..\\venv\\data\\users_reviews_etl_comprimido.parquet')
    
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



