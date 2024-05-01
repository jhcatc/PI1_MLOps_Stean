import pandas as pd
import numpy as np
from fastapi import FastAPI
from fastapi import Request
from fastapi.templating import Jinja2Templates

app = FastAPI()

# Página Principal
templates = Jinja2Templates(directory='code')
@app.get('/', tags=['Página principal, Proyecto Integrador 1, MLOps'])

async def read_root(request: Request):
    return templates.TemplateResponse('index.html', {'request': request})    


# 'PlayTimeGenre' año con horas jugadas por Género
@app.get('/consulta1', tags= ['Año con mas horas jugadas por Genero'])

async def PlayTimeGenre(genero:str):
    return()


# 'UserForGenre' usuario con mas horas jugadas por Género
@app.get('/consulta2', tags= ['Usuario con mas horas jugadas por Genero'])

async def UserForGenre(genero:str):
    return()


# 'UsersRecommend' top 3 de juegos MAS recomendados por año
@app.get('/consulta3', tags= ['Top 3 de Juegos MAS recomendados por año'])

async def UsersRecommend(año:int):
    return()


# 'UsersNotRecommend' top 3 de juegos MENOS recomendados por año
@app.get('/consulta4', tags= ['Top 3 de Juegos MENOS recomendados por año'])

async def UsersNotRecommend(año:int):
    return()


# 'sentiment_analysis' cantidad de reseñas por año de lanzamiento
@app.get('/consulta5', tags= ['Cantidad de reseñas por año'])

async def sentiment_analysis(año:int):
    return()



