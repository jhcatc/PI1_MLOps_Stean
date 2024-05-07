FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia la carpeta venv desde tu repositorio de GitHub al contenedor
COPY venv /app/venv

# Activa el entorno virtual
SHELL ["/bin/bash", "-c"]
RUN source /app/venv/bin/activate

# Copia los archivos de tu aplicación al contenedor
COPY . .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir nltk textblob decorator fastapi matplotlib-inline numpy pandas scikit-learn scikit-metrics scipy uvicorn wcwidth requests wordcloud typing

# Expone el puerto en el que se ejecutará tu API
EXPOSE 5000

# Comando de inicio para ejecutar tu aplicación
CMD ["python", "app.py"]