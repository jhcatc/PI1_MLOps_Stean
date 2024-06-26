FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Instala virtualenv
RUN pip install virtualenv

# Crea un nuevo entorno virtual
RUN virtualenv venv

# Activa el entorno virtual
ENV PATH="/app/venv/bin:$PATH"

# Copia los archivos de tu aplicación al contenedor
COPY . .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir nltk textblob decorator fastapi matplotlib-inline numpy pandas scikit-learn scikit-metrics scipy uvicorn wcwidth requests wordcloud typing pyarrow fastparquet

# Expone el puerto en el que se ejecutará tu API
EXPOSE 8000

# Comando de inicio para ejecutar tu aplicación
CMD ["python", "main.py"]