FROM python:3.9-slim

# Establece la variable de entorno para desactivar la advertencia de pip
ENV PIP_NO_WARN_SCRIPT_LOCATION=1

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia la carpeta venv desde tu repositorio de GitHub al contenedor
COPY venv /app/venv

# Establece el PATH para que apunte al binario del entorno virtual
ENV PATH="/app/venv/bin:$PATH"

# Copia los archivos de tu aplicación al contenedor
COPY . .

# Instala las dependencias necesarias
RUN pip install --no-cache-dir --disable-pip-version-check nltk textblob decorator fastapi matplotlib-inline numpy pandas scikit-learn scikit-metrics scipy uvicorn wcwidth requests wordcloud typing

# Expone el puerto en el que se ejecutará tu API
EXPOSE 5000

# Comando de inicio para ejecutar tu aplicación
CMD ["python", "app.py"]