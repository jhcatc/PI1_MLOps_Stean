# Utiliza una imagen base más ligera
FROM python:3.9-slim

# Establece el directorio de trabajo dentro del contenedor
WORKDIR /app

# Copia solo los archivos necesarios de tu aplicación al contenedor
COPY requirements.txt .

# Instala las dependencias desde un archivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copia el resto de los archivos de tu aplicación
COPY . .

# Expone el puerto en el que se ejecutará tu API
EXPOSE 5000

# Comando de inicio para ejecutar tu aplicación
CMD ["python", "app.py"]