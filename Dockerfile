# 1. Usar una imagen base oficial de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Actualizar el gestor de paquetes e instalar pandoc y git
# Es una buena práctica hacerlo en un solo RUN para reducir las capas de la imagen
RUN apt-get update && apt-get install -y \
    pandoc \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar el fichero de dependencias de Python
COPY requirements.txt .

# 5. Instalar las dependencias de Python
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar todo el código de tu aplicación al contenedor
COPY . .

# 7. Comando para ejecutar la aplicación cuando se inicie el contenedor
# Render proporcionará la variable de entorno $PORT, que usamos aquí.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
