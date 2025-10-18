# 1. Usar una imagen base oficial de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Añadir el directorio de trabajo al PYTHONPATH para que encuentre los módulos
ENV PYTHONPATH "${PYTHONPATH}:/app"

# 3. Actualizar el gestor de paquetes e instalar pandoc y git
RUN apt-get update && apt-get install -y \
    pandoc \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar el fichero de dependencias e instalar
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 6. Copiar todo el código de tu aplicación al contenedor
COPY . .

# 7. Comando para ejecutar la aplicación
CMD ["python", "main.py"]