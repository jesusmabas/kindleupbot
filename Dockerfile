# 1. Usar una imagen base oficial de Python
FROM python:3.11-slim

# 2. Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# 3. Actualizar el gestor de paquetes e instalar pandoc y git
RUN apt-get update && apt-get install -y \
    pandoc \
    git \
    --no-install-recommends \
    && rm -rf /var/lib/apt/lists/*

# 4. Copiar e instalar las dependencias
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar todo el paquete de la aplicación
# La carpeta 'kindleupbot' de tu repo se copia a '/app/kindleupbot' en el contenedor
COPY kindleupbot/ ./kindleupbot/

# 6. Comando para ejecutar la aplicación COMO UN MÓDULO
# Esto le dice a Python que busque y ejecute main.py dentro del paquete kindleupbot
CMD ["python", "-m", "kindleupbot.main"]