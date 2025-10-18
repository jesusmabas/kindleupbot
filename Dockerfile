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

# 4. Copiar e instalar las dependencias PRIMERO para aprovechar el cache de Docker
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. Copiar TODO el código fuente de la aplicación a la raíz del WORKDIR
COPY . .

# 6. Comando para ejecutar la aplicación
CMD ["python", "main.py"]