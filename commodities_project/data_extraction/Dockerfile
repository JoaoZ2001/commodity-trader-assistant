# Python image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /data_extraction

# Copia o arquivo requirements.txt para o diretório atual
COPY requirements.txt .

# Instala as dependências listadas no arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code into the container
COPY . .

# Default command to run the script
CMD ["python", "data_extraction.py"]
