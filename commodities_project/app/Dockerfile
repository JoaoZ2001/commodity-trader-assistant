# Python image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copia o arquivo requirements.txt para o diretório atual
COPY requirements.txt .

# Instala as dependências listadas no arquivo requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app code into the container
COPY . .

# Exposes the default Streamlit port
EXPOSE 8501

# Command to run the script and streamlit
CMD ["streamlit", "run", "commodities_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
