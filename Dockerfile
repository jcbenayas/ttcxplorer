# Usa una imagen oficial de Python
FROM python:3.11.7-slim

# Establece el directorio de trabajo
WORKDIR /app

# Copia los archivos del proyecto
COPY . /app

# Actualiza pip y wheel
RUN pip install --upgrade pip wheel

# Instala las dependencias
RUN pip install -r requirements.txt

# Expone el puerto que usará Gradio
EXPOSE 7860

# Comando para lanzar la app
CMD ["python", "TTCExplorer.py"]