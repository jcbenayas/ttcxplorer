# TTC Xplorer

Explora y analiza señales de los **(TTC)** mediante una interfaz web interactiva desarrollada con **Gradio**.  
El backend ejecuta consultas **Polars** generadas a partir de lenguaje natural usando un modelo LLM servido a través de **OpenWebUI**.  
El proyecto está preparado para desplegarse fácilmente en un contenedor **Docker** y dejar el servicio accesible en tu navegador en cuestión de segundos.

---

## ✨ Características

- Chat en lenguaje natural que convierte preguntas a consultas Polars automáticas.  
- Gráficas interactivas con _matplotlib_ para voltajes, bloqueos, MOC, MLIB y más.  
- Persistencia opcional de datos de entrada/resultado usando volúmenes Docker.  
- Sin dependencias locales: todo se ejecuta dentro de un contenedor ligero basado en **Python 3.11**.  
- Preparado para producción con variables de entorno configurables.

---

## 🐳 Despliegue rápido con Docker

1. **Clona el repositorio**  
   ```bash
   git clone https://github.com/tu-usuario/ttcxplorer.git
   cd ttcxplorer
   ```

2. **Copia tu archivo `.env`** (opcional)  
   Crea un fichero `.env` en la raíz con tus variables privadas:
   ```env
   OPENWEBUI_URL=https://tu-servidor-openwebui
   API_KEY=eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...
   MODEL_NAME=ttcxplorer
   ```
   > También puedes pasar las variables con `docker run -e CLAVE=valor …`.

3. **Construye la imagen**  
   ```bash
   docker build -t ttcxplorer .
   ```

4. **Arranca el contenedor**  
   ```bash
   docker run --rm -p 7860:7860 \
     --env-file .env \
     ttcxplorer
   ```
   Abre <http://localhost:7860> en tu navegador y empieza a preguntar.

---

## 📄 Variables de entorno

| Variable          | Ejemplo                         | Descripción                                                         |
|-------------------|---------------------------------|---------------------------------------------------------------------|
| `OPENWEBUI_URL`   | `https://tesmadslapp31`         | URL del endpoint OpenWebUI (`/api/chat/completions`)                |
| `API_KEY`         | `eyJh…`                         | Token **JWT** con permisos de lectura/ejecución sobre el modelo     |
| `MODEL_NAME`      | `ttcxplorer`                    | Nombre del modelo a emplear en la petición                          |
| `PORT`            | `7860`                          | Puerto interno que expone Gradio (opcional, por defecto 7860)       |

---

## 🛠️ Docker Compose (opcional)

```yaml
version: "3.9"
services:
  ttcxplorer:
    build: .
    ports:
      - "7860:7860"
    env_file:
      - .env
    restart: unless-stopped
```
Arranca todo con:
```bash
docker compose up --build -d
```

---

## 🚧 Ejecución local sin Docker

Si ya tienes Python ≥ 3.11:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python TTCExplorer.py
```
> Necesitarás `chromium` o `firefox` si usas la opción de abrir el navegador automáticamente.

---

## 🤝 Contribuir

1. Crea un _fork_ y una rama descriptiva (`feat/nueva-funcionalidad`).
2. Asegúrate de que `make test` (o los notebooks) pasan sin errores.
3. Envía un *pull request* explicando la motivación y los cambios.

---

## 📄 Licencia

Publicado bajo la licencia MIT. Consulta el fichero [`LICENSE`](LICENSE) para más información.