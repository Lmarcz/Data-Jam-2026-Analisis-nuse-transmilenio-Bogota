FROM python:3.12-slim

WORKDIR /app
COPY requirements-shiny.txt .
RUN pip install --no-cache-dir -r requirements-shiny.txt

COPY dashboard/ dashboard/
COPY salidas_analisis/ salidas_analisis/
COPY salidas_modelo/ salidas_modelo/
COPY *.geojson ./

EXPOSE 8000
CMD ["python", "-m", "shiny", "run", "dashboard/app.py", "--host", "0.0.0.0", "--port", "8000"]
