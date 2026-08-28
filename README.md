# Análisis Espacial y Predictivo: Dinámicas de TransMilenio y Emergencias NUSE (Línea 123) en Bogotá

Este repositorio contiene la solución analítica desarrollada para el **DataJam Edición 3 (2026)**. El proyecto aborda la pregunta central: **¿El perfil de uso de las estaciones de una UPZ se asocia a más llamadas al 123 en esa misma UPZ?** Para resolverlo, se integra el flujo de pasajeros en el sistema TransMilenio con el registro de incidentes y emergencias del Número Único de Seguridad y Emergencias (NUSE - Línea 123) a escala de UPZ en Bogotá.

---

## 1. Descripción del Problema Abordado

El fenómeno urbano analizado vincula la movilidad y el transporte masivo (**TransMilenio**, cuyo flujo se registra a nivel de estación) con el comportamiento de la seguridad y las emergencias urbanas (registradas en el **NUSE**, cuyo desenlace territorial se consolida a nivel de **UPZ**). La propuesta evalúa cómo las dinámicas operativas y de validación de pasajeros en las estaciones impactan o se correlacionan con la concentración de llamadas de emergencia en el territorio circundante, permitiendo generar insumos orientados a la toma de decisiones públicas a escala distrital.

---

## 2. Fuentes de Datos Utilizadas

El análisis integra múltiples fuentes de información pública provenientes del **Portal de Datos Abiertos de Bogotá (IDECA)**:
*   **TransMilenio:** Archivos de validaciones y salidas de pasajeros por estación (`troncal_*.csv`, `salidas_*.csv`, `Dim_estaciones.csv`).
*   **Emergencias NUSE:** Registros de llamadas tramitadas por la línea 123 (C4 Bogotá).
*   **Cartografía oficial:** Delimitación de Unidades de Planeamiento Zonal / Local (`upz.geojson`).
*   *Nota:* Las fuentes primarias y secundarias cumplen con el criterio de utilizar un mínimo de dos fuentes públicas independientes y medibles en el tiempo.

---

## 3. Metodología General

La arquitectura del proyecto se divide en fases técnicas reproducibles:
1.  **ETL e Integración (`integracion_panel.py`):** Cruce espacial y temporal de las fuentes de TransMilenio y NUSE para construir un panel unificado a nivel de UPZ y mes.
2.  **Análisis Estadístico (`analisis_upz.py`):** Generación de rankings, análisis de correlación (Spearman) y diagramas de dispersión para identificar patrones espaciales y sectoriales.
3.  **Modelado Predictivo (`modelo_predictivo.py`):** Implementación de modelos de Machine Learning (Bosques Aleatorios / Random Forest y redes neuronales) para la predicción de llamadas e incidentes orientados al periodo 2026.
4.  **Visualización e Interfaz (`dashboard/app.py`):** Desarrollo de un visor interactivo en **Shiny (Python)** para la exploración de resultados y pronósticos.

---

## 4. Estructura del Repositorio

```text
tm-nuse-bogota/
│
├── dashboard/
│   ├── __init__.py
│   ├── app.py                      # Tablero interactivo Shiny
│   └── exportar_informe.py         # Script para compilación de reportes HTML
│
├── salidas_analisis/               # Resultados estadísticos, rankings y figuras de correlación
├── salidas_modelo/                 # Métricas de rendimiento, importancia de variables y predicciones 2026
├── salidas_entregable/             # Informes analíticos en formato HTML (informe_hurto y violencia)
├── datos_crudos/                   # Metadatos o notas sobre las fuentes originales
│
├── integracion_panel.py            # Script ETL (integra TM + NUSE + datos abiertos)
├── analisis_upz.py                 # Script de análisis exploratorio, rankings y Spearman
├── modelo_predictivo.py            # Script de entrenamiento y evaluación (Bosques y Redes)
├── Dim_estaciones.csv              # Maestro de estaciones de TransMilenio
├── upz.geojson                     # Archivo geoespacial de UPZ
├── requirements.txt                # Dependencias completas para reproducción del análisis
├── requirements-shiny.txt          # Dependencias ligeras exclusivas para el despliegue del Dashboard
├── Dockerfile                      # Configuración de contenedores para despliegue reproducible
└── README.md                       # Documentación principal del proyecto
