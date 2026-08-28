import streamlit as st
import pandas as pd
import geopandas as gpd
import folium
import branca.colormap as cm
from streamlit_folium import st_folium
import py7zr
import os
import plotly.express as px
import osmnx as ox

# Configuración de la página del dashboard
st.set_page_config(page_title="Llamadas Bogotá", layout="wide")
st.title("Análisis Espacial y Temporal de Llamadas de Emergencia - Bogotá")

# Carga de datos
@st.cache_data
def load_data():
    # 🌟 Obtenemos la ruta absoluta de la carpeta donde está este script (app.py)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Rutas relativas seguras basadas en la ubicación del app.py
    ruta_zip = os.path.join(current_dir, 'ept_upz.7z')
    ruta_extraccion = os.path.join(current_dir, 'upz_shapefile')
    
    # 2. Extraer UPZs
    if not os.path.exists(ruta_extraccion):
        with py7zr.SevenZipFile(ruta_zip, mode='r') as z:
            z.extractall(path=ruta_extraccion)
    
    archivos_shp = [os.path.join(r, f) for r, d, files in os.walk(ruta_extraccion) for f in files if f.endswith(".shp")]
    gdf_upz = gpd.read_file(archivos_shp[0]).to_crs("EPSG:4326")
    
    # 3. Cargar CSVs usando la ruta segura
    ruta_llamadas = os.path.join(current_dir, 'Llamadas_UPZ_Estaciones.csv')
    ruta_estaciones = os.path.join(current_dir, 'Dim_estaciones.csv')
    
    llamadas_df = pd.read_csv(ruta_llamadas)
    estaciones_df = pd.read_csv(ruta_estaciones)

    
    # IGUALAR NOMBRE DE COLUMNAS: Renombramos upz_id a CODIGO_UPZ para cruces perfectos
    llamadas_df['upz_id'] = llamadas_df['upz_id'].astype(str).str.replace('UPZ', '', regex=False).str.strip()
    llamadas_df = llamadas_df.rename(columns={'upz_id': 'CODIGO_UPZ'})
    
    # Asegurar que los IDs sean texto
    gdf_upz['CODIGO_UPZ'] = gdf_upz['CODIGO_UPZ'].astype(str)
    llamadas_df['CODIGO_UPZ'] = llamadas_df['CODIGO_UPZ'].astype(str)
    
    # 3. Descargar el límite oficial de Bogotá
    try:
        bogota_bounds = ox.geocode_to_gdf("Bogotá, Colombia")
    except:
        bogota_bounds = None
        
    return gdf_upz, llamadas_df, estaciones_df, bogota_bounds

gdf_upz, llamadas_df, estaciones_df, bogota_bounds = load_data()

# ==========================================
# PROCESAMIENTO Y FILTROS
# ==========================================
idx_inicio = llamadas_df.columns.get_loc('ABRIR DOMICILIO')
causas = list(llamadas_df.columns[idx_inicio:])
if 'Total_Llamadas' not in llamadas_df.columns:
    llamadas_df['Total_Llamadas'] = llamadas_df[causas].sum(axis=1)

opciones_filtro = ['Total_Llamadas'] + causas

# Contenedor de filtros
col_f1, col_f2 = st.columns(2)
with col_f1:
    anios_disponibles = sorted(llamadas_df['anio'].unique())
    idx_2023 = anios_disponibles.index(2023) if 2023 in anios_disponibles else 0
    anio_seleccionado = st.selectbox("Selecciona el Año:", anios_disponibles, index=idx_2023)

with col_f2:
    filtro_seleccionado = st.selectbox("Selecciona el tipo de evento:", opciones_filtro)

# Filtrar base por año seleccionado
df_filtrado = llamadas_df[llamadas_df['anio'] == anio_seleccionado]

# ==========================================
# SERIES DE TIEMPO MENSUALES (SEPARADAS EN 4 GRÁFICOS: 5 CON TOTAL MAYOR Y 5 CON TOTAL MENOR)
# ==========================================
st.markdown("---")

# Calcular Totales para Rankings
totales_upz = df_filtrado.groupby('CODIGO_UPZ')[filtro_seleccionado].sum().reset_index()
totales_upz = totales_upz.sort_values(by=filtro_seleccionado, ascending=False)
top5_upz = totales_upz.head(5)['CODIGO_UPZ'].tolist()
bottom5_upz = totales_upz.tail(5)['CODIGO_UPZ'].tolist()

totales_est = df_filtrado.groupby('Estacion')[filtro_seleccionado].sum().reset_index()
totales_est = totales_est.sort_values(by=filtro_seleccionado, ascending=False)
top5_est = totales_est.head(5)['Estacion'].tolist()
bottom5_est = totales_est.tail(5)['Estacion'].tolist()

# Fila 1: TOP 5
col_t1, col_t2 = st.columns(2)
with col_t1:
    st.subheader(f"Top 5: UPZ con mayor total de llamadas")
    mensual_top_upz = df_filtrado[df_filtrado['CODIGO_UPZ'].isin(top5_upz)]
    serie_top_upz = mensual_top_upz.groupby(['mes', 'CODIGO_UPZ'])[filtro_seleccionado].sum().reset_index()
    fig_t1 = px.line(serie_top_upz, x='mes', y=filtro_seleccionado, color='CODIGO_UPZ', markers=True)
    fig_t1.update_xaxes(tickmode='linear', dtick=1)
    st.plotly_chart(fig_t1, use_container_width=True)

with col_t2:
    st.subheader(f"Top 5: Estaciones con mayor total de llamadas")
    mensual_top_est = df_filtrado[df_filtrado['Estacion'].isin(top5_est)]
    serie_top_est = mensual_top_est.groupby(['mes', 'Estacion'])[filtro_seleccionado].sum().reset_index()
    fig_t2 = px.line(serie_top_est, x='mes', y=filtro_seleccionado, color='Estacion', markers=True)
    fig_t2.update_xaxes(tickmode='linear', dtick=1)
    st.plotly_chart(fig_t2, use_container_width=True)

# Fila 2: BOTTOM 5
col_b1, col_b2 = st.columns(2)
with col_b1:
    st.subheader(f"Top 5: UPZ con menor total de llamadas")
    mensual_bot_upz = df_filtrado[df_filtrado['CODIGO_UPZ'].isin(bottom5_upz)]
    serie_bot_upz = mensual_bot_upz.groupby(['mes', 'CODIGO_UPZ'])[filtro_seleccionado].sum().reset_index()
    fig_b1 = px.line(serie_bot_upz, x='mes', y=filtro_seleccionado, color='CODIGO_UPZ', markers=True)
    fig_b1.update_xaxes(tickmode='linear', dtick=1)
    st.plotly_chart(fig_b1, use_container_width=True)

with col_b2:
    st.subheader(f"Top 5: Estaciones con menor total de llamadas")
    mensual_bot_est = df_filtrado[df_filtrado['Estacion'].isin(bottom5_est)]
    serie_bot_est = mensual_bot_est.groupby(['mes', 'Estacion'])[filtro_seleccionado].sum().reset_index()
    fig_b2 = px.line(serie_bot_est, x='mes', y=filtro_seleccionado, color='Estacion', markers=True)
    fig_b2.update_xaxes(tickmode='linear', dtick=1)
    st.plotly_chart(fig_b2, use_container_width=True)

# ==========================================
# MAPA DUAL: COROPLETA (UPZ) + PUNTOS CON GRADIENTE (ESTACIONES)
# ==========================================
st.markdown("---")
st.subheader("Mapa de Distribución Espacial")

# 1. Métricas UPZ
upz_stats = df_filtrado.groupby('CODIGO_UPZ')[filtro_seleccionado].sum().reset_index()
estaciones_por_upz = llamadas_df.groupby('CODIGO_UPZ')['Estacion'].nunique().reset_index(name='Num_Estaciones')
upz_stats = upz_stats.merge(estaciones_por_upz, on='CODIGO_UPZ', how='left')

gdf_upz_merged = gdf_upz.merge(upz_stats, on='CODIGO_UPZ', how='left')
gdf_upz_merged[filtro_seleccionado] = gdf_upz_merged[filtro_seleccionado].fillna(0)
gdf_upz_merged['Num_Estaciones'] = gdf_upz_merged['Num_Estaciones'].fillna(0)

# Inicializar Mapa
mapa = folium.Map(location=[4.64, -74.1], zoom_start=11, tiles="OpenStreetMap")

if bogota_bounds is not None:
    folium.GeoJson(
        bogota_bounds,
        name="Límite Bogotá",
        style_function=lambda x: {'fillColor': '#cccccc', 'color': '#666666', 'weight': 1, 'fillOpacity': 0.15}
    ).add_to(mapa)

# CAPA 1: UPZ (Tonos Morados para no interferir con las estaciones)
coropleta_upz= folium.Choropleth(
    geo_data=gdf_upz,
    name="Llamadas por UPZ",
    data=upz_stats,
    columns=["CODIGO_UPZ", filtro_seleccionado],
    key_on="feature.properties.CODIGO_UPZ",
    fill_color="Purples", 
    fill_opacity=0.6,
    line_opacity=0.4,
    legend_name=f"Intensidad UPZ: {filtro_seleccionado}",
    nan_fill_color='white',
    bins = 3    ,
)
coropleta_upz.color_scale.width = 300 
coropleta_upz.add_to(mapa)

# Tooltip UPZ
folium.GeoJson(
    gdf_upz_merged,
    name="Información UPZ",
    style_function=lambda x: {'fillColor': 'transparent', 'color': 'black', 'weight': 1},
    tooltip=folium.features.GeoJsonTooltip(
        fields=['CODIGO_UPZ', filtro_seleccionado, 'Num_Estaciones'],
        aliases=['UPZ:', f'Llamadas ({filtro_seleccionado}):', 'Estaciones en la UPZ:'],
        labels=True
    )
).add_to(mapa)

# CAPA 2: Estaciones con Color Dinámico
estaciones_coords = pd.merge(totales_est, estaciones_df, left_on='Estacion', right_on='nom_est', how='inner')

# Configurar el gradiente de colores para las estaciones (Amarillo -> Naranja -> Rojo)
min_val = estaciones_coords[filtro_seleccionado].min()
max_val = estaciones_coords[filtro_seleccionado].max()
if min_val == max_val: max_val += 1  # Evitar error de división si todos los valores son cero

colormap_est = cm.LinearColormap(
    colors=['#fee08b', '#d53e4f', '#9e0142'],
    vmin=min_val,
    vmax=max_val,
    caption=f'Intensidad Estaciones: {filtro_seleccionado}'
).to_step(n=3)  # 3 pasos de color
colormap_est.width = 300
mapa.add_child(colormap_est)

# Añadir los puntos al mapa
for idx, row in estaciones_coords.iterrows():
    valor = row[filtro_seleccionado]
    folium.CircleMarker(
        location=[row['latitud'], row['longitud']],
        radius=7,  
        color='black', 
        weight=1,
        fill=True,
        fill_color=colormap_est(valor),
        fill_opacity=0.9,
        # ¡Aquí agregamos la cantidad de llamadas al cursor!
        tooltip=f"Estación: {row['Estacion']} | Llamadas: {int(valor)}",
        popup=f"<b>Estación:</b> {row['Estacion']}<br><b>Llamadas:</b> {int(valor)}"
    ).add_to(mapa)

# Renderizar Mapa
folium.LayerControl().add_to(mapa)
st_folium(mapa, width=1100, height=600)
