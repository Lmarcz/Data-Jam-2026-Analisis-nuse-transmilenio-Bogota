"""
Dashboard Shiny: hallazgos descriptivos + pronóstico UPZ.

Desde la carpeta DataJAM:
  python -m shiny run dashboard/app.py --port 8000
  python dashboard/exportar_informe.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from shiny import App, reactive, render, ui
from shinywidgets import output_widget, render_plotly

BASE = Path(__file__).resolve().parent.parent
AN = BASE / "salidas_analisis"
MO = BASE / "salidas_modelo"
INFORME = BASE / "salidas_entregable"

AZUL, ROJO, GRIS = "#1f4e79", "#a03d3d", "#6b7280"
FAM_COLS = {
    "hurto": "upz_nuse_hurto",
    "violencia": "upz_nuse_violencia",
    "orden_publico": "upz_nuse_orden_publico",
}
FAM_TITULO = {
    "hurto": "Hurto",
    "violencia": "Violencia",
    "orden_publico": "Orden público",
}
SP_Y = FAM_COLS
ETIQ_X = {
    "n_estaciones": "N° estaciones",
    "pct_finde_pond": "% fin de semana",
    "pct_nocturno_pond": "% nocturno",
    "flujo_total_upz": "Flujo TM",
    "log_flujo_upz": "Flujo TM (log)",
    "gini_pond": "Concentración horaria",
    "pct_pico_am_pond": "% pico mañana",
    "pct_pico_pm_pond": "% pico tarde",
    "dist_cai_mediana_m": "Distancia al CAI",
    "n_cai_500m_media": "CAI a 500 m",
    "indice_seguridad_nocturna_2019": "Índice nocturno 2019",
    "y_lag1": "123 del mes previo",
    "y_rm3": "123 (prom. 3 meses)",
    "n_estaciones_lag1": "N° estaciones",
    "log_flujo_upz_lag1": "Flujo TM",
    "pct_nocturno_pond_lag1": "% nocturno",
    "pct_finde_pond_lag1": "% fin de semana",
    "pct_pico_am_pond_lag1": "% pico mañana",
    "pct_pico_pm_pond_lag1": "% pico tarde",
    "gini_pond_lag1": "Concentración horaria",
    "dist_cai_mediana_m_lag1": "Distancia al CAI",
    "mes_sin": "Mes (seno)",
    "mes_cos": "Mes (coseno)",
}
ACENTOS = {
    "El Rincon": "El Rincón",
    "Los Alcazares": "Los Alcázares",
    "Boyaca Real": "Boyacá Real",
    "San Cristobal": "San Cristóbal",
    "Antonio Narino": "Antonio Nariño",
    "Ciudad Bolivar": "Ciudad Bolívar",
    "Usaquen": "Usaquén",
    "Engativa": "Engativá",
    "Fontibon": "Fontibón",
    "Los Martires": "Los Mártires",
}
CFG_PLOTLY = {"displayModeBar": False, "responsive": True}


def nom_propio(texto) -> str:
    t = str(texto).strip().title()
    for malo, bueno in {
        "De": "de", "Del": "del", "La": "la", "Las": "las",
        "Los": "los", "El": "el", "Y": "y",
    }.items():
        t = t.replace(f" {malo} ", f" {bueno} ")
    t = t[:1].upper() + t[1:] if t else t
    return ACENTOS.get(t, t)


def _cargar() -> dict:
    nombres = pd.read_csv(AN / "nombres_upz.csv")
    nombres["upz_id"] = nombres["upz_id"].astype(str).str.strip()
    nombres["upz_nombre"] = nombres["upz_nombre"].map(nom_propio)

    upz = pd.read_csv(AN / "panel_upz_mes_agregado.csv")
    upz["upz_id"] = upz["upz_id"].astype(str).str.strip()
    upz = upz.merge(nombres, on="upz_id", how="left")
    upz["etiqueta"] = upz["upz_nombre"].fillna(upz["upz_id"])
    upz["localidad"] = upz["localidad"].map(nom_propio)

    loc = pd.read_csv(AN / "ranking_localidad.csv")
    loc["localidad"] = loc["localidad"].map(nom_propio)
    sp = pd.read_csv(AN / "spearman.csv")
    pred = pd.read_csv(MO / "predicciones_2026.csv") if (MO / "predicciones_2026.csv").exists() else pd.DataFrame()
    met = pd.read_csv(MO / "metricas.csv") if (MO / "metricas.csv").exists() else pd.DataFrame()
    met_loc = pd.read_csv(MO / "metricas_localidad.csv") if (MO / "metricas_localidad.csv").exists() else pd.DataFrame()
    imp = pd.read_csv(MO / "importancia.csv") if (MO / "importancia.csv").exists() else pd.DataFrame()
    if not pred.empty:
        pred["etiqueta"] = pred["etiqueta"].map(nom_propio)
        pred["localidad"] = pred["localidad"].map(nom_propio)

    geo = None
    for p in BASE.glob("*.geojson"):
        if "UPZ" in p.name.upper() or "upz" in p.name.lower():
            try:
                geo = json.loads(p.read_text(encoding="utf-8"))
                break
            except Exception:
                geo = None
    return {"upz": upz, "loc": loc, "sp": sp, "pred": pred, "met": met, "met_loc": met_loc, "imp": imp, "geo": geo}


DATOS = _cargar()

ETIQ_MOD = {
    "persistencia": "Persistencia",
    "persistencia (mes previo)": "Persistencia",
    "ridge": "Ridge",
    "bosque": "Bosque",
    "mlp": "MLP",
    "red_residual": "Red residual",
}


def _fmt_met(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "modelo" in out.columns:
        out["modelo"] = out["modelo"].map(lambda x: ETIQ_MOD.get(str(x), str(x)))
    for c in ("mae", "rmse"):
        if c in out.columns:
            out[c] = out[c].round(1)
    if "r2" in out.columns:
        out["r2"] = out["r2"].round(3)
    if "skill_vs_persistencia" in out.columns:
        out["skill_%"] = (100 * out["skill_vs_persistencia"]).round(1)
        out = out.drop(columns=["skill_vs_persistencia"])
    if "familia_modelo" in out.columns:
        out["familia_modelo"] = out["familia_modelo"].map({
            "tradicional": "Tradicional",
            "red_neuronal": "Red neuronal",
            "linea_base": "Línea base",
        }).fillna(out["familia_modelo"])
    rename = {
        "n_test": "n", "mae": "MAE", "rmse": "RMSE", "r2": "R²",
        "skill_%": "Skill vs mes previo (%)", "modelo": "Modelo",
        "localidad": "Localidad", "familia_modelo": "Tipo",
    }
    out = out.rename(columns={k: v for k, v in rename.items() if k in out.columns})
    keep = [c for c in ("Modelo", "Tipo", "Localidad", "n", "MAE", "RMSE", "R²", "Skill vs mes previo (%)") if c in out.columns]
    return out[keep]


def html_tabla(df: pd.DataFrame) -> object:
    if df.empty:
        return ui.p("Sin métricas. Corre python modelo_predictivo.py")
    return ui.HTML(df.to_html(index=False, classes="table table-sm tabla-ajuste", border=0, justify="left"))


def fig_limpia(fig: go.Figure, titulo: str, alto: int = 440) -> go.Figure:
    fig.update_layout(
        title=dict(text=titulo, x=0, font=dict(size=15, color="#1a1a1a")),
        font=dict(family="Segoe UI, sans-serif", size=12, color="#222"),
        margin=dict(l=56, r=24, t=52, b=48),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=alto,
        coloraxis_colorbar=dict(title="", len=0.7),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        modebar=dict(remove=["lasso2d", "select2d"]),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#eef0f3", zeroline=False)
    fig.update_yaxes(showgrid=True, gridcolor="#eef0f3", zeroline=False)
    fig._config = CFG_PLOTLY  # type: ignore[attr-defined]
    return fig


def color_fam(fam: str) -> str:
    return ROJO if fam == "violencia" else AZUL


def panel_filtrado(fam: str, anio: str) -> pd.DataFrame:
    d = DATOS["upz"].copy()
    if anio != "todos":
        d = d.loc[d["anio"] == int(anio)]
    y = FAM_COLS[fam]
    return (
        d.groupby(["upz_id", "etiqueta", "localidad"], as_index=False)
        .agg(
            y=(y, "mean"),
            flujo=("flujo_total_upz", "mean"),
            n_est=("n_estaciones", "median"),
            pct_noc=("pct_nocturno_pond", "mean"),
            pct_finde=("pct_finde_pond", "mean"),
        )
        .sort_values("y", ascending=False)
    )


def fig_ranking(fam: str, anio: str) -> go.Figure:
    r = panel_filtrado(fam, anio).head(12).iloc[::-1]
    fig = px.bar(
        r, x="y", y="etiqueta", orientation="h",
        color_discrete_sequence=[color_fam(fam)],
        hover_data={"localidad": True, "n_est": ":.0f", "y": ":.0f"},
    )
    fig.update_layout(xaxis_title="Llamadas al 123 / mes", yaxis_title="")
    return fig_limpia(fig, f"{FAM_TITULO[fam]} reportado, por UPZ", 520)


def fig_mapa(fam: str, anio: str) -> go.Figure:
    r = panel_filtrado(fam, anio)
    geo = DATOS["geo"]
    if not geo or r.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin polígonos de UPZ", showarrow=False)
        return fig_limpia(fig, "Mapa", 480)
    fig = px.choropleth(
        r, geojson=geo, locations="upz_id", featureidkey="properties.UPLCODIGO",
        color="y", hover_name="etiqueta",
        color_continuous_scale="Reds" if fam == "violencia" else "Blues",
    )
    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(coloraxis_colorbar_title="")
    return fig_limpia(fig, f"{FAM_TITULO[fam]} en el entorno", 520)


def fig_tasa() -> go.Figure:
    loc = DATOS["loc"].nsmallest(10, "rank_tasa_hurto").sort_values("tasa_hurto_media")
    fig = px.bar(loc, x="tasa_hurto_media", y="localidad", orientation="h", color_discrete_sequence=[AZUL])
    fig.update_layout(xaxis_title="Hurto NUSE / 100 mil hab. / mes", yaxis_title="")
    return fig_limpia(fig, "Tasa de hurto, por localidad", 420)


def fig_scatter(fam: str, anio: str) -> go.Figure:
    r = panel_filtrado(fam, anio).assign(pct_noche=lambda d: d["pct_noc"] * 100)
    fig = px.scatter(
        r, x="pct_noche", y="y", size="flujo", hover_name="etiqueta",
        color_discrete_sequence=[color_fam(fam)],
    )
    fig.update_layout(xaxis_title="% del flujo 19:00–05:59", yaxis_title="Llamadas / mes")
    titulo = "¿Más noche, más hurto?" if fam == "hurto" else "Noche y llamados"
    return fig_limpia(fig, titulo, 460)


def fig_rho(fam: str) -> go.Figure:
    sp = DATOS["sp"]
    s = sp.loc[(sp["muestra"] == "media_por_upz") & (sp["y"] == SP_Y[fam])].copy()
    s["lab"] = s["x"].map(ETIQ_X).fillna(s["x"])
    s = s.dropna(subset=["rho"]).sort_values("rho")
    fig = go.Figure(go.Bar(
        x=s["rho"], y=s["lab"], orientation="h",
        marker_color=np.where(s["rho"] >= 0, color_fam(fam), GRIS),
    ))
    fig.add_vline(x=0, line_color="black", line_width=1)
    fig.update_layout(xaxis_title="Spearman ρ  ·  58 UPZ", yaxis_title="")
    return fig_limpia(fig, f"Qué se mueve con {FAM_TITULO[fam].lower()}", 460)


def fig_pred(fam: str, cual: str = "red") -> go.Figure:
    pred = DATOS["pred"]
    if pred.empty or fam == "orden_publico":
        fig = go.Figure()
        fig.add_annotation(text="Sin predicciones para esta familia", showarrow=False)
        return fig_limpia(fig, "Observado vs predicho", 420)
    d = pred.loc[pred["familia"] == fam].copy()
    col = _col_pred(d, cual)
    if col is None:
        fig = go.Figure()
        fig.add_annotation(text="Sin columna de predicción", showarrow=False)
        return fig_limpia(fig, "Observado vs predicho", 420)
    etiqueta = "Red neuronal" if cual == "red" else "Bosque (tradicional)"
    fig = px.scatter(
        d, x="observado", y=col, hover_name="etiqueta",
        hover_data={"periodo": True}, color_discrete_sequence=[color_fam(fam)], opacity=0.7,
    )
    lo = float(min(d["observado"].min(), d[col].min()))
    hi = float(max(d["observado"].max(), d[col].max()))
    fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", line=dict(color=GRIS, dash="dash"), name="ideal"))
    fig.update_layout(xaxis_title="Observado 2026", yaxis_title="Predicho", showlegend=False)
    return fig_limpia(fig, f"Prueba 2026 · {etiqueta}", 420)


def _col_pred(d: pd.DataFrame, cual: str) -> str | None:
    if cual == "red":
        for c in ("pred_red_elegida", "pred_red", "pred_mlp"):
            if c in d.columns:
                return c
    for c in ("pred_tradicional", "pred_bosque", "pred_ganador"):
        if c in d.columns:
            return c
    return None


def fig_imp(fam: str) -> go.Figure:
    imp = DATOS["imp"]
    if imp.empty or fam == "orden_publico":
        fig = go.Figure()
        fig.add_annotation(text="Sin importancia para esta familia", showarrow=False)
        return fig_limpia(fig, "Importancia", 420)
    d = imp.loc[imp["familia"] == fam].copy()
    d["lab"] = d["variable"].map(ETIQ_X).fillna(d["variable"])
    d = d.sort_values("importancia_mae")
    fig = px.bar(d, x="importancia_mae", y="lab", orientation="h", color_discrete_sequence=[color_fam(fam)])
    fig.update_layout(xaxis_title="Caída de MAE al permutar", yaxis_title="")
    return fig_limpia(fig, "El mes previo manda", 420)


def fig_serie(fam: str, upz_id: str) -> go.Figure:
    pred = DATOS["pred"]
    if pred.empty or not upz_id or fam == "orden_publico":
        fig = go.Figure()
        fig.add_annotation(text="Elige hurto o violencia y una UPZ", showarrow=False)
        return fig_limpia(fig, "Serie 2026", 380)
    d = pred.loc[(pred["familia"] == fam) & (pred["upz_id"] == upz_id)].sort_values("mes")
    if d.empty:
        fig = go.Figure()
        fig.add_annotation(text="Sin serie para esa UPZ", showarrow=False)
        return fig_limpia(fig, "Serie 2026", 380)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=d["periodo"], y=d["observado"], mode="lines+markers", name="Observado", line=dict(color=AZUL)))
    col_r = _col_pred(d, "red")
    col_t = _col_pred(d, "bosque")
    if col_r:
        fig.add_trace(go.Scatter(x=d["periodo"], y=d[col_r], mode="lines+markers", name="Red", line=dict(color=ROJO, dash="dot")))
    if col_t:
        fig.add_trace(go.Scatter(x=d["periodo"], y=d[col_t], mode="lines+markers", name="Bosque", line=dict(color="#2a7f62", dash="dash")))
    fig.add_trace(go.Scatter(x=d["periodo"], y=d["pred_persistencia"], mode="lines", name="Mes previo", line=dict(color=GRIS, dash="dash")))
    fig.update_layout(yaxis_title="Llamadas / mes", xaxis_title="")
    return fig_limpia(fig, str(d["etiqueta"].iloc[0]), 380)


def texto_brecha(fam: str) -> str:
    if fam == "hurto":
        return (
            "La brecha: el hurto al 123 se concentra en UPZ céntricas (La Sabana), "
            "no en las más nocturnas. Más estaciones TM van con más llamados: centralidad, no culpa del andén."
        )
    if fam == "violencia":
        return (
            "La brecha: la violencia dibuja otro mapa (El Rincón, Bosa Central). "
            "Ahí sí se parece al fin de semana y a la noche del TransMilenio."
        )
    return "Orden público es demanda de convivencia, no de hurto. No mezclar las tres familias en un solo índice."


def construir_informe_html(fam: str = "hurto", anio: str = "todos") -> Path:
    INFORME.mkdir(exist_ok=True)
    r = panel_filtrado(fam, anio)
    top = r.iloc[0] if not r.empty else None
    piezas = [
        fig_ranking(fam, anio),
        fig_mapa(fam, anio),
        fig_tasa(),
        fig_scatter(fam, anio),
        fig_rho(fam),
        fig_pred(fam, "red"),
        fig_imp(fam),
    ]
    bloques = []
    js = True
    for fig in piezas:
        bloques.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if js else False, config=CFG_PLOTLY))
        js = False
    met = DATOS["met"]
    locm = DATOS.get("met_loc", pd.DataFrame())
    tab_g = _fmt_met(met.loc[met["familia"] == fam]) if not met.empty and "familia" in met.columns else pd.DataFrame()
    tab_l = locm.loc[locm["familia"] == fam] if not locm.empty and "familia" in locm.columns else pd.DataFrame()
    html_g = tab_g.to_html(index=False, classes="tabla-ajuste", border=0) if not tab_g.empty else ""
    html_l = ""
    if not tab_l.empty:
        orden = [c for c in ("persistencia", "bosque", "red_residual") if c in set(tab_l["modelo"])]
        mae = tab_l.pivot_table(index="localidad", columns="modelo", values="mae").reindex(columns=orden)
        mae.columns = [f"MAE {ETIQ_MOD.get(c, c).lower()}" for c in mae.columns]
        n = tab_l.groupby("localidad")["n_test"].first().rename("n")
        wide = pd.concat([n, mae.round(1)], axis=1).reset_index().rename(columns={"localidad": "Localidad"})
        html_l = wide.to_html(index=False, classes="tabla-ajuste", border=0)
    bloques.append(f"<h2>Ajuste global (prueba 2026)</h2>{html_g}<h2>Ajuste por localidad</h2>{html_l}")
    kpi = (
        f"<p><b>UPZ al frente:</b> {top['etiqueta']} · "
        f"<b>Llamadas/mes:</b> {top['y']:.0f} · <b>Estaciones:</b> {top['n_est']:.0f}</p>"
        if top is not None else ""
    )
    html = f"""<!DOCTYPE html>
<html lang="es"><head>
<meta charset="utf-8"/>
<title>Informe TM × NUSE</title>
<style>
body {{ font-family: Segoe UI, sans-serif; max-width: 980px; margin: 24px auto; color: #222; padding: 0 16px; }}
h1 {{ font-size: 1.6rem; }} h2 {{ font-size: 1.2rem; margin-top: 2rem; }}
.aviso {{ background: #eef3f8; padding: 12px 16px; border-radius: 8px; }}
</style></head><body>
<h1>TransMilenio y el 123</h1>
<p class="aviso">NUSE son llamadas al 123, no delitos juzgados ni hechos en la estación.
Unidad: UPZ. Familia: {FAM_TITULO[fam]}. Periodo: {anio}.</p>
<p>{texto_brecha(fam)}</p>
{kpi}
{"".join(bloques)}
<p style="color:#666;font-size:0.9rem">Fuentes: validaciones/salidas TransMilenio (datos abiertos) y NUSE línea 123
(Datos Abiertos Bogotá), más población SDS, DAI/IR y CAI del mismo portal.</p>
</body></html>"""
    destino = INFORME / f"informe_{fam}.html"
    destino.write_text(html, encoding="utf-8")
    return destino


CSS = """
.bslib-page-sidebar, .bslib-sidebar-layout { font-family: 'Segoe UI', sans-serif; }
.bslib-sidebar-layout { --bslib-sidebar-width: 280px; }
.bslib-sidebar-layout .main { overflow-y: auto !important; overflow-x: hidden !important; }
.bslib-sidebar-layout .sidebar { overflow-y: auto; }
.value-box, .bslib-value-box { overflow: hidden !important; min-height: 92px !important; height: auto !important; }
.value-box .value-box-area, .bslib-value-box .value-box-area { overflow: hidden !important; }
.card { overflow: visible !important; }
.html-fill-item, .html-fill-container { overflow: visible !important; }
.js-fill-item { overflow: visible !important; }
#kpi_fila .value-box-title { font-size: 0.8rem; }
.nav-link { font-size: 0.95rem; }
.tabla-ajuste { font-size: 0.84rem; width: 100%; }
.tabla-ajuste th, .tabla-ajuste td { padding: 0.4rem 0.55rem; white-space: nowrap; }
.tabla-ajuste thead { background: #eef3f8; }
"""

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h4("TransMilenio y el 123"),
        ui.accordion(
            ui.accordion_panel(
                "Qué es NUSE",
                ui.p(
                    "Llamadas al 123. No es un delito juzgado ni un hecho en el andén. "
                    "Llega por UPZ: varias estaciones comparten el mismo número."
                ),
            ),
            open=False,
        ),
        ui.input_select(
            "familia", "Familia",
            {"hurto": "Hurto", "violencia": "Violencia", "orden_publico": "Orden público"},
        ),
        ui.input_select(
            "anio", "Año",
            {"todos": "2023–2026", "2023": "2023", "2024": "2024", "2025": "2025", "2026": "2026 (ene–jun)"},
        ),
        ui.p(ui.output_text("brecha")),
        ui.download_button("dl_html", "Descargar informe HTML"),
        ui.p(ui.tags.small("Unidad: UPZ. Soacha queda fuera.")),
        width=280,
    ),
    ui.tags.style(CSS),
    ui.layout_columns(
        ui.value_box("UPZ al frente", ui.output_ui("kpi_top"), theme="primary", showcase=None),
        ui.value_box("Llamadas / mes", ui.output_ui("kpi_valor"), theme="secondary", showcase=None),
        ui.value_box("Estaciones TM", ui.output_ui("kpi_est"), theme="bg-gradient-blue-purple", showcase=None),
        col_widths=(4, 4, 4),
        fill=False,
        id="kpi_fila",
    ),
    ui.navset_card_underline(
        ui.nav_panel(
            "Dónde se concentra",
            ui.card(ui.card_header("Ranking por UPZ"), output_widget("graf_ranking"), full_screen=True),
            ui.card(ui.card_header("Mapa del entorno"), output_widget("graf_mapa"), full_screen=True),
            ui.card(ui.card_header("Tasa por localidad (100 mil hab.)"), output_widget("graf_tasa"), full_screen=True),
        ),
        ui.nav_panel(
            "Asociaciones",
            ui.card(ui.card_header("Noche y llamados"), output_widget("graf_scatter"), full_screen=True),
            ui.card(ui.card_header("Spearman (58 UPZ)"), output_widget("graf_rho"), full_screen=True),
        ),
        ui.nav_panel(
            "Pronóstico 2026",
            ui.output_ui("texto_modelo"),
            ui.input_radio_buttons(
                "curva",
                "Curva del gráfico",
                {"red": "Red neuronal", "bosque": "Bosque (tradicional)"},
                selected="red",
                inline=True,
            ),
            ui.card(ui.card_header("Observado vs predicho"), output_widget("graf_pred"), full_screen=True),
            ui.card(ui.card_header("Ajuste global (prueba 2026)"), ui.output_ui("tabla_global")),
            ui.card(ui.card_header("Ajuste por localidad"), ui.output_ui("tabla_localidad")),
            ui.card(ui.card_header("Qué usa el bosque"), output_widget("graf_imp"), full_screen=True),
            ui.card(
                ui.card_header("Serie de una UPZ"),
                ui.input_selectize("upz_sel", "UPZ", choices=[], multiple=False),
                output_widget("graf_serie"),
                full_screen=True,
            ),
        ),
        footer=None,
    ),
    title="Entorno de estaciones",
    fillable=False,
)


def server(input, output, session):
    @reactive.calc
    def ranking():
        return panel_filtrado(input.familia(), input.anio())

    @reactive.effect
    def _llenar_upz():
        r = ranking()
        opciones = dict(zip(r["upz_id"].astype(str), r["etiqueta"].astype(str)))
        actual = input.upz_sel()
        sel = actual if actual in opciones else (next(iter(opciones), None))
        ui.update_selectize("upz_sel", choices=opciones, selected=sel)

    @render.text
    def brecha():
        return texto_brecha(input.familia())

    @render.ui
    def kpi_top():
        r = ranking()
        return ui.strong("—" if r.empty else str(r.iloc[0]["etiqueta"]))

    @render.ui
    def kpi_valor():
        r = ranking()
        return ui.strong("—" if r.empty else f"{r.iloc[0]['y']:.0f}")

    @render.ui
    def kpi_est():
        r = ranking()
        return ui.strong("—" if r.empty else f"{r.iloc[0]['n_est']:.0f}")

    @render.download(filename=lambda: f"informe_{input.familia()}.html")
    def dl_html():
        path = construir_informe_html(input.familia(), input.anio())
        with path.open("rb") as f:
            yield f.read()

    @render_plotly
    def graf_ranking():
        return fig_ranking(input.familia(), input.anio())

    @render_plotly
    def graf_mapa():
        return fig_mapa(input.familia(), input.anio())

    @render_plotly
    def graf_tasa():
        return fig_tasa()

    @render_plotly
    def graf_scatter():
        return fig_scatter(input.familia(), input.anio())

    @render_plotly
    def graf_rho():
        return fig_rho(input.familia())

    @render.ui
    def texto_modelo():
        fam = input.familia()
        met = DATOS["met"]
        if fam == "orden_publico" or met.empty:
            return ui.p("El pronóstico cubre hurto y violencia, no orden público.")
        sub = met.loc[met["familia"] == fam]
        if sub.empty:
            return ui.p("Corre antes: python modelo_predictivo.py")

        def fila(nombre):
            hit = sub.loc[sub["modelo"] == nombre]
            return None if hit.empty else hit.iloc[0]

        naive = fila("persistencia")
        if naive is None:
            hit = sub.loc[sub["modelo"].str.contains("persistencia", na=False)]
            naive = None if hit.empty else hit.iloc[0]
        bosque = fila("bosque")
        red_tab = sub.loc[sub["modelo"].isin(["red_residual", "mlp"])].sort_values("mae")
        red = None if red_tab.empty else red_tab.iloc[0]
        partes = [
            ui.p(
                ui.strong(f"{FAM_TITULO[fam]}. "),
                "Train 2023–2025, test 2026. Se conserva el bosque (tradicional) y se contrastan dos redes: ",
                "MLP directo y red residual (corrige el mes previo). No predice hechos en la estación.",
            )
        ]
        txt = ""
        if naive is not None:
            txt = f"Persistencia MAE {naive['mae']:.1f}."
        if bosque is not None:
            txt += f" Bosque {bosque['mae']:.1f} (skill {100*bosque['skill_vs_persistencia']:.1f}%)."
        if red is not None:
            txt += f" Mejor red ({red['modelo']}) {red['mae']:.1f} (skill {100*red['skill_vs_persistencia']:.1f}%)."
        partes.append(ui.p(txt))
        return ui.div(*partes)

    @render.ui
    def tabla_global():
        fam = input.familia()
        met = DATOS["met"]
        if met.empty or fam == "orden_publico":
            return ui.p("Sin ajuste para esta familia.")
        sub = met.loc[met["familia"] == fam].sort_values("mae")
        return html_tabla(_fmt_met(sub))

    @render.ui
    def tabla_localidad():
        fam = input.familia()
        locm = DATOS.get("met_loc", pd.DataFrame())
        if locm.empty or fam == "orden_publico":
            return ui.p("Sin desglose por localidad.")
        sub = locm.loc[locm["familia"] == fam]
        met = DATOS["met"]
        nn = "red_residual"
        if not met.empty:
            cand = met.loc[(met["familia"] == fam) & (met["modelo"].isin(["mlp", "red_residual"]))]
            if not cand.empty:
                nn = str(cand.sort_values("mae").iloc[0]["modelo"])
        orden = [c for c in ("persistencia", "bosque", nn) if c in set(sub["modelo"])]
        mae = sub.pivot_table(index="localidad", columns="modelo", values="mae")
        mae = mae.reindex(columns=orden)
        mae.columns = [f"MAE {ETIQ_MOD.get(c, c).lower()}" for c in mae.columns]
        n = sub.groupby("localidad")["n_test"].first().rename("n")
        out = pd.concat([n, mae.round(1)], axis=1).reset_index().rename(columns={"localidad": "Localidad"})
        for modelo, col in (("bosque", "Skill bosque (%)"), (nn, "Skill red (%)")):
            if modelo in set(sub["modelo"]):
                sk = (100 * sub.loc[sub["modelo"] == modelo].set_index("localidad")["skill_vs_persistencia"]).round(1)
                out[col] = out["Localidad"].map(sk)
        out = out.sort_values("n", ascending=False)
        return html_tabla(out)

    @render_plotly
    def graf_pred():
        return fig_pred(input.familia(), input.curva())

    @render_plotly
    def graf_imp():
        return fig_imp(input.familia())

    @render_plotly
    def graf_serie():
        return fig_serie(input.familia(), input.upz_sel() or "")


app = App(app_ui, server)
