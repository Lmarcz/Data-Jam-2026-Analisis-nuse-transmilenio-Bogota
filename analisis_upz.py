"""
Análisis del panel integrado: unidad UPZ–mes, no estación.

Pasos:
  1. Filtra meses incompletos y estaciones Soacha (sin UPZ Bogotá).
  2. Agrega features de TransMilenio de estación → UPZ–mes
     (suma de flujo; % nocturno / finde / picos ponderados por flujo).
  3. Rankings de UPZ (conteos NUSE) y de localidad (tasas por 100 mil).
  4. Spearman: asociación perfil TM ↔ NUSE de la UPZ.
  5. Robustez: sin portales, 2023–2025, hurto vs violencia vs orden público,
     corte transversal (media por UPZ) y within (desvío respecto a la media de la UPZ).

Uso:
  python analisis_upz.py
  python analisis_upz.py --panel salidas_integracion/panel_estacion_mes.csv
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parent
PANEL_DEFAULT = BASE / "salidas_integracion" / "panel_estacion_mes.csv"
SALIDA = BASE / "salidas_analisis"

Y_FAMILIAS = (
    "upz_nuse_hurto",
    "upz_nuse_violencia",
    "upz_nuse_orden_publico",
)
X_PERFIL = (
    "flujo_total_upz",
    "log_flujo_upz",
    "pct_nocturno_pond",
    "pct_finde_pond",
    "pct_pico_am_pond",
    "pct_pico_pm_pond",
    "gini_pond",
    "n_estaciones",
    "dist_cai_mediana_m",
    "n_cai_500m_media",
    "indice_seguridad_nocturna_2019",
)

LOG = logging.getLogger("analisis_upz")


def _bool_col(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s.fillna(False)
    return s.astype(str).str.lower().isin(("1", "true", "yes", "si", "sí"))


def filtrar_calidad(est: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    n0 = len(est)
    soacha = _bool_col(est["flag_soacha"])
    incompleto = _bool_col(est["mes_incompleto"])
    sin_upz = est["upz_id"].isna() | (est["upz_id"].astype(str).str.strip() == "")
    keep = (~soacha) & (~incompleto) & (~sin_upz)
    out = est.loc[keep].copy()
    meta = {
        "filas_entrada": int(n0),
        "filas_soacha": int(soacha.sum()),
        "filas_mes_incompleto": int(incompleto.sum()),
        "filas_sin_upz": int(sin_upz.sum()),
        "filas_analisis": int(len(out)),
        "estaciones_analisis": int(out["cod_estacion"].nunique()),
        "upz_analisis": int(out["upz_id"].nunique()),
    }
    return out, meta


def agregar_upz_mes(est: pd.DataFrame) -> pd.DataFrame:
    """Una fila por UPZ–mes. Outcomes se copian (son de UPZ); TM se agrega."""
    g = est.copy()
    w = g["flujo_total_mes"].clip(lower=0).fillna(0)
    g["_w"] = w
    for col, alias in (
        ("pct_flujo_nocturno", "nocturno"),
        ("pct_flujo_finde", "finde"),
        ("pct_flujo_pico_am", "pico_am"),
        ("pct_flujo_pico_pm", "pico_pm"),
        ("concentracion_horaria_gini", "gini"),
    ):
        g[f"_num_{alias}"] = pd.to_numeric(g[col], errors="coerce") * w
    g["_es_portal"] = g["nom_est"].astype(str).str.contains("portal", case=False, na=False)

    keys = ["upz_id", "anio", "mes"]
    agg = g.groupby(keys, as_index=False).agg(
        n_estaciones=("cod_estacion", "nunique"),
        n_portales=("_es_portal", "sum"),
        flujo_total_upz=("flujo_total_mes", "sum"),
        validaciones_upz=("validaciones_mes", "sum"),
        salidas_upz=("salidas_mes", "sum"),
        w_sum=("_w", "sum"),
        num_nocturno=("_num_nocturno", "sum"),
        num_finde=("_num_finde", "sum"),
        num_pico_am=("_num_pico_am", "sum"),
        num_pico_pm=("_num_pico_pm", "sum"),
        num_gini=("_num_gini", "sum"),
        dist_cai_mediana_m=("dist_cai_m", "median"),
        n_cai_500m_media=("n_cai_500m", "mean"),
        upz_nuse_hurto=("upz_nuse_hurto", "first"),
        upz_nuse_violencia=("upz_nuse_violencia", "first"),
        upz_nuse_orden_publico=("upz_nuse_orden_publico", "first"),
        upz_nuse_otros=("upz_nuse_otros", "first"),
        upz_nuse_total=("upz_nuse_total", "first"),
        loc_nuse_hurto=("loc_nuse_hurto", "first"),
        loc_nuse_violencia=("loc_nuse_violencia", "first"),
        loc_nuse_total=("loc_nuse_total", "first"),
        poblacion_localidad=("poblacion_localidad", "first"),
        loc_tasa_nuse_hurto_100k_hab=("loc_tasa_nuse_hurto_100k_hab", "first"),
        loc_tasa_nuse_violencia_100k_hab=("loc_tasa_nuse_violencia_100k_hab", "first"),
        dai_total=("dai_total", "first"),
        dai_hurto_personas=("dai_hurto_personas", "first"),
        ir_total=("ir_total", "first"),
        indice_seguridad_nocturna_2019=("indice_seguridad_nocturna_2019", "first"),
        cod_localidad=("cod_localidad", "first"),
        localidad=("localidad", "first"),
    )
    den = agg["w_sum"].replace(0, np.nan)
    agg["pct_nocturno_pond"] = agg["num_nocturno"] / den
    agg["pct_finde_pond"] = agg["num_finde"] / den
    agg["pct_pico_am_pond"] = agg["num_pico_am"] / den
    agg["pct_pico_pm_pond"] = agg["num_pico_pm"] / den
    agg["gini_pond"] = agg["num_gini"] / den
    agg["log_flujo_upz"] = np.where(agg["flujo_total_upz"] > 0, np.log(agg["flujo_total_upz"]), np.nan)
    agg["periodo"] = agg["anio"].astype(int).astype(str) + "-" + agg["mes"].astype(int).astype(str).str.zfill(2)
    return (
        agg.drop(columns=["w_sum", "num_nocturno", "num_finde", "num_pico_am", "num_pico_pm", "num_gini"])
        .sort_values(["upz_id", "anio", "mes"])
        .reset_index(drop=True)
    )


def ranking_upz(upz: pd.DataFrame) -> pd.DataFrame:
    g = (
        upz.groupby(["upz_id", "cod_localidad", "localidad"], dropna=False)
        .agg(
            n_meses=("periodo", "nunique"),
            n_estaciones=("n_estaciones", "median"),
            flujo_medio=("flujo_total_upz", "mean"),
            pct_nocturno_medio=("pct_nocturno_pond", "mean"),
            hurto_mensual_medio=("upz_nuse_hurto", "mean"),
            hurto_suma=("upz_nuse_hurto", "sum"),
            violencia_mensual_media=("upz_nuse_violencia", "mean"),
            orden_publico_mensual_medio=("upz_nuse_orden_publico", "mean"),
            nuse_total_mensual_medio=("upz_nuse_total", "mean"),
            tasa_loc_hurto_media=("loc_tasa_nuse_hurto_100k_hab", "mean"),
            dist_cai_mediana_m=("dist_cai_mediana_m", "median"),
            indice_nocturno_2019=("indice_seguridad_nocturna_2019", "mean"),
        )
        .reset_index()
    )
    g["rank_hurto_medio"] = g["hurto_mensual_medio"].rank(ascending=False, method="min")
    g["rank_violencia_media"] = g["violencia_mensual_media"].rank(ascending=False, method="min")
    g["rank_orden_publico_medio"] = g["orden_publico_mensual_medio"].rank(ascending=False, method="min")
    g["rank_tasa_loc_hurto"] = g["tasa_loc_hurto_media"].rank(ascending=False, method="min")
    return g.sort_values(["rank_hurto_medio", "upz_id"]).reset_index(drop=True)


def ranking_localidad(upz: pd.DataFrame) -> pd.DataFrame:
    loc = (
        upz.drop_duplicates(["cod_localidad", "anio", "mes"])
        .groupby(["cod_localidad", "localidad"], dropna=False)
        .agg(
            n_meses=("periodo", "nunique"),
            n_upz=("upz_id", "nunique"),
            poblacion_media=("poblacion_localidad", "mean"),
            tasa_hurto_media=("loc_tasa_nuse_hurto_100k_hab", "mean"),
            tasa_violencia_media=("loc_tasa_nuse_violencia_100k_hab", "mean"),
            loc_nuse_hurto_medio=("loc_nuse_hurto", "mean"),
            dai_hurto_personas_anual=("dai_hurto_personas", "mean"),
        )
        .reset_index()
    )
    loc["rank_tasa_hurto"] = loc["tasa_hurto_media"].rank(ascending=False, method="min")
    loc["rank_tasa_violencia"] = loc["tasa_violencia_media"].rank(ascending=False, method="min")
    return loc.sort_values(["rank_tasa_hurto", "cod_localidad"]).reset_index(drop=True)


def _spearman_par(x: pd.Series, y: pd.Series) -> dict:
    d = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(d)
    if n < 10:
        return {"n": int(n), "rho": np.nan, "p": np.nan}
    try:
        from scipy.stats import spearmanr

        if d["x"].nunique() < 2 or d["y"].nunique() < 2:
            return {"n": int(n), "rho": np.nan, "p": np.nan}
        rho, p = spearmanr(d["x"], d["y"])
        return {"n": int(n), "rho": float(rho), "p": float(p)}
    except Exception:
        pass
    rho = d["x"].corr(d["y"], method="spearman")
    if pd.isna(rho):
        return {"n": int(n), "rho": np.nan, "p": np.nan}
    if abs(float(rho)) >= 1:
        p = 0.0
    else:
        from math import erfc, sqrt

        t = float(rho) * np.sqrt((n - 2) / (1.0 - float(rho) ** 2))
        p = float(erfc(abs(t) / sqrt(2)))
    return {"n": int(n), "rho": float(rho), "p": p}


def matriz_spearman(df: pd.DataFrame, etiqueta: str) -> pd.DataFrame:
    filas = []
    xs = [c for c in X_PERFIL if c in df.columns]
    ys = [c for c in Y_FAMILIAS if c in df.columns]
    for y in ys:
        for x in xs:
            res = _spearman_par(df[x], df[y])
            filas.append({"muestra": etiqueta, "y": y, "x": x, **res})
    out = pd.DataFrame(filas)
    out["abs_rho"] = out["rho"].abs()
    return out.sort_values(["y", "abs_rho"], ascending=[True, False]).drop(columns="abs_rho")


def within_upz(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    w = df.copy()
    for c in cols:
        if c in w.columns:
            w[c] = w[c] - w.groupby("upz_id")[c].transform("mean")
    return w


def robustez(upz: pd.DataFrame, est: pd.DataFrame) -> pd.DataFrame:
    bloques = [matriz_spearman(upz, "panel_upz_mes")]

    corte = upz.groupby("upz_id", as_index=False).mean(numeric_only=True)
    # reatachar nombres
    nombres = upz.drop_duplicates("upz_id")[["upz_id", "localidad", "cod_localidad"]]
    corte = corte.merge(nombres, on="upz_id", how="left")
    bloques.append(matriz_spearman(corte, "media_por_upz"))

    cols_w = [c for c in list(X_PERFIL) + list(Y_FAMILIAS) if c in upz.columns]
    bloques.append(matriz_spearman(within_upz(upz, cols_w), "within_upz"))

    hist = upz.loc[upz["anio"] <= 2025]
    bloques.append(matriz_spearman(hist, "solo_2023_2025"))

    es_portal = est["nom_est"].astype(str).str.contains("portal", case=False, na=False)
    sin_portal = agregar_upz_mes(est.loc[~es_portal])
    bloques.append(matriz_spearman(sin_portal, "sin_portales"))

    return pd.concat(bloques, ignore_index=True)


def figuras(upz: pd.DataFrame, rank: pd.DataFrame, sp: pd.DataFrame, carpeta: Path) -> list[str]:
    try:
        import matplotlib.pyplot as plt
    except Exception as exc:
        LOG.warning("matplotlib no disponible (%s); no se generan figuras.", exc)
        return []

    carpeta.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    top = rank.nsmallest(15, "rank_hurto_medio")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["upz_id"].iloc[::-1], top["hurto_mensual_medio"].iloc[::-1], color="#1f4e79")
    ax.set_xlabel("NUSE hurto mensual medio (UPZ)")
    ax.set_title("Ranking UPZ — hurto NUSE (no es hurto en la estación)")
    fig.tight_layout()
    p = carpeta / "ranking_upz_hurto.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(
        upz["pct_nocturno_pond"],
        upz["upz_nuse_hurto"],
        s=np.clip(upz["flujo_total_upz"] / upz["flujo_total_upz"].median() * 18, 8, 80),
        alpha=0.35,
        c="#1f4e79",
        linewidths=0,
    )
    ax.set_xlabel("% flujo nocturno (ponderado por flujo de la UPZ)")
    ax.set_ylabel("NUSE hurto de la UPZ (mes)")
    ax.set_title("Asociación descriptiva · tamaño = flujo TM de la UPZ")
    fig.tight_layout()
    p = carpeta / "scatter_nocturno_hurto.png"
    fig.savefig(p, dpi=140)
    plt.close(fig)
    paths.append(str(p))

    base = sp.loc[sp["muestra"] == "media_por_upz"].copy()
    if not base.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        sub = base.loc[base["y"] == "upz_nuse_hurto"].sort_values("rho")
        colores = np.where(sub["rho"] >= 0, "#1f4e79", "#a03d3d")
        ax.barh(sub["x"], sub["rho"], color=colores)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Spearman ρ (media por UPZ)")
        ax.set_title("Hurto NUSE UPZ vs perfil TM — corte transversal")
        fig.tight_layout()
        p = carpeta / "spearman_hurto_media_upz.png"
        fig.savefig(p, dpi=140)
        plt.close(fig)
        paths.append(str(p))

    return paths


def hallazgo(sp: pd.DataFrame) -> str:
    """Una frase acotada: usa el corte transversal (N = n° de UPZ), no el panel inflado."""
    sub = sp.loc[
        (sp["muestra"] == "media_por_upz")
        & (sp["y"] == "upz_nuse_hurto")
        & (sp["x"] == "pct_nocturno_pond")
    ]
    if sub.empty or sub["rho"].isna().all():
        return (
            "No hubo suficientes datos para estimar la asociación entre "
            "% nocturno y hurto NUSE a nivel UPZ."
        )
    rho = float(sub["rho"].iloc[0])
    p = float(sub["p"].iloc[0]) if pd.notna(sub["p"].iloc[0]) else np.nan
    n = int(sub["n"].iloc[0])
    sig = "asociación" if (pd.notna(p) and p < 0.05) else "asociación débil o no detectable"
    direccion = "positiva" if rho > 0 else "negativa"
    return (
        f"Corte transversal ({n} UPZ): {sig} {direccion} entre % de flujo nocturno "
        f"ponderado y hurto NUSE medio de la UPZ (Spearman rho={rho:.2f}"
        + (f", p~{p:.3f}" if pd.notna(p) else "")
        + "). No implica que el hurto ocurra en la estación."
    )


def construir(panel_path: Path) -> dict:
    SALIDA.mkdir(exist_ok=True)
    est = pd.read_csv(panel_path)
    est, meta = filtrar_calidad(est)
    LOG.info("Filtro calidad: %s", meta)

    upz = agregar_upz_mes(est)
    rank_u = ranking_upz(upz)
    rank_l = ranking_localidad(upz)
    sp = robustez(upz, est)

    figs = figuras(upz, rank_u, sp, SALIDA / "figuras")

    upz_path = SALIDA / "panel_upz_mes_agregado.csv"
    rank_u_path = SALIDA / "ranking_upz.csv"
    rank_l_path = SALIDA / "ranking_localidad.csv"
    sp_path = SALIDA / "spearman.csv"
    upz.to_csv(upz_path, index=False, encoding="utf-8-sig")
    rank_u.to_csv(rank_u_path, index=False, encoding="utf-8-sig")
    rank_l.to_csv(rank_l_path, index=False, encoding="utf-8-sig")
    sp.to_csv(sp_path, index=False, encoding="utf-8-sig")

    frase = hallazgo(sp)
    reporte = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "panel": str(panel_path),
        "calidad": meta,
        "filas_upz_mes": int(len(upz)),
        "hallazgo_acotado": frase,
        "como_leer": [
            "Y es NUSE de la UPZ. X es el perfil TM agregado de las estaciones de esa UPZ.",
            "media_por_upz es el corte más honesto (N = número de UPZ).",
            "panel_upz_mes infla N (meses no son independientes); usar solo como descriptivo.",
            "within_upz pregunta si, en la misma UPZ, meses con más flujo nocturno tienen más NUSE.",
            "DAI/IR no entran al Spearman mensual: son anuales.",
            "rank_tasa_loc_hurto es de la localidad, no de la UPZ; varias UPZ comparten la misma tasa.",
        ],
        "archivos": {
            "panel_upz_mes_agregado": str(upz_path),
            "ranking_upz": str(rank_u_path),
            "ranking_localidad": str(rank_l_path),
            "spearman": str(sp_path),
            "figuras": figs,
        },
    }
    rep_path = SALIDA / "reporte_analisis.json"
    rep_path.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Listo: %s", upz_path)
    return reporte


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Agregación UPZ + ranking + Spearman")
    parser.add_argument("--panel", type=Path, default=PANEL_DEFAULT)
    args = parser.parse_args()
    if not args.panel.exists():
        raise SystemExit(f"No encuentro el panel: {args.panel}")
    reporte = construir(args.panel)
    texto = json.dumps(reporte, ensure_ascii=False, indent=2)
    try:
        print(texto)
    except UnicodeEncodeError:
        print(texto.encode("cp1252", errors="replace").decode("cp1252"))


if __name__ == "__main__":
    main()
