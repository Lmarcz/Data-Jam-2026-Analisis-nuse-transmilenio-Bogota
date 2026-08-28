"""
Reconstruye el panel estación–mes / UPZ–mes con rigor explícito.

Correcciones respecto de Merge.ipynb:
  - fillna(0) antes de sumar flujo (validaciones + salidas).
  - Ventanas horarias disjuntas: pico AM 06–08, pico PM 17–18, nocturno 19–05.
  - Gini solo en horas de operación 04–23.
  - Corte temporal alineado a NUSE (hasta 2026-06-30).
  - Flag de mes incompleto (< 20 días calendario con dato).
  - NUSE agrupado en familias (hurto / violencia / orden público); no 84 columnas crudas.
  - Prefijo upz_ / loc_ en outcomes: no se fingen incidentes a nivel estación.
  - Tasas con denominador de la misma unidad (localidad × población localidad).
  - Orígenes adicionales: solo ZIP/CSV de Datos Abiertos Bogotá (sin oaiee.scj.gov.co).

Uso:
  python integracion_panel.py
  python integracion_panel.py --skip-download   # reutiliza cache/
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import unicodedata
import warnings
import zipfile
from datetime import date, datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

BASE = Path(__file__).resolve().parent
CACHE = BASE / "cache_abiertos"
SALIDA = BASE / "salidas_integracion"
CKAN = "https://datosabiertos.bogota.gov.co"

ANIOS_TM = (2023, 2024, 2025, 2026)
CORTE_NUSE = date(2026, 6, 30)
DIAS_MIN_MES = 20
HORAS_OPERACION = list(range(4, 24))  # 04:00–23:00
RADIO_CAI_M = 500.0

PICO_AM = {6, 7, 8}
PICO_PM = {17, 18}  # no incluye 19
NOCTURNO = set(range(19, 24)) | set(range(0, 6))  # 19–05

COD_EST_RE = re.compile(r"\((\d+)\)")

# Familias NUSE (TIPO_DETALLE). El resto queda en nuse_otros.
FAMILIAS_NUSE: dict[str, frozenset[str]] = {
    "hurto": frozenset({"HURTO EFECTUADO", "HURTO EN PROCESO"}),
    "violencia": frozenset(
        {
            "RIÑA",
            "LESIONES PERSONALES",
            "DISPAROS",
            "HERIDO",
            "VIOLENCIA SEXUAL",
            "MALTRATO",
            "PORTE DE ARMAS",
            "RAPTO",
            "SECUESTRO",
            "AMENAZA DE SUICIDIO",
            "INTENTO DE SUICIDIO",
            "EXHIBICIONES O ACTOS OBSCENOS",
        }
    ),
    "orden_publico": frozenset(
        {
            "EMBRIAGUEZ",
            "HABITANTE DE LA CALLE",
            "PANDILLAS",
            "RUIDO",
            "PERSONA O VEHÍCULO SOSPECHOSO",
            "MANIFESTACIÓN O MOTÍN",
            "NARCÓTICOS",
            "DELINCUENTE CAPTURADO POR CIVIL",
            "SOLICITUD DE APOYO O DESACATO",
            "VERIFICAR SITUACIÓN",
            "VEHÍCULO ABANDONADO",
            "DAÑOS EN PROPIEDAD PÚBLICA O PRIVADA",
        }
    ),
}

# Festivos Colombia (lunes de decreto) 2023–2026. Respaldo si no hay `holidays`.
FESTIVOS_CO_ISO = {
    "2023-01-01", "2023-01-09", "2023-03-20", "2023-04-06", "2023-04-07",
    "2023-05-01", "2023-05-22", "2023-06-12", "2023-06-19", "2023-07-03",
    "2023-07-20", "2023-08-07", "2023-08-21", "2023-10-16", "2023-11-06",
    "2023-11-13", "2023-12-08", "2023-12-25",
    "2024-01-01", "2024-01-08", "2024-03-25", "2024-03-28", "2024-03-29",
    "2024-05-01", "2024-05-13", "2024-06-03", "2024-06-10", "2024-07-01",
    "2024-07-20", "2024-08-07", "2024-08-19", "2024-10-14", "2024-11-04",
    "2024-11-11", "2024-12-08", "2024-12-25",
    "2025-01-01", "2025-01-06", "2025-03-24", "2025-04-17", "2025-04-18",
    "2025-05-01", "2025-06-02", "2025-06-23", "2025-06-30", "2025-07-20",
    "2025-08-07", "2025-08-18", "2025-10-13", "2025-11-03", "2025-11-17",
    "2025-12-08", "2025-12-25",
    "2026-01-01", "2026-01-12", "2026-03-23", "2026-04-02", "2026-04-03",
    "2026-05-01", "2026-05-18", "2026-06-08", "2026-06-15", "2026-06-29",
    "2026-07-20", "2026-08-07", "2026-08-17", "2026-10-12", "2026-11-02",
    "2026-11-16", "2026-12-08", "2026-12-25",
}

LOG = logging.getLogger("integracion")


# ---------------------------------------------------------------------------
# Utilidades
# ---------------------------------------------------------------------------

def _norm(texto) -> str:
    if texto is None or (isinstance(texto, float) and np.isnan(texto)):
        return ""
    s = unicodedata.normalize("NFKD", str(texto))
    s = "".join(c for c in s if not unicodedata.combining(c))
    return re.sub(r"[^a-z0-9]", "", s.lower())


def _col(df: pd.DataFrame, *claves: str) -> str | None:
    mapa = {_norm(c): c for c in df.columns}
    for clave in claves:
        hit = mapa.get(_norm(clave))
        if hit:
            return hit
    for clave in claves:
        cn = _norm(clave)
        for k, orig in mapa.items():
            if cn and cn in k:
                return orig
    return None


def leer_csv_auto(path: Path) -> pd.DataFrame:
    """SDS y otros CSV del portal a veces vienen con ; y a veces con ,."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def festivos_colombia(anios: Iterable[int]) -> set[date]:
    try:
        import holidays

        return set(holidays.CO(years=list(anios)).keys())
    except Exception:
        LOG.warning("Paquete holidays no disponible; uso calendario embebido 2023–2026.")
        return {date.fromisoformat(x) for x in FESTIVOS_CO_ISO}


def gini(valores: np.ndarray) -> float:
    v = np.sort(np.asarray(valores, dtype=float))
    v = v[np.isfinite(v) & (v >= 0)]
    if v.size == 0 or v.sum() <= 0:
        return np.nan
    n = v.size
    cum = np.cumsum(v)
    return float((n + 1 - 2 * np.sum(cum) / cum[-1]) / n)


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    r = 6371000.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * r * np.arcsin(np.sqrt(a))


def http_get(url: str, **kwargs) -> requests.Response:
    """Reintenta sin verificación SSL (redes corporativas con proxy)."""
    kwargs.setdefault("timeout", 60)
    try:
        r = requests.get(url, **kwargs)
        r.raise_for_status()
        return r
    except requests.exceptions.SSLError:
        warnings.filterwarnings("ignore", message="Unverified HTTPS request")
        kwargs["verify"] = False
        r = requests.get(url, **kwargs)
        r.raise_for_status()
        return r


def descargar(url: str, destino: Path, timeout: int = 180) -> Path:
    destino.parent.mkdir(parents=True, exist_ok=True)
    if destino.exists() and destino.stat().st_size > 0:
        LOG.info("Cache hit: %s", destino.name)
        return destino
    LOG.info("Descargando %s", url)
    r = http_get(url, stream=True, timeout=timeout)
    tmp = destino.with_suffix(destino.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    tmp.replace(destino)
    return destino


def ckan_package(ident: str) -> dict:
    r = http_get(f"{CKAN}/api/3/action/package_show", params={"id": ident})
    body = r.json()
    if not body.get("success"):
        raise RuntimeError(f"CKAN package_show falló para {ident}")
    return body["result"]


def recurso_descargable(pkg: dict, formatos: tuple[str, ...] = ("GEOJSON", "GPKG", "SHP", "CSV")) -> dict:
    """Elige un recurso subido al portal, nunca REST/WMS/WFS de oaiee."""
    prohibido = ("rest", "wms", "wfs", "oaiee.scj.gov.co")
    candidatos = []
    for res in pkg.get("resources", []):
        url = (res.get("url") or "").lower()
        fmt = (res.get("format") or "").upper()
        if any(p in url for p in prohibido):
            continue
        if res.get("url_type") not in (None, "upload") and "datosabiertos.bogota.gov.co" not in url:
            continue
        if not any(f in fmt for f in formatos):
            continue
        candidatos.append(res)
    if not candidatos:
        raise RuntimeError(f"Sin recurso descargable en {pkg.get('name')}")

    def score(res: dict) -> tuple:
        fmt = (res.get("format") or "").upper()
        nombre = (res.get("name") or "").lower()
        pref = {"GEOJSON": 3, "GPKG": 2, "CSV": 2, "SHP": 1}.get(fmt, 0)
        reciente = 1 if any(x in nombre for x in ("2026", "julio", "enero - julio")) else 0
        return (pref, reciente, res.get("last_modified") or "")

    candidatos.sort(key=score, reverse=True)
    return candidatos[0]


def descomprimir_si_zip(path: Path, prefer_nombre: str | None = None) -> Path:
    if path.suffix.lower() != ".zip" and not zipfile.is_zipfile(path):
        return path
    dest = path.with_suffix("")
    dest.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as zf:
        zf.extractall(dest)
    for ext in (".geojson", ".json", ".gpkg", ".shp"):
        hits = list(dest.rglob(f"*{ext}"))
        if prefer_nombre:
            pref = [h for h in hits if prefer_nombre.lower() in h.name.lower()]
            if pref:
                return pref[0]
        if hits:
            return hits[0]
    raise RuntimeError(f"ZIP sin vector en {path}")


def leer_propiedades_vector(path: Path, prefer_nombre: str | None = None) -> pd.DataFrame:
    path = descomprimir_si_zip(path, prefer_nombre=prefer_nombre)
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path)
        return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    except Exception as exc:
        LOG.info("geopandas no usó %s (%s); intento JSON.", path.name, exc)
    if path.suffix.lower() in {".geojson", ".json"}:
        data = json.loads(path.read_text(encoding="utf-8"))
        feats = data["features"] if isinstance(data, dict) else data
        return pd.DataFrame([f.get("properties") or {} for f in feats])
    raise RuntimeError(f"No pude leer atributos de {path}")


def puntos_geojson(path: Path) -> pd.DataFrame:
    path = descomprimir_si_zip(path)
    try:
        import geopandas as gpd

        gdf = gpd.read_file(path).to_crs(epsg=4326)
        gdf["lon"] = gdf.geometry.x
        gdf["lat"] = gdf.geometry.y
        return pd.DataFrame(gdf.drop(columns="geometry", errors="ignore"))
    except Exception:
        data = json.loads(path.read_text(encoding="utf-8"))
        filas = []
        for feat in data.get("features", []):
            geom = feat.get("geometry") or {}
            coords = geom.get("coordinates") or [None, None]
            props = dict(feat.get("properties") or {})
            if geom.get("type") == "Point":
                props["lon"], props["lat"] = coords[0], coords[1]
                filas.append(props)
        if not filas:
            raise RuntimeError(f"Sin puntos en {path}")
        return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# TransMilenio
# ---------------------------------------------------------------------------

def cargar_estaciones() -> pd.DataFrame:
    est = pd.read_csv(BASE / "Dim_estaciones.csv")
    est["num_est"] = pd.to_numeric(est["num_est"], errors="coerce").astype("Int64")
    est["upz_id"] = est["UPLCODIGO"].astype(str).str.strip()
    est["sin_upz"] = est["upz_id"].isin(["", "nan", "None"]) | est["upz_id"].isna()
    # Soacha: las 4 estaciones del notebook original, más cualquier otra sin UPZ.
    soacha_nombres = {
        "león xiii", "leon xiii",
        "terreros - hospital cardio vascular", "terreros",
        "la despensa",
        "san mateo - cc unisur", "san mateo",
    }
    est["nom_norm"] = est["nom_est"].map(_norm)
    est["flag_soacha"] = est["nom_norm"].isin({_norm(x) for x in soacha_nombres}) | est["sin_upz"]
    return est[
        [
            "num_est", "nom_est", "longitud", "latitud", "upz_id",
            "num_acc", "acc_puent", "flag_soacha", "sin_upz",
        ]
    ]


def extraer_codigo(serie: pd.Series) -> pd.Series:
    return pd.to_numeric(serie.astype(str).str.extract(COD_EST_RE, expand=False), errors="coerce")


def leer_tm_anual(anio: int) -> pd.DataFrame:
    val = pd.read_csv(
        BASE / f"troncal_{anio}.csv",
        usecols=["Estacion_Parada", "fecha", "hora", "validaciones"],
        parse_dates=["fecha"],
    )
    sal = pd.read_csv(
        BASE / f"salidas_{anio}.csv",
        usecols=["Estacion_Parada", "fecha", "hora", "salidas"],
        parse_dates=["fecha"],
    )
    val["cod_estacion"] = extraer_codigo(val["Estacion_Parada"])
    sal["cod_estacion"] = extraer_codigo(sal["Estacion_Parada"])
    val = val.dropna(subset=["cod_estacion"])
    sal = sal.dropna(subset=["cod_estacion"])
    val["cod_estacion"] = val["cod_estacion"].astype(int)
    sal["cod_estacion"] = sal["cod_estacion"].astype(int)
    val["hora"] = pd.to_numeric(val["hora"], errors="coerce")
    sal["hora"] = pd.to_numeric(sal["hora"], errors="coerce")

    keys = ["cod_estacion", "fecha", "hora"]
    df = pd.merge(
        val[keys + ["validaciones"]],
        sal[keys + ["salidas"]],
        on=keys,
        how="outer",
    )
    df["validaciones"] = pd.to_numeric(df["validaciones"], errors="coerce").fillna(0)
    df["salidas"] = pd.to_numeric(df["salidas"], errors="coerce").fillna(0)
    df["flujo"] = df["validaciones"] + df["salidas"]
    df = df.dropna(subset=["hora"])
    df["hora"] = df["hora"].astype(int)
    df = df.loc[df["fecha"].dt.date <= CORTE_NUSE]
    return df


def tipo_dia_vec(fechas: pd.Series, festivos: set[date]) -> pd.Series:
    d = fechas.dt.date
    es_fest = d.isin(festivos)
    dow = fechas.dt.dayofweek
    out = pd.Series("laboral", index=fechas.index)
    out = out.mask(dow == 5, "sabado")
    out = out.mask((dow == 6) | es_fest, "domingo_festivo")
    return out


def features_mensuales_estacion(horario: pd.DataFrame, festivos: set[date]) -> pd.DataFrame:
    horario = horario.copy()
    horario["tipo_dia"] = tipo_dia_vec(horario["fecha"], festivos)
    horario["anio"] = horario["fecha"].dt.year
    horario["mes"] = horario["fecha"].dt.month

    filas = []
    grupos = horario.groupby(["cod_estacion", "anio", "mes"], sort=False)
    for (cod, anio, mes), g in grupos:
        total = g["flujo"].sum()
        val = g["validaciones"].sum()
        sal = g["salidas"].sum()
        n_dias = g["fecha"].dt.date.nunique()
        noct = g.loc[g["hora"].isin(NOCTURNO), "flujo"].sum()
        am = g.loc[g["hora"].isin(PICO_AM), "flujo"].sum()
        pm = g.loc[g["hora"].isin(PICO_PM), "flujo"].sum()
        finde = g.loc[g["tipo_dia"].isin(["sabado", "domingo_festivo"]), "flujo"].sum()
        perfil = (
            g.loc[g["hora"].isin(HORAS_OPERACION)]
            .groupby("hora")["flujo"]
            .sum()
            .reindex(HORAS_OPERACION, fill_value=0)
        )
        filas.append(
            {
                "cod_estacion": int(cod),
                "anio": int(anio),
                "mes": int(mes),
                "n_dias_con_dato": int(n_dias),
                "mes_incompleto": n_dias < DIAS_MIN_MES,
                "validaciones_mes": val,
                "salidas_mes": sal,
                "flujo_total_mes": total,
                "pct_flujo_nocturno": noct / total if total > 0 else np.nan,
                "pct_flujo_finde": finde / total if total > 0 else np.nan,
                "pct_flujo_pico_am": am / total if total > 0 else np.nan,
                "pct_flujo_pico_pm": pm / total if total > 0 else np.nan,
                "concentracion_horaria_gini": gini(perfil.to_numpy()),
            }
        )
    return pd.DataFrame(filas)


# ---------------------------------------------------------------------------
# NUSE
# ---------------------------------------------------------------------------

def familia_nuse(detalle: str) -> str:
    d = str(detalle).strip().upper()
    for fam, miembros in FAMILIAS_NUSE.items():
        if d in miembros:
            return fam
    return "otros"


def cargar_nuse() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = BASE / (
        "llamadastramitadas-c4-bogota_numerounicodeseguridadyemergencias-nuse_"
        "linea-123-al-30-junio-2026.csv"
    )
    nuse = pd.read_csv(path, sep=";", dtype=str, low_memory=False)
    nuse["ANIO"] = pd.to_numeric(nuse["ANIO"], errors="coerce")
    nuse["MES"] = pd.to_numeric(nuse["MES"], errors="coerce")
    nuse["CANT_INCIDENTES"] = pd.to_numeric(nuse["CANT_INCIDENTES"], errors="coerce").fillna(0)
    nuse = nuse.loc[nuse["ANIO"].between(2023, 2026)]
    nuse = nuse.loc[~nuse["TIPO_INCIDENTE"].str.upper().isin(["SIMULACRO", "BROMA"])]
    nuse = nuse.loc[nuse["COD_UPZ"].str.upper() != "UPZ999"]
    nuse = nuse.loc[nuse["COD_LOCALIDAD"] != "99"]
    nuse["familia"] = nuse["TIPO_DETALLE"].map(familia_nuse)
    nuse["upz_id"] = nuse["COD_UPZ"].astype(str).str.strip()

    upz = (
        nuse.pivot_table(
            index=["ANIO", "MES", "upz_id"],
            columns="familia",
            values="CANT_INCIDENTES",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"ANIO": "anio", "MES": "mes"})
    )
    for fam in ("hurto", "violencia", "orden_publico", "otros"):
        if fam not in upz.columns:
            upz[fam] = 0
    upz["nuse_total"] = upz[["hurto", "violencia", "orden_publico", "otros"]].sum(axis=1)
    upz = upz.rename(
        columns={
            "hurto": "upz_nuse_hurto",
            "violencia": "upz_nuse_violencia",
            "orden_publico": "upz_nuse_orden_publico",
            "otros": "upz_nuse_otros",
            "nuse_total": "upz_nuse_total",
        }
    )

    loc = (
        nuse.pivot_table(
            index=["ANIO", "MES", "COD_LOCALIDAD", "LOCALIDAD"],
            columns="familia",
            values="CANT_INCIDENTES",
            aggfunc="sum",
            fill_value=0,
        )
        .reset_index()
        .rename(columns={"ANIO": "anio", "MES": "mes", "COD_LOCALIDAD": "cod_localidad", "LOCALIDAD": "localidad"})
    )
    for fam in ("hurto", "violencia", "orden_publico", "otros"):
        if fam not in loc.columns:
            loc[fam] = 0
    loc["nuse_total"] = loc[["hurto", "violencia", "orden_publico", "otros"]].sum(axis=1)
    loc = loc.rename(
        columns={
            "hurto": "loc_nuse_hurto",
            "violencia": "loc_nuse_violencia",
            "orden_publico": "loc_nuse_orden_publico",
            "otros": "loc_nuse_otros",
            "nuse_total": "loc_nuse_total",
        }
    )

    cruz = (
        nuse.groupby(["upz_id", "COD_LOCALIDAD", "LOCALIDAD"])
        .size()
        .reset_index(name="n")
        .sort_values("n", ascending=False)
        .drop_duplicates("upz_id")
        .rename(columns={"COD_LOCALIDAD": "cod_localidad", "LOCALIDAD": "localidad"})
        [["upz_id", "cod_localidad", "localidad"]]
    )
    return upz, loc, cruz


# ---------------------------------------------------------------------------
# Fuentes abiertas
# ---------------------------------------------------------------------------

def poblacion_localidad() -> pd.DataFrame:
    url_directo = (
        "https://datosabiertos.bogota.gov.co/dataset/85bf790d-84d1-4eda-bd6f-40af62e71d95/"
        "resource/37e58cb3-c870-4608-8c37-ce45db0eb7c1/download/osb_demografia-poblacion-localidad.csv"
    )
    try:
        pkg = ckan_package("piramide-poblacional-bogota-d-c")
        rec = next(
            r for r in pkg["resources"]
            if "localidad" in (r.get("name") or "").lower() and (r.get("format") or "").upper() == "CSV"
        )
        url = rec["url"]
    except Exception as exc:
        LOG.warning("package_show población falló (%s); uso URL directa.", exc)
        url = url_directo
    path = descargar(url, CACHE / "poblacion_localidad.csv")
    pob = leer_csv_auto(path)
    c_anio = _col(pob, "ANO", "ANIO", "AÑO", "YEAR")
    c_cod = _col(pob, "CODIGO_LOCALIDAD", "COD_LOCALIDAD", "LOCALIDAD_COD")
    c_nom = _col(pob, "NOMBRE_LOCALIDAD", "LOCALIDAD", "NOM_LOCALIDAD")
    c_pob = _col(pob, "POBLACION", "POBLACIÓN", "POB")
    if not all([c_anio, c_pob]) or (c_cod is None and c_nom is None):
        raise RuntimeError(f"Columnas inesperadas en población: {list(pob.columns)}")
    pob[c_anio] = pd.to_numeric(pob[c_anio], errors="coerce")
    pob[c_pob] = pd.to_numeric(pob[c_pob], errors="coerce")
    keys = [c_anio] + ([c_cod] if c_cod else []) + ([c_nom] if c_nom else [])
    agg = pob.groupby(keys, as_index=False)[c_pob].sum()
    out = pd.DataFrame({"anio": agg[c_anio], "poblacion_localidad": agg[c_pob]})
    if c_nom:
        out["localidad"] = agg[c_nom].astype(str).str.strip()
        out["localidad_norm"] = out["localidad"].map(_norm)
    if c_cod:
        out["cod_localidad"] = (
            pd.to_numeric(agg[c_cod], errors="coerce").fillna(agg[c_cod]).astype(str)
            .str.replace(r"\.0$", "", regex=True).str.zfill(2)
        )
        # Código 00 = total Bogotá en la pirámide SDS; no es una localidad.
        out = out.loc[out["cod_localidad"] != "00"].copy()
    LOG.info("Población: %s filas localidad-año, años %s", len(out), sorted(out["anio"].dropna().unique().astype(int))[-6:])
    return out


DAI_PREFIJOS = {
    "CMH": "homicidio",
    "CMLP": "lesiones_personales",
    "CMHP": "hurto_personas",
    "CMHR": "hurto_residencias",
    "CMHA": "hurto_automotores",
    "CMHB": "hurto_bicicletas",
    "CMHC": "hurto_comercio",
    "CMHCE": "hurto_celulares",
    "CMHM": "hurto_motocicletas",
    "CMDS": "delitos_sexuales",
    "CMVI": "violencia_intrafamiliar",
}

IR_PREFIJOS = {
    "CMR": "incidente_reportado",
    "CMN": "nuse_despachado",
    "CMAOP": "amenaza_orden_publico",
    "CMMM": "maltrato_mujer",
    "CMM": "otros_ir",
    "CMD": "delito_ir",
    "CMPIA": "pia",
    "CMH": "hurto_ir",
    "CMHC": "hurto_cel_ir",
}

COL_ANIO_RE = re.compile(r"^([A-Z]+?)(\d{2})CONT?$", re.I)


def _melt_cont_anual(df: pd.DataFrame, mapa_prefijo: dict[str, str], prefijo_out: str) -> pd.DataFrame:
    """Capas CifrasSCJ: una fila por localidad; años en columnas CMxxYYCONT."""
    c_cod = _col(df, "CMIULOCAL", "COD_LOCALIDAD", "LOCCODIGO", "CLOCCODIGO")
    c_nom = _col(df, "CMNOMLOCAL", "LOCALIDAD", "LOCNOMBRE", "NOMBRE_LOCALIDAD")
    pares: list[tuple[str, int, str]] = []
    for col in df.columns:
        m = COL_ANIO_RE.match(str(col).replace(" ", ""))
        if not m:
            continue
        pref, yy = m.group(1).upper(), int(m.group(2))
        anio = 2000 + yy
        if anio < 2018 or anio > 2026:
            continue
        nombre = mapa_prefijo.get(pref, f"otro_{pref.lower()}")
        pares.append((col, anio, nombre))
    if not pares:
        return pd.DataFrame()

    filas = []
    for _, row in df.iterrows():
        cod = None if c_cod is None else str(row[c_cod]).replace(".0", "").zfill(2)
        nom = None if c_nom is None else str(row[c_nom]).strip()
        por_anio: dict[int, dict] = {}
        for col, anio, nombre in pares:
            val = pd.to_numeric(row[col], errors="coerce")
            bucket = por_anio.setdefault(anio, {"anio": anio, "cod_localidad": cod, "localidad": nom})
            bucket[f"{prefijo_out}_{nombre}"] = bucket.get(f"{prefijo_out}_{nombre}", 0) + (0 if pd.isna(val) else float(val))
        filas.extend(por_anio.values())
    out = pd.DataFrame(filas)
    if out.empty:
        return out
    metricas = [c for c in out.columns if c.startswith(f"{prefijo_out}_")]
    out[f"{prefijo_out}_total"] = out[metricas].sum(axis=1)
    if "localidad" in out.columns:
        out["localidad_norm"] = out["localidad"].map(_norm)
    out["dai_es_anual" if prefijo_out == "dai" else "ir_es_anual"] = True
    return out


def _agregar_capa_anual(df: pd.DataFrame, prefijo: str) -> pd.DataFrame:
    mapa = DAI_PREFIJOS if prefijo == "dai" else IR_PREFIJOS
    melted = _melt_cont_anual(df, mapa, prefijo)
    if not melted.empty:
        LOG.info("%s melt: %s filas, años %s", prefijo, len(melted), sorted(melted["anio"].unique()))
        return melted
    c_anio = _col(df, "ANIO", "AÑO", "YEAR", "VIGENCIA", "DAIANIO")
    c_cod = _col(df, "COD_LOCALIDAD", "CODIGO_LOCALIDAD", "CLOCCODIGO", "LOCODIGO", "LOCCODIGO", "CMIULOCAL")
    c_nom = _col(df, "LOCALIDAD", "LOCNOMBRE", "NOMBRE_LOCALIDAD", "LOCNOM", "CMNOMLOCAL")
    if c_anio is None:
        LOG.warning("%s: no pude interpretar años. Columnas=%s", prefijo, list(df.columns)[:30])
        return pd.DataFrame()
    df = df.copy()
    df["_anio"] = pd.to_numeric(df[c_anio], errors="coerce")
    num_cols = [
        c for c in df.columns
        if c not in {c_anio, c_cod, c_nom} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not num_cols:
        LOG.warning("%s: sin columnas numéricas.", prefijo)
        return pd.DataFrame()
    keys = ["_anio"]
    rename = {"_anio": "anio"}
    if c_cod:
        df["_cod"] = (
            pd.to_numeric(df[c_cod], errors="coerce").fillna(df[c_cod]).astype(str)
            .str.replace(r"\.0$", "", regex=True).str.zfill(2)
        )
        keys.append("_cod")
        rename["_cod"] = "cod_localidad"
    if c_nom:
        df["_nom"] = df[c_nom].astype(str).str.strip()
        df["_nomn"] = df["_nom"].map(_norm)
        keys.extend(["_nom", "_nomn"])
        rename["_nom"] = "localidad"
        rename["_nomn"] = "localidad_norm"
    df["_total"] = df[num_cols].sum(axis=1, min_count=1)
    agg = df.groupby(keys, as_index=False).agg(**{f"{prefijo}_total": ("_total", "sum")})
    return agg.rename(columns=rename)


def capa_localidad_zip(package_id: str, archivo: str, prefijo: str) -> pd.DataFrame:
    pkg = ckan_package(package_id)
    rec = recurso_descargable(pkg)
    ext = Path(rec["url"]).suffix or ".zip"
    path = descargar(rec["url"], CACHE / f"{archivo}{ext}")
    prefer = "DAILoc" if prefijo == "dai" else "IRLoc" if prefijo == "ir" else None
    props = leer_propiedades_vector(path, prefer_nombre=prefer)
    LOG.info("%s columnas: %s", prefijo, list(props.columns)[:25])
    return _agregar_capa_anual(props, prefijo)


def distancia_cai(estaciones: pd.DataFrame) -> pd.DataFrame:
    pkg = ckan_package("centro-de-atencion-accion-para-bogota-d-c")
    rec = recurso_descargable(pkg, formatos=("GEOJSON", "GPKG", "SHP"))
    ext = Path(rec["url"]).suffix or ".geojson"
    path = descargar(rec["url"], CACHE / f"cai{ext}")
    cai = puntos_geojson(path)
    est = estaciones.dropna(subset=["longitud", "latitud"]).copy()
    lon_c = cai["lon"].to_numpy(float)
    lat_c = cai["lat"].to_numpy(float)
    dist_min, n500 = [], []
    for lon, lat in zip(est["longitud"].to_numpy(float), est["latitud"].to_numpy(float)):
        d = haversine_m(lon, lat, lon_c, lat_c)
        dist_min.append(float(np.nanmin(d)))
        n500.append(int(np.sum(d <= RADIO_CAI_M)))
    return pd.DataFrame(
        {
            "num_est": est["num_est"].to_numpy(),
            "dist_cai_m": dist_min,
            "n_cai_500m": n500,
        }
    )


def indice_nocturno_localidad() -> pd.DataFrame:
    """Índice 2019 a localidad (el GeoJSON UPZ local no trae el atributo)."""
    pkg_id = "indice-de-condiciones-de-seguridad-nocturna-bogota-d-c"
    try:
        pkg = ckan_package(pkg_id)
        rec = recurso_descargable(pkg)
    except Exception as exc:
        LOG.warning("Índice nocturno CKAN: %s", exc)
        return pd.DataFrame()
    ext = Path(rec["url"]).suffix or ".zip"
    path = descargar(rec["url"], CACHE / f"nocturno_loc{ext}")
    props = leer_propiedades_vector(path, prefer_nombre="Localidad")
    LOG.info("Índice nocturno columnas: %s", list(props.columns))
    c_cod = _col(props, "LOCCODIGO", "CODIGO_LOCALIDAD", "CMIULOCAL")
    if c_cod is None:
        return pd.DataFrame()
    pcols = [c for c in props.columns if str(c).upper().startswith("P_")]
    out = pd.DataFrame()
    out["cod_localidad"] = (
        pd.to_numeric(props[c_cod], errors="coerce").fillna(props[c_cod]).astype(str)
        .str.replace(r"\.0$", "", regex=True).str.zfill(2)
    )
    for c in pcols:
        out[f"nocturno2019_{_norm(c)}"] = pd.to_numeric(props[c], errors="coerce")
    metricas = [c for c in out.columns if c.startswith("nocturno2019_")]
    if metricas:
        out["indice_seguridad_nocturna_2019"] = out[metricas].mean(axis=1)
    return out.drop_duplicates("cod_localidad")


# ---------------------------------------------------------------------------
# Integración
# ---------------------------------------------------------------------------

def unir_localidad(panel: pd.DataFrame, loc_tbl: pd.DataFrame, on_cod: bool) -> pd.DataFrame:
    if loc_tbl.empty:
        return panel
    if on_cod and "cod_localidad" in loc_tbl.columns:
        a = panel.copy()
        b = loc_tbl.copy()
        a["_k"] = a["cod_localidad"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        b["_k"] = b["cod_localidad"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
        extra = [c for c in b.columns if c not in {"cod_localidad", "localidad", "localidad_norm", "_k", "anio"}]
        return (
            a.merge(b[["_k", "anio"] + extra].drop_duplicates(["_k", "anio"]), on=["_k", "anio"], how="left")
            .drop(columns="_k")
        )
    if "localidad_norm" not in loc_tbl.columns:
        return panel
    a = panel.copy()
    a["localidad_norm"] = a["localidad"].map(_norm)
    extra = [c for c in loc_tbl.columns if c not in {"cod_localidad", "localidad", "localidad_norm", "anio"}]
    return a.merge(
        loc_tbl[["localidad_norm", "anio"] + extra].drop_duplicates(["localidad_norm", "anio"]),
        on=["localidad_norm", "anio"],
        how="left",
    )


def diccionario() -> pd.DataFrame:
    filas = [
        ("cod_estacion", "estación", "Código TM (entero, sin ceros a la izquierda)"),
        ("nom_est", "estación", "Nombre de estación (Dim_estaciones)"),
        ("upz_id", "UPZ", "Código UPZ del join espacial; vacío en Soacha"),
        ("cod_localidad", "localidad", "Código localidad modal de la UPZ según NUSE"),
        ("localidad", "localidad", "Nombre localidad modal de la UPZ según NUSE"),
        ("flag_soacha", "estación", "Estación fuera de UPZ Bogotá"),
        ("mes_incompleto", "calidad", f"Menos de {DIAS_MIN_MES} días con dato en el mes"),
        ("flujo_total_mes", "estación", "validaciones + salidas tras fillna(0)"),
        ("pct_flujo_nocturno", "estación", "Flujo 19:00–05:59 / flujo total"),
        ("pct_flujo_pico_am", "estación", "Flujo 06–08 / flujo total"),
        ("pct_flujo_pico_pm", "estación", "Flujo 17–18 / flujo total (19 no entra)"),
        ("pct_flujo_finde", "estación", "Sábado + domingo/festivo / flujo total"),
        ("concentracion_horaria_gini", "estación", "Gini del perfil 04–23 del mes"),
        ("upz_nuse_hurto", "UPZ", "NUSE hurto efectuado + en proceso. Misma cifra para todas las estaciones de la UPZ."),
        ("upz_nuse_violencia", "UPZ", "NUSE riña, lesiones, disparos, herido, violencia sexual, etc."),
        ("upz_nuse_orden_publico", "UPZ", "NUSE embriaguez, pandillas, ruido, sospechoso, etc."),
        ("upz_nuse_total", "UPZ", "Suma de las 4 familias NUSE (sin simulacro/broma/UPZ999)"),
        ("loc_nuse_hurto", "localidad", "NUSE hurto a nivel localidad (unidad coherente con población)"),
        ("poblacion_localidad", "localidad", "Población SDS proyectada, suma del año"),
        ("loc_tasa_nuse_hurto_100k_hab", "localidad", "Hurto NUSE localidad / población × 100 mil"),
        ("upz_hurto_por_100k_val_estacion", "mixto", "UPZ hurto NUSE / validaciones de ESTA estación × 100 mil. Exposición de estación, outcome de UPZ."),
        ("dai_total", "localidad", "Suma de campos numéricos de Delito de Alto Impacto (ZIP portal)"),
        ("ir_total", "localidad", "Suma de campos numéricos de Incidente Reportado (ZIP portal)"),
        ("dist_cai_m", "estación", "Distancia haversine al CAI más cercano (m)"),
        ("n_cai_500m", "estación", f"CAI a ≤ {int(RADIO_CAI_M)} m"),
        ("indice_seguridad_nocturna_2019", "UPZ", "Índice SDMujer 2019 si el archivo trae el atributo"),
        ("unidad_ecologica", "meta", "Siempre UPZ o localidad para outcomes de seguridad, nunca estación"),
    ]
    return pd.DataFrame(filas, columns=["variable", "unidad", "definicion"])


def construir(skip_download: bool) -> dict:
    CACHE.mkdir(exist_ok=True)
    SALIDA.mkdir(exist_ok=True)
    notas = []
    festivos = festivos_colombia(ANIOS_TM)
    notas.append(f"festivos={len(festivos)} fechas")

    estaciones = cargar_estaciones()
    LOG.info("Estaciones: %s | sin UPZ: %s", len(estaciones), int(estaciones["sin_upz"].sum()))

    bloques = []
    for anio in ANIOS_TM:
        LOG.info("TM %s …", anio)
        horario = leer_tm_anual(anio)
        bloques.append(features_mensuales_estacion(horario, festivos))
        del horario
    tm = pd.concat(bloques, ignore_index=True)

    tm = tm.merge(
        estaciones,
        left_on="cod_estacion",
        right_on="num_est",
        how="left",
    )
    tm["match_dim"] = tm["nom_est"].notna()
    n_fuera = int(tm.loc[~tm["match_dim"], "cod_estacion"].nunique())
    notas.append(f"tm_filas_brutas={len(tm)} codigos_fuera_catalogo={n_fuera}")
    tm = tm.loc[tm["match_dim"]].copy()
    notas.append(f"tm_filas_catalogo={len(tm)} estaciones={tm['cod_estacion'].nunique()}")

    LOG.info("NUSE …")
    nuse_upz, nuse_loc, cruz_upz_loc = cargar_nuse()
    tm = tm.merge(cruz_upz_loc, on="upz_id", how="left")
    tm = tm.merge(nuse_upz, on=["anio", "mes", "upz_id"], how="left")
    tm = tm.merge(
        nuse_loc.drop(columns=["localidad"], errors="ignore"),
        on=["anio", "mes", "cod_localidad"],
        how="left",
        suffixes=("", "_locdup"),
    )

    extras_ok = {}
    if not skip_download:
        try:
            pob = poblacion_localidad()
            tm = unir_localidad(tm, pob, on_cod="cod_localidad" in pob.columns)
            extras_ok["poblacion"] = True
        except Exception as exc:
            extras_ok["poblacion"] = str(exc)
            LOG.warning("Población: %s", exc)

        try:
            dai = capa_localidad_zip("delito-de-alto-impacto-bogota-d-c", "dai", "dai")
            tm = unir_localidad(tm, dai, on_cod="cod_localidad" in dai.columns)
            extras_ok["delito_alto_impacto"] = True
        except Exception as exc:
            extras_ok["delito_alto_impacto"] = str(exc)
            LOG.warning("DAI: %s", exc)

        try:
            ir = capa_localidad_zip("incidente-reportado-bogota-d-c", "ir", "ir")
            tm = unir_localidad(tm, ir, on_cod="cod_localidad" in ir.columns)
            extras_ok["incidente_reportado"] = True
        except Exception as exc:
            extras_ok["incidente_reportado"] = str(exc)
            LOG.warning("IR: %s", exc)

        try:
            cai = distancia_cai(estaciones)
            tm = tm.merge(cai, on="num_est", how="left")
            extras_ok["cai"] = True
        except Exception as exc:
            extras_ok["cai"] = str(exc)
            LOG.warning("CAI: %s", exc)

        try:
            idx = indice_nocturno_localidad()
            if not idx.empty:
                tm["cod_localidad"] = (
                    tm["cod_localidad"].astype(str).str.replace(r"\.0$", "", regex=True).str.zfill(2)
                )
                tm = tm.merge(idx, on="cod_localidad", how="left")
            extras_ok["indice_nocturno"] = (not idx.empty)
        except Exception as exc:
            extras_ok["indice_nocturno"] = str(exc)
            LOG.warning("Índice nocturno: %s", exc)
    else:
        notas.append("skip_download=True")

    hab = tm.get("poblacion_localidad")
    if hab is not None:
        tm["loc_tasa_nuse_hurto_100k_hab"] = np.where(
            hab > 0, tm["loc_nuse_hurto"] / hab * 1e5, np.nan
        )
        tm["loc_tasa_nuse_violencia_100k_hab"] = np.where(
            hab > 0, tm["loc_nuse_violencia"] / hab * 1e5, np.nan
        )
        if "dai_total" in tm.columns:
            tm["loc_tasa_dai_100k_hab"] = np.where(hab > 0, tm["dai_total"] / hab * 1e5, np.nan)

    tm["upz_hurto_por_100k_val_estacion"] = np.where(
        tm["validaciones_mes"] > 0,
        tm["upz_nuse_hurto"] / tm["validaciones_mes"] * 1e5,
        np.nan,
    )
    soacha = tm["flag_soacha"].fillna(False).astype(bool)
    tm["unidad_ecologica"] = np.where(soacha, "sin_upz_bogota", "UPZ")
    cols_upz = [c for c in tm.columns if c.startswith("upz_nuse_")]
    cols_upz_panel = (
        ["upz_id", "cod_localidad", "localidad", "anio", "mes"]
        + cols_upz
        + [
            c for c in (
                "poblacion_localidad", "dai_total", "dai_hurto_personas",
                "ir_total", "indice_seguridad_nocturna_2019",
            )
            if c in tm.columns
        ]
    )
    panel_upz = (
        tm.loc[~soacha, cols_upz_panel]
        .drop_duplicates(["upz_id", "anio", "mes"])
    )

    orden = [
        "cod_estacion", "nom_est", "upz_id", "cod_localidad", "localidad",
        "anio", "mes", "flag_soacha", "sin_upz", "mes_incompleto", "n_dias_con_dato",
        "validaciones_mes", "salidas_mes", "flujo_total_mes",
        "pct_flujo_nocturno", "pct_flujo_finde", "pct_flujo_pico_am", "pct_flujo_pico_pm",
        "concentracion_horaria_gini",
        "upz_nuse_hurto", "upz_nuse_violencia", "upz_nuse_orden_publico", "upz_nuse_otros", "upz_nuse_total",
        "loc_nuse_hurto", "loc_nuse_violencia", "loc_nuse_orden_publico", "loc_nuse_total",
        "poblacion_localidad", "loc_tasa_nuse_hurto_100k_hab", "loc_tasa_nuse_violencia_100k_hab",
        "upz_hurto_por_100k_val_estacion",
        "dai_total", "dai_hurto_personas", "ir_total", "loc_tasa_dai_100k_hab",
        "dist_cai_m", "n_cai_500m", "indice_seguridad_nocturna_2019",
        "num_acc", "acc_puent", "longitud", "latitud", "unidad_ecologica", "match_dim",
    ]
    orden = [c for c in orden if c in tm.columns] + [c for c in tm.columns if c not in orden and c != "num_est"]
    tm = tm[orden].sort_values(["cod_estacion", "anio", "mes"])

    panel_path = SALIDA / "panel_estacion_mes.csv"
    upz_path = SALIDA / "panel_upz_mes.csv"
    dic_path = SALIDA / "diccionario_variables.csv"
    tm.to_csv(panel_path, index=False, encoding="utf-8-sig")
    panel_upz.to_csv(upz_path, index=False, encoding="utf-8-sig")
    diccionario().to_csv(dic_path, index=False, encoding="utf-8-sig")

    reporte = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "corte_nuse": CORTE_NUSE.isoformat(),
        "filas_estacion_mes": int(len(tm)),
        "estaciones": int(tm["cod_estacion"].nunique()),
        "upz": int(tm["upz_id"].nunique()),
        "pct_mes_incompleto": float(tm["mes_incompleto"].mean()),
        "pct_sin_nuse_upz": float(tm["upz_nuse_total"].isna().mean()) if "upz_nuse_total" in tm.columns else None,
        "pct_con_poblacion": float(tm["poblacion_localidad"].notna().mean()) if "poblacion_localidad" in tm.columns else 0.0,
        "pct_con_tasa_hurto": (
            float(tm["loc_tasa_nuse_hurto_100k_hab"].notna().mean())
            if "loc_tasa_nuse_hurto_100k_hab" in tm.columns else 0.0
        ),
        "solape_pct_nocturno_pico_pm": 0.0,
        "extras": extras_ok,
        "notas": notas + [
            "Outcomes de seguridad viven en upz_* o loc_*. No interpretarlos como hechos en la estación.",
            "Ventanas: nocturno 19-05, pico AM 06-08, pico PM 17-18 (disjuntas).",
            "Gini calculado en horas 04-23.",
            "Fuentes extra: ZIP/CSV de datosabiertos.bogota.gov.co. Nada de oaiee.scj.gov.co.",
            "DAI/IR son anuales a localidad (mismo total copiado a cada mes del año). No interpretarlos como mensuales.",
            "Población SDS: suma sexo×edad por localidad-año; se excluye código 00 (total Bogotá).",
        ],
        "archivos": {
            "panel_estacion_mes": str(panel_path),
            "panel_upz_mes": str(upz_path),
            "diccionario": str(dic_path),
        },
    }
    rep_path = SALIDA / "reporte_integracion.json"
    rep_path.write_text(json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8")
    LOG.info("Listo: %s", panel_path)
    return reporte


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Panel TM × NUSE × datos abiertos")
    parser.add_argument("--skip-download", action="store_true")
    args = parser.parse_args()
    reporte = construir(skip_download=args.skip_download)
    print(json.dumps(reporte, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
