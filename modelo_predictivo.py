"""
Pronóstico de demanda NUSE a nivel UPZ–mes.

No predice hurto en la estación. Predice llamados al 123 de la UPZ
en el mes t, usando solo información del mes t-1 (NUSE y perfil TM).

Modelos tradicionales: ridge y bosque.
Redes: MLP directo y red residual (aprende la corrección sobre el mes previo).

Línea base: persistencia (el valor del mes anterior).
Entrenamiento: 2023–2025. Prueba: enero–junio 2026.

Uso:
  python modelo_predictivo.py
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

BASE = Path(__file__).resolve().parent
PANEL = BASE / "salidas_analisis" / "panel_upz_mes_agregado.csv"
NOMBRES = BASE / "salidas_analisis" / "nombres_upz.csv"
SALIDA = BASE / "salidas_modelo"

TARGETS = {
    "hurto": "upz_nuse_hurto",
    "violencia": "upz_nuse_violencia",
}

TM_LAG = (
    "n_estaciones",
    "log_flujo_upz",
    "pct_nocturno_pond",
    "pct_finde_pond",
    "pct_pico_am_pond",
    "pct_pico_pm_pond",
    "gini_pond",
    "dist_cai_mediana_m",
)

LOG = logging.getLogger("modelo")
torch.manual_seed(42)
np.random.seed(42)


class MLPResidual(nn.Module):
    """Red chica: predice la corrección sobre el mes previo, no el nivel."""

    def __init__(self, d_in: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, 96),
            nn.GELU(),
            nn.Dropout(0.12),
            nn.Linear(96, 48),
            nn.GELU(),
            nn.Dropout(0.08),
            nn.Linear(48, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _loader3(x, y, lag, batch=48, shuffle=True):
    ds = TensorDataset(
        torch.tensor(x, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32),
        torch.tensor(lag, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch, shuffle=shuffle)


def _val_temporal(train: pd.DataFrame) -> tuple[pd.Index, pd.Index]:
    """Validación = nov–dic 2025 (corte temporal, no aleatorio)."""
    va = train.loc[(train["anio"] == 2025) & (train["mes"] >= 11)]
    if len(va) < 80:
        va = train.loc[train["anio"] == 2025]
    tr = train.loc[~train.index.isin(va.index)]
    if tr.empty or va.empty:
        return train.index, train.index
    return tr.index, va.index


def entrenar_red_residual(
    x_tr: np.ndarray,
    y_tr: np.ndarray,
    lag_tr: np.ndarray,
    x_va: np.ndarray,
    y_va: np.ndarray,
    lag_va: np.ndarray,
    x_te: np.ndarray,
    lag_te: np.ndarray,
) -> np.ndarray:
    """Optimiza el nivel: pred = clip(mes_previo + red(x), 0)."""
    model = MLPResidual(x_tr.shape[1])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=8e-5)
    loss_fn = nn.SmoothL1Loss()
    best_state, best_va, espera = None, np.inf, 0
    tr_loader = _loader3(x_tr, y_tr, lag_tr)
    xva = torch.tensor(x_va, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.float32)
    lva = torch.tensor(lag_va, dtype=torch.float32)

    def _nivel(mod, xb, lag):
        return torch.clamp(lag + mod(xb), min=0.0)

    for _ in range(280):
        model.train()
        for xb, yb, lb in tr_loader:
            opt.zero_grad()
            loss_fn(_nivel(model, xb, lb), yb).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            va = float(loss_fn(_nivel(model, xva, lva), yva).item())
        if va + 1e-6 < best_va:
            best_va, best_state, espera = va, {k: v.cpu().clone() for k, v in model.state_dict().items()}, 0
        else:
            espera += 1
            if espera >= 22:
                break
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        xte = torch.tensor(x_te, dtype=torch.float32)
        lte = torch.tensor(lag_te, dtype=torch.float32)
        hat = _nivel(model, xte, lte).numpy()
    return np.clip(hat, 0, None)


def nom_propio(texto: str) -> str:
    t = str(texto).strip().title()
    for malo, bueno in {
        "De": "de", "Del": "del", "La": "la", "Las": "las",
        "Los": "los", "El": "el", "Y": "y",
    }.items():
        t = t.replace(f" {malo} ", f" {bueno} ")
    t = t[:1].upper() + t[1:] if t else t
    acentos = {
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
    return acentos.get(t, t)


def cargar_panel() -> pd.DataFrame:
    df = pd.read_csv(PANEL)
    df["upz_id"] = df["upz_id"].astype(str).str.strip()
    if NOMBRES.exists():
        nom = pd.read_csv(NOMBRES)
        nom["upz_id"] = nom["upz_id"].astype(str).str.strip()
        nom["upz_nombre"] = nom["upz_nombre"].map(nom_propio)
        df = df.merge(nom, on="upz_id", how="left")
    df["etiqueta"] = df.get("upz_nombre", pd.Series(df["upz_id"])).fillna(df["upz_id"])
    df["localidad"] = df["localidad"].map(nom_propio)
    df["t"] = df["anio"].astype(int) * 12 + df["mes"].astype(int)
    return df.sort_values(["upz_id", "t"]).reset_index(drop=True)


def con_rezagos(df: pd.DataFrame, ycol: str) -> pd.DataFrame:
    out = df.copy()
    out["y"] = out[ycol]
    g = out.groupby("upz_id", group_keys=False)
    out["y_lag1"] = g["y"].shift(1)
    out["y_rm3"] = g["y"].transform(lambda s: s.shift(1).rolling(3, min_periods=2).mean())
    for c in TM_LAG:
        if c in out.columns:
            out[f"{c}_lag1"] = g[c].shift(1)
    out["mes_sin"] = np.sin(2 * np.pi * out["mes"] / 12)
    out["mes_cos"] = np.cos(2 * np.pi * out["mes"] / 12)
    return out


def feats_pronostico(df: pd.DataFrame) -> list[str]:
    cols = ["y_lag1", "y_rm3", "mes_sin", "mes_cos"]
    cols += [f"{c}_lag1" for c in TM_LAG if f"{c}_lag1" in df.columns]
    return cols


def split_temporal(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    d = df.dropna(subset=cols + ["y"]).copy()
    train = d.loc[d["anio"] <= 2025]
    test = d.loc[d["anio"] == 2026]
    return train, test


def metricas(y_true, y_pred, etiqueta: str, n: int) -> dict:
    mae = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {
        "modelo": etiqueta,
        "n_test": int(n),
        "mae": mae,
        "rmse": rmse,
        "r2": float(r2_score(y_true, y_pred)),
    }


def metricas_grupo(df: pd.DataFrame, ycol: str, predcol: str, modelo: str) -> dict:
    n = int(len(df))
    if n < 3:
        return None
    return metricas(df[ycol], df[predcol], modelo, n)


def ajuste_por_localidad(pred: pd.DataFrame, cols_pred: dict[str, str]) -> pd.DataFrame:
    filas = []
    for loc, g in pred.groupby("localidad", dropna=False):
        for modelo, col in cols_pred.items():
            fila = metricas_grupo(g, "observado", col, modelo)
            if fila is None:
                continue
            fila["localidad"] = loc if pd.notna(loc) else "(sin localidad)"
            filas.append(fila)
    if not filas:
        return pd.DataFrame()
    tab = pd.DataFrame(filas)
    naive = tab.loc[tab["modelo"] == "persistencia"].set_index("localidad")["mae"]
    tab["skill_vs_persistencia"] = tab.apply(
        lambda r: np.nan if r["localidad"] not in naive.index or naive[r["localidad"]] == 0
        else 1 - r["mae"] / naive[r["localidad"]],
        axis=1,
    )
    return tab.sort_values(["localidad", "mae"])


def entrenar_familia(df: pd.DataFrame, nombre: str, ycol: str) -> dict:
    panel = con_rezagos(df, ycol)
    cols = feats_pronostico(panel)
    train, test = split_temporal(panel, cols)
    if train.empty or test.empty:
        raise RuntimeError(f"Sin datos train/test para {nombre}")

    x_tr, y_tr = train[cols], train["y"]
    x_te, y_te = test[cols], test["y"]
    naive = test["y_lag1"].to_numpy(float)

    ridge = Pipeline([("esc", StandardScaler()), ("mod", Ridge(alpha=2.0))])
    ridge.fit(x_tr, y_tr)

    rf = RandomForestRegressor(
        n_estimators=280, max_depth=8, min_samples_leaf=8,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf.fit(x_tr, y_tr)

    lag_tr = train["y_lag1"].to_numpy(float)
    mlp = Pipeline([
        ("esc", StandardScaler()),
        ("mod", MLPRegressor(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=1e-3,
            learning_rate_init=1e-3,
            max_iter=700,
            early_stopping=True,
            validation_fraction=0.12,
            n_iter_no_change=25,
            random_state=42,
        )),
    ])
    mlp.fit(x_tr, (y_tr.to_numpy(float) - lag_tr))

    tr_i, va_i = _val_temporal(train)
    esc = StandardScaler().fit(train.loc[tr_i, cols])
    pred_red = entrenar_red_residual(
        esc.transform(train.loc[tr_i, cols]),
        train.loc[tr_i, "y"].to_numpy(float),
        train.loc[tr_i, "y_lag1"].to_numpy(float),
        esc.transform(train.loc[va_i, cols]),
        train.loc[va_i, "y"].to_numpy(float),
        train.loc[va_i, "y_lag1"].to_numpy(float),
        esc.transform(x_te),
        naive,
    )

    pred_ridge = ridge.predict(x_te)
    pred_rf = rf.predict(x_te)
    pred_mlp = np.clip(naive + mlp.predict(x_te), 0, None)

    filas = [
        metricas(y_te, naive, "persistencia", len(test)),
        metricas(y_te, pred_ridge, "ridge", len(test)),
        metricas(y_te, pred_rf, "bosque", len(test)),
        metricas(y_te, pred_mlp, "mlp", len(test)),
        metricas(y_te, pred_red, "red_residual", len(test)),
    ]
    tab = pd.DataFrame(filas)
    mae_naive = tab.loc[tab["modelo"] == "persistencia", "mae"].iloc[0]
    tab["skill_vs_persistencia"] = 1 - tab["mae"] / mae_naive
    tab.insert(0, "familia", nombre)
    tab["familia_modelo"] = np.where(
        tab["modelo"].isin(["ridge", "bosque"]), "tradicional",
        np.where(tab["modelo"].isin(["mlp", "red_residual"]), "red_neuronal", "linea_base"),
    )

    nn_nom = tab.loc[tab["modelo"].isin(["mlp", "red_residual"])].sort_values("mae").iloc[0]["modelo"]
    trad_nom = tab.loc[tab["modelo"].isin(["ridge", "bosque"])].sort_values("mae").iloc[0]["modelo"]
    mapa_nn = {"mlp": pred_mlp, "red_residual": pred_red}
    mapa_tr = {"ridge": pred_ridge, "bosque": pred_rf}
    pred_nn = mapa_nn[nn_nom]
    pred_tr = mapa_tr[trad_nom]

    perm = permutation_importance(
        rf, x_te, y_te, n_repeats=8, random_state=42, scoring="neg_mean_absolute_error"
    )
    imp = pd.DataFrame(
        {
            "familia": nombre,
            "variable": cols,
            "importancia_mae": perm.importances_mean,
            "sd": perm.importances_std,
        }
    ).sort_values("importancia_mae", ascending=False)

    pred = test[["upz_id", "etiqueta", "localidad", "anio", "mes", "periodo", "y", "y_lag1"]].copy()
    pred["pred_persistencia"] = naive
    pred["pred_ridge"] = pred_ridge
    pred["pred_bosque"] = pred_rf
    pred["pred_mlp"] = pred_mlp
    pred["pred_red"] = pred_red
    pred["pred_tradicional"] = pred_tr
    pred["pred_red_elegida"] = pred_nn
    pred["tradicional_elegido"] = trad_nom
    pred["red_elegida"] = nn_nom
    pred["familia"] = nombre
    pred["error_red"] = pred["pred_red_elegida"] - pred["y"]
    pred = pred.rename(columns={"y": "observado"})

    loc_tab = ajuste_por_localidad(
        pred,
        {
            "persistencia": "pred_persistencia",
            "bosque": "pred_bosque",
            "red_residual": "pred_red",
            "mlp": "pred_mlp",
        },
    )
    loc_tab.insert(0, "familia", nombre)

    joblib.dump(
        {"modelo": rf, "columnas": cols, "familia": nombre, "tipo": "bosque"},
        SALIDA / f"modelo_{nombre}_bosque.joblib",
    )
    joblib.dump(
        {"scaler": esc, "familia": nombre, "tipo": "red_residual", "columnas": cols},
        SALIDA / f"modelo_{nombre}_red.joblib",
    )

    skill_nn = float(tab.loc[tab["modelo"] == nn_nom, "skill_vs_persistencia"].iloc[0])
    skill_tr = float(tab.loc[tab["modelo"] == trad_nom, "skill_vs_persistencia"].iloc[0])
    frase = (
        f"{nombre.capitalize()}: tradicional={trad_nom} (skill {100*skill_tr:.1f}%). "
        f"Red={nn_nom} (skill {100*skill_nn:.1f}%)."
    )
    return {
        "metricas": tab,
        "localidad": loc_tab,
        "importancia": imp,
        "pred": pred,
        "frase": frase,
        "ganador": nn_nom,
        "skill": skill_nn,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    SALIDA.mkdir(exist_ok=True)
    df = cargar_panel()
    LOG.info("Panel %s filas, %s UPZ", len(df), df["upz_id"].nunique())

    metricas_all, loc_all, imp_all, pred_all = [], [], [], []
    resumen = {}
    for nombre, ycol in TARGETS.items():
        LOG.info("Familia %s …", nombre)
        res = entrenar_familia(df, nombre, ycol)
        metricas_all.append(res["metricas"])
        loc_all.append(res["localidad"])
        imp_all.append(res["importancia"])
        pred_all.append(res["pred"])
        resumen[nombre] = {
            "ganador": res["ganador"],
            "skill_vs_persistencia": res["skill"],
            "n_train": res["n_train"],
            "n_test": res["n_test"],
            "frase": res["frase"],
        }
        LOG.info("%s", res["frase"])

    met = pd.concat(metricas_all, ignore_index=True)
    loc = pd.concat(loc_all, ignore_index=True)
    imp = pd.concat(imp_all, ignore_index=True)
    pred = pd.concat(pred_all, ignore_index=True)
    met.to_csv(SALIDA / "metricas.csv", index=False, encoding="utf-8-sig")
    loc.to_csv(SALIDA / "metricas_localidad.csv", index=False, encoding="utf-8-sig")
    imp.to_csv(SALIDA / "importancia.csv", index=False, encoding="utf-8-sig")
    pred.to_csv(SALIDA / "predicciones_2026.csv", index=False, encoding="utf-8-sig")

    reporte = {
        "generado": datetime.now().isoformat(timespec="seconds"),
        "unidad": "UPZ-mes",
        "corte_train": "2023-2025",
        "corte_test": "2026-01 a 2026-06",
        "features": "rezagos t-1 de NUSE y del perfil TM; mes cíclico. Sin DAI/IR (anuales).",
        "aviso": "No predice hechos en la estación. Predice llamados al 123 de la UPZ.",
        "familias": resumen,
        "archivos": {
            "metricas": str(SALIDA / "metricas.csv"),
            "metricas_localidad": str(SALIDA / "metricas_localidad.csv"),
            "importancia": str(SALIDA / "importancia.csv"),
            "predicciones": str(SALIDA / "predicciones_2026.csv"),
        },
    }
    (SALIDA / "reporte_modelo.json").write_text(
        json.dumps(reporte, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(reporte, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
