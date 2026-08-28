"""Genera un HTML autónomo para compartir (requisito 3.3: exportable)."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from dashboard.app import construir_informe_html  # noqa: E402

if __name__ == "__main__":
    for fam in ("hurto", "violencia"):
        p = construir_informe_html(fam, "todos")
        print(p)
