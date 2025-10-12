# -*- coding: utf-8 -*-
"""
INS BES 2020–2024 — Verificación + Descarga de PDFs por semana
Estrategia:
  1) Intento directo con patrones oficiales (BibliotecaDigital / Buscador-Eventos).
  2) Fallback: búsqueda web (Bing) limitada a site:ins.gov.co, (año, semana).
  3) Validación: GET directo (sin HEAD), content-type pdf/octet-stream, tamaño ≥ 20KB.
  4) Auditoría CSV con URL final usado por semana.

Salidas:
  - BASE_OUT/pdf/YYYY.wWW.pdf
  - BASE_OUT/out/downloads_index_2020_2024.csv

Requisitos:
  pip install requests tqdm pandas beautifulsoup4 lxml
"""

import os, re, urllib.parse, time
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm
import pandas as pd
from datetime import date

# =========================
# PARÁMETROS
# =========================
BASE_OUT   = r"C:\Users\juanz\Desktop\incidencia_dengue_INS_2020_2024_ok"
PDF_DIR    = os.path.join(BASE_OUT, "pdf")
OUT_DIR    = os.path.join(BASE_OUT, "out")
os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

YEAR_START, YEAR_END = 2020, 2024
TIMEOUT_S  = 45
UA         = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) dengue-ins-scraper/4.0"

# Si ya conoces URLs exactas para semanas “raras”, ponlas aquí:
MANUAL_OVERRIDES = {
    # (2020, 33): "https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/2020_Boletin_epidemiologico_semana_33.pdf",
    # (2024, 52): "https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/2024_Boletin_epidemiologico_semana_52.pdf",
}

MIN_BYTES_OK = 20_000  # para evitar guardar HTML de error como .pdf

# =========================
# UTILIDADES
# =========================
def iso_weeks_in_year(y: int) -> int:
    dec28 = date(y, 12, 28)
    return dec28.isocalendar()[1]

def enc(u: str) -> str:
    return urllib.parse.quote(u, safe="/._-%?=&")

def candidate_urls(year: int, week: int):
    """Patrones más comunes 2020–2024 (observados y usados por INS)."""
    w2 = f"{week:02d}"
    cands = [
        # BibliotecaDigital (muy frecuente 2021+)
        f"https://www.ins.gov.co/BibliotecaDigital/{year}-boletin-epidemiologico-semana-{w2}.pdf",
        # Buscador-eventos (varias variantes reales)
        f"https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/{year}_Boletin_epidemiologico_semana_{w2}.pdf",
        f"https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/Boletin%20epidemiologico%20semana%20{w2}%20de%20{year}.pdf",
        # Con acento en Boletín (URL-encoded)
        f"https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/{year}_Bolet%C3%ADn_epidemiologico_semana_{w2}.pdf",
        f"https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/Bolet%C3%ADn%20epidemiologico%20semana%20{w2}%20de%20{year}.pdf",
        f"https://www.ins.gov.co/buscador-eventos/BoletinEpidemiologico/Bolet%C3%ADn_epidemiologico_semana_{w2}_{year}.pdf",
    ]
    return [enc(u) for u in cands]

def is_probably_pdf_url(u: str) -> bool:
    u = u.lower()
    return u.endswith(".pdf") and ("boletin" in u and "epidemiolog" in u)

def get_pdf(session: requests.Session, url: str, outpath: str):
    """GET directo con barra de bytes; valida tamaño mínimo."""
    try:
        r = session.get(url, timeout=TIMEOUT_S, stream=True, allow_redirects=True)
        code = r.status_code
        if code == 200 and url.lower().endswith(".pdf"):
            ctype = (r.headers.get("Content-Type") or "").lower()
            if ("pdf" in ctype) or ("octet-stream" in ctype) or (ctype == ""):
                total = int(r.headers.get('Content-Length', 0) or 0)
                with open(outpath, "wb") as f:
                    if total > 0:
                        pbar = tqdm(total=total, unit="B", unit_scale=True, leave=False, desc="  ↓ bytes")
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk: f.write(chunk); pbar.update(len(chunk))
                        pbar.close()
                    else:
                        for chunk in r.iter_content(chunk_size=1 << 16):
                            if chunk: f.write(chunk)
                if os.path.getsize(outpath) >= MIN_BYTES_OK:
                    return True, code, os.path.getsize(outpath)
                else:
                    try: os.remove(outpath)
                    except Exception: pass
                    return False, code, 0
        return False, code, 0
    except requests.RequestException as e:
        return False, getattr(e.response, "status_code", -1), 0

# =========================
# BÚSQUEDA WEB (Bing) — site:ins.gov.co
# =========================
BING = "https://www.bing.com/search"

def bing_search_pdf_candidates(session: requests.Session, year: int, week: int, max_pages: int = 2):
    """
    Busca en Bing: site:ins.gov.co PDF de "boletin epidemiologico semana {week} {year}".
    Devuelve lista de URLs .pdf candidatos (sin duplicados).
    """
    query_terms = [
        f"site:ins.gov.co {year} boletin epidemiologico semana {week:02d} filetype:pdf",
        f"site:ins.gov.co {year} boletin epidemiologico semana {week} filetype:pdf",
        f"site:ins.gov.co {year} boletín epidemiologico semana {week:02d} filetype:pdf",
    ]
    found = []
    seen = set()
    headers = {"User-Agent": UA}
    for q in query_terms:
        for page in range(max_pages):
            params = {"q": q, "first": str(1 + page*10)}
            try:
                r = session.get(BING, params=params, headers=headers, timeout=TIMEOUT_S)
            except requests.RequestException:
                continue
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")
            for a in soup.select("li.b_algo h2 a, a[href]"):
                href = a.get("href") or ""
                if "ins.gov.co" in href and is_probably_pdf_url(href):
                    u = enc(href)
                    if u not in seen:
                        seen.add(u)
                        found.append(u)
            time.sleep(0.2)
    return found

# =========================
# PIPELINE PRINCIPAL
# =========================
def download_verified_2020_2024():
    session = requests.Session()
    session.headers.update({"User-Agent": UA})
    audit_rows = []

    years = list(range(YEAR_START, YEAR_END + 1))
    year_bar = tqdm(years, desc="Año", ncols=100)

    for y in year_bar:
        max_w = iso_weeks_in_year(y)
        weeks = list(range(1, max_w + 1))
        week_bar = tqdm(weeks, desc=f"  Semanas {y} (hasta {max_w})", ncols=100, leave=False)

        for w in week_bar:
            outpdf = os.path.join(PDF_DIR, f"{y}.w{w:02d}.pdf")
            if os.path.exists(outpdf) and os.path.getsize(outpdf) >= MIN_BYTES_OK:
                audit_rows.append({"year": y, "week": w, "status": "cached", "source": "(cached)", "http": 200, "bytes": os.path.getsize(outpdf)})
                continue

            # 0) override manual
            if (y, w) in MANUAL_OVERRIDES:
                u = MANUAL_OVERRIDES[(y, w)]
                ok, code, nb = get_pdf(session, u, outpdf)
                audit_rows.append({"year": y, "week": w, "status": "downloaded" if ok else "override_fail", "source": u, "http": code, "bytes": nb})
                if ok:
                    continue

            # 1) patrones oficiales
            found = False
            for u in candidate_urls(y, w):
                ok, code, nb = get_pdf(session, u, outpdf)
                audit_rows.append({"year": y, "week": w, "status": "downloaded" if ok else "miss_try", "source": u, "http": code, "bytes": nb})
                if ok:
                    found = True
                    break
            if found:
                continue

            # 2) fallback: búsqueda Bing site:ins.gov.co
            pdfs = bing_search_pdf_candidates(session, y, w, max_pages=2)
            hit = False
            for u in pdfs:
                # extra verificación: que mencione año y semana en la URL
                if (str(y) in u) and ((f"_{w:02d}.pdf" in u) or (f"-{w:02d}.pdf" in u) or (f"%20{w:02d}.pdf" in u) or (f"_{w}.pdf" in u) or (f"-{w}.pdf" in u)):
                    ok, code, nb = get_pdf(session, u, outpdf)
                    audit_rows.append({"year": y, "week": w, "status": "downloaded" if ok else "bing_try", "source": u, "http": code, "bytes": nb})
                    if ok:
                        hit = True
                        break
            if hit:
                continue

            # 3) no encontrado
            audit_rows.append({"year": y, "week": w, "status": "missing", "source": "", "http": -1, "bytes": 0})

    # Auditoría
    audit = pd.DataFrame(audit_rows)
    audit_path = os.path.join(OUT_DIR, "downloads_index_2020_2024.csv")
    audit.to_csv(audit_path, index=False, encoding="utf-8")
    print("Auditoría →", audit_path)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    print("=== Verificación + Descarga BES (2020–2024) ===")
    download_verified_2020_2024()
    print("[Hecho]")
