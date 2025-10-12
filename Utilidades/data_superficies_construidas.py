# -*- coding: utf-8 -*-
# GAIA v10 → % construido por celda ~1 km para Colombia (2007–2019)
# Estrategia: densidad con reduceNeighborhood (1 km) en proyección nativa,
# y export a rejilla estable EPSG:4326 usando crsTransform (0.01° ≈ 1.11 km).

import ee, time

# ── Init ─────────────────────────────────────────────────────
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

DATASET_ID   = "Tsinghua/FROM-GLC/GAIA/v10"  # banda: change_year_index (0=nunca; 34..1=1985..2018)
COUNTRY_NAME = "Colombia"
START_YEAR   = 2007
END_YEAR     = 2019                          # 2019 = mismo umbral que 2018
DRIVE_FOLDER = "GAIA_BUILT_PCT_1km_CO_2007_2019"

# Rejilla de salida: EPSG:4326 con píxel cuadrado de 0.01° (~1.11 km en ecuador)
CRS_OUT      = "EPSG:4326"
DEG_STEP     = 0.01  # tamaño de píxel en grados
# Nota: exportamos con 'crsTransform' para fijar grid y evitar reproyecciones pesadas.

# Geometría país (polígono finito)
countries    = ee.FeatureCollection("FAO/GAUL_SIMPLIFIED_500m/2015/level0")
country      = countries.filter(ee.Filter.eq("ADM0_NAME", COUNTRY_NAME)).geometry()
country_rect = country.bounds(1)

gaia   = ee.Image(DATASET_ID)
yr_idx = gaia.select("change_year_index")

# ── 0/1 “ya construido en el año y” ──────────────────────────
def built01_for_year(y: int) -> ee.Image:
    y_eff = min(y, 2018)                 # GAIA llega a 2018
    thr   = 2019 - y_eff                 # 1985..2018 → 34..1
    return yr_idx.gt(0).And(yr_idx.gte(thr)).toFloat()

# ── Densidad a ~1 km sin reproject pesado ────────────────────
# Usamos una media en vecindad cuadrada de 500 m (lado → 1 km).
def percent_built_1km(y: int) -> ee.Image:
    built01 = built01_for_year(y)
    dens = built01.reduceNeighborhood(
        reducer      = ee.Reducer.mean(),
        kernel       = ee.Kernel.square(500, "meters", True),  # 1 km (lado)
        skipMasked   = True
    )
    built_pct = dens.multiply(100.0).rename(f"built_pct_{y}").clip(country).unmask(0)
    return built_pct

# ── Resumen rápido (km² y % país) con backups ────────────────
def summary_area_km2(y: int) -> dict:
    # sumamos área de pixeles con built01==1 (aprox)
    built01 = built01_for_year(y)
    built_m2_img = built01.multiply(ee.Image.pixelArea())
    # intento 1: 2 km
    try:
        red = built_m2_img.reduceRegion(
            reducer   = ee.Reducer.sum(),
            geometry  = country,
            scale     = 2000,
            tileScale = 4,
            bestEffort=True,
            maxPixels = 1_000_000_000
        )
        built_m2 = ee.Number(red.get(built_m2_img.bandNames().get(0)))
        built_km2 = built_m2.divide(1e6)
        pct = built_m2.divide(country.area(1)).multiply(100.0)
        return {"built_km2": built_km2, "pct_country": pct, "method": "reduce_2km"}
    except Exception:
        pass
    # intento 2: 5 km
    red = built_m2_img.reduceRegion(
        reducer   = ee.Reducer.sum(),
        geometry  = country,
        scale     = 5000,
        tileScale = 8,
        bestEffort=True,
        maxPixels = 1_000_000_000
    )
    built_m2 = ee.Number(red.get(built_m2_img.bandNames().get(0)))
    built_km2 = built_m2.divide(1e6)
    pct = built_m2.divide(country.area(1)).multiply(100.0)
    return {"built_km2": built_km2, "pct_country": pct, "method": "reduce_5km"}

# ── Export por año con crsTransform (0.01°) ──────────────────
def export_year(y: int):
    img = percent_built_1km(y)
    # Construimos la transformación affine del grid EPSG:4326 (0.01°)
    # crsTransform = [x_res, 0, x_min, 0, -y_res, y_max]
    # Tomamos la bbox del país en EPSG:4326 para anclar la rejilla.
    bbox = country.bounds().coordinates().get(0).getInfo()
    xs = [pt[0] for pt in bbox]
    ys = [pt[1] for pt in bbox]
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    crs_transform = [DEG_STEP, 0, xmin, 0, -DEG_STEP, ymax]

    desc = f"GAIA_BUILT_PCT_1km_CO_{y}"
    task = ee.batch.Export.image.toDrive(
        image          = img,
        description    = desc,
        folder         = DRIVE_FOLDER,
        fileNamePrefix = desc,
        region         = country_rect,        # polígono finito
        crs            = CRS_OUT,
        crsTransform   = crs_transform,       # fija grilla → sin reproyecciones pesadas
        fileFormat     = "GeoTIFF",
        formatOptions  = {"cloudOptimized": True},
        maxPixels      = 1_000_000_000_000
    )
    task.start()
    return task

# ── Run ──────────────────────────────────────────────────────
print("[INFO] Encolando GAIA % construido a ~1 km (kernel 1 km, grid 0.01°)…")
tasks = []
for y in range(START_YEAR, END_YEAR + 1):
    try:
        sm = ee.Dictionary(summary_area_km2(y)).getInfo()
        print(f"  {y}: built≈{sm['built_km2']:.1f} km²  ({sm['pct_country']:.2f}% del país)  [{sm['method']}]")
    except Exception as e:
        print(f"  {y}: resumen no disponible ({str(e)[:80]}…)")

    tasks.append(export_year(y))

# Espera opcional
while True:
    states = [t.status().get("state", "UNKNOWN") for t in tasks]
    print("Estados:", {s: states.count(s) for s in set(states)})
    if all(s in ("COMPLETED","FAILED","CANCELLED") for s in states):
        break
    time.sleep(30)

# Detalles de errores si los hay
for t in tasks:
    st = t.status()
    if st.get("state") != "COMPLETED":
        print(f"[{st.get('description')}] {st.get('state')} :: {st.get('error_message')}")

print("\n[OK] 1 GeoTIFF por año en Google Drive /", DRIVE_FOLDER)
print("Banda: built_pct_YYYY (Float32, 0..100 = % construido por celda de ~1 km).")
