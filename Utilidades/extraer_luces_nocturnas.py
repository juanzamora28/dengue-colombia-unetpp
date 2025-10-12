# -*- coding: utf-8 -*-
"""
VIIRS-DNB mensual → Semanal sin NaN para Colombia (2018–2019)
Fuente GEE: NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG (band: 'avg_rad', nW/cm^2/sr)
- Cada semana toma la imagen del mes correspondiente (step temporal).
- Salida: float32 (radiancia original), alineado EXACTO a grilla Incidencia 1024×1024.
- Sin NaN dentro de Colombia: unmask(0) + máscara binaria Colombia en la proyección objetivo.
"""

import ee, datetime, time
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# ─────────────────────────────────────────────────────────────────────────────
# Parámetros
# ─────────────────────────────────────────────────────────────────────────────
START_DATE   = '2018-01-01'
END_DATE     = '2019-12-31'

DRIVE_FOLDER     = 'NTL_WEEKLY_CO_2018_2019_VIIRS_VCMSLCFG_F32'
FILENAME_PREFIX  = 'NTL_WEEKLY_CO_'

POLL_SEC      = 20
DRY_RUN_N     = None  # ej. 6 para probar

# Grilla objetivo (tu grilla de Incidencia 1024×1024)
TARGET_CRS        = 'EPSG:4326'
TARGET_WIDTH      = 1024
TARGET_HEIGHT     = 1024
TARGET_TRANSFORM  = [0.0166015625, 0.0, -82.0, 0.0, -0.0166015625, 12.5]
TARGET_BBOX       = [-82.0, -4.5, -65.0, 12.5]
TARGET_REGION     = ee.Geometry.Rectangle(TARGET_BBOX, proj=TARGET_CRS, geodesic=False)

# Colección VIIRS mensual (promedios mensuales stray-light corrected)
C_VIIRS = ee.ImageCollection('NOAA/VIIRS/DNB/MONTHLY_V1/VCMSLCFG')

# ─────────────────────────────────────────────────────────────────────────────
# Geometría de Colombia y máscara binaria en la proyección objetivo
# ─────────────────────────────────────────────────────────────────────────────
GAUL_L0_ID   = 'FAO/GAUL/2015/level0'
GAUL_COUNTRY = 'Colombia'
countries = ee.FeatureCollection(GAUL_L0_ID)
co_raw = countries.filter(ee.Filter.eq('ADM0_NAME', GAUL_COUNTRY)).geometry()
# simplifica y repara pequeñas auto-intersecciones
co = co_raw.simplify(1000).buffer(0, 100)

# Máscara binaria 1=Colombia alineada a la grilla objetivo
co_mask_target = ee.Image.constant(1).clip(co).reproject(
    crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM
)

# ─────────────────────────────────────────────────────────────────────────────
# Utilidades temporales
# ─────────────────────────────────────────────────────────────────────────────
def dates_weekly(start_str, end_str, step_days=7):
    start = datetime.date.fromisoformat(start_str)
    end   = datetime.date.fromisoformat(end_str)
    out = []
    cur = start
    delta = datetime.timedelta(days=step_days)
    while cur <= end:
        nxt = cur + delta
        out.append((cur.isoformat(), min(nxt, end + datetime.timedelta(days=1)).isoformat()))
        cur = nxt
    return out

def month_bounds(date_iso):
    d = datetime.date.fromisoformat(date_iso)
    first = d.replace(day=1)
    if d.month == 12:
        nxt_first = d.replace(year=d.year+1, month=1, day=1)
    else:
        nxt_first = d.replace(month=d.month+1, day=1)
    return first.isoformat(), nxt_first.isoformat()

# ─────────────────────────────────────────────────────────────────────────────
# Construcción semanal (step: imagen mensual del mes que contiene la semana)
# ─────────────────────────────────────────────────────────────────────────────
def get_monthly_image_for(date_iso):
    """Devuelve la imagen mensual (avg_rad) que corresponde al mes de date_iso."""
    mstart, mend = month_bounds(date_iso)
    col = (C_VIIRS
           .filterDate(mstart, mend)
           .select(['avg_rad']))  # banda de radiancia (nW/cm2/sr)
    # en algunos meses hay 1 imagen; si hay >1, usa mean (debería ser 1)
    img = col.mean()
    # Asegura máscara y proyección objetivo; unmask(0) elimina NaN en Colombia
    img = (img
           .clip(co)                       # recorta a Colombia geométricamente
           .unmask(0)                      # RELLENA NaN a 0 (SOLO donde no hay dato)
           .updateMask(co_mask_target)     # asegura 1 dentro de Colombia, 0 fuera
           .resample('bilinear')           # continua
           .reproject(crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM))
    return img.rename('NTL_week_nW')

def weekly_image(ws_iso, we_iso):
    """Imagen semanal = imagen del mes de la fecha de inicio ws_iso."""
    img_m = get_monthly_image_for(ws_iso)
    # metadatos semanales
    return (img_m
            .set({'system:time_start': ee.Date(ws_iso).millis(),
                  'week_start': ws_iso,
                  'week_end': we_iso}))

# ─────────────────────────────────────────────────────────────────────────────
# Estadísticas y cobertura (para logging)
# ─────────────────────────────────────────────────────────────────────────────
def week_stats(img_real, scale_for_stats=5000):
    stat = img_real.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=co, scale=scale_for_stats, maxPixels=1_000_000_000,
        bestEffort=True, tileScale=4
    ).getInfo() or {}
    return {
        "min": float(stat.get('NTL_week_nW_min', float('nan'))),
        "mean": float(stat.get('NTL_week_nW_mean', float('nan'))),
        "max": float(stat.get('NTL_week_nW_max', float('nan')))
    }

def mask_coverage(img_real, scale_for_stats=20000):
    # como la máscara final es co_mask_target, la cobertura debe ser ≈1.0
    cov = img_real.updateMask(co_mask_target).gt(-1e9).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=co, scale=scale_for_stats,
        maxPixels=1_000_000_000, bestEffort=True, tileScale=4
    ).get('NTL_week_nW')
    try:
        return float(cov.getInfo() or 0.0)
    except Exception:
        return float('nan')

# ─────────────────────────────────────────────────────────────────────────────
# Export a Drive — Float32, proyección exacta
# ─────────────────────────────────────────────────────────────────────────────
def build_task(wimg_f32, name):
    return ee.batch.Export.image.toDrive(
        image         = wimg_f32.float(),
        description   = name,
        folder        = DRIVE_FOLDER,
        fileNamePrefix= name,
        region        = TARGET_REGION,
        crs           = TARGET_CRS,
        crsTransform  = TARGET_TRANSFORM,
        dimensions    = f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        maxPixels     = 1_000_000_000,
        fileFormat    = 'GeoTIFF',
        formatOptions = {'cloudOptimized': True}
    )

def wait_tasks(tasks, poll_sec=POLL_SEC):
    done = False
    seen = set()
    while not done:
        time.sleep(poll_sec)
        states = {}
        for t in tasks:
            st = t.status()
            state = st.get('state', 'NA')
            states[state] = states.get(state, 0) + 1
            tid = st.get('id', '')
            desc = st.get('description', '')
            if tid not in seen and state in ('FAILED', 'COMPLETED', 'CANCELLED'):
                seen.add(tid)
                print(f"[TASK] {state:10s} | {desc} | id={tid} | {st.get('error_message','')}")
        print("[TAREAS] " + "  ".join(f"{k}:{v}" for k,v in states.items()))
        done = all(s in ('COMPLETED','FAILED','CANCELLED') for s in states.keys())

# ─────────────────────────────────────────────────────────────────────────────
# Lote principal
# ─────────────────────────────────────────────────────────────────────────────
weeks = dates_weekly(START_DATE, END_DATE, step_days=7)
if DRY_RUN_N is not None:
    weeks = weeks[:DRY_RUN_N]

print(f"Semanas a exportar: {len(weeks)}")
print(f"[GRID OBJETIVO] {TARGET_CRS}  {TARGET_WIDTH}×{TARGET_HEIGHT}  transform={TARGET_TRANSFORM}  bbox={TARGET_BBOX}")

tasks = []
for (ws, we) in weeks:
    d = datetime.date.fromisoformat(ws)
    iso_year, iso_week, _ = d.isocalendar()
    name = f"{FILENAME_PREFIX}{iso_year:04d}_W{iso_week:02d}"

    wimg = weekly_image(ws, we)
    stats = week_stats(wimg)
    cov   = mask_coverage(wimg)

    print(f"[EXPORT] {name}  semana=[{ws}..{we})  "
          f"min={stats['min']:.4f}  mean={stats['mean']:.4f}  max={stats['max']:.4f}  "
          f"mask_coverage_in_CO={cov:.4f}")

    try:
        task = build_task(wimg, name)
        task.start()
        tasks.append(task)
    except Exception as e:
        print(f"[ERROR start] {name}: {e}")

print("→ Tareas enviadas. Monitor…")
wait_tasks(tasks, poll_sec=POLL_SEC)
print("Hecho.")
