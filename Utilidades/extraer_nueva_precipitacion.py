# -*- coding: utf-8 -*-
"""
Precipitación semanal (CHIRPS) para Colombia (2007–2019) – EXPORT 1024×1024 ALINEADO A INCIDENCIA
- Fuente: UCSB-CHG/CHIRPS/DAILY  (mm/día, ~0.05°)
- Composición: SUMA semanal (mm) dentro de Colombia
- Salida: uint8 0..255 con mapeo lineal [PREC_MIN_MM, PREC_MAX_MM] (clamp)
- Exporta EXACTAMENTE en la grilla de Incidencia: 1024×1024, EPSG:4326,
  transform = [ 0.0166015625, 0, -82.0,  0, -0.0166015625, 12.5 ]
  bbox = [-82.0, -4.5, -65.0, 12.5]
- Orden de exportación: INVERSO (2019→2007)
"""

import ee, datetime, time

# ──────────────────────────────────────────────────────────────────────────────
# 0) Auth / Init
# ──────────────────────────────────────────────────────────────────────────────
try:
    ee.Initialize()
except Exception:
    ee.Authenticate()
    ee.Initialize()

# ──────────────────────────────────────────────────────────────────────────────
# 1) Parámetros
# ──────────────────────────────────────────────────────────────────────────────
START_DATE   = '2007-01-01'
END_DATE     = '2019-12-31'

# Mapeo a 8 bits: 0..PREC_MAX_MM (mm/semana) → 0..255
# 450mm/sem suele evitar saturación en Colombia; ajusta si lo deseas.
PREC_MIN_MM  = 0.0
PREC_MAX_MM  = 350.0

DRIVE_FOLDER     = 'PRECIP_SEMANAL_CO_2007_2019_350_ALIGNED_1024'
FILENAME_PREFIX  = 'PRECIP_WEEKLY_CO_'

# Polling de tareas
POLL_SEC      = 20
DRY_RUN_N     = None                 # p.ej. 8 para probar 8 semanas

# Región: Colombia (GAUL Level-0) + “fix” geométrico
GAUL_L0_ID   = 'FAO/GAUL/2015/level0'
GAUL_COUNTRY = 'Colombia'

# ── GRILLA DE INCIDENCIA (OBJETIVO 1024×1024) ────────────────────────────────
TARGET_CRS        = 'EPSG:4326'
TARGET_WIDTH      = 1024
TARGET_HEIGHT     = 1024
# Affine (a,b,c,d,e,f): [sx, 0, x0, 0, -sy, y0]
TARGET_TRANSFORM  = [0.0166015625, 0.0, -82.0, 0.0, -0.0166015625, 12.5]
# bbox coherente con transform y dims
TARGET_BBOX       = [-82.0, -4.5, -65.0, 12.5]   # xmin, ymin, xmax, ymax
TARGET_REGION     = ee.Geometry.Rectangle(TARGET_BBOX, proj=TARGET_CRS, geodesic=False)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Geometría Colombia
# ──────────────────────────────────────────────────────────────────────────────
countries = ee.FeatureCollection(GAUL_L0_ID)
co_raw = countries.filter(ee.Filter.eq('ADM0_NAME', GAUL_COUNTRY)).geometry()
co = co_raw.simplify(1000).buffer(0, 100)  # simplifica 1 km y repara (maxError=100 m)
# Máscara 1=Colombia reproyectada a la grilla objetivo (para que el borde sea nítido)
co_mask_target = ee.Image.constant(1).clip(co).reproject(
    crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) CHIRPS diario preprocesado (mm/día, máscara Colombia)
# ──────────────────────────────────────────────────────────────────────────────
C_CHIRPS = (ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
            .filterDate(START_DATE, END_DATE)
            .filterBounds(co))

def prep_chirps_daily(img):
    p = img.select('precipitation').rename('P_mm_day').max(0)  # sin negativos
    p = p.updateMask(ee.Image.constant(1).clip(co))
    # Conserva timestamp
    return ee.Image(p.copyProperties(img, ['system:time_start']))

chirps_day = C_CHIRPS.map(prep_chirps_daily)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Fechas/semanas y composiciones
# ──────────────────────────────────────────────────────────────────────────────
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

def weekly_sum_mm(start_iso, end_iso):
    col = chirps_day.filterDate(start_iso, end_iso).filterBounds(co)
    summed = col.sum().rename('P_mm_week')
    cnt    = col.count().rename('cnt')
    img = summed.updateMask(cnt.gt(0)).updateMask(ee.Image.constant(1).clip(co))
    # Fija proyección a la grilla 1024×1024 con bilinear (continua)
    img = img.resample('bilinear').reproject(
        crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM
    )
    # Metadatos
    img = img.set({
        'system:time_start': ee.Date(start_iso).millis(),
        'week_start': start_iso,
        'week_end': end_iso
    })
    return img

def scale_to_255_precip(img_mm):
    p = img_mm.select('P_mm_week')
    p_clamped = p.max(PREC_MIN_MM).min(PREC_MAX_MM)
    scaled = p_clamped.subtract(PREC_MIN_MM).divide(PREC_MAX_MM - PREC_MIN_MM).multiply(255.0)
    # Aplica máscara de Colombia ya en la proyección objetivo
    return scaled.rename('PRECIP_8b').toUint8().updateMask(co_mask_target)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Utilidades de depuración y export
# ──────────────────────────────────────────────────────────────────────────────
def week_stats_mm(img_mm, scale_for_stats=10000):
    stat = img_mm.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=co,
        scale=scale_for_stats,
        maxPixels=1_000_000_000,
        bestEffort=True,
        tileScale=4
    ).getInfo() or {}
    return {
        "min_mm": float(stat.get('P_mm_week_min', float('nan'))),
        "max_mm": float(stat.get('P_mm_week_max', float('nan'))),
        "mean_mm": float(stat.get('P_mm_week_mean', float('nan')))
    }

def clip_fractions_8b(img_u8, scale_for_stats=20000):
    # Aproxima fracción de píxeles en 0 y 255 dentro de Colombia
    z0 = img_u8.eq(0).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=co, scale=scale_for_stats,
        maxPixels=1_000_000_000, bestEffort=True, tileScale=4
    ).get('PRECIP_8b')
    z255 = img_u8.eq(255).reduceRegion(
        reducer=ee.Reducer.mean(), geometry=co, scale=scale_for_stats,
        maxPixels=1_000_000_000, bestEffort=True, tileScale=4
    ).get('PRECIP_8b')
    try:
        return float(z0.getInfo() or 0.0), float(z255.getInfo() or 0.0)
    except Exception:
        return float('nan'), float('nan')

def build_task(wimg_u8, name):
    # Exporta EXACTAMENTE en la grilla objetivo
    return ee.batch.Export.image.toDrive(
        image         = wimg_u8,
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

# ──────────────────────────────────────────────────────────────────────────────
# 6) Bucle principal (orden inverso) con DEPURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
weeks = dates_weekly(START_DATE, END_DATE, step_days=7)
weeks_rev = list(reversed(weeks))
if DRY_RUN_N is not None:
    weeks_rev = weeks_rev[:DRY_RUN_N]
print(f"Semanas a exportar (orden inverso): {len(weeks_rev)}")
print(f"[GRID OBJETIVO] {TARGET_CRS}  {TARGET_WIDTH}×{TARGET_HEIGHT}  transform={TARGET_TRANSFORM}  bbox={TARGET_BBOX}")

tasks = []
for (ws, we) in weeks_rev:
    d = datetime.date.fromisoformat(ws)
    iso_year, iso_week, _ = d.isocalendar()
    name = f"{FILENAME_PREFIX}{iso_year:04d}_W{iso_week:02d}"

    wimg_mm = weekly_sum_mm(ws, we)
    stats   = week_stats_mm(wimg_mm)

    wimg_u8 = scale_to_255_precip(wimg_mm)
    frac0, frac255 = clip_fractions_8b(wimg_u8)

    print(f"[EXPORT] {name}  semana=[{ws}..{we})  "
          f"min={stats['min_mm']:.2f}  mean={stats['mean_mm']:.2f}  max={stats['max_mm']:.2f}  "
          f"frac0≈{(100*frac0 if not (frac0!=frac0) else float('nan')):.2f}%  "
          f"frac255≈{(100*frac255 if not (frac255!=frac255) else float('nan')):.2f}%")

    try:
        task = build_task(wimg_u8, name)
        task.start()
        tasks.append(task)
    except Exception as e:
        print(f"[ERROR start] {name}: {e}")

print("→ Tareas enviadas. Inicia monitor…")
wait_tasks(tasks, poll_sec=POLL_SEC)
print("Hecho.")
