# -*- coding: utf-8 -*-
"""
NDVI semanal (NOAA CDR AVHRR/VIIRS) para Colombia (2007–2019) – EXPORT 1024×1024 ALINEADO A INCIDENCIA
- Fuentes:
    * 2007–2013: NOAA/CDR/AVHRR/NDVI/V5  (band: 'NDVI', escala 1e-4 → [-1,1])
    * 2014–2019: NOAA/CDR/VIIRS/NDVI/V1  (band: 'NDVI', escala 1e-4 → [-1,1])
- Composición:
    * Diario: se toma la media (o mediana) de los disponibles (AVHRR/VIIRS) por día
    * Semanal: MEDIANA semanal de los NDVI diarios ([-1,1])
- Salida:
    * uint8 0..255 mapeando NDVI en [-1,1] → [0,1] → [0,255] (banda 'NDVI_8b')
    * Exporta EXACTAMENTE en la grilla de Incidencia (1024×1024, EPSG:4326)
      transform = [ 0.0166015625, 0, -82.0,  0, -0.0166015625, 12.5 ]
      bbox      = [-82.0, -4.5, -65.0, 12.5]
- Orden de exportación: INVERSO (2019→2007) para consistencia con otros productos
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

DRIVE_FOLDER     = 'NDVI_SEMANAL_CO_2007_2019_ALIGNED_1024'
FILENAME_PREFIX  = 'NDVI_WEEKLY_CO_'

# Export/poll
POLL_SEC      = 20
DRY_RUN_N     = None       # p.ej. 6 para probar 6 semanas
ORDER_REVERSED = True      # exportar 2019→2007

# Región: Colombia (GAUL Level-0) + “fix” geométrico
GAUL_L0_ID   = 'FAO/GAUL/2015/level0'
GAUL_COUNTRY = 'Colombia'

# ── GRILLA DE INCIDENCIA (OBJETIVO 1024×1024) ────────────────────────────────
TARGET_CRS        = 'EPSG:4326'
TARGET_WIDTH      = 1024
TARGET_HEIGHT     = 1024
# Affine (a,b,c,d,e,f): [sx, 0, x0, 0, -sy, y0]
TARGET_TRANSFORM  = [0.0166015625, 0.0, -82.0, 0.0, -0.0166015625, 12.5]
TARGET_BBOX       = [-82.0, -4.5, -65.0, 12.5]   # xmin, ymin, xmax, ymax
TARGET_REGION     = ee.Geometry.Rectangle(TARGET_BBOX, proj=TARGET_CRS, geodesic=False)

# ──────────────────────────────────────────────────────────────────────────────
# 2) Geometría Colombia
# ──────────────────────────────────────────────────────────────────────────────
countries = ee.FeatureCollection(GAUL_L0_ID)
co_raw = countries.filter(ee.Filter.eq('ADM0_NAME', GAUL_COUNTRY)).geometry()
co = co_raw.simplify(1000).buffer(0, 100)  # simplifica ~1 km y repara topología

# Máscara 1=Colombia reproyectada a la grilla objetivo (borde nítido)
co_mask_target = ee.Image.constant(1).clip(co).reproject(
    crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM
)

# ──────────────────────────────────────────────────────────────────────────────
# 3) Colecciones NOAA CDR NDVI (diario) y preprocesado
# ──────────────────────────────────────────────────────────────────────────────
C_AVHRR = (ee.ImageCollection('NOAA/CDR/AVHRR/NDVI/V5')
           .filterDate('2007-01-01', '2013-12-31')
           .filterBounds(co))

C_VIIRS = (ee.ImageCollection('NOAA/CDR/VIIRS/NDVI/V1')
           .filterDate('2014-01-01', '2019-12-31')
           .filterBounds(co))

def prep_ndvi(img):
    """
    Toma banda 'NDVI', aplica factor de escala 1e-4 → [-1,1], recorta a Colombia,
    y preserva 'system:time_start'. Devuelve banda 'NDVI' en float32 [-1,1].
    """
    ndvi = img.select('NDVI').multiply(0.0001).float()
    # clamp suave para outliers y máscara válida
    ndvi = ndvi.where(ndvi.lt(-1.0), -1.0).where(ndvi.gt(1.0), 1.0)
    ndvi = ndvi.updateMask(ee.Image.constant(1).clip(co))
    ndvi = ee.Image(ndvi.copyProperties(img, ['system:time_start']))
    return ndvi.rename('NDVI')

avhrr_ndvi = C_AVHRR.map(prep_ndvi)
viirs_ndvi = C_VIIRS.map(prep_ndvi)

# ──────────────────────────────────────────────────────────────────────────────
# 4) Composición diaria homogénea (media de fuentes por día)
# ──────────────────────────────────────────────────────────────────────────────
def date_list_daily(start_str, end_str):
    start = datetime.date.fromisoformat(start_str)
    end   = datetime.date.fromisoformat(end_str)
    out = []
    cur = start
    delta = datetime.timedelta(days=1)
    while cur <= end:
        out.append(ee.Image().set('system:time_start',
                                  ee.Date.fromYMD(cur.year, cur.month, cur.day).millis()))
        cur += delta
    return ee.ImageCollection(out)

daily_targets = date_list_daily(START_DATE, END_DATE)

def merge_daily(date_img):
    """
    Para una fecha, toma las observaciones disponibles (AVHRR/VIIRS) y
    devuelve la media diaria; si no hay, retorna todo en máscara (seguro para medianas).
    """
    t0 = ee.Date(date_img.get('system:time_start'))
    t1 = t0.advance(1, 'day')

    col = (avhrr_ndvi.filterDate(t0, t1).merge(viirs_ndvi.filterDate(t0, t1)))
    daily = ee.Image(col.mean())

    daily = ee.Image(ee.Algorithms.If(
        daily.bandNames().size().gt(0),
        daily.rename('NDVI').updateMask(ee.Image.constant(1).clip(co)),
        ee.Image.constant(0).rename('NDVI').updateMask(ee.Image.constant(0))
    ))
    return daily.set({'system:time_start': t0.millis()})

daily_ndvi = daily_targets.map(merge_daily)

# ──────────────────────────────────────────────────────────────────────────────
# 5) Semanas y composites
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

def weekly_median_ndvi(start_iso, end_iso):
    col = daily_ndvi.filterDate(start_iso, end_iso).filterBounds(co)
    cnt = col.count().rename('cnt')
    img = col.median().rename('NDVI')
    img = img.updateMask(cnt.gt(0)).updateMask(ee.Image.constant(1).clip(co))
    # Reproyección EXACTA a la grilla objetivo (continuo → bilinear)
    img = img.resample('bilinear').reproject(
        crs=TARGET_CRS, crsTransform=TARGET_TRANSFORM
    )
    img = img.set({
        'system:time_start': ee.Date(start_iso).millis(),
        'week_start': start_iso,
        'week_end': end_iso
    })
    return img

def ndvi_to_uint8(img_ndvi):
    """
    NDVI [-1,1] → [0,1] con (ndvi+1)/2 y luego [0,255] → uint8.
    Aplica máscara nítida de Colombia ya en la proyección objetivo.
    """
    nd = img_ndvi.select('NDVI')
    nd01 = nd.add(1.0).divide(2.0).clamp(0.0, 1.0)
    u8 = nd01.multiply(255.0).toUint8().rename('NDVI_8b')
    return u8.updateMask(co_mask_target)

# ──────────────────────────────────────────────────────────────────────────────
# 6) Utilidades de depuración y export
# ──────────────────────────────────────────────────────────────────────────────
def week_stats_ndvi(img_ndvi, scale_for_stats=10000):
    stat = img_ndvi.reduceRegion(
        reducer=ee.Reducer.minMax().combine(ee.Reducer.mean(), sharedInputs=True),
        geometry=co,
        scale=scale_for_stats,
        maxPixels=1_000_000_000,
        bestEffort=True, tileScale=4
    ).getInfo() or {}
    return {
        "min": float(stat.get('NDVI_min', float('nan'))),
        "max": float(stat.get('NDVI_max', float('nan'))),
        "mean": float(stat.get('NDVI_mean', float('nan')))
    }

def build_task(wimg_u8, name):
    # Exporta EXACTAMENTE en la grilla objetivo 1024×1024 (COG)
    return ee.batch.Export.image.toDrive(
        image          = wimg_u8,
        description    = name,
        folder         = DRIVE_FOLDER,
        fileNamePrefix = name,
        region         = TARGET_REGION,
        crs            = TARGET_CRS,
        crsTransform   = TARGET_TRANSFORM,
        dimensions     = f"{TARGET_WIDTH}x{TARGET_HEIGHT}",
        maxPixels      = 1_000_000_000,
        fileFormat     = 'GeoTIFF',
        formatOptions  = {'cloudOptimized': True}
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
# 7) Bucle principal (orden inverso) con DEPURACIÓN
# ──────────────────────────────────────────────────────────────────────────────
weeks = dates_weekly(START_DATE, END_DATE, step_days=7)
weeks_iter = list(reversed(weeks)) if ORDER_REVERSED else weeks
if DRY_RUN_N is not None:
    weeks_iter = weeks_iter[:DRY_RUN_N]

print(f"Semanas a exportar ({'inverso' if ORDER_REVERSED else 'normal'}): {len(weeks_iter)}")
print(f"[GRID OBJETIVO] {TARGET_CRS}  {TARGET_WIDTH}×{TARGET_HEIGHT}  transform={TARGET_TRANSFORM}  bbox={TARGET_BBOX}")

tasks = []
for (ws, we) in weeks_iter:
    d = datetime.date.fromisoformat(ws)
    iso_year, iso_week, _ = d.isocalendar()
    name = f"{FILENAME_PREFIX}{iso_year:04d}_W{iso_week:02d}"

    wimg_nd = weekly_median_ndvi(ws, we)      # float32 NDVI [-1,1]
    stats   = week_stats_ndvi(wimg_nd)         # para logging
    wimg_u8 = ndvi_to_uint8(wimg_nd)           # uint8 [0..255] (NDVI 0..1)

    print(f"[EXPORT] {name}  semana=[{ws}..{we})  "
          f"NDVI_min={stats['min']:.3f}  mean={stats['mean']:.3f}  max={stats['max']:.3f}")

    try:
        task = build_task(wimg_u8, name)
        task.start()
        tasks.append(task)
    except Exception as e:
        print(f"[ERROR start] {name}: {e}")

print("→ Tareas enviadas. Inicia monitor…")
wait_tasks(tasks, poll_sec=POLL_SEC)
print("Hecho.")
