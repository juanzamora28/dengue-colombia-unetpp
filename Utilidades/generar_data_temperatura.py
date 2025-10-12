# -*- coding: utf-8 -*-
"""
Temperatura semanal (LST día) para Colombia (2007-2019) usando MODIS LST diario (Terra + Aqua) en GEE.
Exporta GeoTIFFs a Google Drive (uint8 0..255) ya recortados a un rango de °C configurable.

Colecciones:
  - MODIS/061/MOD11A1 (Terra LST Daily 1km)  | bandas: LST_Day_1km, QC_Day
  - MODIS/061/MYD11A1 (Aqua  LST Daily 1km)  | bandas: LST_Day_1km, QC_Day

Escalado LST:
  - Valor real (Kelvin) = LST_Day_1km * 0.02
  - °C = Kelvin - 273.15

QA (simple):
  - Bits 0-1 de 'QC_Day' (Mandatory QA) == 00 (good) → se conserva
  - Además, se descartan valores no positivos o fuera de rango físico.

Salida:
  - uint8 0..255 según mapeo lineal de [TEMP_MIN_C, TEMP_MAX_C] → [0,255] con clamp.
"""

import ee
import datetime

# ──────────────────────────────────────────────────────────────────────────────
# 0) Autenticación/Init
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

# Rango de temperatura (°C) que se mapeará a 0..255
# Ajusta si lo necesitas; este rango suele cubrir LST diurna en Colombia.
TEMP_MIN_C   = 5.0
TEMP_MAX_C   = 45.0

# Export
DRIVE_FOLDER     = 'TEMP_SEMANAL_CO_2007_2019'
FILENAME_PREFIX  = 'TEMP_WEEKLY_CO_'
CRS_EXPORT       = 'EPSG:4326'
SCALE_M          = 1000   # 1 km MODIS

# Región: Colombia (GAUL Level-0)
GAUL_L0_ID   = 'FAO/GAUL/2015/level0'
GAUL_COUNTRY = 'Colombia'

# ──────────────────────────────────────────────────────────────────────────────
# 2) Geometría Colombia
# ──────────────────────────────────────────────────────────────────────────────
countries = ee.FeatureCollection(GAUL_L0_ID)
co = countries.filter(ee.Filter.eq('ADM0_NAME', GAUL_COUNTRY)).geometry()
# co = co.simplify(100)  # opcional

# ──────────────────────────────────────────────────────────────────────────────
# 3) Colecciones MODIS diarias (Terra/Aqua)
# ──────────────────────────────────────────────────────────────────────────────
C_TERRA = ee.ImageCollection('MODIS/061/MOD11A1') \
            .filterDate(START_DATE, END_DATE).filterBounds(co)
C_AQUA  = ee.ImageCollection('MODIS/061/MYD11A1') \
            .filterDate(START_DATE, END_DATE).filterBounds(co)

def mask_mod11_day(img):
    """
    Aplica máscara usando QA:
    - Mandatory QA bits (0-1) == 00 (good)
    También descarta LST <= 0 o nodata.
    """
    lst = img.select('LST_Day_1km')
    qc  = img.select('QC_Day')
    # bits 0-1 == 00
    good_mandatory = qc.bitwiseAnd(3).eq(0)
    # LST > 0 (evita nodata)
    valid_lst = lst.gt(0)
    mask = good_mandatory.And(valid_lst)
    return img.updateMask(mask)

def lstK_to_C(lstK_band):
    """Kelvin → °C."""
    return lstK_band.multiply(0.02).subtract(273.15)  # (K * 0.02) - 273.15

def daily_LST_C(img, platform='TERRA'):
    """
    Toma imagen MODIS diaria, aplica máscara y convierte a °C.
    Devuelve una ee.Image con banda 'LST_C'.
    """
    # 1) Máscara QA + LST>0
    img_m = mask_mod11_day(img)  # mantiene el tipo ee.Image

    # 2) Kelvin → °C y renombra
    lstC = lstK_to_C(img_m.select('LST_Day_1km')).rename('LST_C')

    # 3) Recorte/máscara a Colombia
    lstC = lstC.updateMask(ee.Image.constant(1).clip(co))

    # 4) Copia de propiedades (cast explícito a Image tras copyProperties)
    #    (limito a system:time_start para evitar inflar el objeto)
    lstC = ee.Image(lstC.copyProperties(img, ['system:time_start']))

    # 5) Plataforma como propiedad adicional
    return lstC.set({'platform': platform})


# Mapear colecciones a LST en °C
terra_C = C_TERRA.map(lambda im: daily_LST_C(im, 'TERRA'))
aqua_C  = C_AQUA.map(lambda im: daily_LST_C(im, 'AQUA'))

# Unimos Terra + Aqua
# Para un día con ambas, tomaremos la media de los valores disponibles
def merge_daily_mean(date_img):
    """
    Para la fecha de 'date_img', calcula la media de Terra/Aqua si existen; si no hay ninguna,
    devuelve una banda 'LST_C' totalmente enmascarada (segura para median/mean posteriores).
    """
    date = ee.Date(date_img.get('system:time_start'))
    day_start = date.update(hour=0, minute=0, second=0)
    day_end   = day_start.advance(1, 'day')

    # Colecciones ya en °C y recortadas a Colombia (terra_C / aqua_C)
    tday = terra_C.filterDate(day_start, day_end)
    aday = aqua_C.filterDate(day_start, day_end)

    merged = tday.merge(aday).select(['LST_C']).mean()

    # Asegurar que siempre retornamos una Image con banda 'LST_C'
    merged = ee.Image(merged)
    merged = ee.Image(ee.Algorithms.If(
        merged.bandNames().size().gt(0),
        merged.rename('LST_C'),
        # banda vacía segura (todo máscara)
        ee.Image.constant(0).rename('LST_C').updateMask(ee.Image.constant(0))
    ))

    # Clip/máscara a Colombia y timestamp
    merged = merged.updateMask(ee.Image.constant(1).clip(co)) \
                   .set({'system:time_start': day_start.millis()})
    return merged


# Construimos una lista de días para forzar “composición diaria” homogénea
def date_list_daily(start_str, end_str):
    start = datetime.date.fromisoformat(start_str)
    end   = datetime.date.fromisoformat(end_str)
    out = []
    cur = start
    delta = datetime.timedelta(days=1)
    while cur <= end:
        out.append(ee.Image().set('system:time_start', ee.Date.fromYMD(cur.year, cur.month, cur.day).millis()))
        cur += delta
    return ee.ImageCollection(out)

daily_targets = date_list_daily(START_DATE, END_DATE)
daily_mean = daily_targets.map(merge_daily_mean)
# ──────────────────────────────────────────────────────────────────────────────
# 4) Semanas y composites
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

def weekly_composite(start_iso, end_iso):
    col = daily_mean.filterDate(start_iso, end_iso).filterBounds(co)
    # mediana semanal robusta
    img = col.median().rename('LST_C')
    img = img.set({
        'system:time_start': ee.Date(start_iso).millis(),
        'week_start': start_iso,
        'week_end': end_iso
    })
    # recorte a Colombia
    img = img.updateMask(ee.Image.constant(1).clip(co))
    return img

# ──────────────────────────────────────────────────────────────────────────────
# 5) Escalado 0..255 (uint8)
# ──────────────────────────────────────────────────────────────────────────────
def scale_to_255(img_celsius):
    """
    Mapea linealmente [TEMP_MIN_C, TEMP_MAX_C] → [0,255], con clamp a los extremos.
    Devuelve banda 'TEMP_8b' como uint8.
    """
    c = img_celsius.select('LST_C')
    c_clamped = c.max(TEMP_MIN_C).min(TEMP_MAX_C)
    scaled = c_clamped.subtract(TEMP_MIN_C).divide(TEMP_MAX_C - TEMP_MIN_C).multiply(255.0)
    return scaled.rename('TEMP_8b').toUint8()

# ──────────────────────────────────────────────────────────────────────────────
# 6) Export semanal
# ──────────────────────────────────────────────────────────────────────────────
weeks = dates_weekly(START_DATE, END_DATE, step_days=7)
print(f"Semanas a exportar: {len(weeks)} (desde {START_DATE} hasta {END_DATE})")

tasks = []
for (ws, we) in weeks:
    wimg_C = weekly_composite(ws, we)
    wimg_u8 = scale_to_255(wimg_C)

    # Nombre por ISO week
    d = datetime.date.fromisoformat(ws)
    iso_year, iso_week, _ = d.isocalendar()
    name = f"{FILENAME_PREFIX}{iso_year:04d}_W{iso_week:02d}"

    task = ee.batch.Export.image.toDrive(
        image         = wimg_u8,
        description   = name,
        folder        = DRIVE_FOLDER,
        fileNamePrefix= name,
        region        = co.bounds(1),   # bbox de Colombia
        crs           = CRS_EXPORT,
        scale         = SCALE_M,
        maxPixels     = 1_000_000_000,
        fileFormat    = 'GeoTIFF',
        formatOptions = {'cloudOptimized': True}
    )
    task.start()
    tasks.append(task)
    print("Iniciado:", name)

print("→ Tareas enviadas a Google Drive. Revisa el panel 'Tasks' en el Code Editor o ee.batch.Task.list().")
# (Opcional) Monitor:
# import time
# while any(t.status()['state'] in ('READY', 'RUNNING') for t in tasks):
#     states = [t.status()['state'] for t in tasks]
#     print("Estados:", {s: states.count(s) for s in set(states)})
#     time.sleep(30)
# print("Finalizado.")
