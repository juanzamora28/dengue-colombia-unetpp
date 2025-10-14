
# ──────────────────────────────────────────────────────────────────────────────
# 1) IMPORTS y CONFIGURACIÓN GLOBAL
# ──────────────────────────────────────────────────────────────────────────────
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

os.environ["TENSORBOARD_NO_TF"] = "1" 
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True,max_split_size_mb:64"

os.environ["GDAL_NUM_THREADS"] = "1"   # evita multiproceso dentro del warp
os.environ["OMP_NUM_THREADS"] = "1"    # BLAS/OMP agresivo puede inflar memoria


os.environ.setdefault("GDAL_CACHEMAX", "1024")    # MB de cache interno GDAL
os.environ.setdefault("CPL_DEBUG", "OFF")
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast
from torch.cuda.amp import GradScaler   
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=r".*nll_loss2d_forward_out_cuda_template.*deterministic.*")

import re
from rasterio.warp import reproject, Resampling
import rasterio
import geopandas as gpd
import matplotlib.pyplot as plt
from tqdm import tqdm
from matplotlib.colors import ListedColormap, BoundaryNorm
from PIL import Image
from rasterio.features import rasterize
import random
from scipy.ndimage import binary_erosion, binary_closing, binary_dilation
import hashlib
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import List, Tuple

ERODE_ITERS = 1          # contrae 1 px el borde válido
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")  # PyTorch 2.x
    except Exception:
        pass
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True, warn_only=True)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

device_type = 'cuda' if torch.cuda.is_available() else 'cpu'

def md5_labels_of(ds):
    h = hashlib.md5()
    for b in ds.basename_list:
        with rasterio.open(os.path.join(ds.data_dir, "Incidencia", f"{b}.tif")) as src:
            arr = src.read(1).astype(np.uint8)  # {0,1,2,3,255}
        h.update(arr.tobytes())
    return h.hexdigest()

def erode_valid(mask_bool_or_float: torch.Tensor,
                iters: int = ERODE_ITERS) -> torch.Tensor:

    if iters <= 0:
        return mask_bool_or_float.float()
    np_mask = mask_bool_or_float.cpu().numpy().astype(bool)
    np_er   = np.stack(
        [binary_erosion(m, iterations=iters) for m in np_mask], 0
    )
    return torch.from_numpy(np_er.astype(np.float32)).to(mask_bool_or_float.device)



def _choose_gn_groups(num_channels: int, max_groups: int = 8) -> int:
    """Elige el mayor divisor de `num_channels` ≤ max_groups para GroupNorm."""
    g = min(max_groups, num_channels)
    while g > 1 and (num_channels % g) != 0:
        g -= 1
    return g  # si nada divide, queda 1 (LayerNorm por canal)

def bn_to_gn(module: nn.Module, max_groups: int = 8):
    """Recorre recursivamente el módulo y sustituye BatchNorm2d por GroupNorm."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            C = child.num_features
            g = _choose_gn_groups(C, max_groups=max_groups)
            setattr(module, name, nn.GroupNorm(num_groups=g, num_channels=C))
        else:
            bn_to_gn(child, max_groups=max_groups)

# ──────────────────────────────────────────────────────────────────────────────
# Calibración por temperatura (binaria, softmax/logits → probas calibradas)
# ──────────────────────────────────────────────────────────────────────────────
class TemperatureScaler(nn.Module):

    def __init__(self, init_T: float = 1.0):
        super().__init__()
        self.log_T = nn.Parameter(torch.tensor(np.log(init_T), dtype=torch.float32))

    @property
    def T(self):
        return torch.exp(self.log_T)

    def forward(self, logits):
        return logits / self.T.clamp(min=1e-3, max=100.0)

def _nll_bin_from_logits(logits, targets_01, ignore_index=-100):

    C = logits.shape[1]
    assert C == 2, "Calibración binaria: se esperan 2 logits"
    loss = F.cross_entropy(
        logits, targets_01.long(),
        ignore_index=ignore_index, reduction='mean'
    )
    return loss

@torch.no_grad()
def _gather_logits_targets_bin(model_head, val_loader, to_device, build_targets_fn, preprocess_X=None):

    model_head.eval()
    logits_list, targets_list = [], []
    for Xv, yv, mv in val_loader:
        Xv = Xv.to(to_device, non_blocking=True)
        if callable(preprocess_X):
            preprocess_X(Xv)  # in-place
        # targets binarios para el stage correspondiente:
        tbin = build_targets_fn(yv, mv).to(to_device, non_blocking=True)
        out = model_head(Xv)
        out = out[-1] if isinstance(out, list) else out
        logits_list.append(out.float().detach().cpu())
        targets_list.append(tbin.detach().cpu())
    logits = torch.cat(logits_list, dim=0)
    targets = torch.cat(targets_list, dim=0)
    return logits, targets

def calibrate_temperature_binary_head(
        model_head, val_loader, device, build_targets_fn,
        *, max_iter=200, lr=0.05, verbose=True, preprocess_X=None
    ):

    # Recolectar logits y targets (en CPU para ahorrar VRAM)
    logits, targets = _gather_logits_targets_bin(model_head, val_loader, device, build_targets_fn, preprocess_X=preprocess_X)

    # Mover a device para optimizar
    logits = logits.to(device)
    targets = targets.to(device)

    scaler = TemperatureScaler(init_T=1.0).to(device)
    opt = torch.optim.LBFGS([scaler.log_T], lr=lr, max_iter=max_iter, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad(set_to_none=True)
        logits_scaled = scaler(logits)
        loss = _nll_bin_from_logits(logits_scaled, targets, ignore_index=-100)
        loss.backward()
        return loss

    last_loss = float('inf')
    for _ in range(3):  # algunas pasadas LBFGS es suficiente
        loss = opt.step(closure)
        last_loss = float(loss.item())

    if verbose:
        print(f"[Calibración] T óptima ≈ {scaler.T.item():.4f}  (NLL={last_loss:.4f})")

    # Devolver en CPU (más cómodo para guardar)
    scaler = scaler.cpu()
    return scaler



# ──────────────────────────────────────────────────────────────────────────────
# Feature Configurations
# ──────────────────────────────────────────────────────────────────────────────
base_cfg = {
    "use_precipitation": True,
    "use_temperature": True,

    "use_sexo_h": False,
    "use_sexo_m": False,
    "use_edad_menos5": False,
    "use_edad_5_14": False,
    "use_edad_15_49": False,
    "use_edad_50_69": False,
    "use_edad_70mas": False,

    "use_nbi_mise": False,
    "use_nbi_serv": False,

    # Vegetación (como ya tenías)
    "use_vegetacion": False,
    "vegetation_scale_to_unit": True,
    "vegetation_missing_fallback": "zeros",  # 'zeros' | 'nan'

    # Incidencia t-1 (como ya tenías)
    "use_prev_incidence": False,
    "prev_inc_mode": "auto",
    "prev_one_hot": False,
    "prev_fallback": "zeros",
    "prev_pred_provider": None,

    # Superficies Construidas (anual, % 0..100) ─────────────────────
    # Carpeta esperada: <periodo>/Superficies_Construidas/<YYYY>.tif
    # Se alinea a Incidencia con reproyección bilinear y máscara nodata.
    "use_superficies": False,
    "superficies_scale_to_unit": True,        # 0..100 → 0..1
    "superficies_missing_fallback": "zeros",  # 'zeros' | 'nan'

    "use_luz_nocturna": False,
    "luznoct_scale_to_unit": False,      
    "luznoct_missing_fallback": "zeros", # 'zeros' | 'nan'

}


base_cfg.update({
    # Índice de hábitat H(x)
    "use_habitat_index": False,                  
    "habitat_w_built": 0.60,                     # pesos por defecto
    "habitat_w_nbi_serv": 0.25,
    "habitat_w_veg": 0.15,                       # se usa (1 - vegetación) para urbano

    # NTL dinámica: anomalía causal (num/den)
    "use_ntl_anomaly": False,                   
    "ntl_anom_num": (1, 4),                      # media t-1..t-4
    "ntl_anom_den": (5, 26),                     # media t-5..t-26 (baseline causal)
    "ntl_eps": 1e-3,                             # para evitar div. por 0

    # Interacciones exógenas
    "use_exo_interactions": False,              
    "inter_FxH": True,                           # F=P/T ventanas
    "inter_FxE": True,
    "inter_HxE": True,

    # Asegura continuo en [0,1] para P/T/NTL
    "scale_continuous_to_unit": True,
})



# Construcciones incrementalmente iguales a las tuyas
# ───────────────── Incremental con SOLO variables reales (+ lags incidencia) ─────────────
# Base común: P/T actuales + incidencia previa one-hot (4 lags)
_cfg_base_real = {
    **base_cfg,

    # Asegura P/T actuales activadas en el punto de partida
    "use_precipitation": True,
    "use_temperature": True,
    "scale_continuous_to_unit": True,

    # Incidencia previa SIEMPRE ON
    "use_prev_incidence": True,
    "prev_inc_mode": "gt",
    "prev_one_hot": True,
    "prev_fallback": "zeros",
    "prev_pred_provider": None,
    "prev_lags": 4,
    "prev_atten": [0.0, 0.45, 0.30, 0.20],

    # APAGA todo lo “inventado/derivado”
    "use_habitat_index": False,
    "use_ntl_anomaly": False,
    "use_exo_interactions": False,
    "use_climate_windows": False,
    "use_pt_stacks": False,
    "use_precip_extremes": False,
    "use_temp_quant_feats": False,
    "use_ndvi_anomaly": False,
    "use_dynamic_demographics": False,
    "use_nbi_dynamic": False,
    "use_nbi_mise_dynamic": False,
    "use_seasonal_posenc": False,

    # Por si en otro lado quedaron encendidas:
    "dasim_alpha": 0.0,
    "viirs_calibrate_to_nbi": False,
}

# cfg1: P/T + lags incidencia (nada más)
cfg1 = {**_cfg_base_real}

# cfg2: + sexo (H/M)
cfg2 = {**cfg1,
    "use_sexo_h": True,
    "use_sexo_m": True,
}

# cfg3: + edades
cfg3 = {**cfg2,
    "use_edad_menos5": True,
    "use_edad_5_14": True,
    "use_edad_15_49": True,
    "use_edad_50_69": True,
    "use_edad_70mas": True,
}

# cfg4: + NBI MISe
cfg4 = {**cfg3,
    "use_nbi_mise": True,
}

# cfg5: + NBI Servicios
cfg5 = {**cfg4,
    "use_nbi_serv": True,
}

# cfg6: + Vegetación (NDVI)
cfg6 = {**cfg5,
    "use_vegetacion": True,
    "vegetation_scale_to_unit": True,
    "vegetation_missing_fallback": "zeros",
}

# cfg7: + Superficies Construidas
cfg7 = {**cfg6,
    "use_superficies": True,
    "superficies_scale_to_unit": True,
    "superficies_missing_fallback": "zeros",
}

# cfg8: + Luz nocturna (VIIRS) — SIN ventanas climáticas ni nada inventado
cfg8 = {**cfg7,
    "use_luz_nocturna": True,
    "luznoct_scale_to_unit": False,
    "luznoct_fixed_max": 255.0,
    "luznoct_missing_fallback": "zeros",
}



cfg_EXO_PLUS = {
    **base_cfg,
    # clima por ventanas (lo que te dio bien)
    "use_precipitation": False, "use_temperature": False,
    "use_climate_windows": True,
    "p_acc1": (1,5), "p_acc2": (1,4),
    "t_avg_weeks": 5, "degday_tau_q": 0.40,

    "use_sexo_h": True, "use_sexo_m": True,
    "use_edad_menos5": True, "use_edad_5_14": True,
    "use_edad_15_49": True, "use_edad_50_69": True, "use_edad_70mas": True,

    # stacks crudos P/T
    "use_pt_stacks": True, "p_stack_weeks": 4, "t_stack_weeks": 4,

    # extremos lluvia
    "use_precip_extremes": True, "p_ext_window": (1,5), "p_heavy_thr": 0.60,

    # cuantiles temperatura
    "use_temp_quant_feats": True,

    # vegetación/NTL
    "use_vegetacion": True,
    "use_superficies": True,
    "use_nbi_serv": True,
    "use_habitat_index": True,

    "use_luz_nocturna": True,
    "luznoct_scale_to_unit": True,
    "use_ntl_anomaly": True,

    # NDVI anomalía
    "use_ndvi_anomaly": True, "ndvi_anom_num": (1,4), "ndvi_anom_den": (5,26),

    # demografía/NBI dinámicos
    "use_dynamic_demographics": True, "dyn_demo_with": ("E","P"),
    "use_nbi_dynamic": True, "nbi_dyn_with": ("E","P"),

    # estacionalidad
    "use_seasonal_posenc": True,

    "use_nbi_mise": True,
    "use_nbi_mise_dynamic": True,
    "nbi_mise_dyn_with": ("E","P"),   # o solo ("E",) / ("P",)

    "use_luz_nocturna": True,
    "luznoct_scale_to_unit": True,
    "dasim_alpha": 0.8,   

    # LAGS 
    "use_prev_incidence": True,
    "prev_inc_mode": "gt",
    "prev_one_hot": True,
    "prev_fallback": "zeros",
    "prev_pred_provider": None,
    "prev_lags": 4,
    "prev_atten": [0.0, 0.45, 0.30, 0.20],

}


# ── ORIGINAL (solo capas base + lags one-hot) ─────────────────────────────────
cfg_ORIGINAL_RAW = {
    **base_cfg,

    # Climáticas semanales crudas (reproyectadas, 0..1)
    "use_precipitation": True,
    "use_temperature": True,
    "scale_continuous_to_unit": True,

    # Demografía estática por cuartiles (0..3)/3 → [0,1]
    "use_sexo_h": True, "use_sexo_m": True,
    "use_edad_menos5": True, "use_edad_5_14": True,
    "use_edad_15_49": True, "use_edad_50_69": True, "use_edad_70mas": True,

    # NBI estático (labels 0..4)/4 → [0,1]
    "use_nbi_mise": True,
    "use_nbi_serv": True,

    # Vegetación semanal (NDVI recortado y a [0,1])
    "use_vegetacion": True,
    "vegetation_scale_to_unit": True,
    "vegetation_missing_fallback": "zeros",

    # Superficies construidas anual (0..100 → [0,1])
    "use_superficies": True,
    "superficies_scale_to_unit": True,
    "superficies_missing_fallback": "zeros",

    # Luz nocturna semanal (canal crudo reescalado a [0,1]; sin anomalías ni calibración)
    "use_luz_nocturna": True,
    "luznoct_scale_to_unit": True,
    "luznoct_missing_fallback": "zeros",
    "luznoct_fixed_max": 255.0,

    # Incidencia pasada como canales (one-hot) — “GT” para evitar fuga de pred
    "use_prev_incidence": True,
    "prev_inc_mode": "gt",
    "prev_one_hot": True,
    "prev_fallback": "zeros",
    "prev_pred_provider": None,
    "prev_lags": 4,
    "prev_atten": [0.0, 0.45, 0.30, 0.20],


    # VARIABLES DINÁMICAS APAGADAS
    "use_habitat_index": False,
    "use_ntl_anomaly": False,
    "use_exo_interactions": False,
    "use_climate_windows": False,
    "use_pt_stacks": False,
    "use_precip_extremes": False,
    "use_temp_quant_feats": False,
    "use_ndvi_anomaly": False,
    "use_dynamic_demographics": False,
    "use_nbi_dynamic": False,
    "use_nbi_mise_dynamic": False,
    "use_seasonal_posenc": False,
}


# ── ORIGINAL con P/T como lags (stacks) en lugar de P/T actuales ─────────────
cfg_ORIGINAL_RAW_PTSTACKS = {
    **cfg_ORIGINAL_RAW,

    # Desactiva P/T actuales
    "use_precipitation": False,
    "use_temperature": False,

    "use_pt_stacks": True,
    "p_stack_weeks": 4,  
    "t_stack_weeks": 4,  

    # Asegúrate de NO activar nada “inventado”
    "use_climate_windows": False,
    "use_precip_extremes": False,
    "use_temp_quant_feats": False,
}

FEATURE_CONFIGS = [
#    ("TEMP_PREC", cfg1),
#    ("TEMP_PREC_SEX", cfg2),
#    ("TEMP_PREC_SEX_AGE", cfg3),
#    ("TEMP_PREC_SEX_AGE_MISE", cfg4),
#    ("TEMP_PREC_SEX_AGE_MISE_SERV", cfg5),
#    ("TEMP_PREC_SEX_AGE_MISE_SERV_VEG", cfg6),
#    ("TEMP_PREC_SEX_AGE_MISE_SERV_VEG_SUP", cfg7),
#    ("TEMP_PREC_SEX_AGE_MISE_SERV_VEG_NTL_SUP", cfg8),
#    ("EXO_ONLY_PLUS", cfg_EXO_PLUS),
    ("ORIGINAL_RAW", cfg_ORIGINAL_RAW),
#    ("ORIGINAL_RAW_PTSTACKS", cfg_ORIGINAL_RAW_PTSTACKS),

]


# ──────────────────────────────────────────────────────────────────────────────
# 3) CLASE DengueDataset 
# ──────────────────────────────────────────────────────────────────────────────
class DengueDataset(Dataset):
    def __init__(
        self,
        data_dir,
        mode='train',
        # Continuas base
        use_precipitation=True,
        use_temperature=True,
        # NBI
        use_nbi_mise=True,
        use_nbi_serv=True,
        # Demográficas
        use_sexo_h=True,
        use_sexo_m=True,
        use_edad_menos5=True,
        use_edad_5_14=True,
        use_edad_15_49=True,
        use_edad_50_69=True,
        use_edad_70mas=True,
        # Vegetación (NDVI semanal)
        use_vegetacion=False,
        vegetation_scale_to_unit=True,           # [-1,1] → [0,1]
        vegetation_missing_fallback="zeros",     # 'zeros' | 'nan'
        # Superficies Construidas (anual, %)
        use_superficies=False,
        superficies_scale_to_unit=True,          # 0..100 → [0,1]
        superficies_missing_fallback="zeros",    # 'zeros' | 'nan'
        # Tamaño destino y splits
        target_size=None,
        train_split=0.7,
        val_split=0.15,
        scale_continuous_to_unit=True,           # [0,255] → [0,1] para P/T
        add_valid_mask_channel=False,            # añade la máscara como canal extra
        # Incidencia t-1
        use_prev_incidence=False,
        prev_inc_mode="auto",                    # 'gt'|'pred'|'zeros'|'auto'
        prev_one_hot=False,                      # 1 canal (ordinal) o 4 canales one-hot
        prev_fallback="zeros",                   # 'zeros'|'repeat0'
        prev_pred_provider=None,                 # callable(basename)->ndarray HxW {0..3,255}
        prev_lags=1,                             # K lags (1..K)
        prev_sched_p=0.0,                        # prob. de usar pred en TRAIN si mode='auto'
        prev_atten=None,                         # lista de pesos por lag; None → default
        # ── LUZ NOCTURNA (VIIRS) ──
        use_luz_nocturna=False,
        luznoct_scale_to_unit=False,             # normalización robusta [0,1] con p99 si True
        luznoct_missing_fallback="zeros",        # 'zeros' | 'nan'
        luznoct_fixed_max=None,
        # Dasimetría & clima
        dasim_alpha=0.0,                         # 0.0 = OFF; 0.8 recomendado
        dasim_eps=1e-3,
        viirs_calibrate_to_nbi=False,            # True para calibrar VIIRS→NBI (municipal)
        use_climate_windows=False,               # True para derivar canales P/T

        dept_name=None,
        dept_code=None,
        p_acc1=(2,6),
        p_acc2=(3,8),
        t_avg_weeks=4,
        degday_tau_q=0.40,
        # Hábitat H(x) y NTL-anomaly E(x,t) e interacciones ──
        use_habitat_index=False,
        habitat_w_built=0.60,
        habitat_w_nbi_serv=0.25,
        habitat_w_veg=0.15,                      # se usa (1 - vegetación)
        use_ntl_anomaly=False,
        ntl_anom_num=(1,4),                      # t-1..t-4
        ntl_anom_den=(5,26),                     # t-5..t-26
        ntl_eps=1e-3,
        use_exo_interactions=False,
        inter_FxH=True,
        inter_FxE=True,
        inter_HxE=True,
        use_seasonal_posenc=False,               # agrega sin/cos(semana_del_año)
        use_pt_stacks=False, p_stack_weeks=4, t_stack_weeks=4,  # P/T crudas t-1..t-k como canales
        use_precip_extremes=False, p_ext_window=(1,5), p_heavy_thr=0.60,  # picos y %semanas>umbral
        use_temp_quant_feats=False,              # proporción semanas T > tau (además de deg-days)
        use_ndvi_anomaly=False, ndvi_anom_num=(1,4), ndvi_anom_den=(5,26),  # NDVI(t-1..4) - NDVI(t-5..t-26)
        use_dynamic_demographics=False,          # multipl. demografía × dinámica (E o P)
        dyn_demo_with=("E","P"),                 # opciones: 'E' (NTL anomaly), 'P' (precip ventana)
        use_nbi_dynamic=False,                   # NBI_serv × dinámica
        nbi_dyn_with=("E","P"),
        use_nbi_mise_dynamic=False,
        nbi_mise_dyn_with=("E","P"),


    ):
        self.data_dir            = data_dir
        self.mode                = mode

        # Flags de uso
        self.use_precipitation   = use_precipitation
        self.use_temperature     = use_temperature
        self.use_nbi_mise        = use_nbi_mise
        self.use_nbi_serv        = use_nbi_serv
        self.use_sexo_h          = use_sexo_h
        self.use_sexo_m          = use_sexo_m
        self.use_edad_menos5     = use_edad_menos5
        self.use_edad_5_14       = use_edad_5_14
        self.use_edad_15_49      = use_edad_15_49
        self.use_edad_50_69      = use_edad_50_69
        self.use_edad_70mas      = use_edad_70mas

        # Vegetación
        self.use_vegetacion              = use_vegetacion
        self.vegetation_scale_to_unit    = vegetation_scale_to_unit
        self.vegetation_missing_fallback = vegetation_missing_fallback

        # Superficies Construidas
        self.use_superficies               = use_superficies
        self.superficies_scale_to_unit     = superficies_scale_to_unit
        self.superficies_missing_fallback  = superficies_missing_fallback

        # LUZ NOCTURNA
        self.use_luz_nocturna          = use_luz_nocturna
        self.luznoct_scale_to_unit     = luznoct_scale_to_unit
        self.luznoct_missing_fallback  = luznoct_missing_fallback
        self.luznoct_fixed_max         = luznoct_fixed_max

        # Dasimetría & clima
        self.dasim_alpha = float(dasim_alpha)
        self.dasim_eps   = float(dasim_eps)
        self.viirs_calibrate_to_nbi = bool(viirs_calibrate_to_nbi)
        self.use_climate_windows = bool(use_climate_windows)
        self.p_acc1 = tuple(p_acc1)
        self.p_acc2 = tuple(p_acc2)
        self.t_avg_weeks = int(t_avg_weeks)
        self.degday_tau_q = float(degday_tau_q)

        # NUEVO: H(x), E(x,t) e interacciones
        self.use_habitat_index  = bool(use_habitat_index)
        self.habitat_w_built    = float(habitat_w_built)
        self.habitat_w_nbi_serv = float(habitat_w_nbi_serv)
        self.habitat_w_veg      = float(habitat_w_veg)

        self.use_ntl_anomaly    = bool(use_ntl_anomaly)
        self.ntl_anom_num       = tuple(ntl_anom_num)
        self.ntl_anom_den       = tuple(ntl_anom_den)
        self.ntl_eps            = float(ntl_eps)

        self.use_exo_interactions = bool(use_exo_interactions)
        self.inter_FxH          = bool(inter_FxH)
        self.inter_FxE          = bool(inter_FxE)
        self.inter_HxE          = bool(inter_HxE)

        # Caches
        self._muni_id_cache = {}       # geom key -> ndarray[int32]
        self._dasi_factor_cache = {}   # basename -> ndarray[float32]
        self._ntl_calib_cache  = {}    # basename -> ndarray[float32]

        # LRU por semana y tipo (depende de geometría)
        self._pt_cache   = OrderedDict()
        self._tt_cache   = OrderedDict()
        self._ntl_cache  = OrderedDict()
        self._ndvi_cache = OrderedDict()
        self._cache_cap  = 256  # ajustable

        # LRU de cubo completo por basename
        self._cube_cache = OrderedDict()
        self._cube_cap   = 3    # ajustable

        self.use_seasonal_posenc   = bool(use_seasonal_posenc)
        self.use_pt_stacks         = bool(use_pt_stacks)
        self.p_stack_weeks         = int(p_stack_weeks)
        self.t_stack_weeks         = int(t_stack_weeks)

        self.use_precip_extremes   = bool(use_precip_extremes)
        self.p_ext_window          = tuple(p_ext_window)
        self.p_heavy_thr           = float(p_heavy_thr)

        self.use_temp_quant_feats  = bool(use_temp_quant_feats)

        self.use_ndvi_anomaly      = bool(use_ndvi_anomaly)
        self.ndvi_anom_num         = tuple(ndvi_anom_num)
        self.ndvi_anom_den         = tuple(ndvi_anom_den)

        self.use_dynamic_demographics = bool(use_dynamic_demographics)
        self.dyn_demo_with            = tuple(dyn_demo_with)

        self.use_nbi_dynamic = bool(use_nbi_dynamic)
        self.nbi_dyn_with    = tuple(nbi_dyn_with)
        self.use_nbi_mise_dynamic = bool(use_nbi_mise_dynamic)
        self.nbi_mise_dyn_with    = tuple(nbi_mise_dyn_with)



        # Otros parámetros
        def _norm_target_size(ts):
            if ts is None:
                return None
            if isinstance(ts, int):
                return (int(ts), int(ts))
            if isinstance(ts, (list, tuple)):
                if len(ts) == 0:
                    return None
                if len(ts) == 1:
                    return (int(ts[0]), int(ts[0]))
                return (int(ts[0]), int(ts[1]))
            return None

        self.target_size = _norm_target_size(target_size)
        self.train_split         = train_split
        self.val_split           = val_split
        self.scale_continuous_to_unit  = scale_continuous_to_unit
        self.add_valid_mask_channel    = add_valid_mask_channel

        # t-1..t-K
        self.use_prev_incidence = use_prev_incidence
        self.prev_inc_mode      = prev_inc_mode
        self.prev_one_hot       = prev_one_hot
        self.prev_fallback      = prev_fallback
        self.prev_pred_provider = prev_pred_provider

        self.prev_lags       = int(prev_lags)
        self.prev_sched_p    = float(prev_sched_p)
        self.prev_atten      = prev_atten
        if (self.prev_atten is None) and (self.prev_lags > 0):
            base = [1.00, 0.70, 0.50, 0.35, 0.25, 0.18]
            self.prev_atten = base[:self.prev_lags]

        self.prev_ch_slices: list = []
        self.prev_all_idx:   list = []
        self.prev_lag1_onehot_slice = None

        # Split cronológico
        self.basename_list = self._split_data()
        self._index_by_name = {b: i for i, b in enumerate(sorted(self._get_basename_list()))}

        self.dept_name = dept_name
        # Si pasan name y no code, resolver con el dict global
        if (dept_code is None) and (dept_name is not None):
            if (dept_name in DEPARTMENTS):
                dept_code = DEPARTMENTS[dept_name]["code"]
            else:
                raise ValueError(f"dept_name '{dept_name}' no está en DEPARTMENTS")
        self.dept_code = int(dept_code) if dept_code is not None else None

        # Cache para rasterizar máscara por depto en grids de referencia distintos
        self._dept_mask_cache = {}

    # -------- helpers de cache --------
    def _cache_get(self, cache, key):
        v = cache.get(key)
        if v is not None:
            cache.move_to_end(key)
        return v

    def _cache_put(self, cache, key, val):
        cache[key] = val
        cache.move_to_end(key)
        if len(cache) > self._cache_cap:
            cache.popitem(last=False)

    # ------------------------- Utilidades internas -------------------------
    def _get_basename_list(self):
        def base_set(folder):
            p = os.path.join(self.data_dir, folder)
            return set(os.path.splitext(f)[0] for f in os.listdir(p) if f.lower().endswith('.tif'))
        pre = base_set('Precipitacion')
        tmp = base_set('Temperatura')
        inc = base_set('Incidencia')
        return sorted(pre & tmp & inc)

    def _split_data(self):
        basenames = sorted(self._get_basename_list())
        n_total   = len(basenames)
        n_train   = int(n_total * self.train_split)
        n_val     = int(n_total * self.val_split)
        train_split = basenames[:n_train]
        val_split   = basenames[n_train:n_train + n_val]
        test_split  = basenames[n_train + n_val:]
        splits = {'train': train_split, 'val': val_split, 'test': test_split}

        def _compute_dist(file_list):
            cnt = Counter(); total = 0
            for b in file_list:
                with rasterio.open(os.path.join(self.data_dir, "Incidencia", f"{b}.tif")) as src:
                    arr = src.read(1)
                valid = arr < 255
                for cls in range(4):
                    c = int(((arr == cls) & valid).sum())
                    cnt[cls] += c; total += c
            return cnt, total

        for name, files in splits.items():
            cnt, total = _compute_dist(files)
            print(f"\n[Distribución en {name.upper()}] {len(files)} archivos, {total:,} píxeles válidos")
            for cls in range(4):
                c = cnt.get(cls, 0)
                pct = 100 * c / total if total else 0.0
                print(f"  Clase {cls}: {c:,} píxeles ({pct:.2f} %)")

        return splits[self.mode]

    @staticmethod
    def _resize_np(arr, size_hw, resample):
        pil = Image.fromarray(arr)
        pil = pil.resize((size_hw[1], size_hw[0]), resample=resample)
        return np.array(pil)

    @staticmethod
    def _warp_to_ref(src_path, ref_height, ref_width, ref_transform, ref_crs,
                     resampling, dst_dtype=None):
        with rasterio.open(src_path) as src:
            arr_src = src.read(1)
            if dst_dtype is None:
                dst_dtype = arr_src.dtype
            dst = np.zeros((ref_height, ref_width), dtype=dst_dtype)
            reproject(
                source=arr_src,
                destination=dst,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=ref_transform, dst_crs=ref_crs,
                resampling=resampling,
                num_threads=1,
                warp_mem_limit=256  # más grande para menos overhead
            )
        return dst

    def _year_from_basename(self, b):
        try:
            return int(b.split('.')[0])
        except Exception:
            return int(b[:4])

    def _year_week_from_basename(self, b):
        m = re.search(r"(20\d{2}).*?[wW](\d{1,2})", b)
        if not m:
            return self._year_from_basename(b), None
        return int(m.group(1)), int(m.group(2))

    def _prev_basename(self, basename):
        if basename not in self._index_by_name:
            return None
        idx = self._index_by_name[basename]
        if idx == 0:
            return None
        ordered = sorted(self._get_basename_list())
        return ordered[idx - 1]

    def _prev_basename_k(self, basename, k: int):
        if basename not in self._index_by_name: return None
        ordered = sorted(self._get_basename_list())
        idx = self._index_by_name[basename]
        k = int(k)
        if idx - k < 0: return None
        return ordered[idx - k]

    def _load_prev_incidence(self, basename, ref_h, ref_w, ref_transform, ref_crs):
        mode = self.prev_inc_mode
        prev_b = self._prev_basename(basename)

        def _read_inc(b):
            ipath = os.path.join(self.data_dir, "Incidencia", f"{b}.tif")
            if not os.path.exists(ipath):
                return None
            with rasterio.open(ipath) as src:
                arr = self._warp_to_ref(
                    ipath, ref_h, ref_w, ref_transform, ref_crs,
                    resampling=Resampling.nearest, dst_dtype=np.uint8
                ).astype(np.uint8)
            return arr

        if mode == "gt":
            arr_prev = _read_inc(prev_b) if prev_b else None
        elif mode == "pred":
            arr_prev = None
            if callable(self.prev_pred_provider):
                arr_prev = self.prev_pred_provider(basename)
        elif mode == "zeros":
            arr_prev = None
        else:  # 'auto'
            arr_prev = _read_inc(prev_b) if prev_b else None
            if arr_prev is None and callable(self.prev_pred_provider):
                arr_prev = self.prev_pred_provider(basename)

        if arr_prev is None:
            if self.prev_fallback == "repeat0":
                arr_prev = np.zeros((ref_h, ref_w), dtype=np.uint8)
            else:
                arr_prev = np.full((ref_h, ref_w), 255, dtype=np.uint8)
        return arr_prev

    def _load_prev_incidence_k(self, basename, k, ref_h, ref_w, ref_transform, ref_crs):
        mode = self.prev_inc_mode
        prev_b = self._prev_basename_k(basename, k)

        def _read_inc(b):
            if not b: return None
            ipath = os.path.join(self.data_dir, "Incidencia", f"{b}.tif")
            if not os.path.exists(ipath): return None
            return self._warp_to_ref(
                ipath, ref_h, ref_w, ref_transform, ref_crs,
                resampling=Resampling.nearest, dst_dtype=np.uint8
            ).astype(np.uint8)

        use_pred = False
        if mode == "pred":
            use_pred = True
        elif mode == "auto" and self.mode == "train" and callable(self.prev_pred_provider):
            use_pred = (np.random.rand() < self.prev_sched_p)

        arr_prev = None
        if use_pred and callable(self.prev_pred_provider):
            bprev = prev_b
            if bprev is not None:
                arr_prev = self.prev_pred_provider(bprev)
                if arr_prev is not None:
                    arr_prev = arr_prev.astype(np.uint8)
        if arr_prev is None:
            if mode == "zeros":
                arr_prev = None
            else:
                arr_prev = _read_inc(prev_b)

        if arr_prev is None:
            if self.prev_fallback == "repeat0":
                arr_prev = np.zeros((ref_h, ref_w), dtype=np.uint8)
            else:
                arr_prev = np.full((ref_h, ref_w), 255, dtype=np.uint8)
        return arr_prev

    def _muni_ids(self, ref_h, ref_w, ref_transform, ref_crs):
        key = (ref_h, ref_w, str(ref_crs), repr(ref_transform))
        if key in self._muni_id_cache:
            return self._muni_id_cache[key]

        gdf = muni_gdf.to_crs(ref_crs)  # global del main
        shapes = [(geom, i+1) for i, geom in enumerate(gdf.geometry) if geom is not None]
        ids = rasterize(
            shapes=shapes,
            out_shape=(ref_h, ref_w),
            transform=ref_transform,
            fill=0,
            dtype=np.int32
        )
        self._muni_id_cache[key] = ids
        return ids
    
    def _dept_mask(self, ref_h, ref_w, ref_transform, ref_crs):
        """
        Máscara booleana [H,W] True sólo dentro del departamento seleccionado.
        Cacheada por (grid,crs,transform,dept_code).
        """
        if self.dept_code is None:
            return np.ones((ref_h, ref_w), dtype=bool)

        key = (ref_h, ref_w, str(ref_crs), repr(ref_transform), int(self.dept_code))
        if key in self._dept_mask_cache:
            return self._dept_mask_cache[key]

        # Tomamos geometría del depto y rasterizamos a la grilla de referencia
        gdf_dept = muni_gdf[muni_gdf["DPTO_Code"] == int(self.dept_code)]
        if gdf_dept.empty:
            raise ValueError(f"DPTO_Code={self.dept_code} no existe en muni_gdf")

        gdf_dept = gdf_dept.to_crs(ref_crs)
        shapes = [(geom, 1) for geom in gdf_dept.geometry if geom is not None]
        if not shapes:
            # fallback: todo True (no limita)
            mask = np.ones((ref_h, ref_w), dtype=bool)
        else:
            mask = rasterize(
                shapes=shapes,
                out_shape=(ref_h, ref_w),
                transform=ref_transform,
                fill=0,
                dtype=np.uint8
            ).astype(bool)

        self._dept_mask_cache[key] = mask
        return mask


    def _compute_dasi_factor(self, basename, ntl_unit, valid_mask, muni_ids):
        if basename in self._dasi_factor_cache:
            return self._dasi_factor_cache[basename]

        a = max(0.0, min(1.0, self.dasim_alpha))
        if a <= 0.0:
            F = np.ones_like(ntl_unit, dtype=np.float32)
            self._dasi_factor_cache[basename] = F
            return F

        w = np.where(valid_mask, ntl_unit, 0.0).astype(np.float32)
        eps = float(self.dasim_eps)
        ids = muni_ids
        max_id = int(ids.max()) if ids.size > 0 else 0
        sums = np.bincount(ids[valid_mask].ravel(), weights=w[valid_mask].ravel(), minlength=max_id+1).astype(np.float64)
        cnts = np.bincount(ids[valid_mask].ravel(), minlength=max_id+1).astype(np.float64)
        means = (sums + eps) / (cnts + eps)

        mean_map = means[ids]
        F = ((1.0 - a) + a * (w + eps) / mean_map).astype(np.float32)
        F[~valid_mask] = 1.0
        self._dasi_factor_cache[basename] = F
        return F

    def _calibrate_ntl_to_nbi(self, basename, ntl_unit, valid_mask, muni_ids, nbi_m=None, nbi_s=None):
        if basename in self._ntl_calib_cache:
            return self._ntl_calib_cache[basename]

        if (nbi_m is None) and (nbi_s is None):
            self._ntl_calib_cache[basename] = ntl_unit.astype(np.float32)
            return self._ntl_calib_cache[basename]

        ref_list = []
        if nbi_m is not None: ref_list.append(nbi_m.astype(np.float32))
        if nbi_s is not None: ref_list.append(nbi_s.astype(np.float32))
        nbi_ref = np.mean(ref_list, axis=0).astype(np.float32)

        ids = muni_ids
        valid = valid_mask
        eps = 1e-6

        max_id = int(ids.max()) if ids.size > 0 else 0
        ntl_sum = np.bincount(ids[valid].ravel(), weights=ntl_unit[valid].ravel(), minlength=max_id+1).astype(np.float64)
        ntl_cnt = np.bincount(ids[valid].ravel(), minlength=max_id+1).astype(np.float64)
        ntl_mean = (ntl_sum + eps) / (ntl_cnt + eps)

        nbi_sum = np.bincount(ids[valid].ravel(), weights=nbi_ref[valid].ravel(), minlength=max_id+1).astype(np.float64)
        nbi_mean = (nbi_sum + eps) / (ntl_cnt + eps)

        scale = (nbi_mean + eps) / (ntl_mean + eps)
        scale_map = scale[ids].astype(np.float32)

        ntl_cal = np.clip(ntl_unit * scale_map, 0.0, 1.0).astype(np.float32)
        ntl_cal[~valid] = 0.0
        self._ntl_calib_cache[basename] = ntl_cal
        return ntl_cal

    # ------------------------- PyTorch Dataset API -------------------------
    def __getitem__(self, idx):
        basename = self.basename_list[idx]

        # Si el cubo ya está en cache (misma config y target_size), devuélvelo
        v = self._cube_cache.get(basename)
        if v is not None:
            self._cube_cache.move_to_end(basename)
            return v

        X, y, m = self._build_full_cube(basename)
        # guarda en LRU de cubos
        self._cube_cache[basename] = (X, y, m)
        self._cube_cache.move_to_end(basename)
        if len(self._cube_cache) > self._cube_cap:
            self._cube_cache.popitem(last=False)
        return X, y, m

    def _build_full_cube(self, basename):
        year     = self._year_from_basename(basename)
        chans    = []
        cont_flags = []

        # 1) Incidencia (rejilla ref)
        inc_path = os.path.join(self.data_dir, "Incidencia", f"{basename}.tif")
        with rasterio.open(inc_path) as ref:
            ref_h, ref_w            = ref.height, ref.width
            ref_transform, ref_crs  = ref.transform, ref.crs
            y_ref                   = ref.read(1).astype(np.int64)

        # Máscara válida por nodata
        mask_local = (y_ref != 255)

        #limitar al departamento ───────────────────────────────────────────
        dept_mask = self._dept_mask(ref_h, ref_w, ref_transform, ref_crs)
        # todo lo fuera del depto queda como ignorado
        y_ref[~dept_mask] = 255
        # la máscara de entrenamiento/val también respeta el depto
        mask_local = mask_local & dept_mask

        geom_key = (ref_h, ref_w, str(ref_crs), repr(ref_transform))


        # helpers locales
        def _warp_cont_masked(src_path, resampling):
            with rasterio.open(src_path) as src:
                arr_src = src.read(1)
                dst = np.zeros((ref_h, ref_w), dtype=np.float32)
                reproject(
                    source=arr_src,
                    destination=dst,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=ref_transform, dst_crs=ref_crs,
                    resampling=resampling,
                    num_threads=1,
                    warp_mem_limit=256
                )
            dst = dst.astype(np.float32)
            return np.where(mask_local, dst, np.nan).astype(np.float32)

        def _parse_yw(bname):
            m = re.search(r"(?P<year>20\d{2})\D*[wW](?P<week>\d{1,2})", bname)
            if m:
                try:
                    return int(m.group("year")), int(m.group("week"))
                except Exception:
                    pass
            return self._year_from_basename(bname), None

        # 1.5) VIIRS
        ntl_unit = None
        if getattr(self, 'use_luz_nocturna', False):
            key = ("NTL", basename, geom_key, bool(self.luznoct_scale_to_unit), self.luznoct_fixed_max)
            cached = self._cache_get(self._ntl_cache, key)
            if cached is not None:
                ntl_unit = cached
            else:
                ntl_path = os.path.join(self.data_dir, "Luz_Nocturna", f"{basename}.tif")
                if os.path.exists(ntl_path):
                    ntl_tmp = _warp_cont_masked(ntl_path, Resampling.bilinear).astype(np.float32)
                    ntl_tmp = np.where(np.isfinite(ntl_tmp), ntl_tmp, 0.0)
                    ntl_tmp[ntl_tmp < 0.0] = 0.0
                    if getattr(self, 'luznoct_scale_to_unit', False):
                        vmax_cfg = getattr(self, 'luznoct_fixed_max', None)
                        if (vmax_cfg is not None) and np.isfinite(vmax_cfg) and (vmax_cfg > 0):
                            vmax = float(vmax_cfg)
                        else:
                            vmax = float(np.percentile(ntl_tmp[ntl_tmp > 0], 99)) if np.any(ntl_tmp > 0) else 1.0
                        vmax = max(vmax, 1e-6)
                        ntl_unit = np.clip(ntl_tmp, 0.0, vmax) / vmax
                    else:
                        ntl_unit = ntl_tmp
                else:
                    ntl_unit = np.zeros((ref_h, ref_w), dtype=np.float32)
                self._cache_put(self._ntl_cache, key, ntl_unit)

        muni_ids = self._muni_ids(ref_h, ref_w, ref_transform, ref_crs) if getattr(self, 'use_luz_nocturna', False) else None

        # 1.6) Pos-encoding estacional
        if getattr(self, 'use_seasonal_posenc', False):
            y_cur, w_cur = _parse_yw(basename)
            if (w_cur is None) or (w_cur < 1) or (w_cur > 53):
                w_cur = 26
            phi = 2.0 * np.pi * (w_cur - 1) / 52.0
            sinw = np.full((ref_h, ref_w), np.sin(phi), np.float32) * mask_local.astype(np.float32)
            cosw = np.full((ref_h, ref_w), np.cos(phi), np.float32) * mask_local.astype(np.float32)
            chans.append(sinw); cont_flags.append(True)
            chans.append(cosw); cont_flags.append(True)

        # 2) Continuas P/T con posible shift
        def _shifted_basename(bname: str) -> str:
            k = int(getattr(DengueDataset, "_pt_shift_weeks", 0))
            if k > 0:
                pb = self._prev_basename_k(bname, k)
                if pb is not None:
                    return pb
            return bname

        b_pt = _shifted_basename(basename)

        # loaders con cache por semana
        def _load_week_P_unit(bname):
            key = ("P", bname, geom_key, self.scale_continuous_to_unit)
            v = self._cache_get(self._pt_cache, key)
            if v is not None: return v
            pth = os.path.join(self.data_dir, "Precipitacion", f"{bname}.tif")
            if not os.path.exists(pth): return None
            arr = _warp_cont_masked(pth, Resampling.bilinear)
            if getattr(self, 'scale_continuous_to_unit', True): arr = arr / 255.0
            arr = np.nan_to_num(arr, nan=0.0) * mask_local.astype(np.float32)
            self._cache_put(self._pt_cache, key, arr.astype(np.float32))
            return arr

        def _load_week_T_unit(bname):
            key = ("T", bname, geom_key, self.scale_continuous_to_unit)
            v = self._cache_get(self._tt_cache, key)
            if v is not None: return v
            pth = os.path.join(self.data_dir, "Temperatura", f"{bname}.tif")
            if not os.path.exists(pth): return None
            arr = _warp_cont_masked(pth, Resampling.bilinear)
            if getattr(self, 'scale_continuous_to_unit', True): arr = arr / 255.0
            arr = np.nan_to_num(arr, nan=0.0) * mask_local.astype(np.float32)
            self._cache_put(self._tt_cache, key, arr.astype(np.float32))
            return arr

        def _load_week_NDVI_unit(bname):
            key = ("NDVI", bname, geom_key, self.vegetation_scale_to_unit)
            v = self._cache_get(self._ndvi_cache, key)
            if v is not None: return v
            pth = os.path.join(self.data_dir, "Vegetacion", f"{bname}.tif")
            if not os.path.exists(pth): return None
            arr = self._warp_to_ref(
                pth, ref_h, ref_w, ref_transform, ref_crs,
                resampling=Resampling.bilinear, dst_dtype=np.float32
            ).astype(np.float32)
            arr[(arr < -1.2) | (arr > 1.2)] = np.nan
            arr = 0.5 * (arr + 1.0) if getattr(self, 'vegetation_scale_to_unit', True) else arr
            arr = np.where(mask_local, arr, np.nan).astype(np.float32)
            self._cache_put(self._ndvi_cache, key, arr)
            return arr

        if getattr(self, 'use_precipitation', False):
            p_path = os.path.join(self.data_dir, "Precipitacion", f"{b_pt}.tif")
            if not os.path.exists(p_path):
                p_path = os.path.join(self.data_dir, "Precipitacion", f"{basename}.tif")
            p = _warp_cont_masked(p_path, Resampling.bilinear)
            if getattr(self, 'scale_continuous_to_unit', True):
                p = p / 255.0
            p = np.nan_to_num(p, nan=0.0) * mask_local.astype(np.float32)
            chans.append(p.astype(np.float32)); cont_flags.append(True)

        if getattr(self, 'use_temperature', False):
            t_path = os.path.join(self.data_dir, "Temperatura", f"{b_pt}.tif")
            if not os.path.exists(t_path):
                t_path = os.path.join(self.data_dir, "Temperatura", f"{basename}.tif")
            t = _warp_cont_masked(t_path, Resampling.bilinear)
            if getattr(self, 'scale_continuous_to_unit', True):
                t = t / 255.0
            t = np.nan_to_num(t, nan=0.0) * mask_local.astype(np.float32)
            chans.append(t.astype(np.float32)); cont_flags.append(True)

        # 2.bis) Stacks P/T — canales fijos
        if getattr(self, 'use_pt_stacks', False):
            for k in range(1, int(getattr(self, 'p_stack_weeks', 0)) + 1):
                pbk = self._prev_basename_k(basename, k)
                arr = _load_week_P_unit(pbk) if pbk is not None else None
                if arr is None: arr = np.zeros((ref_h, ref_w), np.float32)
                chans.append(arr.astype(np.float32)); cont_flags.append(True)

            for k in range(1, int(getattr(self, 't_stack_weeks', 0)) + 1):
                pbk = self._prev_basename_k(basename, k)
                arr = _load_week_T_unit(pbk) if pbk is not None else None
                if arr is None: arr = np.zeros((ref_h, ref_w), np.float32)
                chans.append(arr.astype(np.float32)); cont_flags.append(True)

        # 2.x) Ventanas climáticas
        pa, t_mean, E, H = None, None, None, None
        if getattr(self, 'use_climate_windows', False):

            def _load_PT(folder, bname, loader):
                if bname is None: return None
                arr = loader(bname)
                return arr

            def _mean_no_warn(arr_list):
                xs = [a for a in arr_list if a is not None]
                if not xs:
                    return (np.zeros((ref_h, ref_w), dtype=np.float32) * mask_local.astype(np.float32))
                st = np.stack([np.asarray(a, np.float32) for a in xs], axis=0)
                msk = np.isfinite(st)
                cnt = msk.sum(axis=0)
                sumv = np.where(msk, st, 0.0).sum(axis=0)
                out = np.divide(sumv, cnt, out=np.zeros((ref_h, ref_w), np.float32), where=(cnt > 0))
                out = out * mask_local.astype(np.float32)
                return out.astype(np.float32)

            # P medias en ventanas
            def _mean_window_P(a, b):
                xs = []
                for kk in range(a, b+1):
                    pb = self._prev_basename_k(basename, kk)
                    xs.append(_load_PT("Precipitacion", pb, _load_week_P_unit))
                return _mean_no_warn(xs)

            pa = _mean_window_P(self.p_acc1[0], self.p_acc1[1])
            pb_ = _mean_window_P(self.p_acc2[0], self.p_acc2[1])
            chans.append(pa);  cont_flags.append(True)
            chans.append(pb_); cont_flags.append(True)

            # Extremos P (máx y %>umbral) — sin warnings
            if getattr(self, 'use_precip_extremes', False):
                sa, sb = getattr(self, 'p_ext_window', (1,5))
                xs = []
                for kk in range(sa, sb+1):
                    pbx = self._prev_basename_k(basename, kk)
                    if pbx is None: 
                        continue
                    arr = _load_week_P_unit(pbx)
                    if arr is not None:
                        xs.append(arr)

                if xs:
                    st = np.stack(xs, axis=0).astype(np.float32)  # [T,H,W]
                    st_maxsafe = np.where(np.isfinite(st), st, -1.0)
                    p_max = st_maxsafe.max(axis=0)
                    p_max = np.where(p_max >= 0.0, p_max, 0.0)
                    thr = float(getattr(self, 'p_heavy_thr', 0.6))
                    heavy_stack = np.where(np.isfinite(st) & (st >= thr), 1.0, 0.0)
                    heavy = heavy_stack.mean(axis=0).astype(np.float32)
                else:
                    p_max = np.zeros((ref_h, ref_w), np.float32)
                    heavy = np.zeros((ref_h, ref_w), np.float32)

                p_max = np.nan_to_num(p_max, nan=0.0) * mask_local.astype(np.float32)
                heavy = np.nan_to_num(heavy, nan=0.0) * mask_local.astype(np.float32)
                chans.append(p_max.astype(np.float32)); cont_flags.append(True)
                chans.append(heavy.astype(np.float32)); cont_flags.append(True)

            # T medias y degree-days
            def _stack_T_last(w):
                xs = []
                for kk in range(1, w+1):
                    pb = self._prev_basename_k(basename, kk)
                    xs.append(_load_week_T_unit(pb) if pb is not None else None)
                return xs

            t_list = _stack_T_last(self.t_avg_weeks)
            t_mean = _mean_no_warn(t_list)

            tv = t_mean[mask_local]
            tv = tv[np.isfinite(tv)]
            tau = float(np.quantile(np.clip(tv, 0.0, 1.0), self.degday_tau_q)) if tv.size > 0 else 0.5

            dd_list = [np.clip(a - tau, 0.0, 1.0) for a in t_list if a is not None]
            degday  = _mean_no_warn(dd_list)

            chans.append(t_mean); cont_flags.append(True)
            chans.append(degday);  cont_flags.append(True)

            # proporción T > tau
            if getattr(self, 'use_temp_quant_feats', False):
                abv = [(a > tau).astype(np.float32) for a in t_list if a is not None]
                prop_above = _mean_no_warn(abv) if abv else np.zeros((ref_h, ref_w), np.float32)
                chans.append(prop_above.astype(np.float32)); cont_flags.append(True)

            # NDVI en ventana + anomalía
            if getattr(self, 'use_vegetacion', False):
                xs = []
                for kk in range(2, 5):  # t-2..t-4
                    pbv = self._prev_basename_k(basename, kk)
                    if pbv is None: continue
                    v = _load_week_NDVI_unit(pbv)
                    if v is not None: xs.append(v)
                ndvi_w = _mean_no_warn(xs)
                ndvi_w = np.nan_to_num(ndvi_w, nan=0.0)
                chans.append(ndvi_w); cont_flags.append(True)

                if getattr(self, 'use_ndvi_anomaly', False):
                    def _mean_window_ndvi(a, b):
                        ys = []
                        for kk in range(a, b+1):
                            pbv = self._prev_basename_k(basename, kk)
                            if pbv is None: continue
                            v = _load_week_NDVI_unit(pbv)
                            if v is not None: ys.append(v)
                        if not ys:
                            return np.zeros((ref_h, ref_w), np.float32)
                        st = np.stack(ys, axis=0)
                        msk = np.isfinite(st); cnt = msk.sum(axis=0)
                        sumv = np.where(msk, st, 0.0).sum(axis=0)
                        out = np.divide(sumv, cnt, out=np.zeros((ref_h, ref_w), np.float32), where=(cnt>0))
                        return out.astype(np.float32)
                    a1,b1 = getattr(self, 'ndvi_anom_num', (1,4))
                    a2,b2 = getattr(self, 'ndvi_anom_den', (5,26))
                    ndvi_num = _mean_window_ndvi(a1,b1)
                    ndvi_den = _mean_window_ndvi(a2,b2)
                    ndvi_anom = (ndvi_num - ndvi_den)
                    ndvi_anom = np.where(np.isfinite(ndvi_anom), ndvi_anom, 0.0) * mask_local.astype(np.float32)
                    chans.append(ndvi_anom.astype(np.float32)); cont_flags.append(True)

            # H(x)
            if getattr(self, 'use_habitat_index', False):
                y_cur, _ = _parse_yw(basename)
                parts = []

                if getattr(self, 'use_superficies', False):
                    year_file = f"{y_cur}.tif"
                    sup_path  = os.path.join(self.data_dir, "Superficies_Construidas", year_file)
                    if os.path.exists(sup_path):
                        with rasterio.open(sup_path) as src:
                            src_ma   = src.read(1, masked=True)
                            src_data = src_ma.filled(np.float32(-9999.0))
                            dst      = np.full((ref_h, ref_w), -9999.0, np.float32)
                            reproject(
                                source=src_data,
                                destination=dst,
                                src_transform=src.transform, src_crs=src.crs,
                                dst_transform=ref_transform, dst_crs=ref_crs,
                                resampling=Resampling.bilinear,
                                src_nodata=-9999.0,
                                dst_nodata=-9999.0,
                                num_threads=1,
                                warp_mem_limit=256,
                            )
                        sup = dst
                        sup[sup == -9999.0] = np.nan
                        sup[(sup < 0.0) | (sup > 100.0)] = np.nan
                        if getattr(self, 'superficies_scale_to_unit', True):
                            sup = sup / 100.0
                        parts.append(getattr(self, 'habitat_w_built', 0.60) * np.nan_to_num(sup, nan=0.0))

                if getattr(self, 'use_vegetacion', False):
                    vg_path = os.path.join(self.data_dir, "Vegetacion", f"{basename}.tif")
                    urb = None
                    if os.path.exists(vg_path):
                        vg = self._warp_to_ref(vg_path, ref_h, ref_w, ref_transform, ref_crs,
                                               resampling=Resampling.bilinear, dst_dtype=np.float32).astype(np.float32)
                        vg[(vg < -1.2) | (vg > 1.2)] = np.nan
                        vg = 0.5 * (vg + 1.0) if getattr(self, 'vegetation_scale_to_unit', True) else vg
                        urb = np.clip(1.0 - vg, 0.0, 1.0)
                    if urb is not None:
                        parts.append(getattr(self, 'habitat_w_veg', 0.15) * np.nan_to_num(urb, nan=0.0))

                if getattr(self, 'use_nbi_serv', False):
                    fn_nbi = "nbi_serv_2005_labels.tif" if y_cur < 2018 else "nbi_serv_2018_labels.tif"
                    nbi_path = os.path.join(self.data_dir, "nbi", fn_nbi)
                    if os.path.exists(nbi_path):
                        nbi_tmp = self._warp_to_ref(nbi_path, ref_h, ref_w, ref_transform, ref_crs,
                                                    resampling=Resampling.nearest, dst_dtype=np.int16).astype(np.float32) / 4.0
                        nbi_tmp = np.clip(nbi_tmp, 0.0, 1.0)
                        parts.append(getattr(self, 'habitat_w_nbi_serv', 0.25) * nbi_tmp)

                if parts:
                    H = np.clip(np.nansum(np.stack(parts, axis=0), axis=0), 0.0, 1.0).astype(np.float32)
                    H = np.where(mask_local, H, 0.0).astype(np.float32)
                else:
                    H = np.zeros((ref_h, ref_w), np.float32)
                chans.append(H); cont_flags.append(True)

            # E(x,t) NTL anomaly
            if getattr(self, 'use_ntl_anomaly', False):
                def _load_ntl_week(bname):
                    if bname is None: return None
                    key = ("NTL", bname, geom_key, bool(self.luznoct_scale_to_unit), self.luznoct_fixed_max)
                    cached = self._cache_get(self._ntl_cache, key)
                    if cached is not None: return cached
                    pth = os.path.join(self.data_dir, "Luz_Nocturna", f"{bname}.tif")
                    if not os.path.exists(pth): return None
                    arr = _warp_cont_masked(pth, Resampling.bilinear)
                    if getattr(self, 'luznoct_scale_to_unit', False):
                        finite = arr[np.isfinite(arr)]
                        if finite.size:
                            mx = float(np.nanpercentile(finite, 99))
                        else:
                            mx = 1.0
                        mx = max(mx, 1e-6)
                        arr = np.clip(arr, 0.0, mx) / mx
                    self._cache_put(self._ntl_cache, key, arr.astype(np.float32))
                    return arr

                def _mean_window_ntl(a, b):
                    xs = []
                    for kk in range(a, b+1):
                        pbk = self._prev_basename_k(basename, kk)
                        if pbk is None: continue
                        arr = _load_ntl_week(pbk)
                        if arr is not None: xs.append(arr)
                    if not xs:
                        return np.zeros((ref_h, ref_w), np.float32)
                    st = np.stack(xs, axis=0)
                    msk = np.isfinite(st); cnt = msk.sum(axis=0)
                    sumv = np.where(msk, st, 0.0).sum(axis=0)
                    out = np.divide(sumv, cnt, out=np.zeros((ref_h, ref_w), np.float32), where=(cnt > 0))
                    return out.astype(np.float32)

                a1, b1 = getattr(self, 'ntl_anom_num', (1,4))
                a2, b2 = getattr(self, 'ntl_anom_den', (5,26))
                num = _mean_window_ntl(a1, b1)
                den = _mean_window_ntl(a2, b2)
                E = np.divide(num, (den + getattr(self, 'ntl_eps', 1e-3)),
                              out=np.zeros_like(num), where=np.isfinite(den))
                E = np.clip(E, 0.0, 2.0).astype(np.float32)
                E = np.where(mask_local, E, 0.0).astype(np.float32)
                chans.append(E); cont_flags.append(True)

            # Interacciones exógenas
            if getattr(self, 'use_exo_interactions', False):
                if (pa is not None) and (H is not None) and getattr(self, 'inter_FxH', True):
                    chans.append((pa * H).astype(np.float32));      cont_flags.append(True)
                    if t_mean is not None:
                        chans.append((t_mean * H).astype(np.float32)); cont_flags.append(True)
                if (pa is not None) and (E is not None) and getattr(self, 'inter_FxE', True):
                    chans.append((pa * E).astype(np.float32));      cont_flags.append(True)
                    if t_mean is not None:
                        chans.append((t_mean * E).astype(np.float32)); cont_flags.append(True)
                if (H is not None) and (E is not None) and getattr(self, 'inter_HxE', True):
                    chans.append((H * E).astype(np.float32));        cont_flags.append(True)

        # 3) NBI estático (+ dasimetría si procede)
        F = None
        if getattr(self, 'use_luz_nocturna', False) and (getattr(self, 'dasim_alpha', 0.0) > 0.0) and (ntl_unit is not None):
            F = self._compute_dasi_factor(basename, ntl_unit, mask_local, muni_ids)

        nbi_m, nbi_s = None, None

        if getattr(self, 'use_nbi_mise', False):
            fn = "nbi_mise_2005_labels.tif" if year < 2018 else "nbi_mise_2018_labels.tif"
            nbi_m = self._warp_to_ref(
                os.path.join(self.data_dir, "nbi", fn),
                ref_h, ref_w, ref_transform, ref_crs,
                resampling=Resampling.nearest, dst_dtype=np.int16
            ).astype(np.float32) / 4.0
            if F is not None: nbi_m = np.clip(nbi_m * F, 0.0, 1.0)
            chans.append(nbi_m); cont_flags.append(False)

        if getattr(self, 'use_nbi_serv', False):
            fn = "nbi_serv_2005_labels.tif" if year < 2018 else "nbi_serv_2018_labels.tif"
            nbi_s = self._warp_to_ref(
                os.path.join(self.data_dir, "nbi", fn),
                ref_h, ref_w, ref_transform, ref_crs,
                resampling=Resampling.nearest, dst_dtype=np.int16
            ).astype(np.float32) / 4.0
            if F is not None: nbi_s = np.clip(nbi_s * F, 0.0, 1.0)
            chans.append(nbi_s); cont_flags.append(False)

        if getattr(self, 'use_nbi_mise_dynamic', False) and (nbi_m is not None):
            dyn_with = tuple(getattr(self, 'nbi_mise_dyn_with', ("E","P")))
            if ("E" in dyn_with) and ('E' in locals()) and (E is not None):
                chans.append((nbi_m * E).astype(np.float32));  cont_flags.append(True)
            if ("P" in dyn_with) and ('pa' in locals()) and (pa is not None):
                chans.append((nbi_m * pa).astype(np.float32)); cont_flags.append(True)

        # 4) Demográficas (+ dinámicas)
        demo_cfg = [
            (getattr(self, 'use_sexo_h', False),      "Sexo_hombres"),
            (getattr(self, 'use_sexo_m', False),      "Sexo_mujeres"),
            (getattr(self, 'use_edad_menos5', False), "Edad_menos5"),
            (getattr(self, 'use_edad_5_14', False),   "Edad_5_14"),
            (getattr(self, 'use_edad_15_49', False),  "Edad_15_49"),
            (getattr(self, 'use_edad_50_69', False),  "Edad_50_69"),
            (getattr(self, 'use_edad_70mas', False),  "Edad_70mas"),
        ]
        for flag, folder in demo_cfg:
            if not flag:
                continue
            dem = self._warp_to_ref(
                os.path.join(self.data_dir, folder, f"{year}.tif"),
                ref_h, ref_w, ref_transform, ref_crs,
                resampling=Resampling.nearest, dst_dtype=np.int16
            ).astype(np.float32)
            dem /= 3.0
            if F is not None:
                dem = np.clip(dem * F, 0.0, 1.0)
            chans.append(dem); cont_flags.append(False)

            if getattr(self, 'use_dynamic_demographics', False):
                dyn_with = tuple(getattr(self, 'dyn_demo_with', ("E","P")))
                if ("E" in dyn_with) and ('E' in locals()) and (E is not None):
                    chans.append((dem * E).astype(np.float32)); cont_flags.append(True)
                if ("P" in dyn_with) and (pa is not None):
                    chans.append((dem * pa).astype(np.float32)); cont_flags.append(True)

        # 4.1) NBI dinámico
        if getattr(self, 'use_nbi_dynamic', False) and (nbi_s is not None):
            nbi_with = tuple(getattr(self, 'nbi_dyn_with', ("E","P")))
            if ("E" in nbi_with) and ('E' in locals()) and (E is not None):
                chans.append((nbi_s * E).astype(np.float32)); cont_flags.append(True)
            if ("P" in nbi_with) and (pa is not None):
                chans.append((nbi_s * pa).astype(np.float32)); cont_flags.append(True)

        # 4.5) Incidencia pasada
        if getattr(self, 'use_prev_incidence', False) and (int(getattr(self, 'prev_lags', 0)) > 0):
            lag_slices = []
            for k in range(1, int(self.prev_lags) + 1):
                arr_prev = self._load_prev_incidence_k(basename, k, ref_h, ref_w, ref_transform, ref_crs)
                mask_curr = (y_ref != 255)
                mask_prev = (arr_prev != 255)
                w_k = float(self.prev_atten[min(k-1, len(self.prev_atten)-1)])

                if getattr(self, 'prev_one_hot', False):
                    s0 = len(chans)
                    for c in (0, 1, 2, 3):
                        ch = (arr_prev == c).astype(np.float32)
                        ch = ch * mask_prev.astype(np.float32)
                        ch = ch * mask_curr.astype(np.float32)
                        ch *= w_k
                        chans.append(ch); cont_flags.append(False)
                    s1 = len(chans)
                    lag_slices.append(slice(s0, s1))
                    if (k == 1) and (self.prev_lag1_onehot_slice is None):
                        self.prev_lag1_onehot_slice = slice(s0, s1)
                else:
                    ch = arr_prev.astype(np.float32)
                    ch[~mask_prev] = 0.0
                    ch[mask_prev] /= 3.0
                    ch = ch * mask_curr.astype(np.float32)
                    ch *= w_k
                    s0 = len(chans); chans.append(ch); cont_flags.append(False); s1 = len(chans)
                    lag_slices.append(slice(s0, s1))

            if not self.prev_ch_slices:
                self.prev_ch_slices = lag_slices[:]
                idxs = []
                for sl in lag_slices:
                    idxs.extend(list(range(sl.start, sl.stop)))
                self.prev_all_idx = idxs

        # 4.6) Vegetación (semana actual)
        if getattr(self, 'use_vegetacion', False):
            key = ("NDVI", basename, geom_key, self.vegetation_scale_to_unit)
            veg = self._cache_get(self._ndvi_cache, key)
            if veg is None:
                veg_path = os.path.join(self.data_dir, "Vegetacion", f"{basename}.tif")
                if os.path.exists(veg_path):
                    veg = self._warp_to_ref(
                        veg_path, ref_h, ref_w, ref_transform, ref_crs,
                        resampling=Resampling.bilinear, dst_dtype=np.float32
                    ).astype(np.float32)
                    veg[(veg < -1.2) | (veg > 1.2)] = np.nan
                    if getattr(self, 'vegetation_scale_to_unit', True):
                        veg = 0.5 * (veg + 1.0)
                    if getattr(self, 'vegetation_missing_fallback', "zeros") == "zeros":
                        veg = np.nan_to_num(veg, nan=0.0)
                else:
                    veg = np.zeros((ref_h, ref_w), dtype=np.float32) if \
                          getattr(self, 'vegetation_missing_fallback', "zeros") == "zeros" else \
                          np.full((ref_h, ref_w), np.nan, dtype=np.float32)
                self._cache_put(self._ndvi_cache, key, veg)
            chans.append(veg.astype(np.float32)); cont_flags.append(True)

        # 4.7) Superficies_Construidas
        if getattr(self, 'use_superficies', False):
            year_file = f"{year}.tif"
            sup_path  = os.path.join(self.data_dir, "Superficies_Construidas", year_file)
            if os.path.exists(sup_path):
                with rasterio.open(sup_path) as src:
                    src_ma   = src.read(1, masked=True)
                    src_data = src_ma.filled(np.float32(-9999.0))
                    dst      = np.full((ref_h, ref_w), -9999.0, np.float32)
                    reproject(
                        source=src_data,
                        destination=dst,
                        src_transform=src.transform, src_crs=src.crs,
                        dst_transform=ref_transform, dst_crs=ref_crs,
                        resampling=Resampling.bilinear,
                        src_nodata=-9999.0,
                        dst_nodata=-9999.0,
                        num_threads=1,
                        warp_mem_limit=256,
                    )
                sup = dst
                sup[sup == -9999.0] = np.nan
                sup[(sup < 0.0) | (sup > 100.0)] = np.nan
                if getattr(self, 'superficies_scale_to_unit', True):
                    sup = sup / 100.0
                if getattr(self, 'superficies_missing_fallback', "zeros") == "zeros":
                    sup = np.nan_to_num(sup, nan=0.0)
            else:
                sup = np.zeros((ref_h, ref_w), dtype=np.float32) if \
                      getattr(self, 'superficies_missing_fallback', "zeros") == "zeros" else \
                      np.full((ref_h, ref_w), np.nan, dtype=np.float32)
            chans.append(sup.astype(np.float32)); cont_flags.append(True)

        # 4.8) Luz Nocturna (canal final)
        if getattr(self, 'use_luz_nocturna', False):
            if getattr(self, 'viirs_calibrate_to_nbi', False) and (ntl_unit is not None) and ('muni_ids' in locals()) and (muni_ids is not None):
                ntl_final = self._calibrate_ntl_to_nbi(basename, ntl_unit, mask_local, muni_ids, nbi_m, nbi_s)
            else:
                ntl_final = ntl_unit if ntl_unit is not None else np.zeros((ref_h, ref_w), dtype=np.float32)
            chans.append(ntl_final.astype(np.float32)); cont_flags.append(True)

        # 5) Máscara válida + saneo
        mask = (y_ref != 255).astype(np.float32)
        chans = [np.where(np.isfinite(c), c, 0.0) * mask for c in chans]

        if getattr(self, 'add_valid_mask_channel', False):
            chans.append(mask.astype(np.float32)); cont_flags.append(False)

        # 6) Resize opcional
        need_resize = (getattr(self, 'target_size', None) is not None) and (self.target_size != (ref_h, ref_w))
        if need_resize:
            y_resized = self._resize_np(y_ref.astype(np.int32),
                                        self.target_size, Image.NEAREST).astype(np.int64)
            mask = self._resize_np(mask.astype(np.uint8),
                                   self.target_size, Image.NEAREST).astype(np.float32)
            resized = []
            for arr, is_cont in zip(chans, cont_flags):
                resample = Image.BILINEAR if is_cont else Image.NEAREST
                resized.append(
                    self._resize_np(arr.astype(np.float32),
                                    self.target_size, resample).astype(np.float32)
                )
            chans = resized
            y_ref = y_resized

        # Sanity: #canales estable
        currC = len(chans)
        if not hasattr(self, "_expected_C") or self._expected_C is None:
            self._expected_C = currC
        else:
            assert currC == self._expected_C, f"Channel drift en {basename}: {currC} vs {self._expected_C}"

        # 7) Tensores
        X = torch.from_numpy(np.stack(chans)).float()  # [C,H,W]
        y = torch.from_numpy(y_ref).long()
        m = torch.from_numpy(mask).float()
        return X, y, m

    def __len__(self):
        return len(self.basename_list)

DengueDataset._pt_shift_weeks = 0

def _sliding_windows(H: int, W: int, patch: int, stride: int):
    for y0 in range(0, max(1, H - patch + 1), stride):
        for x0 in range(0, max(1, W - patch + 1), stride):
            yield y0, x0, patch, patch

@dataclass
class PatchRec:
    basename: str
    y0: int
    x0: int
    h: int
    w: int
    dom: int
    fracs: Tuple[float, float, float, float]
    minor_sum: float
    purity: float

def build_train_patch_index(train_ds: DengueDataset,
                            patch_size: int = 128,
                            stride: int = 64) -> List[PatchRec]:

    out: List[PatchRec] = []
    
    for b in tqdm(train_ds.basename_list, desc="Barrido TRAIN para parches", ncols=80):
        inc_path = os.path.join(train_ds.data_dir, "Incidencia", f"{b}.tif")
        with rasterio.open(inc_path) as src:
            y = src.read(1).astype(np.int16)
            ref_h, ref_w = src.height, src.width
            ref_transform, ref_crs = src.transform, src.crs
        # restringe al depto del dataset base
        dept_mask = train_ds._dept_mask(ref_h, ref_w, ref_transform, ref_crs)
        y[~dept_mask] = 255
        valid = (y != 255)
        H, W = y.shape
        for y0, x0, h, w in _sliding_windows(H, W, patch_size, stride):
            win = y[y0:y0+h, x0:x0+w]
            v   = valid[y0:y0+h, x0:x0+w]
            tot = int(v.sum())
            if tot == 0:
                continue

            cnt = Counter()
            for c in (0,1,2,3):
                cnt[c] = int(((win == c) & v).sum())
            if sum(cnt.values()) == 0:
                continue

            fracs = tuple(cnt[c] / tot for c in (0,1,2,3))
            dom   = int(np.argmax(fracs))
            purity = float(max(fracs))
            minor_sum = float(fracs[1] + fracs[2] + fracs[3])

            out.append(PatchRec(b, y0, x0, h, w, dom, fracs, minor_sum, purity))
    return out



def select_boundary_group(index, groupA=(0,1), groupB=(2,3),
                          minA=0.15, minB=0.15, max_purity=0.98,
                          target=360, per_file_cap=10):

    out, used = [], {}
    def frac_group(fracs, grp): return sum(fracs[c] for c in grp)
    cands = [p for p in index
             if frac_group(p.fracs, groupA) >= minA
             and frac_group(p.fracs, groupB) >= minB
             and p.purity <= max_purity]
    # prioriza bordes "equilibrados"
    cands.sort(key=lambda p: -min(frac_group(p.fracs, groupA), frac_group(p.fracs, groupB)))
    for p in cands:
        k = (p.basename, p.dom)
        if used.get(k, 0) >= per_file_cap: continue
        out.append(p); used[k] = used.get(k,0)+1
        if target and len(out) >= target: break
    return out

class PatchCropDataset(Dataset):

    def __init__(self, base: DengueDataset, patch_list: List[PatchRec], patch_size: int, augment: bool = False):
        self.base = base
        self.patch_list = patch_list
        self.patch = patch_size
        self.augment = augment
        # índice rápido: basename -> idx en base
        self._name_to_idx = {b:i for i,b in enumerate(base.basename_list)}
        self._tile_cache = {}        # cache simple por basename
        self._tile_cache_max = 2     # mantén 2 mapas por worker

    def __len__(self):
        return len(self.patch_list)

    def __getitem__(self, i):
        rec = self.patch_list[i]
        idx = self._name_to_idx[rec.basename]
        if rec.basename in self._tile_cache:
            X, y, m = self._tile_cache[rec.basename]
        else:
            X, y, m = self.base[idx]
            # cachea y controla tamaño
            if len(self._tile_cache) >= self._tile_cache_max:
                self._tile_cache.pop(next(iter(self._tile_cache)))
            self._tile_cache[rec.basename] = (X, y, m)
        y0, x0, h, w = rec.y0, rec.x0, rec.h, rec.w

        Xp = X[:, y0:y0+h, x0:x0+w]
        yp = y[y0:y0+h, x0:x0+w]
        mp = m[y0:y0+h, x0:x0+w]

        # seguridad (puede ocurrir borde: patch parcial)
        if Xp.shape[-2:] != (self.patch, self.patch):
            # pad hasta tamaño patch
            pad_h = self.patch - Xp.shape[-2]
            pad_w = self.patch - Xp.shape[-1]
            Xp = F.pad(Xp, (0,pad_w,0,pad_h))
            yp = F.pad(yp, (0,pad_w,0,pad_h), value=255)
            mp = F.pad(mp, (0,pad_w,0,pad_h), value=0.0)

        if self.augment:
            # rotación 0/90/180/270
            k = random.randint(0, 3)
            if k:
                Xp = torch.rot90(Xp, k, dims=[1, 2])   # [C,H,W]
                yp = torch.rot90(yp, k, dims=[0, 1])   # [H,W]
                mp = torch.rot90(mp, k, dims=[0, 1])   # [H,W]

            # flip horizontal
            if random.random() < 0.5:
                Xp = torch.flip(Xp, dims=[2])
                yp = torch.flip(yp, dims=[1])
                mp = torch.flip(mp, dims=[1])

            # flip vertical
            if random.random() < 0.5:
                Xp = torch.flip(Xp, dims=[1])
                yp = torch.flip(yp, dims=[0])
                mp = torch.flip(mp, dims=[0])

        return Xp, yp, mp

class RandomPatchDataset(Dataset):

    def __init__(self, base: DengueDataset, patch_size: int, samples_per_epoch: int = 4000, augment: bool = False):
        self.base = base
        self.patch = patch_size
        self.N = samples_per_epoch
        self.augment = augment
        self._name_to_idx = {b:i for i,b in enumerate(base.basename_list)}  # <<<
        self._cache = {}

    def __len__(self):
        return self.N

    def __getitem__(self, i):
        # elige un basename al azar y carga el tile completo
        b = random.choice(self.base.basename_list)
        idx = self._name_to_idx[b]
        if idx in self._cache:
            X, y, m = self._cache[idx]
        else:
            X, y, m = self.base[idx]
            # cachea último tile visto; si quieres LRU, limita a 2-3 keys
            if len(self._cache) >= 2:
                self._cache.pop(next(iter(self._cache)))
            self._cache[idx] = (X, y, m)

        H, W = y.shape
        P = self.patch

        # umbrales (modificables externamente: dataset.min_* = ...)
        min_valid_frac = float(getattr(self, "min_valid_frac_std", 0.10))   # ≥10% píxeles válidos
        min_minor_frac = float(getattr(self, "min_minor_frac_std", 0.10))   # ≥10% de {1,2,3} dentro del válido
        max_tries = int(getattr(self, "max_tries_std", 80))

        best = None
        best_score = -1.0  # usamos la fracción minor como “score”

        # intenta encontrar un parche con suficiente válido y suficiente minoría
        for _ in range(max_tries):
            y0 = random.randint(0, max(0, H - P))
            x0 = random.randint(0, max(0, W - P))

            mp_win = m[y0:y0+P, x0:x0+P]               # máscara válida (float)
            valid = mp_win.sum().item()                # #píxeles válidos aproximado
            if valid < min_valid_frac * (P * P):
                continue

            yp_win = y[y0:y0+P, x0:x0+P]

            valid_mask = ((yp_win != 255) & (mp_win > 0))   # bool
            valid_count = valid_mask.sum().item()
            if valid_count <= 0:
                continue

            minor_mask = valid_mask & (yp_win > 0)          # clases {1,2,3}
            frac_minor = minor_mask.sum().item() / (valid_count + 1e-6)

            # si cumple el umbral, aceptamos inmediatamente
            if frac_minor >= min_minor_frac:
                Xp = X[:, y0:y0+P, x0:x0+P]
                yp = yp_win
                mp = mp_win
                best = (Xp, yp, mp)
                best_score = frac_minor
                break

            # si no, guarda el mejor candidato visto hasta ahora
            if frac_minor > best_score:
                Xp_tmp = X[:, y0:y0+P, x0:x0+P]
                best = (Xp_tmp, yp_win, mp_win)
                best_score = frac_minor

        # fallback: si no se alcanzó el umbral, usa el mejor parche encontrado
        if best is None:
            # recorte centrado como último recurso
            y0 = max(0, H // 2 - P // 2)
            x0 = max(0, W // 2 - P // 2)
            Xp = X[:, y0:y0+P, x0:x0+P]
            yp = y[y0:y0+P, x0:x0+P]
            mp = m[y0:y0+P, x0:x0+P]
        else:
            Xp, yp, mp = best

        # bordes → pad hasta tamaño P si hace falta
        if Xp.shape[-2:] != (P, P):
            pad_h = P - Xp.shape[-2]
            pad_w = P - Xp.shape[-1]
            Xp = F.pad(Xp, (0, pad_w, 0, pad_h))
            yp = F.pad(yp, (0, pad_w, 0, pad_h), value=255)
            mp = F.pad(mp, (0, pad_w, 0, pad_h), value=0.0)

        # augmentaciones simples (rot/flip) consistentes en X/y/m
        if self.augment:
            k = random.randint(0, 3)
            if k:
                Xp = torch.rot90(Xp, k, dims=[1, 2])
                yp = torch.rot90(yp, k, dims=[0, 1])
                mp = torch.rot90(mp, k, dims=[0, 1])

            if random.random() < 0.5:
                Xp = torch.flip(Xp, dims=[2]); yp = torch.flip(yp, dims=[1]); mp = torch.flip(mp, dims=[1])
            if random.random() < 0.5:
                Xp = torch.flip(Xp, dims=[1]); yp = torch.flip(yp, dims=[0]); mp = torch.flip(mp, dims=[0])

        return Xp, yp, mp
    

# ──────────────────────────────────────────────────────────────────────────────
# 3) Iterador que mezcla dos DataLoaders por proporción (minor/estándar)
# ──────────────────────────────────────────────────────────────────────────────
class MixedBatchIterator:

    def __init__(self, loader_minor, loader_std, p_minor=0.6, steps_per_epoch=1000):
        self.loader_minor = loader_minor
        self.loader_std   = loader_std
        self.p_minor      = float(p_minor)
        self.steps = int(steps_per_epoch)
        self._it_minor = None
        self._it_std   = None

    def __iter__(self):
        self._it_minor = iter(self.loader_minor)
        self._it_std   = iter(self.loader_std)
        for _ in range(self.steps):
            use_minor = (random.random() < self.p_minor)
            it = self._it_minor if use_minor else self._it_std
            try:
                batch = next(it)
            except StopIteration:
                # reiniciar
                if use_minor:
                    self._it_minor = iter(self.loader_minor)
                    batch = next(self._it_minor)
                else:
                    self._it_std = iter(self.loader_std)
                    batch = next(self._it_std)
            yield batch

    def __len__(self):
        return self.steps

class UNetPlusPlusDS(nn.Module):

    def __init__(self, in_channels, n_classes=2, deep_supervision=True, base_ch=32):
        super().__init__()
        self.deep_supervision = deep_supervision

        # Encoder (igual que U-Net pero más compacto y ramificado)
        def CBR(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True)
            )
        chs = [base_ch, base_ch*2, base_ch*4, base_ch*8, base_ch*16]
        self.conv0_0 = CBR(in_channels, chs[0])
        self.conv1_0 = CBR(chs[0],    chs[1])
        self.conv2_0 = CBR(chs[1],    chs[2])
        self.conv3_0 = CBR(chs[2],    chs[3])
        self.conv4_0 = CBR(chs[3],    chs[4])

        self.maxpool = nn.MaxPool2d(2)
        self.up = lambda x, scale: nn.functional.interpolate(x, scale_factor=scale, mode='bilinear', align_corners=True)

        # Nested convs
        self.conv0_1 = CBR(chs[0]+chs[1], chs[0])
        self.conv1_1 = CBR(chs[1]+chs[2], chs[1])
        self.conv2_1 = CBR(chs[2]+chs[3], chs[2])
        self.conv3_1 = CBR(chs[3]+chs[4], chs[3])

        self.conv0_2 = CBR(chs[0]*2+chs[1], chs[0])
        self.conv1_2 = CBR(chs[1]*2+chs[2], chs[1])
        self.conv2_2 = CBR(chs[2]*2+chs[3], chs[2])

        self.conv0_3 = CBR(chs[0]*3+chs[1], chs[0])
        self.conv1_3 = CBR(chs[1]*3+chs[2], chs[1])

        self.conv0_4 = CBR(chs[0]*4+chs[1], chs[0])

        # Classifiers para deep supervision
        self.final1 = nn.Conv2d(chs[0], n_classes, 1)
        self.final2 = nn.Conv2d(chs[0], n_classes, 1)
        self.final3 = nn.Conv2d(chs[0], n_classes, 1)
        self.final4 = nn.Conv2d(chs[0], n_classes, 1)

    def forward(self, x):
        # Encoder
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.maxpool(x0_0))
        x2_0 = self.conv2_0(self.maxpool(x1_0))
        x3_0 = self.conv3_0(self.maxpool(x2_0))
        x4_0 = self.conv4_0(self.maxpool(x3_0))

        # Decoder ++
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0, 2)], 1))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0, 2)], 1))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0, 2)], 1))
        x3_1 = self.conv3_1(torch.cat([x3_0, self.up(x4_0, 2)], 1))

        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1, 2)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1, 2)], 1))
        x2_2 = self.conv2_2(torch.cat([x2_0, x2_1, self.up(x3_1, 2)], 1))

        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2, 2)], 1))
        x1_3 = self.conv1_3(torch.cat([x1_0, x1_1, x1_2, self.up(x2_2, 2)], 1))

        x0_4 = self.conv0_4(torch.cat([x0_0, x0_1, x0_2, x0_3, self.up(x1_3, 2)], 1))

        # Deep supervision: retorna lista de salidas
        if self.deep_supervision:
            output1 = self.final1(x0_1)
            output2 = self.final2(x0_2)
            output3 = self.final3(x0_3)
            output4 = self.final4(x0_4)
            return [output1, output2, output3, output4]
        else:
            output = self.final4(x0_4)
            return output


def seed_worker(worker_id):
    worker_seed = SEED + worker_id
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers de canales para lags de incidencia
# ──────────────────────────────────────────────────────────────────────────────

def restrict_prev_for_s1_inplace(
    X: torch.Tensor,
    base_ds,
    *,
    zero_t1: bool = True,       # anula t-1
    keep_from_lag: int = 2,     # deja desde t-2 en adelante
    smooth_kernel: int = 5,     # suavizado por promedio k×k
    dropout_p: float = 0.30,    # dropout por lag (t>=keep_from_lag)
    scale: float = 0.70         # atenuación global de lags retenidos
) -> None:

    sls = getattr(base_ds, "prev_ch_slices", None)
    if not sls:
        return
    for lag_idx, sl in enumerate(sls, start=1):
        # 1) anula t-1
        if zero_t1 and lag_idx == 1:
            X[:, sl, :, :] = 0.0
            continue
        # 2) descarta lags anteriores al umbral (por si cambias keep_from_lag)
        if lag_idx < int(keep_from_lag):
            X[:, sl, :, :] = 0.0
            continue
        # 3) dropout por lag
        if float(dropout_p) > 0 and torch.rand(1).item() < float(dropout_p):
            X[:, sl, :, :] = 0.0
            continue
        # 4) suavizado espacial (proximidad en vez de contornos duros)
        k = int(smooth_kernel)
        if k > 1:
            pad = k // 2
            X[:, sl, :, :] = F.avg_pool2d(X[:, sl, :, :], kernel_size=k, stride=1, padding=pad)
        # 5) atenuación
        if scale is not None:
            X[:, sl, :, :] *= float(scale)


def main():

    # ──────────────────────────────────────────────────────────────────────────────
    # 6) MÉTRICAS CON RunningScore Y FUNCIÓN evaluation
    # ──────────────────────────────────────────────────────────────────────────────
    class RunningScore:
        def __init__(self, n_classes, ignore_index=255):
            self.n_classes = n_classes
            self.ignore_index = ignore_index
            self.confusion_matrix = np.zeros((n_classes, n_classes), dtype=np.int64)

        def _fast_hist(self, label_true, label_pred):
            mask = (
                (label_true != self.ignore_index) &
                (label_true >= 0) &
                (label_true < self.n_classes) &
                (label_pred >= 0) &
                (label_pred < self.n_classes)
            )
            hist = np.bincount(
                self.n_classes * label_true[mask].astype(int)
                + label_pred[mask].astype(int),
                minlength=self.n_classes ** 2
            ).reshape(self.n_classes, self.n_classes)
            return hist

        def update(self, label_trues, label_preds):
            for lt, lp in zip(label_trues, label_preds):
                self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

        def get_scores(self):
            hist = self.confusion_matrix
            overall_acc = np.diag(hist).sum() / hist.sum()
            with np.errstate(divide='ignore', invalid='ignore'):
                acc_cls = np.diag(hist) / hist.sum(axis=1)
            mean_acc = np.nanmean(acc_cls)
            freq = hist.sum(axis=1) / hist.sum()
            iu = np.diag(hist) / (hist.sum(axis=1) + hist.sum(axis=0) - np.diag(hist))
            mean_iu = np.nanmean(iu)
            fwavacc = np.nansum(freq * iu)
            dice = {}
            for c in range(self.n_classes):
                tp = hist[c, c]
                fp = hist[:, c].sum() - tp
                fn = hist[c, :].sum() - tp
                denom = (2 * tp + fp + fn)
                dice[c] = 2 * tp / denom if denom > 0 else np.nan
            class_iou = dict(zip(range(self.n_classes), iu))
            return {
                "OverallAcc": overall_acc,
                "MeanAcc": mean_acc,
                "FreqWAcc": fwavacc,
                "MeanIoU": mean_iu,
                "Dice": dice
            }, class_iou

    # ──────────────────────────────────────────────────────────────────────────────
    # Evaluación BINARIA (0 vs RESTO)
    # ──────────────────────────────────────────────────────────────────────────────
    @torch.no_grad()
    def eval_binary(model_bin, val_loader, device, *, threshold=0.5, ignore_index=255, tag="VAL_BIN", preprocess_X=None):
        model_bin.eval()
        scorer = RunningScore(n_classes=2, ignore_index=ignore_index)
        for Xv, yv, mv in val_loader:
            Xv = Xv.to(device, non_blocking=True)
            if callable(preprocess_X): 
                preprocess_X(Xv)  # in-place si aplica restricción de lags para S1

            # forward (deep supervision -> tomar el último)
            out = model_bin(Xv)
            out = out[-1] if isinstance(out, list) else out  # [B,2,H,W] logits

            # prob RESTO = softmax[:,1], pred = (p_rest>=thr)
            p = torch.softmax(out, dim=1)[:, 1]              # [B,H,W]
            pred = (p >= float(threshold)).long()            # 0/1

            # target binario con -100=ignore y 0/1 en válidos
            ybin = _bin_targets_s1_0vR(yv.to(device), mv.to(device)).cpu().numpy()
            pred_np = pred.detach().cpu().numpy()

            # mapear -100→255 para el scorer; ya ignora 255
            ybin[ybin < 0] = 255
            scorer.update(ybin, pred_np)

        score, iou = scorer.get_scores()
        miou = float(score["MeanIoU"])
        print(f"[{tag}] mIoU(BIN)={miou:.4f}  IoU0={iou.get(0, float('nan')):.4f}  IoU1={iou.get(1, float('nan')):.4f}")
        return score, iou, scorer.confusion_matrix
        
    # ──────────────────────────────────────────────────────────────────────────────
    # Entrenamiento BINARIO end-to-end (usa UNetPlusPlusDS con n_classes=2)
    # ──────────────────────────────────────────────────────────────────────────────
    def train_binary_end_to_end(
            model_bin: UNetPlusPlusDS,
            mixed_iter: MixedBatchIterator,
            val_loader,
            device,
            *,
            base_train_ds=None,
            epochs=100,
            patience=10,
            lr=1e-4,
            weight_decay=1e-4,
            accum_steps=8,
            max_grad_norm=1.0,
            deep_supervision=True,
            ds_weights=(0.05, 0.15, 0.30, 0.50),
            class_weights_bin=None,            # [w0,w1] o None
            p_minor_fixed=0.85,                # igual que usabas en plan A
            p_minor_warm=0,                    # sin schedule por defecto
            steps_per_epoch=1000,
            threshold=0.5,
            s1_keep_from_lag=2,
            s1_smooth_kernel=5,
            s1_dropout_p=0.30,
            s1_scale=0.70,
            log_prefix="[BIN S1]"):
        model_bin = model_bin.to(device)
        optimizer = torch.optim.AdamW(model_bin.parameters(), lr=lr, weight_decay=weight_decay)
        scaler = GradScaler(enabled=(device.type == "cuda"))

        ce_bin = torch.nn.CrossEntropyLoss(
            weight=(torch.tensor(class_weights_bin, dtype=torch.float32, device=device) 
                    if class_weights_bin is not None else None),
            ignore_index=-100,
            reduction='mean'
        )

        def _sched_p_minor(ep):
            if p_minor_fixed is not None:
                return float(p_minor_fixed)
            t = min(max((ep-1)/(max(1,p_minor_warm)-1), 0.0), 1.0)
            return 0.85 + (0.65 - 0.85) * t

        best_miou = -1.0
        best_state = None
        wait = 0

        for ep in range(1, epochs+1):
            model_bin.train()
            mixed_iter.p_minor = _sched_p_minor(ep)
            running_loss = 0.0
            steps_done = 0
            optimizer.zero_grad(set_to_none=True)

            for step, (X, y, m) in enumerate(mixed_iter, 1):
                X = X.to(device, non_blocking=True)
                y = y.to(device, non_blocking=True)
                m = m.to(device, non_blocking=True)

                # targets binarios para S1
                y1 = _bin_targets_s1_0vR(y, m)  # -100/0/1

                # restricción de lags estable que ya usabas para S1
                X_s1 = X.clone()
                if base_train_ds is not None:
                    restrict_prev_for_s1_inplace(
                        X_s1, base_train_ds,
                        zero_t1=True, keep_from_lag=s1_keep_from_lag,
                        smooth_kernel=s1_smooth_kernel, dropout_p=s1_dropout_p, scale=s1_scale
                    )

                with autocast(device_type=device.type):
                    out = model_bin(X_s1)
                    if isinstance(out, list) and deep_supervision:
                        outs = out
                        last = outs[-1]
                        loss = sum(float(w) * ce_bin(o, y1) for w, o in zip(ds_weights, outs))
                    else:
                        last = out[-1] if isinstance(out, list) else out
                        loss = ce_bin(last, y1)

                scaler.scale(loss / max(1, accum_steps)).backward()
                running_loss += float(loss.item())
                steps_done += 1

                if (step % accum_steps) == 0:
                    if max_grad_norm:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model_bin.parameters(), max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

            # flush resto
            if (steps_done % max(1, accum_steps)) != 0:
                if max_grad_norm:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model_bin.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            avg_loss = running_loss / max(1, steps_done)

            # ---------- VALIDACIÓN binaria ----------
            def _pre_s1(Xin):
                Xs = Xin.clone()
                if base_train_ds is not None:
                    restrict_prev_for_s1_inplace(
                        Xs, base_train_ds,
                        zero_t1=True, keep_from_lag=s1_keep_from_lag,
                        smooth_kernel=s1_smooth_kernel, dropout_p=0.0, scale=s1_scale
                    )
                return Xs

            score, iou, _ = eval_binary(model_bin, val_loader, device, threshold=threshold, tag=f"{log_prefix} Ep {ep:03d}/VAL", preprocess_X=_pre_s1)
            miou = float(score["MeanIoU"])
           # print(f"{log_prefix} Ep {ep:03d}/{epochs}  loss={avg_loss:.4f}  val_mIoU={miou:.4f}  IoU={{{0}: {iou.get(0,0.0):.4f}, {1}: {iou.get(1,0.0):.4f}}}")
            print(f"{ep:03d}/{epochs}  loss={avg_loss:.4f}  ")

            if miou > best_miou + 1e-6:
                best_miou = miou
                best_state = {
                    "model": {k: v.detach().cpu() for k, v in model_bin.state_dict().items()}
                }
                wait = 0
                print(f"   ✓ nuevo BEST mIoU={miou:.4f}")
            else:
                wait += 1
                if wait >= patience:
                    print("   → early stop por paciencia")
                    break

        if best_state is not None:
            model_bin.load_state_dict(best_state["model"])
        return model_bin

    # ──────────────────────────────────────────────────────────────────────────
    # FUNCS de binarización para CALIBRACIÓN
    # ──────────────────────────────────────────────────────────────────────────

    def _bin_targets_s1_0vR(y, m):
        # y: [B,H,W] {0,1,2,3,255}
        # m: [B,H,W] máscara válida (float 0/1)
        ve = erode_valid(m, ERODE_ITERS)
        t  = y.clone()
        valid_base  = (t != 255)
        valid_final = valid_base & (ve > 0)

        # RESTO crudo = {1,2,3}
        resto = (t == 1) | (t == 2) | (t == 3)

        # Construir target binario con ignore = -100
        yb = torch.full_like(t, -100, dtype=torch.long)
        yb[valid_final] = 0               # por defecto clase 0
        yb[resto & valid_final] = 1       # RESTO (dilatado) → 1
        return yb.long()

    def visualize_cascade_maps(dataset, predict_fn, device, save_dir):
        """
        Versión BINARIA (0vs1) con el MISMO diseño que estás usando para exportar:
        - Figura 1920x1080 (dpi=100)
        - Panel izq: "Original Map"
        - Panel der: "Binary Prediction"
        - Colores: ['#808080', '#000080', '#FFAA00'] = [Out, Class 0, Class 1]
        - Leyenda abajo ("Out of Area", "Class 0", "Class 1") con label "0vs1"
        - Supertítulo = Wxx-YYYY
        - Para 'norte_santander' y 'tolima' los paneles SE TOCAN (superposición)
        - Guardado idéntico (sin bbox_inches='tight', dpi=100)
        """
        os.makedirs(save_dir, exist_ok=True)

        # === estilos exactos ===
        CMAP = ListedColormap(['#808080', '#000080', '#FFAA00'])
        NORM = BoundaryNorm([-1.5, -0.5, 0.5, 1.5], 3)

        def _weekyear(basename: str):
            y = re.search(r"(20\d{2})", basename)
            w = re.search(r"[Ww](\d{1,2})", basename)
            return f"W{int(w.group(1))}-{y.group(1)}" if (y and w) else basename

        def _add_axes_pair(fig, dept_name):
            """
            Para 'norte_santander' y 'tolima' hago que los ejes SE TOQUEN
            (superposición como en tu script). El resto usa GridSpec con wspace pequeño.
            """
            if dept_name in ("norte_santander", "tolima"):
                left, bottom, height = 0.03, 0.16, 0.74
                center = 0.50
                overlap = 0.3  # mismo valor que estás usando ahora
                w_left  = 0.47 + overlap/2
                w_right = 0.47 + overlap/2

                axL = fig.add_axes([left, bottom, w_left, height])
                axR = fig.add_axes([center - overlap/2, bottom, w_right, height])

                for ax in (axL, axR):
                    ax.set_facecolor('none')
                    for spine in ax.spines.values():
                        spine.set_visible(False)
                axR.set_zorder(axL.get_zorder() + 1)
            else:
                gs = fig.add_gridspec(1, 2, left=0.03, right=0.97, top=0.92, bottom=0.14, wspace=0.01)
                axL = fig.add_subplot(gs[0, 0])
                axR = fig.add_subplot(gs[0, 1])
            return axL, axR

        dept_name = getattr(dataset, 'dept_name', "")

        for idx in range(len(dataset)):
            X, y, _ = dataset[idx]
            basename = dataset.basename_list[idx]

            # --- inferencia (acepta logits [1,2,H,W] o prob [1,1,H,W]/[1,H,W]) ---
            with torch.no_grad():
                out = predict_fn(X.unsqueeze(0).to(device))
                if not isinstance(out, torch.Tensor):
                    raise RuntimeError("predict_fn debe devolver un torch.Tensor")
                if out.ndim == 4 and out.shape[1] == 2:
                    p1 = torch.softmax(out, dim=1)[:, 1]
                else:
                    p1 = out
                    if p1.ndim == 4 and p1.shape[1] == 1:
                        p1 = p1[:, 0]
                pred_bin = (p1 >= 0.5).long().squeeze(0).cpu().numpy().astype(np.int16)

            # --- GT binaria y enmascarado fuera de área ---
            y_np  = y.numpy().astype(np.int16, copy=True)
            y_bin = np.where(y_np == 255, -1, np.where(y_np == 0, 0, 1)).astype(np.int16)
            pred_bin[y_bin == -1] = -1

            # --- recorte a bbox válida (reduce franjas) ---
            valid = (y_bin != -1)
            if valid.any():
                rows = np.where(valid.any(axis=1))[0]
                cols = np.where(valid.any(axis=0))[0]
                r0, r1 = rows[0], rows[-1] + 1
                c0, c1 = cols[0], cols[-1] + 1
                y_plot    = y_bin[r0:r1, c0:c1]
                pred_plot = pred_bin[r0:r1, c0:c1]
            else:
                y_plot, pred_plot = y_bin, pred_bin

            # --- figura idéntica: 1920x1080 @ 100 dpi ---
            fig = plt.figure(figsize=(19.2, 10.8), dpi=100)
            ax0, ax1 = _add_axes_pair(fig, dept_name)

            ax0.imshow(y_plot,   cmap=CMAP, norm=NORM, origin='upper', interpolation='nearest', aspect='equal')
            ax1.imshow(pred_plot, cmap=CMAP, norm=NORM, origin='upper', interpolation='nearest', aspect='equal')

            ax0.set_title("Original Map", fontsize=20, pad=6); ax0.axis('off')
            ax1.set_title("Binary Prediction", fontsize=20, pad=6); ax1.axis('off')

            # --- leyenda abajo en misma posición y subida 15 px (como tu script) ---
            cax = fig.add_axes([0.20, 0.05, 0.60, 0.03])
            dy = 15.0 / (fig.dpi * fig.get_size_inches()[1])  # 15 px en coords de figura
            pos = cax.get_position()
            cax.set_position([pos.x0, pos.y0 + dy, pos.width, pos.height])

            cb = plt.colorbar(
                plt.cm.ScalarMappable(norm=NORM, cmap=CMAP),
                cax=cax, orientation='horizontal', ticks=[-1, 0, 1]
            )
            cb.ax.set_xticklabels(["Out of Area", "Class 0", "Class 1"])
            cb.ax.tick_params(labelsize=16)
            cb.set_label("0vs1", fontsize=17, labelpad=4)

            # --- supertítulo sólo con Wxx-YYYY ---
            wy = _weekyear(basename)
            fig.suptitle(f"{wy}", fontsize=23, y=0.955)

            # --- guardado idéntico ---
            out_path = os.path.join(save_dir, f"binary_{basename}.png")
            fig.savefig(out_path, dpi=100)
            plt.close(fig)
            print(f"→ Guardado: {out_path}")



    def compute_split_class_freqs(ds: DengueDataset):
        """Devuelve frecuencias por clase [4] ignorando 255."""
        cnt = np.zeros(4, dtype=np.float64)
        tot = 0.0
        for b in ds.basename_list:
            path = os.path.join(ds.data_dir, "Incidencia", f"{b}.tif")
            with rasterio.open(path) as src:
                arr = src.read(1).astype(np.int64)
                ref_h, ref_w = src.height, src.width
                ref_transform, ref_crs = src.transform, src.crs

            # máscara válida por nodata
            valid = (arr != 255)

            # ── NUEVO: limitar al depto si aplica ────────────────────────────────────────
            if ds.dept_code is not None:
                dept_mask = ds._dept_mask(ref_h, ref_w, ref_transform, ref_crs)
                valid = valid & dept_mask
                arr[~dept_mask] = 255  # uniformidad con dataset

            for c in range(4):
                cnt[c] += int(((arr == c) & valid).sum())
            tot += valid.sum()

        if tot == 0:
            return np.ones(4) / 4.0
        return (cnt / cnt.sum()).astype(np.float64)


    # ── Iteración por departamentos seleccionados ─────────────────────────────────
    DEPT_ORDER = list(DEPARTMENTS.keys())  # o subset si quieres
    for _DEPT_NAME in DEPT_ORDER:
        _DEPT_CODE = DEPARTMENTS[_DEPT_NAME]["code"]

        # ─── CICLO AUTOMATIZADO PARA TODOS LOS EXPERIMENTOS ───────────────────────
        for period in PERIODS:
            print(f"\n=== DEPARTAMENTO: {_DEPT_NAME} ===")
            print(f"\n=== PERÍODO: {period} ===")
            periodo_dir = os.path.join(DATA_DIR, period)
            out_base    = os.path.join(OUTPUT_DIR, period)
            os.makedirs(out_base, exist_ok=True)

            for feat_name, feat_cfg in FEATURE_CONFIGS:

                print(f"\n[Feat: {feat_name}]")
                # Carpeta por departamento
                expdir = os.path.join(out_base, f"{feat_name}__dept_{_DEPT_NAME}")
                os.makedirs(expdir, exist_ok=True)

                # 1) Datasets a tamaño nativo (MAPA COMPLETO; sin resize) con FOCUS por dept
                train_ds = DengueDataset(
                    data_dir   = periodo_dir,
                    mode       = 'train',
                    target_size=None,
                    dept_name=_DEPT_NAME, dept_code=_DEPT_CODE,
                    **feat_cfg
                )
                val_ds = DengueDataset(
                    data_dir   = periodo_dir,
                    mode       = 'val',
                    target_size=None,
                    dept_name=_DEPT_NAME, dept_code=_DEPT_CODE,
                    **feat_cfg
                )
                test_ds = DengueDataset(
                    data_dir   = periodo_dir,
                    mode       = 'test',
                    target_size=None,
                    dept_name=_DEPT_NAME, dept_code=_DEPT_CODE,
                    **feat_cfg
                )

                print("len(train), len(val), len(test) =",
                    len(train_ds), len(val_ds), len(test_ds))

                # Chequeo sanidad split
                set_val  = set(val_ds.basename_list)
                set_test = set(test_ds.basename_list)
                intersec = set_val & set_test
                if len(intersec) > 0:
                    print("→ ATENCIÓN: hay solapamiento entre val y test. Ejemplos:", list(sorted(intersec))[:10])
                    raise SystemExit("Abortado: VAL y TEST no son disjuntos.")

                # Persistimos listas para inspección
                debug_dir = os.path.join(expdir, "_debug_split")
                os.makedirs(debug_dir, exist_ok=True)
                with open(os.path.join(debug_dir, "val_basenames.txt"), "w") as f:
                    f.write("\n".join(val_ds.basename_list))
                with open(os.path.join(debug_dir, "test_basenames.txt"), "w") as f:
                    f.write("\n".join(test_ds.basename_list))

                # ─── PARÁMETROS DE PARCHES ───
                PATCH_SIZE = 128
                STRIDE     = 64

                # 2) Indexado TRAIN-only (sin leakage)
                train_index = build_train_patch_index(
                    train_ds, patch_size=PATCH_SIZE, stride=STRIDE
                )
                print(f"→ Index TRAIN generado: {len(train_index)} ventanas totales")

                # ---- Subsets por etapa (SOLO S1 en binario 0 vs RESTO) ----
                minor_s1 = select_boundary_group(
                    train_index, groupA=(0,), groupB=(1,2,3),
                    minA=0.10, minB=0.25, target=480, per_file_cap=8
                )
                if len(minor_s1) == 0:
                    raise RuntimeError("Subset S1 vacío. Ajusta umbrales/target y reintenta.")

                print(f"→ Parches S1={len(minor_s1)}")

                # 3) Datasets/Loaders de parches (train) + Loaders val/test (full tiles)
                minor_ds_s1 = PatchCropDataset(train_ds, minor_s1, patch_size=PATCH_SIZE, augment=True)

                # std con epoch-size >= minor
                std_ds = RandomPatchDataset(
                    train_ds, patch_size=PATCH_SIZE,
                    samples_per_epoch=max(3000, len(minor_ds_s1)),
                    augment=True
                )
                std_ds.min_minor_frac_std = 0.18
                std_ds.max_tries_std      = 150

                BATCH_SIZE = 2
                g = torch.Generator().manual_seed(SEED)

                loader_std = DataLoader(
                    std_ds, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=NUM_WORKERS, pin_memory=True,
                    worker_init_fn=seed_worker, generator=g
                )

                STEPS_PER_EPOCH = 500

                # VALID/TEST con mapas completos
                val_loader  = DataLoader(val_ds,  batch_size=2, shuffle=False, num_workers=0)
                test_loader = DataLoader(test_ds, batch_size=2, shuffle=False, num_workers=0)

                # ───── PLAN A BINARIO: Entrenamiento end-to-end (0 vs RESTO) ─────
                sample_X, _, _ = train_ds[0]
                in_ch = int(sample_X.shape[0])

                # Modelo binario con Deep Supervision activo
                model_bin = UNetPlusPlusDS(in_channels=in_ch, n_classes=2, base_ch=16, deep_supervision=True).to(DEVICE)
                bn_to_gn(model_bin, max_groups=8)
                model_bin = model_bin.to(DEVICE)

                # Mezcla de parches: SOLO minor S1 vs estándar
                loader_minor_combined = DataLoader(
                    minor_ds_s1, batch_size=BATCH_SIZE, shuffle=True,
                    num_workers=6, pin_memory=True, worker_init_fn=seed_worker, generator=g
                )
                mix_end = MixedBatchIterator(loader_minor_combined, loader_std, p_minor=0.85, steps_per_epoch=STEPS_PER_EPOCH)

                # Dilatación S1 fija a 0 (no “limpieza” de positivos aquí)

                planA_dir = os.path.join(expdir, "bin_end2end"); os.makedirs(planA_dir, exist_ok=True)

                # Pesos de clase BINARIOS a partir de la frecuencia del TRAIN
                train_freq = compute_split_class_freqs(train_ds)
                f0  = float(train_freq[0])
                fR  = max(1e-6, 1.0 - f0)
                w0  = 1.0 / max(1e-6, f0)
                w1  = 1.0 / fR
                s   = w0 + w1
                bin_weights = [w0/s, w1/s]

                # ENTRENAR BINARIO
                trained_bin = train_binary_end_to_end(
                    model_bin, mix_end, val_loader, DEVICE,
                    base_train_ds=train_ds,
                    epochs=100, patience=10, lr=1e-4, weight_decay=1e-4,
                    accum_steps=8, max_grad_norm=1.0,
                    deep_supervision=True,
                    ds_weights=(0.05, 0.15, 0.30, 0.50),
                    class_weights_bin=bin_weights,
                    p_minor_fixed=0.85,
                    steps_per_epoch=STEPS_PER_EPOCH,
                    threshold=0.5,
                    s1_keep_from_lag=2, s1_smooth_kernel=5, s1_dropout_p=0.30, s1_scale=0.70,
                    log_prefix="[BIN S1]"
                )

                # Calibración por temperatura (VAL) para la cabeza binaria
                print("\n── Calibración por temperatura (VAL) BIN ──")
                scaler_bin = calibrate_temperature_binary_head(
                    trained_bin, val_loader, DEVICE, build_targets_fn=_bin_targets_s1_0vR, max_iter=200, lr=0.05,
                    preprocess_X=(lambda X: restrict_prev_for_s1_inplace(
                        X, train_ds, zero_t1=True, keep_from_lag=2, smooth_kernel=5, dropout_p=0.0, scale=0.70
                    ))
                )

                # Guardar checkpoint binario y calibración
                torch.save({"state_dict": trained_bin.state_dict()}, os.path.join(planA_dir, "best_bin.ckpt"))
                torch.save({"log_T": scaler_bin.log_T.detach().cpu()}, os.path.join(planA_dir, "temp_scaler_bin.pt"))

                # ─── VALIDACIÓN y TEST BINARIOS ───
                def _pre_s1_val(Xin):
                    Xs = Xin.clone()
                    restrict_prev_for_s1_inplace(
                        Xs, val_loader.dataset, zero_t1=True, keep_from_lag=2, smooth_kernel=5, dropout_p=0.0, scale=0.70
                    )
                    return Xs

                def _pre_s1_test(Xin):
                    Xs = Xin.clone()
                    restrict_prev_for_s1_inplace(
                        Xs, test_loader.dataset, zero_t1=True, keep_from_lag=2, smooth_kernel=5, dropout_p=0.0, scale=0.70
                    )
                    return Xs

                class _WrapWithTemp(nn.Module):
                    def __init__(self, net, log_T):
                        super().__init__()
                        self.net = net
                        self.log_T = nn.Parameter(log_T, requires_grad=False)
                    def forward(self, x):
                        out = self.net(x)
                        out = out[-1] if isinstance(out, list) else out
                        T = torch.exp(self.log_T).to(out.device).clamp(min=1e-3, max=100.0)
                        return out / T

                wrap_val  = _WrapWithTemp(trained_bin, scaler_bin.log_T)
                wrap_test = _WrapWithTemp(trained_bin, scaler_bin.log_T)

                print("\n── VALIDACIÓN (BIN 0vsRESTO) ──")
                val_score, val_iou, _ = eval_binary(wrap_val, val_loader, DEVICE, threshold=0.5, tag="VAL_BIN", preprocess_X=_pre_s1_val)
                with open(os.path.join(planA_dir, "val_metrics_bin.json"), "w") as f:
                    json.dump({
                        "MeanIoU": float(val_score["MeanIoU"]),
                        "IoU": {str(k): float(v) for k, v in val_iou.items()},
                        "OverallAcc": float(val_score["OverallAcc"]),
                        "FreqWAcc": float(val_score["FreqWAcc"]),
                        "Dice": {str(k): float(d) for k, d in val_score["Dice"].items()},
                    }, f, indent=2)

                print("\n── TEST (BIN 0vsRESTO) ──")
                test_score, test_iou, _ = eval_binary(wrap_test, test_loader, DEVICE, threshold=0.5, tag="TEST_BIN", preprocess_X=_pre_s1_test)
                with open(os.path.join(planA_dir, "test_metrics_bin.json"), "w") as f:
                    json.dump({
                        "MeanIoU": float(test_score["MeanIoU"]),
                        "IoU": {str(k): float(v) for k, v in test_iou.items()},
                        "OverallAcc": float(test_score["OverallAcc"]),
                        "FreqWAcc": float(test_score["FreqWAcc"]),
                        "Dice": {str(k): float(d) for k, d in test_score["Dice"].items()},
                    }, f, indent=2)

                # ───── Mapas de TEST (binarios) ─────
                maps_dir = os.path.join(expdir, "binary_maps")
                os.makedirs(maps_dir, exist_ok=True)

                class _PredictorBin:
                    def __init__(self, net_with_temp, device):
                        self.netT = net_with_temp
                        self.device = device
                    def __call__(self, X):
                        with torch.no_grad():
                            X = X.to(self.device)
                            Xs = X.clone()
                            restrict_prev_for_s1_inplace(
                                Xs, test_ds, zero_t1=True, keep_from_lag=2, smooth_kernel=5, dropout_p=0.0, scale=0.70
                            )
                            logits = self.netT(Xs)  # [1,2,H,W]
                            return logits  # visualize_cascade_maps convierte a proba y binariza

                visualize_cascade_maps(
                    test_ds,
                    _PredictorBin(_WrapWithTemp(trained_bin, scaler_bin.log_T), DEVICE),
                    DEVICE,
                    maps_dir
                )

                try:
                    trained_bin.to("cpu")
                except Exception:
                    pass
                del trained_bin, wrap_val, wrap_test, scaler_bin
                import gc; gc.collect(); torch.cuda.empty_cache()

    print("\n[Proceso terminado exitosamente]")

if __name__ == '__main__':
    # Rutas base
    DATA_DIR    = r'C:\Users\DENGUE'
    SHAPE_PATH  = os.path.join(DATA_DIR, "Shape", "Shape", "Municipios", "Municipios.shp")
    PERIODS     = [#"2007-2019",
                   #"2016-2019",
                    "2018-2019",
                   ]
    OUTPUT_DIR  = os.path.join(DATA_DIR, "experiments_results_dengue_binario_depto")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Recursos
    use_cuda      = torch.cuda.is_available()
    DEVICE        = torch.device("cuda" if use_cuda else "cpu")
    NUM_WORKERS   = 2
    torch.backends.cudnn.benchmark       = use_cuda
    torch.set_num_threads(NUM_WORKERS)
    torch.set_num_interop_threads(NUM_WORKERS)

    # --- LEE EL SHAPE SÓLO UNA VEZ EN EL PROCESO PRINCIPAL ---
    muni = gpd.read_file(SHAPE_PATH).to_crs("EPSG:4326")

    # ── Bounds automáticos por departamento ───────────────────────────────────────
    DEPARTMENTS = {}
    for dcode, dname in [
        (15, "boyaca"),
        (25, "cundinamarca"),
        (50, "meta"),
        (54, "norte_santander"),
        (73, "tolima"),
    ]:
        dept = muni[muni["DPTO_Code"] == dcode]
        if dept.empty:
            raise ValueError(f"No encontré el departamento código {dcode}")
        minx, miny, maxx, maxy = dept.total_bounds
        DEPARTMENTS[dname] = {
            "code":  dcode,
            "xmin":  float(minx),
            "ymin":  float(miny),
            "xmax":  float(maxx),
            "ymax":  float(maxy),
        }

    muni_gdf = muni.copy()

    main()
