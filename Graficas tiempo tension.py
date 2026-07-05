import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ─── Rutas de las carpetas ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_2_0          = os.path.join(BASE, "concentracion 2.0")
RUTA_3_5          = os.path.join(BASE, "Concentracion 3.5")
RUTA_SIN_COLAGENO = os.path.join(BASE, "Sin colágeno")

# ─── Cargas y tensiones ───────────────────────────────────────────────────────
MASAS = [25, 75, 125, 175, 225, 275]

SIGMA = {
    25:  (0.5778, 0.0095),
    75:  (1.7335, 0.0286),
    125: (2.8891, 0.0477),
    175: (4.0448, 0.0668),
    225: (5.2004, 0.0859),
    275: (6.3560, 0.1050),
}

# ─── Condiciones: (ruta, número de hilo, etiqueta de columna) ────────────────
CONDICIONES = [
    (RUTA_3_5,           1, "3.5 mg/mL"),
    (RUTA_2_0,           1, "2.0 mg/mL"),
    (RUTA_SIN_COLAGENO,  3, "Sin colágeno"),
]

# ─── Colores: plasma de amarillo (25 g, 0.15) a azul marino (275 g, 0.85) ────
COLORES = {
    m: plt.cm.plasma(v)
    for m, v in zip(MASAS, np.linspace(0.15, 0.85, len(MASAS)))
}

# ─── Estilo ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           9,
    "axes.linewidth":      0.7,
    "axes.facecolor":      "white",
    "axes.grid":           False,
    "figure.facecolor":    "white",
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.major.size":    3.5,
    "ytick.major.size":    3.5,
    "xtick.minor.size":    2,
    "ytick.minor.size":    2,
    "figure.dpi":          150,
})

# ─── Lectura de CSV del VNA ───────────────────────────────────────────────────
def leer_csv(ruta):
    xs, ys = [], []
    with open(ruta, encoding='utf-8', errors='replace') as f:
        lineas = f.readlines()[2:]
    for linea in lineas:
        linea = linea.strip().replace('"', '')
        partes = [p for p in linea.split(',') if p.strip()]
        if len(partes) >= 2:
            try:
                xs.append(float(partes[0]))
                ys.append(float(partes[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(ys)

# ─── Búsqueda de archivo por hilo y masa ─────────────────────────────────────
def buscar_csv(carpeta, hilo_num, masa_g):
    pat_hilo = re.compile(rf'hilo{hilo_num}(?!\d)', re.IGNORECASE)
    pat_masa = re.compile(rf'(?<!\d){masa_g}g(?!\d)')
    for nombre in sorted(os.listdir(carpeta)):
        if not nombre.endswith('.csv'):
            continue
        if 'freq' in nombre.lower():
            continue
        if pat_hilo.search(nombre) and pat_masa.search(nombre):
            return os.path.join(carpeta, nombre)
    return None

# ─── Figura ───────────────────────────────────────────────────────────────────
n_filas = len(MASAS)
n_cols  = len(CONDICIONES)

fig, axes = plt.subplots(
    n_filas, n_cols,
    figsize=(18, 22),
    constrained_layout=True,
    sharey=True,
)
plt.subplots_adjust(hspace=0.08)

# Títulos de columna
for col, (_, _, etiqueta) in enumerate(CONDICIONES):
    axes[0, col].set_title(etiqueta, fontsize=13, fontweight='bold', pad=6)

# ─── Paneles ──────────────────────────────────────────────────────────────────
for fila, masa in enumerate(MASAS):
    color            = COLORES[masa]
    sigma_val, sigma_unc = SIGMA[masa]

    for col, (carpeta, hilo_num, _) in enumerate(CONDICIONES):
        ax   = axes[fila, col]
        ruta = buscar_csv(carpeta, hilo_num, masa)

        if ruta is None:
            ax.text(0.5, 0.5, 'sin datos', transform=ax.transAxes,
                    ha='center', va='center', color='#aaaaaa', fontsize=8)
        else:
            x, y = leer_csv(ruta)
            if len(y) > 0:
                y_norm = y - np.median(y)
                ax.plot(x, y_norm, color=color, linewidth=1.0)

        # Fijar escala Y explícitamente tras plotear
        ax.set_ylim(-1.0, 0.2)

        # Etiqueta de tensión (esquina superior derecha)
        label = f'σ = {sigma_val:.4f} ± {sigma_unc:.4f} MPa'
        ax.text(0.97, 0.05, label, transform=ax.transAxes,
                ha='right', va='bottom', fontsize=14, fontweight='bold',
                color=color,
                bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))

        # Eje X: solo fila inferior
        if fila == n_filas - 1:
            ax.set_xlabel('Tiempo (s)', fontsize=16, fontweight='bold')
        else:
            ax.tick_params(labelbottom=False)

        # Eje Y: solo columna izquierda
        if col == 0:
            ax.set_ylabel(r'$\Delta S_{21}$ (dB)', fontsize=16, fontweight='bold')

        ax.minorticks_on()
        ax.tick_params(labelsize=10)

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, 'figura_4_4.png')
fig.savefig(salida, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Guardado: {salida}')
