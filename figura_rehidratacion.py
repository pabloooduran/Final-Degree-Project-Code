import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from scipy.signal import find_peaks

# ─── CONFIGURACIÓN ────────────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))
NOMBRE_SALIDA = "figura_rehidratacion.png"

# ─── Colores y estilos ────────────────────────────────────────────────────────
# Orden alfabético de los CSVs:
#   índice 0 = 1.0 mg/mL
#   índice 1 = 2.0 mg/mL
#   índice 2 = 3.5 mg/mL
#   índice 3 = Sin colágeno (hilo_aire)
_pal = sns.color_palette("colorblind", 3)
COLORES   = [_pal[0], _pal[1], _pal[2], "black"]
ETIQUETAS = ["1.0 mg/mL", "2.0 mg/mL", "3.5 mg/mL", "Sin colágeno"]
ESTILOS   = ["-",    "--",    ":",     "-"]
ANCHURAS  = [1.5,    1.5,     1.5,    2.0]
ALPHAS    = [0.85,   0.85,    0.85,   1.0]
ORDEN_LEY = [3, 0, 1, 2]  # Leyenda: Sin colágeno primero

# Offset temporal entre curvas (negativo = desplaza a la izquierda)
OFFSET = [-0.001, -0.002, -0.003, 0.000]

# ─── Lectura de CSV del VNA ───────────────────────────────────────────────────
def leer_csv(ruta):
    xs, ys = [], []
    with open(ruta) as f:
        lineas = f.readlines()[2:]
    for linea in lineas:
        linea = linea.strip().replace('"', '')
        partes = [p for p in linea.split(",") if p.strip()]
        if len(partes) >= 2:
            try:
                xs.append(float(partes[0]))
                ys.append(float(partes[1]))
            except ValueError:
                continue
    return np.array(xs), np.array(ys)

# ─── Suavizado: media móvil de 25 puntos ─────────────────────────────────────
def suavizar(y, ventana=25):
    return np.convolve(y, np.ones(ventana) / ventana, mode='same')

# ─── Carga de datos ───────────────────────────────────────────────────────────
datos = {}
for panel in ["izquierdo", "central", "derecho"]:
    carpeta = os.path.join(BASE, f"panel {panel}")
    archivos = sorted(f for f in os.listdir(carpeta) if f.endswith(".csv"))

    print(f"\nCarpeta: {panel}")
    for i, nombre in enumerate(archivos):
        print(f"  índice {i}: {nombre}")

    entradas = []
    for nombre in archivos:
        x, y = leer_csv(os.path.join(carpeta, nombre))

        if panel == "izquierdo":
            y_suav = suavizar(y)
            y_norm = y_suav - y_suav[0]
        else:
            y_norm = y - np.mean(y)

        entradas.append((x, y_norm))
    datos[panel] = entradas

# ─── Alineación temporal al primer mínimo + offset ───────────────────────────
def alinear(entradas):
    resultado = []
    for idx, (x, y) in enumerate(entradas):
        rango = y.max() - y.min()
        picos, _ = find_peaks(-y, prominence=rango * 0.1)
        idx_min = picos[0] if len(picos) > 0 else np.argmin(y)
        x_alin = x - x[idx_min] + OFFSET[idx]
        mascara = x_alin >= 0
        resultado.append((x_alin[mascara], y[mascara]))
    return resultado

datos_cen = alinear(datos["central"])
datos_der = alinear(datos["derecho"])

# ─── Escala Y compartida para paneles de tiempo ───────────────────────────────
todos_y = [v for _, y in datos_cen + datos_der for v in y]
y_min_g, y_max_g = min(todos_y), max(todos_y)
margen = (y_max_g - y_min_g) * 0.06
ylim_tiempo = (y_min_g - margen, y_max_g + margen)

# ─── Estilo para publicación ──────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           9,
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "xtick.minor.size":    2,
    "ytick.minor.size":    2,
    "legend.frameon":      True,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "legend.fontsize":     7.5,
    "figure.dpi":          150,
})

# ─── Figura con tres paneles ──────────────────────────────────────────────────
fig, (ax_izq, ax_cen, ax_der) = plt.subplots(
    1, 3, figsize=(15, 4.2), constrained_layout=True
)

# ─── Panel izquierdo: barrido en frecuencia ───────────────────────────────────
for idx, (x, y) in enumerate(datos["izquierdo"]):
    freq = x / 1e9
    m = (freq >= 1.0) & (freq <= 3.0)
    ax_izq.plot(freq[m], y[m],
                color=COLORES[idx], linestyle=ESTILOS[idx],
                linewidth=ANCHURAS[idx], alpha=ALPHAS[idx])

ax_izq.set_xlabel("Frecuencia (GHz)", fontsize=14)
ax_izq.set_ylabel(r"$\Delta S_{21}$ (dB)", fontsize=14)
ax_izq.tick_params(axis='both', labelsize=12)
ax_izq.set_xlim(1.0, 3.0)
ax_izq.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
ax_izq.xaxis.set_minor_locator(ticker.MultipleLocator(0.25))
ax_izq.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
ax_izq.yaxis.set_minor_locator(ticker.AutoMinorLocator())
handles = [plt.Line2D([0], [0], color=COLORES[i], linestyle=ESTILOS[i],
                      linewidth=ANCHURAS[i], alpha=ALPHAS[i]) for i in ORDEN_LEY]
ax_izq.legend(handles, [ETIQUETAS[i] for i in ORDEN_LEY], loc="lower left", fontsize=12)

# ─── Paneles de tiempo ────────────────────────────────────────────────────────
def dibujar_tiempo(ax, entradas):
    orden = sorted(range(4), key=lambda i: abs(entradas[i][1].min()), reverse=True)
    for idx in orden:
        x, y = entradas[idx]
        ax.plot(x, y,
                color=COLORES[idx], linestyle=ESTILOS[idx],
                linewidth=ANCHURAS[idx], alpha=ALPHAS[idx])
    ax.set_xlabel("Tiempo (s)", fontsize=14)
    ax.set_ylabel(r"$\Delta S_{21}$ (dB)", fontsize=14)
    ax.tick_params(axis='both', labelsize=12)
    ax.set_xlim(0, 0.06)
    ax.set_ylim(ylim_tiempo)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(nbins=5))
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

dibujar_tiempo(ax_cen, datos_cen)
dibujar_tiempo(ax_der, datos_der)

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, NOMBRE_SALIDA)
fig.savefig(salida, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nGuardado en {salida}")
plt.show()
