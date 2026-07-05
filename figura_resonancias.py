import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))

# ─── Datos de frecuencia de resonancia (GHz) ─────────────────────────────────
datos = {
    "Aire": [2.11, 2.11, 2.11, 2.11, 2.12, 2.12, 2.12, 2.16, 2.12, 2.18, 2.14, 2.20,
             2.06, 2.06, 2.06, 2.14, 2.05, 2.11, 2.06, 2.06, 2.12, 2.14, 2.05, 2.17,
             2.10, 2.16, 2.14],
    "Incubación": {
        "1.0": [1.33, 1.33, 1.33, 1.32],
        "2.0": [1.31, 1.31, 1.32, 1.32],
        "3.5": [1.34, 1.33, 1.35, 1.32, 1.32, 1.32, 1.32],
    },
    "Primer secado": {
        "1.0": [2.31, 2.31, 2.14, 2.32],
        "2.0": [2.35, 2.24, 2.32, 2.33],
        "3.5": [2.0, 2.0, 2.0, 2.0, 2.31, 2.32, 2.32, 2.32, 2.33, 2.35,
                2.30, 2.30, 2.35, 2.23, 2.23, 2.23, 2.24, 2.35, 2.30],
    },
    "Rehidratación": {
        "1.0": [1.50, 1.51, 1.50, 1.52],
        "2.0": [1.51, 1.52, 1.51, 1.51],
        "3.5": [1.48, 1.43, 1.43, 1.43, 1.42, 1.49, 1.50, 1.50, 1.50, 1.44,
                1.50, 1.41, 1.40, 1.42, 1.42, 1.36, 1.36, 1.33, 1.33],
    },
    "Segundo secado": {
        "3.5": [2.22, 2.20, 2.14, 2.10, 2.12, 2.15, 2.15, 2.14, 2.00, 2.06, 2.10, 2.14],
    },
}

# ─── Colores ──────────────────────────────────────────────────────────────────
_pal = sns.color_palette("colorblind", 3)
COLOR_10   = _pal[0]   # azul   = 1.0 mg/mL
COLOR_20   = _pal[1]   # naranja = 2.0 mg/mL
COLOR_35   = _pal[2]   # verde  = 3.5 mg/mL
COLOR_AIRE = "#555555"  # gris oscuro = referencia aire

# ─── Geometría de barras ──────────────────────────────────────────────────────
ancho  = 0.20   # anchura de cada barra
gap    = 0.05   # separación entre barras del mismo grupo
# Offset respecto al centro del grupo para grupos de 3 barras
off = {
    "1.0": -(ancho + gap),
    "2.0": 0.0,
    "3.5": +(ancho + gap),
}

# Centros de cada grupo en el eje X (separados 1.2 entre sí)
fases = ["Aire", "Incubación", "Primer secado", "Rehidratación", "Segundo secado"]
centros = {f: i * 1.2 for i, f in enumerate(fases)}

# ─── Estilo publicación ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":         "sans-serif",
    "font.size":           9,
    "axes.linewidth":      0.8,
    "axes.spines.top":     False,
    "axes.spines.right":   False,
    "xtick.direction":     "in",
    "ytick.direction":     "in",
    "xtick.minor.visible": False,
    "ytick.minor.visible": True,
    "xtick.major.size":    4,
    "ytick.major.size":    4,
    "ytick.minor.size":    2,
    "legend.frameon":      True,
    "legend.framealpha":   0.9,
    "legend.edgecolor":    "#cccccc",
    "legend.fontsize":     8,
    "figure.dpi":          150,
})

fig, ax = plt.subplots(figsize=(10, 5), constrained_layout=True)

# ─── Barra de referencia: Aire ────────────────────────────────────────────────
y_aire = np.array(datos["Aire"])
ax.bar(centros["Aire"], y_aire.mean(), ancho * 1.5,
       yerr=y_aire.std(ddof=1), capsize=4,
       color=COLOR_AIRE, edgecolor="white", linewidth=0.5,
       label="Sin colágeno (aire)")

# ─── Barras de concentración ──────────────────────────────────────────────────
concs = [
    ("1.0", COLOR_10, "1.0 mg/mL"),
    ("2.0", COLOR_20, "2.0 mg/mL"),
    ("3.5", COLOR_35, "3.5 mg/mL"),
]

for conc, color, label in concs:
    primera = True
    for fase in ["Incubación", "Primer secado", "Rehidratación", "Segundo secado"]:
        if conc not in datos[fase]:
            continue
        y = np.array(datos[fase][conc])
        # Segundo secado solo tiene 3.5: centrar la barra en el grupo
        if fase == "Segundo secado":
            x_pos = centros[fase]
        else:
            x_pos = centros[fase] + off[conc]
        ax.bar(x_pos, y.mean(), ancho,
               yerr=y.std(ddof=1), capsize=4,
               color=color, edgecolor="white", linewidth=0.5,
               label=label if primera else "_nolegend_")
        primera = False

# ─── Eje X ────────────────────────────────────────────────────────────────────
ax.set_xticks([centros[f] for f in fases])
ax.set_xticklabels(fases, fontsize=14)

# ─── Eje Y ────────────────────────────────────────────────────────────────────
ax.set_ylabel("Frecuencia de resonancia (GHz)", fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.set_ylim(1.0, 2.6)
ax.yaxis.set_major_locator(ticker.MultipleLocator(0.2))
ax.yaxis.set_minor_locator(ticker.MultipleLocator(0.05))

# ─── Leyenda ──────────────────────────────────────────────────────────────────
ax.legend(loc="upper right", fontsize=14, bbox_to_anchor=(1.0, 1.08))

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, "figura_resonancias.png")
fig.savefig(salida, bbox_inches="tight", dpi=300, facecolor="white")
print(f"Guardado en {salida}")
plt.show()
