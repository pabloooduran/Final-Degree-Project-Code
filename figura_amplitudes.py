import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

BASE = os.path.dirname(os.path.abspath(__file__))

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

# ─── Amplitud de un CSV: percentil 2 de la señal centrada ────────────────────
def amplitud_media_csv(ruta):
    _, y = leer_csv(ruta)
    if len(y) == 0:
        return None
    # Centrar la señal restando la media (elimina el offset del VNA)
    y_norm = y - np.mean(y)
    # Mínimo absoluto de la señal normalizada = valor real del pico más profundo
    return abs(y_norm.min())

# ─── Procesar carpeta: una amplitud media por CSV ─────────────────────────────
def procesar_carpeta(carpeta):
    archivos = sorted(f for f in os.listdir(carpeta) if f.endswith(".csv"))
    medias = []
    for nombre in archivos:
        val = amplitud_media_csv(os.path.join(carpeta, nombre))
        if val is not None:
            medias.append(val)
            print(f"    {nombre}: {val:.4f} dB")
    return medias

# ─── Calcular estadísticos por condición ─────────────────────────────────────
resultados = {}

print("\n=== Aire ===")
m = procesar_carpeta(os.path.join(BASE, "Aire"))
resultados["Aire"] = {"media": np.mean(m), "std": np.std(m, ddof=1)}

fases_conc = {
    "Incubación":    ("Incubacion",    ["1.0", "2.0", "3.5"]),
    "Primer secado": ("Seco 1",        ["1.0", "2.0", "3.5"]),
    "Rehidratación": ("Rehidratacion", ["1.0", "2.0", "3.5"]),
}

for fase, (carpeta_nombre, concs) in fases_conc.items():
    resultados[fase] = {}
    for conc in concs:
        print(f"\n=== {fase} {conc} mg/mL ===")
        carpeta = os.path.join(BASE, carpeta_nombre, conc)
        m = procesar_carpeta(carpeta)
        resultados[fase][conc] = {"media": np.mean(m), "std": np.std(m, ddof=1)}

print("\n=== Segundo secado (3.5 mg/mL) ===")
m = procesar_carpeta(os.path.join(BASE, "Seco 2 (3.5)"))
resultados["Segundo secado"] = {"3.5": {"media": np.mean(m), "std": np.std(m, ddof=1)}}

# ─── Imprimir resumen ─────────────────────────────────────────────────────────
print("\n--- Resumen ---")
print(f"Aire:  {resultados['Aire']['media']:.4f} ± {resultados['Aire']['std']:.4f} dB")
for fase in ["Incubación", "Primer secado", "Rehidratación"]:
    for conc in ["1.0", "2.0", "3.5"]:
        v = resultados[fase][conc]
        print(f"{fase} {conc} mg/mL:  {v['media']:.4f} ± {v['std']:.4f} dB")
v = resultados["Segundo secado"]["3.5"]
print(f"Segundo secado 3.5 mg/mL:  {v['media']:.4f} ± {v['std']:.4f} dB")

# ─── Colores ──────────────────────────────────────────────────────────────────
_pal = sns.color_palette("colorblind", 3)
COLOR_10   = _pal[0]
COLOR_20   = _pal[1]
COLOR_35   = _pal[2]
COLOR_AIRE = "#555555"

# ─── Geometría de barras ──────────────────────────────────────────────────────
ancho = 0.20
gap   = 0.05
off = {"1.0": -(ancho + gap), "2.0": 0.0, "3.5": +(ancho + gap)}

fases    = ["Aire", "Incubación", "Primer secado", "Rehidratación", "Segundo secado"]
centros  = {f: i * 1.2 for i, f in enumerate(fases)}

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

# ─── Barra de Aire ────────────────────────────────────────────────────────────
r = resultados["Aire"]
ax.bar(centros["Aire"], r["media"], ancho * 1.5,
       yerr=r["std"], capsize=4,
       color=COLOR_AIRE, edgecolor="white", linewidth=0.5,
       label="Sin colágeno (aire)")

# ─── Barras de concentración ──────────────────────────────────────────────────
concs_info = [
    ("1.0", COLOR_10, "1.0 mg/mL"),
    ("2.0", COLOR_20, "2.0 mg/mL"),
    ("3.5", COLOR_35, "3.5 mg/mL"),
]

for conc, color, label in concs_info:
    primera = True
    for fase in ["Incubación", "Primer secado", "Rehidratación", "Segundo secado"]:
        if conc not in resultados[fase]:
            continue
        r = resultados[fase][conc]
        # Segundo secado: única barra → centrar en el grupo
        x_pos = centros[fase] if fase == "Segundo secado" else centros[fase] + off[conc]
        ax.bar(x_pos, r["media"], ancho,
               yerr=r["std"], capsize=4,
               color=color, edgecolor="white", linewidth=0.5,
               label=label if primera else "_nolegend_")
        primera = False

# ─── Ejes ─────────────────────────────────────────────────────────────────────
ax.set_xticks([centros[f] for f in fases])
ax.set_xticklabels(fases, fontsize=14)
ax.set_ylabel(r"Amplitud media $|S_{21}|$ (dB)", fontsize=16)
ax.tick_params(axis='y', labelsize=14)
ax.yaxis.set_major_locator(ticker.AutoLocator())
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, "figura_amplitudes.png")
fig.savefig(salida, bbox_inches="tight", dpi=300, facecolor="white")
print(f"\nGuardado en {salida}")
plt.show()
