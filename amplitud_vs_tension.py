import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ─── Rutas de las carpetas ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_2_0          = os.path.join(BASE, "concentracion 2.0")
RUTA_3_5          = os.path.join(BASE, "Concentracion 3.5")
RUTA_SIN_COLAGENO = os.path.join(BASE, "Sin colágeno")

# ─── Cargas y tensiones ───────────────────────────────────────────────────────
MASAS = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275]

SIGMA = {
    25:  0.5778,
    50:  1.1557,
    75:  1.7335,
    100: 2.3113,
    125: 2.8891,
    150: 3.4670,
    175: 4.0448,
    200: 4.6226,
    225: 5.2004,
    250: 5.7783,
    275: 6.3560,
}

# ─── Condiciones: (ruta, [hilos], etiqueta, color) ───────────────────────────
CONDICIONES = [
    (RUTA_2_0,          [1, 2], "2.0 mg/mL",   "red"),
    (RUTA_3_5,          [1, 4], "3.5 mg/mL",   "blue"),
    (RUTA_SIN_COLAGENO, [3, 4], "Sin colágeno", "green"),
]

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

# ─── Amplitud media de picos para un CSV ─────────────────────────────────────
def amplitud_hilo(ruta):
    _, y = leer_csv(ruta)
    if len(y) == 0:
        return None, None
    # Normalizar restando la media
    y_norm = y - np.mean(y)
    # Dividir en 8 segmentos (un ciclo por período del campo AC)
    N_ciclos = 8
    segmentos = np.array_split(y_norm, N_ciclos)
    # Amplitud de cada ciclo = mínimo absoluto de ese segmento
    amplitudes = np.array([abs(seg.min()) for seg in segmentos if len(seg) > 0])
    A_bar = np.mean(amplitudes)
    u = np.std(amplitudes, ddof=1) / np.sqrt(len(amplitudes))
    return A_bar, u

# ─── Estilo ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "sans-serif",
    "axes.facecolor":   "white",
    "figure.facecolor": "white",
    "axes.grid":        False,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "xtick.direction":  "in",
    "ytick.direction":  "in",
})

fig, ax = plt.subplots(figsize=(10, 7), constrained_layout=True)

# ─── Cálculo y plot por condición ────────────────────────────────────────────
for carpeta, hilos, etiqueta, color in CONDICIONES:
    sigmas_plot = []
    A_plot      = []
    u_plot      = []

    for masa in MASAS:
        sigma_val = SIGMA[masa]

        resultados = []
        for h in hilos:
            ruta = buscar_csv(carpeta, h, masa)
            if ruta is None:
                continue
            A_bar, u = amplitud_hilo(ruta)
            if A_bar is None:
                continue
            resultados.append((A_bar, u))

        if len(resultados) == 0:
            continue
        elif len(resultados) == 1:
            A_cond = resultados[0][0]
            u_cond = resultados[0][1]
        else:
            A1, u1 = resultados[0]
            A2, u2 = resultados[1]
            A_cond = (A1 + A2) / 2.0
            u_cond = 0.5 * np.sqrt(u1**2 + u2**2)

        sigmas_plot.append(sigma_val)
        A_plot.append(A_cond)
        u_plot.append(u_cond)

    if sigmas_plot:
        # Regresión lineal
        slope, intercept, r_value, p_value, std_err = linregress(sigmas_plot, A_plot)
        u_b = std_err * np.sqrt(np.mean(np.array(sigmas_plot)**2))

        # Línea de regresión
        x_reg = np.linspace(min(sigmas_plot), max(sigmas_plot), 200)
        y_reg = slope * x_reg + intercept

        # Plotear línea de regresión con mismo color, línea discontinua
        ax.plot(x_reg, y_reg, color=color, linestyle='--', linewidth=1.2, alpha=0.7)

        # Añadir R² en la leyenda
        etiqueta_reg = (f"{etiqueta}\n"
                        f"y = ({slope:.4f}±{std_err:.4f})σ + ({intercept:.4f}±{u_b:.4f})\n"
                        f"R² = {r_value**2:.3f}")

        ax.errorbar(sigmas_plot, A_plot, yerr=u_plot,
                    fmt='o', color=color, capsize=4,
                    linewidth=1.5, markersize=5, label=etiqueta_reg)

# ─── Formato de ejes ─────────────────────────────────────────────────────────
ax.set_xlabel('Tensión σ (MPa)', fontsize=15, fontweight='bold')
ax.set_ylabel('Amplitud media de los picos ΔS₂₁ (dB)', fontsize=15, fontweight='bold')
ax.tick_params(labelsize=12)
ax.minorticks_on()
ax.legend(fontsize=10, loc='lower right', framealpha=0.9,
          handlelength=1.5, borderpad=0.8, labelspacing=1.0,
          bbox_to_anchor=(1.0, 0.02))

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, 'amplitud_vs_tension.png')
fig.savefig(salida, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Guardado: {salida}')

print("\n" + "="*70)
print("VERIFICACIÓN: comparación de A_cond y u_cond entre ambos scripts")
print("="*70)
for carpeta, hilos, etiqueta, color in [(RUTA_2_0, [1,2], "2.0 mg/mL", "red"),
                                          (RUTA_3_5, [1,4], "3.5 mg/mL", "blue")]:
    print(f"\n--- {etiqueta} ---")
    for masa in MASAS:
        resultados = []
        for h in hilos:
            ruta = buscar_csv(carpeta, h, masa)
            if ruta is None:
                continue
            A_bar, u = amplitud_hilo(ruta)
            if A_bar is None:
                continue
            resultados.append((A_bar, u))
        if len(resultados) == 1:
            A_cond = resultados[0][0]
            u_cond = resultados[0][1]
            print(f"  masa={masa}g  A_cond={A_cond:.6f}  u_cond={u_cond:.6f}")
        elif len(resultados) == 2:
            A1, u1 = resultados[0]
            A2, u2 = resultados[1]
            A_cond = (A1 + A2) / 2.0
            u_cond = 0.5 * np.sqrt(u1**2 + u2**2)
            print(f"  masa={masa}g  A_cond={A_cond:.6f}  u_cond={u_cond:.6f}")
