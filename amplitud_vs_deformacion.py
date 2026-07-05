import os
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import linregress

# ─── Rutas de las carpetas ────────────────────────────────────────────────────
BASE = os.path.dirname(os.path.abspath(__file__))

RUTA_2_0 = os.path.join(BASE, "concentracion 2.0")
RUTA_3_5 = os.path.join(BASE, "Concentracion 3.5")

# ─── Rutas Instron (Bluehill, curvas tensión-deformación) ────────────────────
RUTA_INSTRON_3_5 = (
    r"C:\Users\FX517\Practicas\IMA\Semana 15-06-2026"
    r"\16-06-2026 (Instron)\Pablo colageno hilo_8.is_tens_Exports"
    r"\Pablo colageno hilo_8_1_monotono.csv"
)


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

# ─── Condiciones: (ruta_vna, [hilos], etiqueta, color, ruta_instron) ─────────
CONDICIONES = [
    (RUTA_3_5, [1, 4], "3.5 mg/mL", "blue", RUTA_INSTRON_3_5),
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

# ─── Lectura de curva tensión-deformación del Instron (formato Bluehill) ─────
def leer_instron(ruta):
    """Devuelve (tension_MPa, deformacion_pct) ordenados por tensión creciente.

    Estructura del CSV: separador ';', decimal ',', 2 filas de cabecera.
    Las filas con datos calculados tienen las columnas 0-3 vacías y datos en
    col 5 (Deformacion, mm/mm) y col 6 (Tension, MPa).
    """
    tensiones    = []
    deformaciones = []
    with open(ruta, encoding='utf-8', errors='replace') as f:
        lineas = f.readlines()

    # Imprimir primeras filas de datos para verificar unidades
    print(f"\n--- Primeros valores del Instron: {os.path.basename(ruta)} ---")
    count = 0
    for linea in lineas[2:]:
        partes = linea.strip().split(';')
        if len(partes) < 7:
            continue
        defo_str = partes[5].replace(',', '.').strip()
        tens_str = partes[6].replace(',', '.').strip()
        if defo_str == '' or tens_str == '':
            continue
        try:
            defo = float(defo_str)
            tens = float(tens_str)
        except ValueError:
            continue
        if count < 5:
            print(f"  Deformacion = {defo:.6f} mm/mm  |  Tension = {tens:.4f} MPa")
            count += 1
        tensiones.append(tens)
        deformaciones.append(defo)   # mm/mm, sin conversión

    tens_arr = np.array(tensiones)
    defo_arr = np.array(deformaciones)

    # Ordenar por tensión creciente para np.interp
    orden    = np.argsort(tens_arr)
    print(f"  Rango tensión: {tens_arr[orden[0]]:.3f} – {tens_arr[orden[-1]]:.3f} MPa")
    print(f"  Rango deformación: {defo_arr[orden[0]]:.4f} – {defo_arr[orden[-1]]:.4f} %")
    return tens_arr[orden], defo_arr[orden]

# ─── Búsqueda del punto discreto más cercano en la curva Instron ─────────────
def sigma_a_deformacion(sigma_val, sigma_instron, defo_instron):
    """Busca el punto discreto del Instron más cercano a sigma_val,
    sin extrapolar más allá de la tensión de rotura."""
    sigma_rotura = sigma_instron[-1]
    if sigma_val > sigma_rotura:
        return None  # no hay dato disponible más allá de la rotura
    idx = np.argmin(np.abs(sigma_instron - sigma_val))
    return defo_instron[idx]

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
    y_norm = y - np.mean(y)
    N_ciclos  = 8
    segmentos = np.array_split(y_norm, N_ciclos)
    amplitudes = np.array([abs(seg.min()) for seg in segmentos if len(seg) > 0])
    A_bar = np.mean(amplitudes)
    u     = np.std(amplitudes, ddof=1) / np.sqrt(len(amplitudes))
    return A_bar, u

# ─── Estilo ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "axes.facecolor":    "white",
    "figure.facecolor":  "white",
    "axes.grid":         False,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
})

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7), constrained_layout=True)

# ─── Cálculo y plot por condición ────────────────────────────────────────────
for carpeta, hilos, etiqueta, color, ruta_instron in CONDICIONES:
    sigma_inst, defo_inst = leer_instron(ruta_instron)

    defo_plot = []
    A_plot    = []
    u_plot    = []

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

        print(f"  [{etiqueta}] masa={masa}g  sigma={sigma_val:.4f} MPa  "
              f"A_cond={A_cond:.6f}  u_cond={u_cond:.6f}")
        eps = sigma_a_deformacion(sigma_val, sigma_inst, defo_inst)
        if eps is None:
            print(f"  [{etiqueta}] sigma={sigma_val:.4f} MPa supera la tension de rotura "
                  f"({sigma_inst[-1]:.3f} MPa) - punto omitido")
            continue
        defo_plot.append(eps)
        A_plot.append(A_cond)
        u_plot.append(u_cond)

    if defo_plot:
        slope, intercept, r_value, p_value, std_err = linregress(defo_plot, A_plot)
        u_b = std_err * np.sqrt(np.mean(np.array(defo_plot)**2))

        x_reg = np.linspace(min(defo_plot), max(defo_plot), 200)
        y_reg = slope * x_reg + intercept

        ax1.plot(x_reg, y_reg, color=color, linestyle='--', linewidth=1.2, alpha=0.7)

        etiqueta_reg = (f"{etiqueta}\n"
                        f"y = ({slope:.4f}±{std_err:.4f})ε + ({intercept:.4f}±{u_b:.4f})\n"
                        f"R² = {r_value**2:.3f}")

        ax1.errorbar(defo_plot, A_plot, yerr=u_plot,
                     fmt='o', color=color, capsize=4,
                     linewidth=1.5, markersize=5, label=etiqueta_reg)

        # ─── Deformación transversal ──────────────────────────────────────────
        L0 = 80.0    # mm (8 cm)
        A0 = 0.5568  # mm² (sección transversal media usada para el cálculo de sigma)
        r0 = np.sqrt(A0 / np.pi)

        defo_t_plot = []
        for eps_l in defo_plot:
            L_t = L0 * (1 + eps_l)
            A_t = A0 * L0 / L_t
            r_t = np.sqrt(A_t / np.pi)
            eps_t = abs((r_t - r0) / r0)
            defo_t_plot.append(eps_t)

        slope_t, intercept_t, r_value_t, p_value_t, std_err_t = linregress(defo_t_plot, A_plot)
        u_b_t = std_err_t * np.sqrt(np.mean(np.array(defo_t_plot)**2))

        x_reg_t = np.linspace(min(defo_t_plot), max(defo_t_plot), 200)
        y_reg_t = slope_t * x_reg_t + intercept_t

        ax2.plot(x_reg_t, y_reg_t, color='blue', linestyle='--', linewidth=1.2, alpha=0.7)

        etiqueta_reg_t = (f"3.5 mg/mL\n"
                          f"y = ({slope_t:.4f}±{std_err_t:.4f})εₜ + ({intercept_t:.4f}±{u_b_t:.4f})\n"
                          f"R² = {r_value_t**2:.3f}")

        ax2.errorbar(defo_t_plot, A_plot, yerr=u_plot,
                     fmt='o', color='blue', capsize=4,
                     linewidth=1.5, markersize=5, label=etiqueta_reg_t)

# ─── Formato de ejes ──────────────────────────────────────────────────────────
ax1.set_xlabel('Deformación longitudinal ε', fontsize=17, fontweight='bold')
ax1.set_ylabel('Amplitud media de los picos ΔS₂₁ (dB)', fontsize=17, fontweight='bold')
ax1.tick_params(labelsize=14)
ax1.minorticks_on()
ax1.legend(fontsize=18, loc='lower right', framealpha=0.9,
           handlelength=1.5, borderpad=0.8, labelspacing=1.0,
           bbox_to_anchor=(1.0, 0.08))

ax2.set_xlabel('Deformación transversal |εₜ|', fontsize=17, fontweight='bold')
ax2.set_ylabel('Amplitud media de los picos ΔS₂₁ (dB)', fontsize=17, fontweight='bold')
ax2.tick_params(labelsize=14)
ax2.minorticks_on()
ax2.legend(fontsize=18, loc='lower right', framealpha=0.9,
           handlelength=1.5, borderpad=0.8, labelspacing=1.0,
           bbox_to_anchor=(1.0, 0.08))

# ─── Guardado ─────────────────────────────────────────────────────────────────
salida = os.path.join(BASE, 'amplitud_vs_deformacion_combinada.png')
fig.savefig(salida, dpi=300, bbox_inches='tight', facecolor='white')
print(f'\nGuardado: {salida}')

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
