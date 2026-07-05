import os
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import linregress

# ─── Parámetros físicos ───────────────────────────────────────────────────────
A       = 0.5568        # mm²
sigmaA  = 0.0074        # mm²
sigmaF  = 0.1           # N
sigmaL0 = 1.0           # mm
sigmadL = 0.0001        # mm

L0 = {1: 80.0, 2: 80.0, 3: 80.0, 6: 40.0, 7: 65.0, 8: 80.0, 11: 80.0}
CONC    = {2: 3.5, 6: 2.0, 7: 1.0, 8: 3.5, 11: 2.0}   # hilo -> concentración mg/mL
COLORES = {1.0: '#1F77B4', 2.0: '#FF7F0E', 3.5: '#2CA02C'}

BASE   = os.path.dirname(os.path.abspath(__file__))
SALIDA = os.path.join(BASE, "analisis_modulo_young")
os.makedirs(SALIDA, exist_ok=True)

# ─── Estilo publicación ───────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         9,
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "xtick.direction":   "in",
    "ytick.direction":   "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size":  4,
    "ytick.major.size":  4,
    "xtick.minor.size":  2,
    "ytick.minor.size":  2,
    "figure.dpi":        150,
})

# ─── Lectura de tabla procesada (cols 5-8 del CSV Instron) ───────────────────
def leer_instron_csv(ruta):
    df = pd.read_csv(ruta, sep=';', decimal=',', encoding='latin-1')

    # Buscar columnas por nombre (case insensitive, sin requerir tildes)
    # Fallback a índice 5 si 'deformac' no se encuentra (ej. 'Desplazamiento.1')
    deformac_cols = [c for c in df.columns if 'deformac' in c.lower()]
    col_eps = deformac_cols[0] if deformac_cols else df.columns[5]
    # 'tensi' coincide con 'Tension' y 'Tensión'
    col_sig = [c for c in df.columns if 'tensi' in c.lower()][0]
    col_mod = [c for c in df.columns if 'modulo' in c.lower()][0]

    # Eliminar filas donde Deformacion o Tension están vacías
    df = df[[col_eps, col_sig, col_mod]].dropna(subset=[col_eps, col_sig])
    df.columns = ['Deformacion', 'Tension', 'Modulo']
    df = df.apply(pd.to_numeric, errors='coerce')

    eps      = df['Deformacion'].values
    sigma    = df['Tension'].values
    mask_mod = df['Modulo'].notna().values

    return eps, sigma, mask_mod

# ─── Propagación de errores ───────────────────────────────────────────────────
def error_E(E, eps_lin, sigma_lin, L0_val):
    """E = Delta_sigma / Delta_eps = (Delta_F/A) / (Delta_L/L0)
    sigma_E / E = sqrt((sigmaF/Delta_F)^2 + (sigmaA/A)^2 + (sigmaL0/L0)^2 + (sigmadL/Delta_L)^2)
    """
    Delta_sigma = sigma_lin.max() - sigma_lin.min()
    Delta_eps   = eps_lin.max()   - eps_lin.min()
    Delta_F     = Delta_sigma * A          # N
    Delta_L     = Delta_eps   * L0_val     # mm
    rel = np.sqrt(
        (sigmaF  / Delta_F)**2 +
        (sigmaA  / A)**2       +
        (sigmaL0 / L0_val)**2  +
        (sigmadL / Delta_L)**2
    )
    return E * rel

def error_UTS(sigma_uts):
    """UTS = F_max / A → sigma_UTS/UTS = sqrt((sigmaF/F_max)^2 + (sigmaA/A)^2)"""
    F_max = sigma_uts * A
    return sigma_uts * np.sqrt((sigmaF / F_max)**2 + (sigmaA / A)**2)

def error_eps_r(eps_r, L0_val):
    """eps_r = dL_r / L0 → sigma_epsr = sqrt((sigmadL/L0)^2 + (eps_r*sigmaL0/L0)^2)"""
    return np.sqrt((sigmadL / L0_val)**2 + (eps_r * sigmaL0 / L0_val)**2)

# ─── Procesado por hilo ───────────────────────────────────────────────────────
resultados = {}

for carpeta_nombre in sorted(os.listdir(BASE)):
    carpeta = os.path.join(BASE, carpeta_nombre)
    if not os.path.isdir(carpeta) or carpeta_nombre == "analisis_modulo_young":
        continue

    csvs = [f for f in os.listdir(carpeta) if f.endswith('.csv')]
    if not csvs:
        continue
    nombre_csv = csvs[0]

    m = re.search(r'hilo_(\d+)', carpeta_nombre)
    if not m:
        print(f"Omitido (sin número de hilo en carpeta): {carpeta_nombre}")
        continue

    hilo_num = int(m.group(1))
    if hilo_num not in L0:
        print(f"Omitido (L0 no definido): hilo {hilo_num}")
        continue

    L0_val = L0[hilo_num]
    ruta   = os.path.join(carpeta, nombre_csv)

    eps, sigma, mask_mod = leer_instron_csv(ruta)
    if len(eps) < 20:
        print(f"AVISO: hilo {hilo_num} — pocos datos ({len(eps)} filas), omitido.")
        continue

    # Eliminar posible región descendente si el archivo no es 100% monotono
    idx_max   = np.argmax(sigma)
    eps       = eps[:idx_max + 1]
    sigma     = sigma[:idx_max + 1]
    mask_mod  = mask_mod[:idx_max + 1]

    sigma_uts = sigma[idx_max]
    eps_r     = eps[idx_max]

    # Región lineal: rango de eps definido por los valores donde Modulo no está vacío
    eps_lin_vals = eps[mask_mod]
    eps_min      = eps_lin_vals.min()
    eps_max      = eps_lin_vals.max()
    mask_rango   = (eps >= eps_min) & (eps <= eps_max)
    eps_lin      = eps[mask_rango]
    sigma_lin    = sigma[mask_rango]

    if len(eps_lin) < 5:
        print(f"AVISO: hilo {hilo_num} — región lineal con pocos puntos, omitido.")
        continue

    slope, intercept, r_val, _, _ = linregress(eps_lin, sigma_lin)
    E  = slope
    R2 = r_val**2

    # Errores
    dE      = error_E(E, eps_lin, sigma_lin, L0_val)
    dUTS    = error_UTS(sigma_uts)
    deps_r  = error_eps_r(eps_r, L0_val)

    resultados[hilo_num] = dict(
        E=E, dE=dE, UTS=sigma_uts, dUTS=dUTS,
        eps_r=eps_r, deps_r=deps_r, R2=R2, L0=L0_val,
        eps=eps, sigma=sigma,
        eps_lin=eps_lin, sigma_lin=sigma_lin,
        intercept=intercept,
    )

    print(f"Hilo {hilo_num:2d} (L0={L0_val:4.0f} mm): "
          f"E = {E:.1f} ± {dE:.1f} MPa  "
          f"UTS = {sigma_uts:.2f} ± {dUTS:.2f} MPa  "
          f"eps_r = {eps_r*100:.1f} ± {deps_r*100:.2f}%  "
          f"R2 = {R2:.4f}")

    # ─── Figura individual ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)

    ax.plot(eps, sigma,
            color='#2196F3', linewidth=1.2)

    # Recta de regresión
    eps_reg   = np.array([eps_lin.min(), eps_lin.max()])
    sigma_reg = E * eps_reg + intercept
    ax.plot(eps_reg, sigma_reg,
            color='#B71C1C', linewidth=1.0, linestyle='--', alpha=0.8, zorder=2)

    # Anotación con resultados (esquina inferior derecha)
    txt = (f'E = {E:.1f} ± {dE:.1f} MPa  (R² = {R2:.4f})\n'
           f'UTS = {sigma_uts:.2f} ± {dUTS:.2f} MPa\n'
           f'εᵣ = {eps_r:.4f} ± {deps_r:.4f}')
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            ha='right', va='bottom', multialignment='left', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.9, edgecolor='#cccccc'))

    ax.set_xlabel('Deformación', fontsize=9)
    ax.set_ylabel('Tensión (MPa)', fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.savefig(os.path.join(SALIDA, f'hilo_{hilo_num:02d}.png'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"  -> Guardado: hilo_{hilo_num:02d}.png")

# ─── Panel reutilizable para figuras combinadas ───────────────────────────────
def dibujar_panel(ax, r, letra):
    ax.plot(r['eps'], r['sigma'], color='#2196F3', linewidth=1.2)

    eps_reg   = np.array([r['eps_lin'].min(), r['eps_lin'].max()])
    sigma_reg = r['E'] * eps_reg + r['intercept']
    ax.plot(eps_reg, sigma_reg, color='#B71C1C', linewidth=1.0,
            linestyle='--', alpha=0.8, zorder=2)

    txt = (f"E = {r['E']:.1f} ± {r['dE']:.1f} MPa  (R² = {r['R2']:.4f})\n"
           f"UTS = {r['UTS']:.2f} ± {r['dUTS']:.2f} MPa\n"
           f"εᵣ = {r['eps_r']:.4f} ± {r['deps_r']:.4f}")
    ax.text(0.97, 0.05, txt, transform=ax.transAxes,
            ha='right', va='bottom', multialignment='left', fontsize=7.5,
            bbox=dict(boxstyle='round,pad=0.35', facecolor='white',
                      alpha=0.9, edgecolor='#cccccc'))

    ax.text(0.03, 0.95, letra, transform=ax.transAxes,
            fontsize=12, fontweight='bold', va='top', ha='left')

    ax.set_xlabel('Deformación', fontsize=9)
    ax.set_ylabel('Tensión (MPa)', fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())

# ─── Figura 1: sin hilo (1 fila × 3 columnas) ─────────────────────────────────
SIN_HILO = [(2, 'A'), (6, 'B'), (7, 'C')]   # 3,5 / 2,0 / 1,0 mg/mL

if all(h in resultados for h, _ in SIN_HILO):
    fig1, axs1 = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    for (hilo_num, letra), ax in zip(SIN_HILO, axs1):
        dibujar_panel(ax, resultados[hilo_num], letra)

    fig1.savefig(os.path.join(SALIDA, 'figura_sin_hilo.png'),
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig1)
    print("Guardado: figura_sin_hilo.png")
else:
    faltantes = [h for h, _ in SIN_HILO if h not in resultados]
    print(f"AVISO: figura_sin_hilo.png omitida, faltan hilos {faltantes}")

# ─── Figura 2: con hilo (1 fila × 2 columnas) ─────────────────────────────────
CON_HILO = [(8, 'A'), (11, 'B')]   # 3,5 / 2,0 mg/mL

if all(h in resultados for h, _ in CON_HILO):
    fig2, axs2 = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)
    for (hilo_num, letra), ax in zip(CON_HILO, axs2):
        dibujar_panel(ax, resultados[hilo_num], letra)

    fig2.savefig(os.path.join(SALIDA, 'figura_con_hilo.png'),
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig2)
    print("Guardado: figura_con_hilo.png")
else:
    faltantes = [h for h, _ in CON_HILO if h not in resultados]
    print(f"AVISO: figura_con_hilo.png omitida, faltan hilos {faltantes}")

# ─── Curva individual para figuras con superposición por concentración ───────
def dibujar_curva(ax, r, hilo_num):
    conc  = CONC[hilo_num]
    color = COLORES[conc]

    ax.plot(r['eps'], r['sigma'], color=color, linewidth=1.3,
            label=f'{conc} mg/mL (E = {r["E"]:.1f} ± {r["dE"]:.1f} MPa)')

    eps_reg   = np.array([r['eps_lin'].min(), r['eps_lin'].max()])
    sigma_reg = r['E'] * eps_reg + r['intercept']
    ax.plot(eps_reg, sigma_reg, color='black', linewidth=1.0,
            linestyle='--', alpha=0.7, zorder=2)

# ─── Figura 3: curvas superpuestas por condición (1 fila × 2 columnas) ────────
SIN_HILO_SUP = [2, 6, 7]    # 3,5 / 2,0 / 1,0 mg/mL
CON_HILO_SUP = [8, 11]      # 3,5 / 2,0 mg/mL

if all(h in resultados for h in SIN_HILO_SUP + CON_HILO_SUP):
    fig3, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    resumen_sup = []

    for hilo_num in SIN_HILO_SUP:
        dibujar_curva(ax1, resultados[hilo_num], hilo_num)
        resumen_sup.append((hilo_num, 'Sin hilo', CONC[hilo_num], resultados[hilo_num]))

    for hilo_num in CON_HILO_SUP:
        dibujar_curva(ax2, resultados[hilo_num], hilo_num)
        resumen_sup.append((hilo_num, 'Con hilo', CONC[hilo_num], resultados[hilo_num]))

    for ax in (ax1, ax2):
        ax.set_xlabel('Deformación', fontsize=14)
        ax.set_ylabel('Tensión (MPa)', fontsize=14)
        ax.tick_params(axis='both', labelsize=12)
        ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.xaxis.set_minor_locator(ticker.AutoMinorLocator())
        ax.legend(loc='upper left', fontsize=12, frameon=True)

    fig3.savefig(os.path.join(SALIDA, 'figura_curvas_superpuestas.png'),
                 bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig3)
    print("Guardado: figura_curvas_superpuestas.png")

    # ─── Resumen de parámetros mecánicos (para el pie de figura) ─────────────
    print(f"\n{'Hilo':<6}{'Condición':<10}{'Conc.':<8}{'E (MPa)':<20}{'UTS (MPa)':<16}{'eps_r':<16}")
    for hilo_num, cond, conc, r in resumen_sup:
        print(f"{hilo_num:<6}{cond:<10}{conc:<8}" +
              f"{r['E']:.1f} ± {r['dE']:.1f}".ljust(20) +
              f"{r['UTS']:.2f} ± {r['dUTS']:.2f}".ljust(16) +
              f"{r['eps_r']:.4f} ± {r['deps_r']:.4f}")

    with open(os.path.join(SALIDA, 'resumen_curvas_superpuestas.txt'), 'w', encoding='utf-8') as f:
        f.write("Resumen para pie de figura — curvas superpuestas por concentración\n")
        f.write("=" * 68 + "\n")
        f.write(f"{'Hilo':<6}{'Condición':<10}{'Conc.':<8}{'E (MPa)':<20}{'UTS (MPa)':<16}{'eps_r':<16}\n")
        for hilo_num, cond, conc, r in resumen_sup:
            f.write(f"{hilo_num:<6}{cond:<10}{conc:<8}" +
                    f"{r['E']:.1f} ± {r['dE']:.1f}".ljust(20) +
                    f"{r['UTS']:.2f} ± {r['dUTS']:.2f}".ljust(16) +
                    f"{r['eps_r']:.4f} ± {r['deps_r']:.4f}\n")
    print("Guardado: resumen_curvas_superpuestas.txt")
else:
    faltantes = [h for h in SIN_HILO_SUP + CON_HILO_SUP if h not in resultados]
    print(f"AVISO: figura_curvas_superpuestas.png omitida, faltan hilos {faltantes}")

# ─── Figura resumen: E por hilo ───────────────────────────────────────────────
if resultados:
    hilos_ord = sorted(resultados)
    E_vals    = [resultados[h]['E']   for h in hilos_ord]
    dE_vals   = [resultados[h]['dE']  for h in hilos_ord]
    etiquetas = [f'Hilo {h}' for h in hilos_ord]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    x = np.arange(len(hilos_ord))
    ax.bar(x, E_vals, yerr=dE_vals, capsize=5,
           color='#1565C0', edgecolor='white', linewidth=0.5,
           error_kw=dict(ecolor='#212121', elinewidth=1.0))
    ax.set_xticks(x)
    ax.set_xticklabels(etiquetas, fontsize=9)
    ax.set_ylabel('Módulo de Young $E$ (MPa)', fontsize=9)
    ax.set_title('Módulo de Young por muestra', fontsize=9)
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())

    fig.savefig(os.path.join(SALIDA, 'resumen_E.png'),
                bbox_inches='tight', dpi=300, facecolor='white')
    plt.close(fig)
    print(f"\nGuardado: resumen_E.png")

# ─── Resumen en texto ─────────────────────────────────────────────────────────
    with open(os.path.join(SALIDA, 'resumen_mecanico.txt'), 'w', encoding='utf-8') as f:
        f.write("Análisis mecánico — Hilos de colágeno\n")
        f.write("=" * 55 + "\n")
        f.write(f"A = {A} ± {sigmaA} mm²   sigmaF = {sigmaF} N   "
                f"sigmaL0 = {sigmaL0} mm\n")
        f.write("Región lineal: puntos con columna Modulo no vacía\n\n")
        for h in sorted(resultados):
            r = resultados[h]
            f.write(f"Hilo {h}  (L0 = {r['L0']:.0f} mm)\n")
            f.write(f"  E     = {r['E']:.1f}  ±  {r['dE']:.1f}  MPa   (R2 = {r['R2']:.4f})\n")
            f.write(f"  UTS   = {r['UTS']:.3f}  ±  {r['dUTS']:.3f}  MPa\n")
            f.write(f"  eps_r = {r['eps_r']*100:.2f}  ±  {r['deps_r']*100:.3f}  %%\n\n")

    print(f"Guardado: resumen_mecanico.txt")
