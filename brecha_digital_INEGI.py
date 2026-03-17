"""
=================================================================
  PARTE II -- ESTADISTICA DESCRIPTIVA: BRECHA DIGITAL
  Dataset : INEGI_datos.xlsx  (INEGI Censo 2020)
  Estado  : Aguascalientes
  Unidad  : 11 municipios

  Objetivo: Identificar y cuantificar la brecha digital
  entre los municipios de Aguascalientes.

  Variables TIC analizadas (% de viviendas):
    - pct_internet        : Acceso a Internet
    - pct_computadora     : Computadora / laptop / tablet
    - pct_celular         : Telefono celular
    - pct_tel_fija        : Linea telefonica fija
    - pct_tv_paga         : Television de paga
    - pct_streaming       : Streaming (peliculas/musica por internet)
    - pct_sin_tic         : Sin ninguna TIC (exclusion total)
    - pct_sin_comp_int    : Sin computadora ni Internet

  Variables de contexto:
    - Grado promedio de escolaridad
    - % Analfabetismo
    - % Poblacion ocupada
    - % Poblacion indigena

  Metricas calculadas SIN .describe():
    media, mediana, varianza, desviacion estandar, CV,
    skewness, curtosis, IQR, percentiles (5,25,50,75,95)

  Instalacion:
    pip install pandas numpy matplotlib seaborn scipy openpyxl
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import unicodedata
import re
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

PALETA = ["#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
          "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#8bc34a"]


# ==============================================================
#  UTILIDAD: nombre de archivo seguro para Windows
# ==============================================================

def limpiar_nombre(nombre: str, max_len: int = 50) -> str:
    """
    Elimina acentos, caracteres especiales y trunca el nombre
    para evitar OSError al guardar archivos en Windows.
    """
    nombre = unicodedata.normalize("NFD", str(nombre))
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    nombre = re.sub(r"[^\w]", "_", nombre.lower())
    nombre = re.sub(r"_+", "_", nombre).strip("_")
    return nombre[:max_len]


# ==============================================================
#  0. CARGA Y PREPARACION
# ==============================================================

# Nombres de columnas originales (INEGI)
COL_VIV_CAR  = "Total de viviendas particulares habitadas con características"
COL_VIV_HAB  = "Total de viviendas particulares habitadas"
COL_POB_T    = "Población total"
COL_POB_15   = "Población de 15 años y más"
COL_POB_IND  = "Población de 3 años y más que habla alguna lengua indígena"
COL_ANALFAB  = "Población de 15 años y más analfabeta"
COL_OCUPADA  = "Población de 12 años y más ocupada"
COL_ECON_ACT = "Población de 12 años y más económicamente activa"
COL_ESCOL    = "Grado promedio de escolaridad"

# Mapa: nombre corto -> columna original TIC
TIC_MAP = {
    "pct_internet":      "Viviendas particulares habitadas que disponen de Internet",
    "pct_computadora":   "Viviendas particulares habitadas que disponen de computadora, laptop o tablet",
    "pct_celular":       "Viviendas particulares habitadas que disponen de teléfono celular",
    "pct_tel_fija":      "Viviendas particulares habitadas que disponen de línea telefónica fija",
    "pct_tv_paga":       "Viviendas particulares habitadas que disponen de servicio de televisión de paga",
    "pct_streaming":     "Viviendas particulares habitadas que disponen de servicio de películas, música o videos de paga por Internet",
    "pct_sin_tic":       "Viviendas particulares habitadas sin tecnologías de la información y de la comunicación (TIC)",
    "pct_sin_comp_int":  "Viviendas particulares habitadas sin computadora ni Internet",
    "pct_sin_tel":       "Viviendas particulares habitadas sin línea telefónica fija ni teléfono celular",
}

ETIQUETAS = {
    "pct_internet":      "% Internet",
    "pct_computadora":   "% Computadora/Tablet",
    "pct_celular":       "% Cel. celular",
    "pct_tel_fija":      "% Tel. fija",
    "pct_tv_paga":       "% TV paga",
    "pct_streaming":     "% Streaming",
    "pct_sin_tic":       "% Sin TIC",
    "pct_sin_comp_int":  "% Sin comp/internet",
    "pct_sin_tel":       "% Sin telefono",
    COL_ESCOL:           "Grado prom. escolaridad",
    "pct_analfabeta":    "% Analfabetismo",
    "pct_ocupada":       "% Ocupacion laboral",
    "pct_indigena":      "% Poblacion indigena",
}

VARS_TIC = list(TIC_MAP.keys())
VARS_CTX = [COL_ESCOL, "pct_analfabeta", "pct_ocupada", "pct_indigena"]


def convertir_numerico(serie: pd.Series) -> pd.Series:
    """Convierte a numerico reemplazando '*' y espacios por NaN."""
    return pd.to_numeric(
        serie.astype(str).str.replace("*", "", regex=False).str.strip(),
        errors="coerce"
    )


def cargar_y_preparar(ruta: str = "INEGI_datos.xlsx") -> pd.DataFrame:
    """
    Carga el dataset y filtra los totales por municipio
    (Clave de localidad == 0, municipio != 0).
    Calcula todos los indicadores TIC como porcentaje de viviendas.
    """
    print(f"\n  Cargando: {ruta}")
    df_raw = pd.read_excel(ruta, sheet_name=0)
    print(f"  Shape original: {df_raw.shape[0]:,} filas x {df_raw.shape[1]} columnas")

    # Filtrar totales municipales
    df = df_raw[
        (df_raw["Clave de localidad"] == 0) &
        (df_raw["Clave de municipio o demarcación territorial"] != 0)
    ].copy().reset_index(drop=True)

    print(f"  Municipios encontrados: {len(df)}")
    print(f"  Estado: {df['Entidad federativa'].iloc[0]}")

    # Convertir todas las columnas numericas
    cols_excluir = {"Entidad federativa", "Municipio o demarcación territorial",
                    "Localidad", "Tamaño de localidad"}
    for col in df.columns:
        if col not in cols_excluir:
            df[col] = convertir_numerico(df[col])

    # ── Indicadores TIC como % de viviendas ───────────────
    denominador = COL_VIV_CAR if COL_VIV_CAR in df.columns else COL_VIV_HAB
    for nuevo, original in TIC_MAP.items():
        if original in df.columns and denominador in df.columns:
            df[nuevo] = (df[original] / df[denominador] * 100).round(2)

    # ── Indicadores de contexto ────────────────────────────
    if COL_ANALFAB in df.columns and COL_POB_15 in df.columns:
        df["pct_analfabeta"] = (df[COL_ANALFAB] / df[COL_POB_15] * 100).round(2)

    if COL_POB_IND in df.columns and COL_POB_T in df.columns:
        df["pct_indigena"] = (df[COL_POB_IND] / df[COL_POB_T] * 100).round(2)

    if COL_OCUPADA in df.columns and COL_ECON_ACT in df.columns:
        df["pct_ocupada"] = (df[COL_OCUPADA] / df[COL_ECON_ACT] * 100).round(2)

    # Columna municipio corta para graficas
    df["Municipio"] = df["Municipio o demarcación territorial"]

    print(f"  Indicadores TIC calculados: {sum(v in df.columns for v in VARS_TIC)}")
    return df


# ==============================================================
#  FUNCIONES MATEMATICAS DESDE CERO (SIN .describe())
# ==============================================================

def calcular_media(x: np.ndarray) -> float:
    """x_bar = (1/n) * sum(xi)"""
    return float(np.sum(x) / len(x))

def calcular_mediana(x: np.ndarray) -> float:
    """Valor central. Promedio de los dos del medio si n es par."""
    xs = np.sort(x); n = len(xs); mid = n // 2
    return float(xs[mid]) if n % 2 == 1 else float((xs[mid-1] + xs[mid]) / 2.0)

def calcular_varianza(x: np.ndarray) -> float:
    """s^2 = (1/(n-1)) * sum((xi - x_bar)^2)  -- varianza muestral"""
    xbar = calcular_media(x)
    return float(np.sum((x - xbar) ** 2) / (len(x) - 1))

def calcular_desv_std(x: np.ndarray) -> float:
    """s = sqrt(s^2)"""
    return float(np.sqrt(calcular_varianza(x)))

def calcular_cv(x: np.ndarray) -> float:
    """CV = (s / |x_bar|) * 100  -- dispersion relativa"""
    media = calcular_media(x)
    return float("nan") if media == 0 else float(calcular_desv_std(x) / abs(media) * 100)

def calcular_skewness(x: np.ndarray) -> float:
    """g1 = [n/((n-1)(n-2))] * sum(((xi - x_bar)/s)^3)  -- asimetria de Fisher"""
    n = len(x)
    if n < 3: return float("nan")
    xbar, s = calcular_media(x), calcular_desv_std(x)
    if s == 0: return float("nan")
    return float(n / ((n-1)*(n-2)) * np.sum(((x - xbar)/s)**3))

def calcular_curtosis(x: np.ndarray) -> float:
    """g2 = [n(n+1)/((n-1)(n-2)(n-3))]*sum(((xi-x_bar)/s)^4) - 3(n-1)^2/((n-2)(n-3))"""
    n = len(x)
    if n < 4: return float("nan")
    xbar, s = calcular_media(x), calcular_desv_std(x)
    if s == 0: return float("nan")
    A = (n*(n+1)) / ((n-1)*(n-2)*(n-3))
    B = np.sum(((x - xbar)/s)**4)
    C = 3*(n-1)**2 / ((n-2)*(n-3))
    return float(A*B - C)

def calcular_percentil(x: np.ndarray, p: float) -> float:
    """Interpolacion lineal: L=(p/100)*(n-1)"""
    xs = np.sort(x); L = (p/100.0)*(len(xs)-1)
    lo, hi = int(np.floor(L)), int(np.ceil(L))
    return float(xs[lo]) if lo == hi else float(xs[lo] + (L-lo)*(xs[hi]-xs[lo]))

def calcular_iqr(x: np.ndarray) -> float:
    """IQR = Q3 - Q1 = P75 - P25"""
    return calcular_percentil(x, 75) - calcular_percentil(x, 25)

def metricas_completas(serie: pd.Series) -> dict:
    """Aplica todas las metricas a una serie numerica."""
    x = serie.dropna().values.astype(float)
    if len(x) < 2: return {}
    return {
        "n":        len(x),
        "media":    round(calcular_media(x), 3),
        "mediana":  round(calcular_mediana(x), 3),
        "varianza": round(calcular_varianza(x), 3),
        "desv_std": round(calcular_desv_std(x), 3),
        "cv_%":     round(calcular_cv(x), 3),
        "skewness": round(calcular_skewness(x), 3),
        "curtosis": round(calcular_curtosis(x), 3),
        "iqr":      round(calcular_iqr(x), 3),
        "min":      round(float(np.min(x)), 3),
        "p5":       round(calcular_percentil(x,  5), 3),
        "p25":      round(calcular_percentil(x, 25), 3),
        "p50":      round(calcular_percentil(x, 50), 3),
        "p75":      round(calcular_percentil(x, 75), 3),
        "p95":      round(calcular_percentil(x, 95), 3),
        "max":      round(float(np.max(x)), 3),
        "rango":    round(float(np.max(x) - np.min(x)), 3),
    }


# ==============================================================
#  1. ESTADISTICA DESCRIPTIVA COMPLETA
# ==============================================================

def estadistica_descriptiva(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  1. ESTADISTICA DESCRIPTIVA -- BRECHA DIGITAL")
    print("=" * 65)

    todas = [v for v in VARS_TIC + VARS_CTX if v in df.columns]
    resultados = {}
    for col in todas:
        res = metricas_completas(df[col])
        if res:
            resultados[ETIQUETAS.get(col, col)] = res

    df_res = pd.DataFrame(resultados).T
    df_res.index.name = "indicador"

    print("\n  METRICAS CENTRALES Y DISPERSION:")
    print(df_res[["n", "media", "mediana", "desv_std", "cv_%", "iqr"]].to_string())

    print("\n  FORMA DE LA DISTRIBUCION (skewness y curtosis):")
    sub = df_res[["skewness", "curtosis"]].copy()
    sub["asimetria"] = sub["skewness"].apply(
        lambda s: "simetrica" if abs(s) < 0.5
        else ("cola_derecha(+)" if s > 0 else "cola_izquierda(-)")
    )
    sub["forma"] = sub["curtosis"].apply(
        lambda k: "mesocurtica" if abs(k) < 0.5
        else ("leptocurtica" if k > 0 else "platicurtica")
    )
    print(sub.to_string())

    print("\n  PERCENTILES (5, 25, 50, 75, 95):")
    print(df_res[["min","p5","p25","p50","p75","p95","max"]].to_string())

    df_res.to_csv("bd_01_estadistica_descriptiva.csv")
    print("\n  Guardado: bd_01_estadistica_descriptiva.csv")
    return df_res


# ==============================================================
#  2. RANKING MUNICIPIOS POR ACCESO DIGITAL
# ==============================================================

def ranking_municipios(df: pd.DataFrame) -> pd.DataFrame:
    print("\n" + "=" * 65)
    print("  2. RANKING DE MUNICIPIOS -- ACCESO DIGITAL")
    print("=" * 65)

    cols_rank = [v for v in VARS_TIC if v in df.columns]
    df_rank = df[["Municipio"] + cols_rank].copy()
    df_rank.columns = ["Municipio"] + [ETIQUETAS.get(c, c) for c in cols_rank]

    # Indice de inclusion digital: promedio de accesos positivos normalizados
    accesos = [ETIQUETAS[c] for c in ["pct_internet","pct_computadora",
                                       "pct_celular","pct_tv_paga"]
               if c in df.columns and ETIQUETAS[c] in df_rank.columns]
    df_rank["indice_inclusion"] = df_rank[accesos].mean(axis=1).round(2)
    df_rank = df_rank.sort_values("indice_inclusion", ascending=False).reset_index(drop=True)
    df_rank.index += 1

    print("\n  Ranking por indice de inclusion digital (mayor = mejor acceso):")
    cols_show = ["Municipio","% Internet","% Computadora/Tablet",
                 "% Cel. celular","% Sin TIC","indice_inclusion"]
    cols_show = [c for c in cols_show if c in df_rank.columns]
    print(df_rank[cols_show].to_string())

    df_rank.to_csv("bd_02_ranking_municipios.csv")
    print("\n  Guardado: bd_02_ranking_municipios.csv")
    return df_rank


# ==============================================================
#  3. GRAFICA: RANKING INTERNET Y SIN TIC
# ==============================================================

def grafica_ranking(df: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # ── Internet ──
    if "pct_internet" in df.columns:
        d = df[["Municipio","pct_internet"]].dropna().sort_values("pct_internet")
        media = calcular_media(d["pct_internet"].values)
        col = ["#e74c3c" if v < media-5 else "#f39c12" if v < media else "#2ecc71"
               for v in d["pct_internet"]]
        bars = axes[0].barh(d["Municipio"], d["pct_internet"],
                            color=col, edgecolor="white", height=0.7)
        for bar, val in zip(bars, d["pct_internet"]):
            axes[0].text(bar.get_width()+0.4, bar.get_y()+bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=9)
        axes[0].axvline(media, color="black", linestyle="--", linewidth=1.5,
                        label=f"Media: {media:.1f}%")
        axes[0].set_title("% Viviendas con Internet", fontsize=12, fontweight="bold")
        axes[0].set_xlabel("% de viviendas")
        axes[0].legend(fontsize=9)

    # ── Sin TIC ──
    if "pct_sin_tic" in df.columns:
        d2 = df[["Municipio","pct_sin_tic"]].dropna().sort_values("pct_sin_tic", ascending=False)
        media2 = calcular_media(d2["pct_sin_tic"].values)
        col2 = ["#e74c3c" if v > media2+2 else "#f39c12" if v > media2 else "#2ecc71"
                for v in d2["pct_sin_tic"]]
        bars2 = axes[1].barh(d2["Municipio"], d2["pct_sin_tic"],
                             color=col2, edgecolor="white", height=0.7)
        for bar, val in zip(bars2, d2["pct_sin_tic"]):
            axes[1].text(bar.get_width()+0.1, bar.get_y()+bar.get_height()/2,
                         f"{val:.1f}%", va="center", fontsize=9)
        axes[1].axvline(media2, color="black", linestyle="--", linewidth=1.5,
                        label=f"Media: {media2:.1f}%")
        axes[1].set_title("% Viviendas SIN ninguna TIC (exclusion total)",
                          fontsize=12, fontweight="bold")
        axes[1].set_xlabel("% de viviendas")
        axes[1].legend(fontsize=9)

    plt.suptitle("Brecha Digital en Aguascalientes -- Por Municipio\n"
                 "Rojo = peor desempeno  |  Verde = mejor desempeno",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig("bd_03_ranking_internet_sin_tic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_03_ranking_internet_sin_tic.png")


# ==============================================================
#  4. HEATMAP PERFIL TIC COMPLETO
# ==============================================================

def heatmap_tic(df: pd.DataFrame):
    cols_heat = [v for v in VARS_TIC if v in df.columns]
    if not cols_heat: return

    df_heat = df[["Municipio"] + cols_heat].set_index("Municipio")
    df_heat.columns = [ETIQUETAS.get(c, c) for c in cols_heat]
    df_heat = df_heat.sort_values("% Internet", ascending=False)

    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(
        df_heat, annot=True, fmt=".1f",
        cmap="RdYlGn", linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 9},
        cbar_kws={"label": "% de viviendas"}
    )
    ax.set_title("Perfil TIC por Municipio -- Aguascalientes\n"
                 "Verde = mayor acceso  |  Rojo = menor acceso / mayor exclusion",
                 fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("bd_04_heatmap_tic_municipios.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_04_heatmap_tic_municipios.png")


# ==============================================================
#  5. DISTRIBUCIONES TIC CON HISTOGRAMA + BOXPLOT
# ==============================================================

def distribuciones_tic(df: pd.DataFrame):
    vars_plot = [v for v in VARS_TIC if v in df.columns]
    n = len(vars_plot)
    ncols = 3
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols * 2,
                              figsize=(ncols*2*4, nrows*3.8))
    axes = axes.reshape(nrows, ncols*2)

    for idx, col in enumerate(vars_plot):
        row       = idx // ncols
        col_start = (idx % ncols) * 2
        ax_h = axes[row, col_start]
        ax_b = axes[row, col_start+1]
        data = df[col].dropna().values.astype(float)
        lbl  = ETIQUETAS.get(col, col)

        # Histograma
        ax_h.hist(data, bins=6, color="#3498db", edgecolor="white",
                  alpha=0.85, density=True)
        media   = calcular_media(data)
        mediana = calcular_mediana(data)
        sk      = calcular_skewness(data)
        cv      = calcular_cv(data)
        ax_h.axvline(media,   color="#2ecc71", linestyle="--",
                     linewidth=1.5, label=f"Media={media:.1f}%")
        ax_h.axvline(mediana, color="#f39c12", linestyle=":",
                     linewidth=1.5, label=f"Med={mediana:.1f}%")
        ax_h.set_title(f"{lbl}\nskew={sk:.2f}  CV={cv:.1f}%", fontsize=8)
        ax_h.legend(fontsize=6)
        ax_h.tick_params(labelsize=7)

        # Boxplot
        ax_b.boxplot(data, vert=True, patch_artist=True,
                     boxprops=dict(facecolor="#3498db", alpha=0.6),
                     medianprops=dict(color="#e74c3c", linewidth=2),
                     flierprops=dict(marker="o", color="#e74c3c",
                                     alpha=0.5, markersize=5))
        p5  = calcular_percentil(data,  5)
        p95 = calcular_percentil(data, 95)
        ax_b.axhline(p5,  color="#9b59b6", linestyle="--",
                     linewidth=1, alpha=0.7, label=f"P5={p5:.1f}")
        ax_b.axhline(p95, color="#9b59b6", linestyle="--",
                     linewidth=1, alpha=0.7, label=f"P95={p95:.1f}")
        ax_b.set_title(lbl, fontsize=8)
        ax_b.legend(fontsize=6)
        ax_b.set_xticks([])

    for idx in range(n, nrows * ncols):
        row = idx // ncols; cs = (idx % ncols) * 2
        axes[row, cs].set_visible(False)
        axes[row, cs+1].set_visible(False)

    plt.suptitle("Distribuciones TIC -- 11 Municipios de Aguascalientes",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bd_05_distribuciones_tic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_05_distribuciones_tic.png")


# ==============================================================
#  6. CORRELACION TIC VS CONTEXTO
# ==============================================================

def correlacion_tic_contexto(df: pd.DataFrame):
    vars_corr = [v for v in VARS_TIC + VARS_CTX if v in df.columns]
    if len(vars_corr) < 3: return

    data_c = df[vars_corr].copy()
    data_c.columns = [ETIQUETAS.get(c, c) for c in vars_corr]
    corr = data_c.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(14, 11))
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mascara, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 8},
        vmin=-1, vmax=1
    )
    ax.set_title("Correlacion: Indicadores TIC vs Factores de Contexto\n"
                 "Rojo=correlacion positiva | Azul=negativa",
                 fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    plt.tight_layout()
    plt.savefig("bd_06_correlacion_tic_contexto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_06_correlacion_tic_contexto.png")

    if "% Internet" in corr.columns:
        top = corr["% Internet"].drop("% Internet").sort_values(key=abs, ascending=False)
        print("\n  Correlaciones con % Internet (r de Pearson):")
        print(top.round(3).to_string())

    corr.to_csv("bd_06_correlacion_tic_contexto.csv")
    print("  Guardado: bd_06_correlacion_tic_contexto.csv")


# ==============================================================
#  7. SCATTER: INTERNET VS FACTORES SOCIOECONOMICOS
# ==============================================================

def scatter_brecha(df: pd.DataFrame):
    pares = [
        (COL_ESCOL,        "pct_internet",
         "Grado prom. escolaridad", "% Viviendas con Internet",
         "Escolaridad vs Acceso a Internet"),
        ("pct_analfabeta",  "pct_sin_tic",
         "% Analfabetismo", "% Sin TIC (exclusion total)",
         "Analfabetismo vs Exclusion Digital"),
        ("pct_indigena",    "pct_internet",
         "% Poblacion indigena", "% Viviendas con Internet",
         "Poblacion indigena vs Acceso a Internet"),
        ("pct_ocupada",     "pct_computadora",
         "% Poblacion ocupada", "% Computadora/Tablet",
         "Empleo vs Acceso a Computadora"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    for idx, (x_col, y_col, x_lbl, y_lbl, titulo) in enumerate(pares):
        if x_col not in df.columns or y_col not in df.columns:
            axes[idx].set_visible(False)
            continue

        ax  = axes[idx]
        sub = df[[x_col, y_col, "Municipio"]].dropna()
        x   = sub[x_col].values.astype(float)
        y   = sub[y_col].values.astype(float)

        for i, (xi, yi, mun) in enumerate(zip(x, y, sub["Municipio"])):
            ax.scatter(xi, yi, color=PALETA[i % len(PALETA)],
                       s=120, zorder=3, edgecolors="white", linewidth=0.8)
            ax.annotate(mun[:14], (xi, yi), fontsize=7,
                        xytext=(4, 4), textcoords="offset points")

        # Linea de tendencia
        try:
            from scipy import stats as sp_stats
            slope, intercept, r, p_val, _ = sp_stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            ax.plot(x_line, slope*x_line + intercept,
                    color="#e74c3c", linewidth=1.8, linestyle="--",
                    label=f"r={r:.3f}  p={p_val:.3f}")
            ax.legend(fontsize=9)
        except Exception:
            pass

        ax.set_xlabel(x_lbl, fontsize=9)
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    plt.suptitle("Brecha Digital: Relacion con Factores Socioeconomicos\n"
                 "Aguascalientes -- Por Municipio",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("bd_07_scatter_brecha_contexto.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_07_scatter_brecha_contexto.png")


# ==============================================================
#  8. GRAFICA MULTIVARIABLE: PERFIL RADAR POR MUNICIPIO
# ==============================================================

def grafica_barras_agrupadas(df: pd.DataFrame):
    """Barras agrupadas de los principales indicadores TIC por municipio."""
    vars_bar = [v for v in ["pct_internet","pct_computadora","pct_celular",
                             "pct_tv_paga","pct_sin_tic"] if v in df.columns]
    if not vars_bar: return

    df_bar = df[["Municipio"] + vars_bar].set_index("Municipio")
    df_bar.columns = [ETIQUETAS.get(c, c) for c in vars_bar]
    df_bar = df_bar.sort_values("% Internet", ascending=False)

    x     = np.arange(len(df_bar))
    n_var = len(df_bar.columns)
    ancho = 0.15
    offsets = np.linspace(-(n_var-1)*ancho/2, (n_var-1)*ancho/2, n_var)

    colores_bar = ["#3498db","#2ecc71","#9b59b6","#f39c12","#e74c3c"]
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, (col, offset, color) in enumerate(zip(df_bar.columns, offsets, colores_bar)):
        bars = ax.bar(x + offset, df_bar[col], ancho,
                      label=col, color=color, alpha=0.85,
                      edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(df_bar.index, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("% de viviendas")
    ax.set_title("Perfil TIC Completo por Municipio -- Aguascalientes\n"
                 "Ordenado por acceso a Internet (mayor a menor)",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig("bd_08_barras_agrupadas_tic.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: bd_08_barras_agrupadas_tic.png")


# ==============================================================
#  9. INDICE DE BRECHA DIGITAL (IBD) Y CLASIFICACION
# ==============================================================

def indice_brecha_digital(df: pd.DataFrame) -> pd.DataFrame:
    """
    Indice de Brecha Digital (IBD):
      - Normaliza cada indicador TIC de 0 a 100
      - Indicadores positivos (acceso): mayor = mejor
      - Indicadores negativos (exclusion): se invierte (100 - valor)
      - IBD = promedio de todos los componentes normalizados
      - Rango: 0 (maxima brecha) a 100 (minima brecha)
    Clasifica municipios en 3 grupos por tercil.
    """
    print("\n" + "=" * 65)
    print("  9. INDICE DE BRECHA DIGITAL (IBD)")
    print("=" * 65)

    componentes = {
        "pct_internet":     True,   # positivo
        "pct_computadora":  True,
        "pct_celular":      True,
        "pct_tv_paga":      True,
        "pct_sin_tic":      False,  # negativo (se invierte)
        "pct_sin_comp_int": False,
    }
    componentes = {k: v for k, v in componentes.items() if k in df.columns}

    df_ibd = df[["Municipio"]].copy()

    for col, positivo in componentes.items():
        col_min = df[col].min(); col_max = df[col].max()
        rango   = col_max - col_min
        if rango == 0:
            df_ibd[col+"_norm"] = 50.0
            continue
        norm = (df[col] - col_min) / rango * 100
        df_ibd[col+"_norm"] = norm if positivo else (100 - norm)

    norm_cols = [c+"_norm" for c in componentes]
    df_ibd["IBD"] = df_ibd[norm_cols].mean(axis=1).round(2)
    df_ibd = df_ibd.sort_values("IBD", ascending=False).reset_index(drop=True)
    df_ibd.index += 1

    # Clasificacion por terciles
    ibd_vals = df_ibd["IBD"].values
    p33 = calcular_percentil(ibd_vals, 33)
    p67 = calcular_percentil(ibd_vals, 67)
    df_ibd["clasificacion"] = df_ibd["IBD"].apply(
        lambda v: "Baja brecha (mejor acceso)"  if v >= p67
        else "Brecha media" if v >= p33
        else "Alta brecha (peor acceso)"
    )

    print("\n  Indice de Brecha Digital (IBD):")
    print("  100 = mejor acceso digital  |  0 = maxima exclusion\n")
    print(df_ibd[["Municipio","IBD","clasificacion"]].to_string())

    # Grafica IBD
    fig, ax = plt.subplots(figsize=(12, 7))
    colores_ibd = ["#2ecc71" if c == "Baja brecha (mejor acceso)"
                   else "#f39c12" if c == "Brecha media"
                   else "#e74c3c"
                   for c in df_ibd["clasificacion"]]
    bars = ax.barh(df_ibd["Municipio"], df_ibd["IBD"],
                   color=colores_ibd, edgecolor="white", height=0.7)
    for bar, val in zip(bars, df_ibd["IBD"]):
        ax.text(bar.get_width()+0.5, bar.get_y()+bar.get_height()/2,
                f"{val:.1f}", va="center", fontsize=9, fontweight="bold")

    ax.axvline(p33, color="#f39c12", linestyle="--", alpha=0.7,
               label=f"P33 = {p33:.1f}")
    ax.axvline(p67, color="#2ecc71", linestyle="--", alpha=0.7,
               label=f"P67 = {p67:.1f}")
    ax.set_xlabel("Indice de Brecha Digital (IBD)")
    ax.set_title("Indice de Brecha Digital por Municipio\n"
                 "Verde = baja brecha  |  Naranja = media  |  Rojo = alta brecha",
                 fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.invert_yaxis()
    plt.tight_layout()
    plt.savefig("bd_09_indice_brecha_digital.png", dpi=150, bbox_inches="tight")
    plt.close()

    df_ibd.to_csv("bd_09_indice_brecha_digital.csv", index=False)
    print("\n  Guardado: bd_09_indice_brecha_digital.png")
    print("  Guardado: bd_09_indice_brecha_digital.csv")
    return df_ibd


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  ANALISIS DE BRECHA DIGITAL")
    print("  Aguascalientes -- 11 Municipios -- INEGI Censo 2020")
    print("=" * 65)

    df = cargar_y_preparar("INEGI_datos.xlsx")

    print("\n  Variables TIC disponibles:")
    for v in VARS_TIC:
        ok = "OK" if v in df.columns else "NO ENCONTRADA"
        print(f"    [{ok}] {ETIQUETAS.get(v, v)}")

    print("\n[1/9] Estadistica descriptiva...")
    df_res = estadistica_descriptiva(df)

    print("\n[2/9] Ranking de municipios...")
    df_rank = ranking_municipios(df)

    print("\n[3/9] Grafica ranking Internet vs Sin TIC...")
    grafica_ranking(df)

    print("\n[4/9] Heatmap perfil TIC...")
    heatmap_tic(df)

    print("\n[5/9] Distribuciones TIC...")
    distribuciones_tic(df)

    print("\n[6/9] Correlacion TIC vs contexto...")
    correlacion_tic_contexto(df)

    print("\n[7/9] Scatter brecha vs factores socioeconomicos...")
    scatter_brecha(df)

    print("\n[8/9] Barras agrupadas TIC por municipio...")
    grafica_barras_agrupadas(df)

    print("\n[9/9] Indice de Brecha Digital...")
    df_ibd = indice_brecha_digital(df)

    print("\n" + "=" * 65)
    print("  ARCHIVOS GENERADOS")
    print("=" * 65)
    for a in [
        "bd_01_estadistica_descriptiva.csv",
        "bd_02_ranking_municipios.csv",
        "bd_03_ranking_internet_sin_tic.png",
        "bd_04_heatmap_tic_municipios.png",
        "bd_05_distribuciones_tic.png",
        "bd_06_correlacion_tic_contexto.png",
        "bd_06_correlacion_tic_contexto.csv",
        "bd_07_scatter_brecha_contexto.png",
        "bd_08_barras_agrupadas_tic.png",
        "bd_09_indice_brecha_digital.png",
        "bd_09_indice_brecha_digital.csv",
    ]:
        print(f"  -> {a}")
    print("=" * 65)