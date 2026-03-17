"""
=================================================================
  PARTE IV -- VISUALIZACION ESTADISTICA OBLIGATORIA
  Dataset : INEGI_datos.xlsx  (INEGI Censo 2020)
  Estado  : Aguascalientes

  Graficos univariados:
    1. Histogramas con KDE
    2. Boxplots
    3. Violin plots

  Graficos multivariados:
    4. Matriz de correlacion
    5. Pairplots
    6. Mapas geograficos (latitud/longitud)
    7. Scatter poblacion vs vivienda
    8. Mapas de calor por municipio

  Distribuciones demograficas:
    9.  Piramide poblacional
    10. Proporcion hombres/mujeres
    11. Densidad poblacional

  Instalacion:
    pip install pandas numpy matplotlib seaborn scipy openpyxl
=================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde
import unicodedata
import re
import warnings

warnings.filterwarnings("ignore")

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.25,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

PALETA_MUN = [
    "#e74c3c","#3498db","#2ecc71","#f39c12","#9b59b6",
    "#1abc9c","#e67e22","#34495e","#e91e63","#00bcd4","#8bc34a"
]


# ==============================================================
#  UTILIDADES
# ==============================================================

def limpiar_nombre(nombre: str, max_len: int = 50) -> str:
    nombre = unicodedata.normalize("NFD", str(nombre))
    nombre = "".join(c for c in nombre if not unicodedata.combining(c))
    nombre = re.sub(r"[^\w]", "_", nombre.lower())
    nombre = re.sub(r"_+", "_", nombre).strip("_")
    return nombre[:max_len]


def num(serie: pd.Series) -> pd.Series:
    """Convierte a numerico reemplazando '*' por NaN."""
    return pd.to_numeric(
        serie.astype(str).str.replace("*", "", regex=False).str.strip(),
        errors="coerce"
    )


def dms_a_decimal(dms_str: str) -> float:
    """Convierte coordenadas DMS (21°52'47.362\" N) a decimal."""
    try:
        nums = re.findall(r"[\d\.]+", str(dms_str))
        if len(nums) < 3:
            return None
        d, m, s = float(nums[0]), float(nums[1]), float(nums[2])
        dec = d + m / 60 + s / 3600
        if "W" in str(dms_str) or "S" in str(dms_str):
            dec = -dec
        return round(dec, 6)
    except Exception:
        return None


# ==============================================================
#  0. CARGA Y PREPARACION
# ==============================================================

# Columnas TIC
COL_VIV_CAR  = "Total de viviendas particulares habitadas con características"
COL_VIV_HAB  = "Total de viviendas particulares habitadas"
COL_POB_T    = "Población total"
COL_POB_FEM  = "Población femenina"
COL_POB_MAS  = "Población masculina"
COL_ESCOL    = "Grado promedio de escolaridad"
COL_ANALFAB  = "Población de 15 años y más analfabeta"
COL_POB_15   = "Población de 15 años y más"
COL_OCUPADA  = "Población de 12 años y más ocupada"
COL_ECON_ACT = "Población de 12 años y más económicamente activa"
COL_POB_IND  = "Población de 3 años y más que habla alguna lengua indígena"
COL_INTERNET = "Viviendas particulares habitadas que disponen de Internet"
COL_COMP     = "Viviendas particulares habitadas que disponen de computadora, laptop o tablet"
COL_CEL      = "Viviendas particulares habitadas que disponen de teléfono celular"
COL_TEL_FIJA = "Viviendas particulares habitadas que disponen de línea telefónica fija"
COL_TV_PAGA  = "Viviendas particulares habitadas que disponen de servicio de televisión de paga"
COL_SIN_TIC  = "Viviendas particulares habitadas sin tecnologías de la información y de la comunicación (TIC)"
COL_SIN_CI   = "Viviendas particulares habitadas sin computadora ni Internet"

# Grupos quinquenales de edad para piramide
GRUPOS_FEM = [
    "Población femenina de 0 a 4 años",
    "Población femenina de 5 a 9 años",
    "Población femenina de 10 a 14 años",
    "Población femenina de 15 a 19 años",
    "Población femenina de 20 a 24 años",
    "Población femenina de 25 a 29 años",
    "Población femenina de 30 a 34 años",
    "Población femenina de 35 a 39 años",
    "Población femenina de 40 a 44 años",
    "Población femenina de 45 a 49 años",
    "Población femenina de 50 a 54 años",
    "Población femenina de 55 a 59 años",
    "Población femenina de 60 a 64 años",
    "Población femenina de 65 a 69 años",
    "Población femenina de 70 a 74 años",
    "Población femenina de 75 a 79 años",
    "Población femenina de 80 a 84 años",
    "Población femenina de 85 años y más",
]
GRUPOS_MAS = [g.replace("femenina", "masculina") for g in GRUPOS_FEM]
ETIQUETAS_EDAD = [
    "0-4","5-9","10-14","15-19","20-24","25-29",
    "30-34","35-39","40-44","45-49","50-54","55-59",
    "60-64","65-69","70-74","75-79","80-84","85+"
]

TIC_VARS = {
    "pct_internet":   COL_INTERNET,
    "pct_comp":       COL_COMP,
    "pct_celular":    COL_CEL,
    "pct_tel_fija":   COL_TEL_FIJA,
    "pct_tv_paga":    COL_TV_PAGA,
    "pct_sin_tic":    COL_SIN_TIC,
    "pct_sin_ci":     COL_SIN_CI,
}
TIC_LABELS = {
    "pct_internet":  "% Internet",
    "pct_comp":      "% Computadora",
    "pct_celular":   "% Celular",
    "pct_tel_fija":  "% Tel. fija",
    "pct_tv_paga":   "% TV paga",
    "pct_sin_tic":   "% Sin TIC",
    "pct_sin_ci":    "% Sin comp/internet",
}


def cargar_datos(ruta: str = "INEGI_datos.xlsx"):
    """
    Retorna tres DataFrames:
      df_mun   : totales por municipio (11 filas)
      df_loc   : todas las localidades con coordenadas
      df_total : fila del total estatal
    """
    print(f"\n  Cargando: {ruta}")
    df_raw = pd.read_excel(ruta, sheet_name=0)
    print(f"  Shape: {df_raw.shape[0]:,} x {df_raw.shape[1]}")

    # Convertir columnas numericas
    cols_texto = {"Entidad federativa", "Municipio o demarcación territorial",
                  "Localidad", "Tamaño de localidad", "Latitud", "Longitud"}
    for col in df_raw.columns:
        if col not in cols_texto:
            df_raw[col] = num(df_raw[col])

    # Total estatal
    df_total = df_raw[
        (df_raw["Clave de municipio o demarcación territorial"] == 0) &
        (df_raw["Clave de localidad"] == 0)
    ].copy()

    # Totales municipales
    df_mun = df_raw[
        (df_raw["Clave de localidad"] == 0) &
        (df_raw["Clave de municipio o demarcación territorial"] != 0)
    ].copy().reset_index(drop=True)
    df_mun["Municipio"] = df_mun["Municipio o demarcación territorial"]

    # Localidades con coordenadas
    df_loc = df_raw[
        (df_raw["Clave de localidad"] != 0) &
        (df_raw["Clave de localidad"] < 9000)
    ].copy().reset_index(drop=True)
    df_loc["lat"] = df_loc["Latitud"].apply(dms_a_decimal)
    df_loc["lon"] = df_loc["Longitud"].apply(dms_a_decimal)
    df_loc = df_loc.dropna(subset=["lat", "lon", COL_POB_T]).copy()
    print(f"  Municipios: {len(df_mun)}  |  Localidades con coords: {len(df_loc)}")

    # Calcular indicadores TIC en municipios
    den = COL_VIV_CAR if COL_VIV_CAR in df_mun.columns else COL_VIV_HAB
    for nuevo, original in TIC_VARS.items():
        if original in df_mun.columns and den in df_mun.columns:
            df_mun[nuevo] = (df_mun[original] / df_mun[den] * 100).round(2)

    if COL_ANALFAB in df_mun.columns and COL_POB_15 in df_mun.columns:
        df_mun["pct_analfabeta"] = (df_mun[COL_ANALFAB] / df_mun[COL_POB_15] * 100).round(2)
    if COL_POB_IND in df_mun.columns:
        df_mun["pct_indigena"] = (df_mun[COL_POB_IND] / df_mun[COL_POB_T] * 100).round(2)
    if COL_OCUPADA in df_mun.columns and COL_ECON_ACT in df_mun.columns:
        df_mun["pct_ocupada"] = (df_mun[COL_OCUPADA] / df_mun[COL_ECON_ACT] * 100).round(2)

    # Calcular pct_internet en localidades (para mapa)
    if COL_INTERNET in df_loc.columns and den in df_loc.columns:
        df_loc["pct_internet"] = (df_loc[COL_INTERNET] / df_loc[den] * 100).round(2)

    return df_mun, df_loc, df_total


# ==============================================================
#  ── UNIVARIADOS ──
# ==============================================================

# 1. HISTOGRAMAS CON KDE
# ==============================================================

def histogramas_kde(df_mun: pd.DataFrame):
    """
    Histograma + curva KDE para cada variable TIC.
    Muestra media (verde) y mediana (naranja).
    """
    vars_plot = [v for v in TIC_VARS if v in df_mun.columns]
    n = len(vars_plot)
    ncols = 4
    nrows = -(-n // ncols)

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 4.5, nrows * 3.8))
    axes = axes.flatten()

    for i, col in enumerate(vars_plot):
        ax   = axes[i]
        data = df_mun[col].dropna().values.astype(float)
        lbl  = TIC_LABELS.get(col, col)

        # Histograma
        ax.hist(data, bins=7, color="#3498db", edgecolor="white",
                alpha=0.8, density=True, label="Frecuencia")

        # KDE (solo si hay varianza)
        if len(data) > 2 and data.std() > 0:
            try:
                kde    = gaussian_kde(data, bw_method="silverman")
                x_line = np.linspace(data.min() * 0.95, data.max() * 1.05, 300)
                ax.plot(x_line, kde(x_line), color="#e74c3c",
                        linewidth=2, label="KDE")
            except Exception:
                pass

        media   = data.mean()
        mediana = np.median(data)
        ax.axvline(media,   color="#2ecc71", linestyle="--",
                   linewidth=1.8, label=f"Media={media:.1f}")
        ax.axvline(mediana, color="#f39c12", linestyle=":",
                   linewidth=1.8, label=f"Med={mediana:.1f}")
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xlabel("% viviendas", fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Histogramas con KDE -- Indicadores TIC\nAguascalientes, 11 municipios",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("p4_01_histogramas_kde.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_01_histogramas_kde.png")


# ==============================================================
# 2. BOXPLOTS
# ==============================================================

def boxplots(df_mun: pd.DataFrame):
    """
    Boxplot horizontal de cada indicador TIC.
    Muestra P5, P95 como lineas adicionales y los valores
    de cada municipio como puntos superpuestos (jitter).
    """
    vars_plot = [v for v in TIC_VARS if v in df_mun.columns]
    data_list  = [df_mun[v].dropna().values.astype(float) for v in vars_plot]
    labels     = [TIC_LABELS.get(v, v) for v in vars_plot]

    fig, ax = plt.subplots(figsize=(14, 7))
    bp = ax.boxplot(
        data_list,
        vert=False,
        patch_artist=True,
        labels=labels,
        medianprops=dict(color="#e74c3c", linewidth=2.5),
        flierprops=dict(marker="D", color="#e74c3c", alpha=0.5, markersize=5),
        whiskerprops=dict(linewidth=1.5, color="#555"),
        capprops=dict(linewidth=1.5, color="#555"),
    )

    colores = ["#3498db","#2ecc71","#9b59b6","#f39c12","#1abc9c","#e74c3c","#e67e22"]
    for patch, color in zip(bp["boxes"], colores):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    # Puntos jitter para ver cada municipio
    np.random.seed(42)
    for i, (v, col) in enumerate(zip(vars_plot, colores), 1):
        data = df_mun[v].dropna().values.astype(float)
        jitter = np.random.uniform(-0.2, 0.2, len(data))
        ax.scatter(data, np.full_like(data, i) + jitter,
                   color=col, s=40, zorder=5, edgecolors="white",
                   linewidth=0.5, alpha=0.9)

    ax.set_xlabel("% de viviendas", fontsize=10)
    ax.set_title("Boxplots de Indicadores TIC -- Distribucion entre Municipios\n"
                 "Puntos = municipios individuales  |  Linea roja = mediana",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig("p4_02_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_02_boxplots.png")


# ==============================================================
# 3. VIOLIN PLOTS
# ==============================================================

def violin_plots(df_mun: pd.DataFrame):
    """
    Violin plot de los principales indicadores TIC.
    Combina KDE + boxplot interno. Util para ver la forma
    de la distribucion con pocos datos (11 municipios).
    """
    vars_plot = [v for v in ["pct_internet","pct_comp","pct_celular",
                              "pct_tv_paga","pct_sin_tic","pct_sin_ci"]
                 if v in df_mun.columns]

    df_long = pd.melt(
        df_mun[["Municipio"] + vars_plot],
        id_vars="Municipio",
        var_name="indicador",
        value_name="valor"
    )
    df_long["indicador"] = df_long["indicador"].map(TIC_LABELS)

    fig, axes = plt.subplots(1, len(vars_plot),
                              figsize=(len(vars_plot) * 3, 7))

    colores = ["#3498db","#2ecc71","#9b59b6","#f39c12","#e74c3c","#e67e22"]

    for i, (v, col_color) in enumerate(zip(vars_plot, colores)):
        ax   = axes[i]
        data = df_mun[v].dropna().values.astype(float)
        lbl  = TIC_LABELS.get(v, v)

        parts = ax.violinplot(data, positions=[0], showmedians=True,
                              showextrema=True, widths=0.8)
        for pc in parts["bodies"]:
            pc.set_facecolor(col_color)
            pc.set_alpha(0.7)
        parts["cmedians"].set_color("#e74c3c")
        parts["cmedians"].set_linewidth(2)
        parts["cmins"].set_color("#555")
        parts["cmaxes"].set_color("#555")
        parts["cbars"].set_color("#555")

        # Superponer puntos individuales
        jitter = np.random.uniform(-0.08, 0.08, len(data))
        ax.scatter(np.zeros(len(data)) + jitter, data,
                   color=col_color, s=50, zorder=5,
                   edgecolors="white", linewidth=0.5, alpha=0.9)

        ax.set_xticks([])
        ax.set_title(lbl, fontsize=8, fontweight="bold")
        ax.set_ylabel("% viviendas" if i == 0 else "", fontsize=8)
        ax.tick_params(labelsize=7)

    plt.suptitle("Violin Plots -- Distribucion de Indicadores TIC\n"
                 "Muestra la forma de distribucion + medianas (rojo)",
                 fontsize=12, y=1.02)
    plt.tight_layout()
    plt.savefig("p4_03_violin_plots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_03_violin_plots.png")


# ==============================================================
#  ── MULTIVARIADOS ──
# ==============================================================

# 4. MATRIZ DE CORRELACION
# ==============================================================

def matriz_correlacion(df_mun: pd.DataFrame):
    """
    Heatmap triangular de correlacion de Pearson entre
    todos los indicadores TIC y variables de contexto.
    """
    vars_corr = [v for v in list(TIC_VARS.keys()) +
                 [COL_ESCOL,"pct_analfabeta","pct_ocupada","pct_indigena"]
                 if v in df_mun.columns]

    labels = {**TIC_LABELS,
              COL_ESCOL:       "Escolaridad",
              "pct_analfabeta":"% Analfabeta",
              "pct_ocupada":   "% Ocupada",
              "pct_indigena":  "% Indigena"}

    data_c = df_mun[vars_corr].copy()
    data_c.columns = [labels.get(c, c) for c in vars_corr]
    corr = data_c.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(13, 10))
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mascara, annot=True, fmt=".2f",
        cmap="coolwarm", center=0,
        linewidths=0.5, linecolor="white",
        ax=ax, annot_kws={"size": 9},
        vmin=-1, vmax=1,
        cbar_kws={"label": "r de Pearson"}
    )
    ax.set_title("Matriz de Correlacion -- TIC y Contexto Socioeconomico\n"
                 "Rojo=positiva | Azul=negativa",
                 fontsize=12, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha="right", fontsize=9)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=9)
    plt.tight_layout()
    plt.savefig("p4_04_matriz_correlacion.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_04_matriz_correlacion.png")


# ==============================================================
# 5. PAIRPLOTS
# ==============================================================

def pairplots(df_mun: pd.DataFrame):
    """
    Pairplot de los 5 principales indicadores TIC.
    Diagonal = KDE. Off-diagonal = scatter con regresion.
    Color por municipio.
    """
    vars_pair = [v for v in ["pct_internet","pct_comp","pct_celular",
                              "pct_sin_tic",COL_ESCOL]
                 if v in df_mun.columns]
    lbl_map = {**TIC_LABELS, COL_ESCOL: "Escolaridad"}

    df_pp = df_mun[["Municipio"] + vars_pair].dropna().copy()
    df_pp.columns = ["Municipio"] + [lbl_map.get(c, c) for c in vars_pair]

    n = len(df_pp.columns) - 1  # sin Municipio
    fig, axes = plt.subplots(n, n, figsize=(n * 3.2, n * 3.2))

    cols = [c for c in df_pp.columns if c != "Municipio"]
    colores_mun = {m: PALETA_MUN[i % len(PALETA_MUN)]
                   for i, m in enumerate(df_pp["Municipio"])}

    for i, col_y in enumerate(cols):
        for j, col_x in enumerate(cols):
            ax = axes[i, j]
            if i == j:
                # Diagonal: KDE
                data = df_pp[col_x].values.astype(float)
                if data.std() > 0:
                    try:
                        kde    = gaussian_kde(data)
                        x_line = np.linspace(data.min(), data.max(), 200)
                        ax.plot(x_line, kde(x_line), color="#3498db", linewidth=2)
                        ax.fill_between(x_line, kde(x_line), alpha=0.2, color="#3498db")
                    except Exception:
                        pass
                ax.set_xlabel("")
            else:
                # Off-diagonal: scatter
                for _, row in df_pp.iterrows():
                    ax.scatter(row[col_x], row[col_y],
                               color=colores_mun[row["Municipio"]],
                               s=50, edgecolors="white", linewidth=0.5, zorder=3)
                # Linea tendencia
                try:
                    from scipy import stats as sp_stats
                    x_ = df_pp[col_x].values.astype(float)
                    y_ = df_pp[col_y].values.astype(float)
                    slope, intercept, r, _, _ = sp_stats.linregress(x_, y_)
                    x_line = np.linspace(x_.min(), x_.max(), 100)
                    ax.plot(x_line, slope * x_line + intercept,
                            color="#e74c3c", linewidth=1.2, alpha=0.8)
                    ax.text(0.97, 0.05, f"r={r:.2f}",
                            transform=ax.transAxes, fontsize=6,
                            ha="right", color="#e74c3c")
                except Exception:
                    pass

            # Etiquetas solo en bordes
            if i == n - 1:
                ax.set_xlabel(col_x, fontsize=7)
            else:
                ax.set_xlabel("")
            if j == 0:
                ax.set_ylabel(col_y, fontsize=7)
            else:
                ax.set_ylabel("")
            ax.tick_params(labelsize=6)

    # Leyenda de municipios
    handles = [mpatches.Patch(color=PALETA_MUN[i], label=m)
               for i, m in enumerate(df_pp["Municipio"])]
    fig.legend(handles=handles, loc="lower center",
               ncol=4, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    plt.suptitle("Pairplot -- Indicadores TIC y Escolaridad\nCada punto = municipio",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("p4_05_pairplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_05_pairplot.png")


# ==============================================================
# 6. MAPA GEOGRAFICO (latitud/longitud)
# ==============================================================

def mapa_geografico(df_loc: pd.DataFrame, df_mun: pd.DataFrame):
    """
    Mapa de dispersion geografica de localidades.
    Tamano del punto = poblacion.
    Color = % acceso a internet (verde=alto, rojo=bajo).
    Superpone etiquetas de cabeceras municipales.
    """
    # Filtrar localidades con poblacion > 0
    df_map = df_loc[df_loc[COL_POB_T] > 0].copy()
    df_map["pct_int"] = (df_map[COL_INTERNET] / df_map[COL_VIV_CAR] * 100
                         if COL_VIV_CAR in df_map.columns
                         else np.nan)

    fig, ax = plt.subplots(figsize=(14, 10))

    # Scatter con tamano proporcional a poblacion y color por % internet
    tamanio = np.clip(np.sqrt(df_map[COL_POB_T].fillna(1)) * 0.4, 5, 300)
    sc = ax.scatter(
        df_map["lon"], df_map["lat"],
        c=df_map["pct_int"].fillna(df_map["pct_int"].median()),
        s=tamanio,
        cmap="RdYlGn",
        alpha=0.75,
        edgecolors="white",
        linewidth=0.3,
        vmin=0, vmax=100,
        zorder=3
    )

    # Etiquetas de las cabeceras municipales (poblacion mas grande por municipio)
    cabeceras = (df_loc.sort_values(COL_POB_T, ascending=False)
                       .drop_duplicates("Municipio o demarcación territorial")
                       .dropna(subset=["lat","lon"]))
    for _, row in cabeceras.iterrows():
        ax.annotate(
            row["Municipio o demarcación territorial"][:16],
            (row["lon"], row["lat"]),
            fontsize=7, fontweight="bold",
            xytext=(5, 5), textcoords="offset points",
            color="#2c3e50",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
                      alpha=0.7, edgecolor="gray", linewidth=0.5)
        )

    cbar = plt.colorbar(sc, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label("% Viviendas con Internet", fontsize=9)

    # Leyenda de tamanos
    for pob, lbl in [(100,"100 hab"),(1000,"1,000 hab"),(10000,"10,000 hab"),(100000,"100,000 hab")]:
        ax.scatter([], [], c="gray", alpha=0.5,
                   s=np.sqrt(pob) * 0.4, label=lbl)
    ax.legend(title="Poblacion", loc="lower left", fontsize=7, title_fontsize=8)

    ax.set_xlabel("Longitud", fontsize=10)
    ax.set_ylabel("Latitud", fontsize=10)
    ax.set_title("Mapa Geografico -- Localidades de Aguascalientes\n"
                 "Tamano = poblacion  |  Color = % acceso a Internet",
                 fontsize=12, fontweight="bold")
    ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.savefig("p4_06_mapa_geografico.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_06_mapa_geografico.png")


# ==============================================================
# 7. SCATTER POBLACION VS VIVIENDA
# ==============================================================

def scatter_pob_vivienda(df_mun: pd.DataFrame):
    """
    Scatter matrix de relaciones entre poblacion, viviendas
    y acceso a Internet por municipio.
    Panel 1: Poblacion total vs Viviendas habitadas
    Panel 2: Poblacion vs % Internet
    Panel 3: Viviendas vs % sin TIC
    Panel 4: Densidad viviendas vs Internet
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.flatten()

    pares = [
        (COL_POB_T,  COL_VIV_HAB, "Poblacion total", "Viviendas particulares habitadas",
         "Poblacion vs Viviendas"),
        (COL_POB_T,  "pct_internet", "Poblacion total", "% Viviendas con Internet",
         "Poblacion vs % Internet"),
        (COL_VIV_HAB,"pct_sin_tic",  "Viviendas habitadas", "% Sin TIC",
         "Viviendas vs % Exclusion Digital"),
        (COL_ESCOL,  "pct_internet", "Grado prom. escolaridad", "% Internet",
         "Escolaridad vs Acceso a Internet"),
    ]

    for idx, (x_col, y_col, x_lbl, y_lbl, titulo) in enumerate(pares):
        if x_col not in df_mun.columns or y_col not in df_mun.columns:
            axes[idx].set_visible(False)
            continue

        ax  = axes[idx]
        sub = df_mun[["Municipio", x_col, y_col]].dropna()

        for i, (_, row) in enumerate(sub.iterrows()):
            ax.scatter(row[x_col], row[y_col],
                       color=PALETA_MUN[i % len(PALETA_MUN)],
                       s=120, zorder=3, edgecolors="white", linewidth=0.8)
            ax.annotate(
                str(row["Municipio"])[:12],
                (row[x_col], row[y_col]),
                fontsize=7, xytext=(5, 4), textcoords="offset points"
            )

        # Linea de tendencia
        try:
            from scipy import stats as sp_stats
            x_ = sub[x_col].values.astype(float)
            y_ = sub[y_col].values.astype(float)
            slope, intercept, r, p_val, _ = sp_stats.linregress(x_, y_)
            x_line = np.linspace(x_.min(), x_.max(), 100)
            ax.plot(x_line, slope * x_line + intercept,
                    color="#e74c3c", linewidth=1.8, linestyle="--",
                    label=f"r={r:.3f}  p={p_val:.3f}")
            ax.legend(fontsize=8)
        except Exception:
            pass

        ax.set_xlabel(x_lbl, fontsize=9)
        ax.set_ylabel(y_lbl, fontsize=9)
        ax.set_title(titulo, fontsize=10, fontweight="bold")
        ax.tick_params(labelsize=8)

    plt.suptitle("Scatter: Poblacion, Vivienda y Acceso Digital\nAguascalientes -- Por Municipio",
                 fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig("p4_07_scatter_pob_vivienda.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_07_scatter_pob_vivienda.png")


# ==============================================================
# 8. MAPAS DE CALOR POR MUNICIPIO
# ==============================================================

def mapa_calor_municipio(df_mun: pd.DataFrame):
    """
    Dos heatmaps:
    A) Perfil TIC completo por municipio (ordenado por Internet)
    B) Contexto socioeconomico por municipio
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # ── A: Perfil TIC ──
    vars_tic = [v for v in TIC_VARS if v in df_mun.columns]
    df_tic = df_mun[["Municipio"] + vars_tic].set_index("Municipio")
    df_tic.columns = [TIC_LABELS.get(c, c) for c in vars_tic]
    df_tic = df_tic.sort_values("% Internet", ascending=False)

    sns.heatmap(
        df_tic, annot=True, fmt=".1f",
        cmap="RdYlGn", linewidths=0.5, linecolor="white",
        ax=axes[0], annot_kws={"size": 9},
        cbar_kws={"label": "% viviendas"}
    )
    axes[0].set_title("Perfil TIC por Municipio\nVerde=mayor acceso | Rojo=mayor exclusion",
                      fontsize=10, fontweight="bold")
    axes[0].set_xticklabels(axes[0].get_xticklabels(),
                             rotation=30, ha="right", fontsize=8)
    axes[0].set_yticklabels(axes[0].get_yticklabels(), rotation=0, fontsize=8)

    # ── B: Contexto socioeconomico ──
    vars_ctx = [v for v in [COL_ESCOL,"pct_analfabeta","pct_ocupada","pct_indigena"]
                if v in df_mun.columns]
    lbl_ctx  = {COL_ESCOL:"Escolaridad","pct_analfabeta":"% Analfab.",
                "pct_ocupada":"% Ocup.","pct_indigena":"% Indigena"}

    df_ctx = df_mun[["Municipio"] + vars_ctx].set_index("Municipio")
    df_ctx.columns = [lbl_ctx.get(c, c) for c in vars_ctx]
    df_ctx = df_ctx.loc[df_tic.index]  # mismo orden

    # Normalizar para comparar (z-score)
    df_ctx_norm = (df_ctx - df_ctx.mean()) / df_ctx.std()

    sns.heatmap(
        df_ctx_norm, annot=df_ctx.round(2), fmt="",
        cmap="coolwarm", linewidths=0.5, linecolor="white",
        ax=axes[1], annot_kws={"size": 9},
        cbar_kws={"label": "Z-score"}
    )
    axes[1].set_title("Contexto Socioeconomico (Z-score)\nRojo=alto | Azul=bajo",
                      fontsize=10, fontweight="bold")
    axes[1].set_xticklabels(axes[1].get_xticklabels(),
                             rotation=30, ha="right", fontsize=8)
    axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0, fontsize=8)

    plt.suptitle("Mapas de Calor por Municipio -- Aguascalientes",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("p4_08_mapas_calor_municipio.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_08_mapas_calor_municipio.png")


# ==============================================================
#  ── DISTRIBUCIONES DEMOGRAFICAS ──
# ==============================================================

# 9. PIRAMIDE POBLACIONAL
# ==============================================================

def piramide_poblacional(df_mun: pd.DataFrame, df_total: pd.DataFrame):
    """
    Piramide de poblacion por grupos quinquenales de edad.
    Barras horizontales: izquierda=hombres, derecha=mujeres.
    Muestra el total estatal y superpone el municipio capital.
    """
    # Usar el total estatal
    df_t = df_total.iloc[0] if len(df_total) > 0 else df_mun.iloc[0]

    fem  = [float(df_t[g]) if g in df_t.index and pd.notna(df_t[g]) else 0
            for g in GRUPOS_FEM]
    mas  = [float(df_t[g]) if g in df_t.index and pd.notna(df_t[g]) else 0
            for g in GRUPOS_MAS]
    pob_t = sum(fem) + sum(mas)

    # Convertir a porcentajes
    fem_pct = [v / pob_t * 100 for v in fem]
    mas_pct = [v / pob_t * 100 for v in mas]

    fig, ax = plt.subplots(figsize=(11, 9))
    y = np.arange(len(ETIQUETAS_EDAD))

    ax.barh(y,  mas_pct, height=0.8, color="#3498db", alpha=0.85, label="Hombres")
    ax.barh(y, [-f for f in fem_pct], height=0.8,
            color="#e91e63", alpha=0.85, label="Mujeres")

    # Linea central
    ax.axvline(0, color="black", linewidth=1.2)

    # Etiquetas de grupos
    ax.set_yticks(y)
    ax.set_yticklabels(ETIQUETAS_EDAD, fontsize=9)

    # Eje x con valores absolutos
    max_val = max(max(mas_pct), max(fem_pct)) * 1.1
    ax.set_xlim(-max_val, max_val)
    ticks = np.linspace(0, max_val * 0.9, 5)
    ax.set_xticks(np.concatenate([-ticks[::-1][:-1], ticks]))
    ax.set_xticklabels([f"{abs(v):.1f}%" for v in
                        np.concatenate([-ticks[::-1][:-1], ticks])], fontsize=8)

    ax.set_xlabel("% de la poblacion total", fontsize=10)
    ax.set_title("Piramide Poblacional -- Aguascalientes (Total Estatal)\n"
                 f"Poblacion total: {int(pob_t):,} habitantes",
                 fontsize=12, fontweight="bold")
    ax.legend(loc="lower right", fontsize=10)

    # Etiquetas de valores en las barras mas grandes
    for i, (f, m) in enumerate(zip(fem_pct, mas_pct)):
        if f > 1.5:
            ax.text(-f - 0.1, i, f"{f:.1f}%", va="center",
                    ha="right", fontsize=6.5, color="#c0392b")
        if m > 1.5:
            ax.text(m + 0.1, i, f"{m:.1f}%", va="center",
                    ha="left", fontsize=6.5, color="#2471a3")

    plt.tight_layout()
    plt.savefig("p4_09_piramide_poblacional.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_09_piramide_poblacional.png")


# ==============================================================
# 10. PROPORCION HOMBRES / MUJERES
# ==============================================================

def proporcion_hombres_mujeres(df_mun: pd.DataFrame):
    """
    Grafico de barras apiladas horizontales con la proporcion
    hombres/mujeres por municipio.
    Incluye el indice de masculinidad (hombres por cada 100 mujeres).
    """
    if COL_POB_FEM not in df_mun.columns or COL_POB_MAS not in df_mun.columns:
        return

    df_prop = df_mun[["Municipio", COL_POB_T, COL_POB_FEM, COL_POB_MAS]].dropna()
    df_prop["pct_fem"] = (df_prop[COL_POB_FEM] / df_prop[COL_POB_T] * 100).round(2)
    df_prop["pct_mas"] = (df_prop[COL_POB_MAS] / df_prop[COL_POB_T] * 100).round(2)
    df_prop["idx_masc"] = (df_prop[COL_POB_MAS] / df_prop[COL_POB_FEM] * 100).round(1)
    df_prop = df_prop.sort_values("pct_fem", ascending=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # ── Barras apiladas ──
    y  = np.arange(len(df_prop))
    axes[0].barh(y, df_prop["pct_fem"], height=0.7,
                 color="#e91e63", alpha=0.85, label="Mujeres")
    axes[0].barh(y, df_prop["pct_mas"], left=df_prop["pct_fem"],
                 height=0.7, color="#3498db", alpha=0.85, label="Hombres")

    axes[0].axvline(50, color="black", linestyle="--",
                    linewidth=1.2, alpha=0.7, label="50%")
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_prop["Municipio"], fontsize=9)
    axes[0].set_xlabel("% de la poblacion", fontsize=10)
    axes[0].set_title("Proporcion Hombres / Mujeres\npor Municipio",
                      fontsize=11, fontweight="bold")
    axes[0].legend(fontsize=9)

    # Etiquetas en barras
    for i, row in enumerate(df_prop.itertuples()):
        axes[0].text(row.pct_fem / 2, i, f"{row.pct_fem:.1f}%",
                     va="center", ha="center", fontsize=7.5,
                     color="white", fontweight="bold")
        axes[0].text(row.pct_fem + row.pct_mas / 2, i,
                     f"{row.pct_mas:.1f}%",
                     va="center", ha="center", fontsize=7.5,
                     color="white", fontweight="bold")

    # ── Indice de masculinidad ──
    colores_idx = ["#e74c3c" if v > 100 else "#3498db"
                   for v in df_prop["idx_masc"]]
    bars2 = axes[1].barh(y, df_prop["idx_masc"],
                         color=colores_idx, alpha=0.8,
                         height=0.7, edgecolor="white")
    axes[1].axvline(100, color="black", linestyle="--",
                    linewidth=1.5, label="Paridad (100)")
    axes[1].set_yticks(y)
    axes[1].set_yticklabels(df_prop["Municipio"], fontsize=9)
    axes[1].set_xlabel("Hombres por cada 100 mujeres", fontsize=10)
    axes[1].set_title("Indice de Masculinidad\n(>100 = mas hombres | <100 = mas mujeres)",
                      fontsize=11, fontweight="bold")
    axes[1].legend(fontsize=9)

    for bar, val in zip(bars2, df_prop["idx_masc"]):
        axes[1].text(bar.get_width() + 0.3,
                     bar.get_y() + bar.get_height() / 2,
                     f"{val:.1f}", va="center", fontsize=8.5)

    plt.suptitle("Distribucion por Sexo -- Aguascalientes",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig("p4_10_proporcion_hombres_mujeres.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_10_proporcion_hombres_mujeres.png")


# ==============================================================
# 11. DENSIDAD POBLACIONAL
# ==============================================================

def densidad_poblacional(df_loc: pd.DataFrame, df_mun: pd.DataFrame):
    """
    Panel 1: Mapa de densidad KDE (hexbin) de localidades
             con peso = poblacion. Muestra concentracion espacial.
    Panel 2: Barras de poblacion total por municipio.
    Panel 3: Scatter de tamano de localidades (distribucion de tamano).
    """
    fig = plt.figure(figsize=(18, 6))
    gs  = gridspec.GridSpec(1, 3, width_ratios=[2, 1, 1], wspace=0.35)
    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1])
    ax3 = fig.add_subplot(gs[2])

    # ── Panel 1: KDE espacial (hexbin) ──
    df_m = df_loc[df_loc[COL_POB_T] > 0].copy()
    hb = ax1.hexbin(
        df_m["lon"], df_m["lat"],
        C=df_m[COL_POB_T],
        reduce_C_function=np.sum,
        gridsize=30, cmap="YlOrRd",
        linewidths=0.2, mincnt=1
    )
    plt.colorbar(hb, ax=ax1, label="Poblacion acumulada")
    ax1.scatter(df_m["lon"], df_m["lat"],
                s=np.clip(np.sqrt(df_m[COL_POB_T]) * 0.3, 2, 100),
                c="white", alpha=0.3, zorder=2, linewidths=0)
    ax1.set_xlabel("Longitud", fontsize=9)
    ax1.set_ylabel("Latitud", fontsize=9)
    ax1.set_title("Densidad Poblacional\n(hexbin, peso=poblacion)",
                  fontsize=10, fontweight="bold")
    ax1.tick_params(labelsize=7)

    # ── Panel 2: Poblacion por municipio ──
    df_bar = df_mun[["Municipio", COL_POB_T]].dropna()
    df_bar = df_bar.sort_values(COL_POB_T, ascending=True)
    colores_bar = PALETA_MUN[:len(df_bar)]
    bars = ax2.barh(df_bar["Municipio"], df_bar[COL_POB_T] / 1000,
                    color=colores_bar, edgecolor="white", height=0.7)
    for bar, val in zip(bars, df_bar[COL_POB_T]):
        ax2.text(bar.get_width() + 2,
                 bar.get_y() + bar.get_height() / 2,
                 f"{val/1000:.0f}K", va="center", fontsize=7.5)
    ax2.set_xlabel("Poblacion (miles)", fontsize=9)
    ax2.set_title("Poblacion por\nMunicipio", fontsize=10, fontweight="bold")
    ax2.tick_params(labelsize=7)

    # ── Panel 3: Distribucion del tamano de localidades ──
    pob_loc = df_loc[COL_POB_T].dropna().values.astype(float)
    pob_loc = pob_loc[pob_loc > 0]

    ax3.hist(np.log10(pob_loc), bins=25,
             color="#9b59b6", edgecolor="white", alpha=0.85)
    ax3.set_xlabel("log10(Poblacion)", fontsize=9)
    ax3.set_ylabel("N localidades", fontsize=9)
    ax3.set_title("Distribucion de\nTamano de Localidades",
                  fontsize=10, fontweight="bold")

    # Marcas de referencia
    for exp, lbl in [(0,"1"), (1,"10"), (2,"100"),
                     (3,"1K"), (4,"10K"), (5,"100K")]:
        ax3.axvline(exp, color="gray", linestyle="--",
                    linewidth=0.8, alpha=0.6)
        ax3.text(exp + 0.05, ax3.get_ylim()[1] * 0.92,
                 lbl, fontsize=6.5, color="gray")
    ax3.tick_params(labelsize=7)

    plt.suptitle("Densidad y Distribucion Poblacional -- Aguascalientes",
                 fontsize=13, fontweight="bold", y=1.02)
    plt.savefig("p4_11_densidad_poblacional.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: p4_11_densidad_poblacional.png")


# ==============================================================
#  MAIN
# ==============================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  PARTE IV -- VISUALIZACION ESTADISTICA OBLIGATORIA")
    print("  Aguascalientes -- INEGI Censo 2020")
    print("=" * 65)

    df_mun, df_loc, df_total = cargar_datos("INEGI_datos.xlsx")

    # ── UNIVARIADOS ──────────────────────────────────────────
    print("\n[1/11] Histogramas con KDE...")
    histogramas_kde(df_mun)

    print("[2/11] Boxplots...")
    boxplots(df_mun)

    print("[3/11] Violin plots...")
    violin_plots(df_mun)

    # ── MULTIVARIADOS ────────────────────────────────────────
    print("[4/11] Matriz de correlacion...")
    matriz_correlacion(df_mun)

    print("[5/11] Pairplots...")
    pairplots(df_mun)

    print("[6/11] Mapa geografico...")
    mapa_geografico(df_loc, df_mun)

    print("[7/11] Scatter poblacion vs vivienda...")
    scatter_pob_vivienda(df_mun)

    print("[8/11] Mapas de calor por municipio...")
    mapa_calor_municipio(df_mun)

    # ── DISTRIBUCIONES DEMOGRAFICAS ──────────────────────────
    print("[9/11] Piramide poblacional...")
    piramide_poblacional(df_mun, df_total)

    print("[10/11] Proporcion hombres/mujeres...")
    proporcion_hombres_mujeres(df_mun)

    print("[11/11] Densidad poblacional...")
    densidad_poblacional(df_loc, df_mun)

    print("\n" + "=" * 65)
    print("  ARCHIVOS GENERADOS")
    print("=" * 65)
    archivos = [
        "p4_01_histogramas_kde.png       -- Univariado",
        "p4_02_boxplots.png              -- Univariado",
        "p4_03_violin_plots.png          -- Univariado",
        "p4_04_matriz_correlacion.png    -- Multivariado",
        "p4_05_pairplot.png              -- Multivariado",
        "p4_06_mapa_geografico.png       -- Multivariado",
        "p4_07_scatter_pob_vivienda.png  -- Multivariado",
        "p4_08_mapas_calor_municipio.png -- Multivariado",
        "p4_09_piramide_poblacional.png  -- Demografico",
        "p4_10_proporcion_hombres_mujeres.png -- Demografico",
        "p4_11_densidad_poblacional.png  -- Demografico",
    ]
    for a in archivos:
        print(f"  -> {a}")
    print("=" * 65)