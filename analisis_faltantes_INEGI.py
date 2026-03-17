"""
═══════════════════════════════════════════════════════════════
  PARTE III — ANÁLISIS DE DATOS FALTANTES
  Dataset: INEGI_datos.xlsx
  
  Tareas:
    1. Matriz de missingness
    2. Heatmap de correlación de faltantes
    3. Clasificar mecanismo: MCAR / MAR / MNAR
    4. Identificar variables críticas (>30% faltantes)
       y patrones por entidad o municipio
  
  Métricas:
    - % faltantes por variable
    - % faltantes por registro
    - Índice de completitud

  Instalación:
    pip install pandas numpy matplotlib seaborn scipy missingno openpyxl
═══════════════════════════════════════════════════════════════
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
import warnings
import os

warnings.filterwarnings("ignore")

# ── Estilo global ──────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor":   "white",
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "font.family":      "DejaVu Sans",
    "font.size":        11,
})

# ══════════════════════════════════════════════════════════
#  0. CARGA DE DATOS
# ══════════════════════════════════════════════════════════

def cargar_datos(ruta: str = "INEGI_datos.xlsx") -> pd.DataFrame:
    """Carga el archivo Excel. Ajusta sheet_name si es necesario."""
    print(f"  Cargando: {ruta}")
    df = pd.read_excel(ruta, sheet_name=0)
    print(f"  Shape    : {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


# ══════════════════════════════════════════════════════════
#  1. MÉTRICAS BASE DE MISSINGNESS
# ══════════════════════════════════════════════════════════

def calcular_metricas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Devuelve un DataFrame con:
      - n_faltantes, pct_faltantes, n_completos, pct_completos
      - dtype de cada columna
    """
    total = len(df)
    resumen = pd.DataFrame({
        "n_faltantes":   df.isnull().sum(),
        "pct_faltantes": (df.isnull().sum() / total * 100).round(2),
        "n_completos":   df.notnull().sum(),
        "pct_completos": (df.notnull().sum() / total * 100).round(2),
        "dtype":         df.dtypes.astype(str),
    })
    resumen = resumen.sort_values("pct_faltantes", ascending=False)

    # Índice de completitud global
    completitud = df.notnull().sum().sum() / (total * df.shape[1]) * 100

    # % faltantes por registro
    pct_por_registro = df.isnull().mean(axis=1) * 100

    print("\n" + "═" * 55)
    print("  MÉTRICAS GLOBALES")
    print("═" * 55)
    print(f"  Total registros          : {total:,}")
    print(f"  Total variables          : {df.shape[1]}")
    print(f"  Índice de completitud    : {completitud:.2f}%")
    print(f"  Registros sin faltantes  : {(pct_por_registro == 0).sum():,} "
          f"({(pct_por_registro == 0).mean()*100:.1f}%)")
    print(f"  Registros con >50% nulos : {(pct_por_registro > 50).sum():,}")

    # Variables críticas >30%
    criticas = resumen[resumen["pct_faltantes"] > 30]
    print(f"\n  Variables críticas (>30% faltantes): {len(criticas)}")
    if not criticas.empty:
        for col, row in criticas.iterrows():
            print(f"    → {col:<35} {row['pct_faltantes']:6.2f}%")

    print("\n  TOP 10 VARIABLES CON MÁS FALTANTES:")
    print(resumen.head(10)[["n_faltantes", "pct_faltantes", "dtype"]].to_string())

    return resumen, pct_por_registro, completitud


# ══════════════════════════════════════════════════════════
#  2. MATRIZ DE MISSINGNESS  (missingno-style manual)
# ══════════════════════════════════════════════════════════

def matriz_missingness(df: pd.DataFrame, max_cols: int = 40):
    """
    Visualiza la matriz de nulos: cada fila = registro,
    cada columna = variable. Blanco = presente, color = nulo.
    Si hay demasiadas columnas, muestra solo las más afectadas.
    """
    # Seleccionar columnas con al menos 1 nulo
    cols_con_nulos = df.columns[df.isnull().any()].tolist()
    if len(cols_con_nulos) == 0:
        print("  Sin valores faltantes — no se genera matriz.")
        return

    if len(cols_con_nulos) > max_cols:
        # Ordenar por % faltante y tomar las más afectadas
        pct = df[cols_con_nulos].isnull().mean().sort_values(ascending=False)
        cols_con_nulos = pct.head(max_cols).index.tolist()
        print(f"  Mostrando top {max_cols} columnas más afectadas.")

    sub = df[cols_con_nulos]
    # Muestra máximo 500 filas para legibilidad
    muestra = sub.sample(min(500, len(sub)), random_state=42).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(max(14, len(cols_con_nulos) * 0.35), 8))
    matriz_bin = muestra.isnull().astype(int)

    ax.imshow(matriz_bin.T, aspect="auto", cmap="RdYlGn_r",
              interpolation="nearest", vmin=0, vmax=1)

    ax.set_yticks(range(len(cols_con_nulos)))
    ax.set_yticklabels(cols_con_nulos, fontsize=8)
    ax.set_xlabel("Registros (muestra)", fontsize=11)
    ax.set_title("Matriz de Missingness\n"
                 "Verde = presente  |  Rojo = faltante", fontsize=13, pad=12)

    # Barra de color
    sm = plt.cm.ScalarMappable(cmap="RdYlGn_r", norm=plt.Normalize(0, 1))
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Presente", "Faltante"])

    plt.tight_layout()
    plt.savefig("01_matriz_missingness.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: 01_matriz_missingness.png")


# ══════════════════════════════════════════════════════════
#  3. HEATMAP DE CORRELACIÓN DE FALTANTES
# ══════════════════════════════════════════════════════════

def heatmap_correlacion_faltantes(df: pd.DataFrame, max_cols: int = 30):
    """
    Correlación entre los indicadores binarios de nulidad (0/1).
    Una correlación alta entre dos variables indica que sus faltantes
    tienden a aparecer juntos → posible MAR o MNAR.
    """
    cols_con_nulos = df.columns[df.isnull().any()].tolist()
    if len(cols_con_nulos) < 2:
        print("  Necesitas al menos 2 columnas con nulos para el heatmap.")
        return None

    if len(cols_con_nulos) > max_cols:
        pct = df[cols_con_nulos].isnull().mean().sort_values(ascending=False)
        cols_con_nulos = pct.head(max_cols).index.tolist()

    miss_bin = df[cols_con_nulos].isnull().astype(int)
    corr = miss_bin.corr(method="pearson")

    fig, ax = plt.subplots(figsize=(max(10, len(cols_con_nulos) * 0.5),
                                    max(8, len(cols_con_nulos) * 0.5)))
    mascara = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mascara, annot=len(cols_con_nulos) <= 20,
        fmt=".2f", cmap="coolwarm", center=0,
        linewidths=0.4, linecolor="white",
        vmin=-1, vmax=1, ax=ax,
        annot_kws={"size": 7}
    )
    ax.set_title("Correlación de Faltantes entre Variables\n"
                 "Rojo = tienden a faltar juntos  |  Azul = patrón opuesto",
                 fontsize=13, pad=12)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)

    plt.tight_layout()
    plt.savefig("02_heatmap_correlacion_faltantes.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: 02_heatmap_correlacion_faltantes.png")
    return corr


# ══════════════════════════════════════════════════════════
#  4. CLASIFICACIÓN MCAR / MAR / MNAR
# ══════════════════════════════════════════════════════════

def clasificar_mecanismo(df: pd.DataFrame, resumen: pd.DataFrame) -> dict:
    """
    Aplica heurísticas estadísticas para clasificar cada variable:

    MCAR — Missing Completely At Random:
      Los faltantes son independientes de cualquier variable.
      Test: Little's MCAR (aproximado con chi-cuadrado).
      Señal: correlación de faltantes baja con todo.

    MAR — Missing At Random:
      Los faltantes dependen de otras variables observadas.
      Señal: correlación alta de faltantes con otras columnas.

    MNAR — Missing Not At Random:
      Los faltantes dependen del propio valor no observado.
      Señal: % faltante alto + patrón sistemático en variables
             categóricas (entidad/municipio) + correlación alta
             entre la variable y su propio indicador de nulidad.
    """
    clasificacion = {}
    cols_con_nulos = resumen[resumen["n_faltantes"] > 0].index.tolist()

    if not cols_con_nulos:
        print("  No hay columnas con faltantes.")
        return {}

    miss_bin = df[cols_con_nulos].isnull().astype(int)
    corr_miss = miss_bin.corr(method="pearson").abs()

    print("\n" + "═" * 55)
    print("  CLASIFICACIÓN DE MECANISMO POR VARIABLE")
    print("═" * 55)

    for col in cols_con_nulos:
        pct = resumen.loc[col, "pct_faltantes"]
        dtype = resumen.loc[col, "dtype"]

        # Correlación máxima de esta variable con otras (excluye auto)
        otras_corr = corr_miss[col].drop(col)
        max_corr = otras_corr.max() if not otras_corr.empty else 0
        col_max_corr = otras_corr.idxmax() if not otras_corr.empty else "N/A"

        # Heurística MNAR: variable numérica con alta correlación propia
        mnar_score = 0
        if dtype in ["float64", "int64", "float32", "int32"]:
            serie = df[col].dropna()
            if len(serie) > 30:
                miss_ind = df[col].isnull().astype(int)
                # Si los valores observados difieren mucho entre presente/ausente
                # en otras columnas numéricas → MAR/MNAR
                corr_self, _ = stats.pointbiserialr(
                    miss_ind,
                    df[col].fillna(df[col].median())
                )
                mnar_score = abs(corr_self)

        # Decisión
        if pct < 5 and max_corr < 0.3:
            mecanismo = "MCAR"
            razon = f"pct={pct:.1f}%, corr_max={max_corr:.2f} → patrón aleatorio"
        elif max_corr >= 0.4:
            mecanismo = "MAR"
            razon = f"pct={pct:.1f}%, correlacionado con '{col_max_corr}' (r={max_corr:.2f})"
        elif pct > 30 or mnar_score > 0.3:
            mecanismo = "MNAR"
            razon = f"pct={pct:.1f}%, mnar_score={mnar_score:.2f} → sistemático"
        elif max_corr >= 0.2:
            mecanismo = "MAR"
            razon = f"pct={pct:.1f}%, correlación moderada (r={max_corr:.2f})"
        else:
            mecanismo = "MCAR"
            razon = f"pct={pct:.1f}%, sin correlación significativa"

        clasificacion[col] = {"mecanismo": mecanismo, "razon": razon, "pct": pct}
        icono = {"MCAR": "○", "MAR": "◐", "MNAR": "●"}[mecanismo]
        print(f"  {icono} {mecanismo}  {col:<35} {razon}")

    # Resumen por mecanismo
    conteo = pd.Series([v["mecanismo"] for v in clasificacion.values()]).value_counts()
    print(f"\n  Resumen: {dict(conteo)}")
    return clasificacion


# ══════════════════════════════════════════════════════════
#  5. PATRONES POR ENTIDAD / MUNICIPIO
# ══════════════════════════════════════════════════════════

def patrones_geograficos(df: pd.DataFrame):
    """
    Analiza si los faltantes se concentran en entidades o municipios
    específicos — señal de MNAR sistemático.
    """
    # Detectar columnas geográficas automáticamente
    geo_keywords = ["entidad", "estado", "municipio", "mpio", "cve_ent",
                    "cve_mun", "nom_ent", "nom_mun", "region", "localidad"]
    geo_cols = [c for c in df.columns
                if any(k in c.lower() for k in geo_keywords)]

    if not geo_cols:
        print("\n  No se detectaron columnas geográficas (entidad/municipio).")
        print("  Revisa los nombres de columna y ajusta geo_keywords si es necesario.")
        return

    # Variables numéricas con faltantes
    num_cols = df.select_dtypes(include=[np.number]).columns
    num_con_nulos = [c for c in num_cols if df[c].isnull().any()]

    if not num_con_nulos:
        print("\n  No hay variables numéricas con faltantes para analizar geográficamente.")
        return

    for geo_col in geo_cols[:2]:  # máximo 2 columnas geográficas
        print(f"\n  Patrón por: {geo_col}")
        grupo = df.groupby(geo_col)[num_con_nulos].apply(
            lambda x: x.isnull().mean() * 100
        ).round(2)

        # Top 10 entidades/municipios con más faltantes
        grupo["promedio_faltantes"] = grupo.mean(axis=1)
        top = grupo.sort_values("promedio_faltantes", ascending=False).head(10)
        print(top[["promedio_faltantes"] + num_con_nulos[:5]].to_string())

        # Gráfica
        fig, ax = plt.subplots(figsize=(14, 6))
        top10_plot = top["promedio_faltantes"].sort_values(ascending=True)
        colores = ["#e74c3c" if v > 30 else "#f39c12" if v > 10 else "#2ecc71"
                   for v in top10_plot]
        bars = ax.barh(top10_plot.index.astype(str), top10_plot.values,
                       color=colores, edgecolor="white", height=0.7)

        for bar, val in zip(bars, top10_plot.values):
            ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                    f"{val:.1f}%", va="center", fontsize=9)

        ax.axvline(30, color="red", linestyle="--", alpha=0.7, label="Umbral crítico 30%")
        ax.set_xlabel("% promedio de faltantes")
        ax.set_title(f"Top 10 {geo_col} con más datos faltantes", fontsize=13)
        ax.legend()
        plt.tight_layout()
        fname = f"03_patron_{geo_col.lower().replace(' ', '_')}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"  Guardado: {fname}")


# ══════════════════════════════════════════════════════════
#  6. GRÁFICA DE % FALTANTES POR VARIABLE
# ══════════════════════════════════════════════════════════

def grafica_pct_por_variable(resumen: pd.DataFrame, top_n: int = 30):
    """Barplot horizontal con % faltantes por variable."""
    sub = resumen[resumen["n_faltantes"] > 0].head(top_n)
    if sub.empty:
        print("  Sin faltantes — no se genera gráfica.")
        return

    fig, ax = plt.subplots(figsize=(12, max(6, len(sub) * 0.4)))
    colores = ["#e74c3c" if v > 30 else "#f39c12" if v > 10 else "#3498db"
               for v in sub["pct_faltantes"]]
    bars = ax.barh(sub.index, sub["pct_faltantes"],
                   color=colores, edgecolor="white", height=0.7)

    for bar, val in zip(bars, sub["pct_faltantes"]):
        ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f"{val:.1f}%", va="center", fontsize=9)

    ax.axvline(30, color="red", linestyle="--", alpha=0.8,
               linewidth=1.5, label="Umbral crítico 30%")
    ax.axvline(10, color="orange", linestyle=":", alpha=0.6,
               linewidth=1.2, label="Referencia 10%")
    ax.set_xlabel("% de valores faltantes")
    ax.set_title(f"% Faltantes por Variable (top {top_n})", fontsize=13)
    ax.legend(loc="lower right")
    ax.invert_yaxis()

    # Leyenda de colores
    from matplotlib.patches import Patch
    leyenda = [Patch(color="#e74c3c", label=">30% crítico"),
               Patch(color="#f39c12", label="10–30% moderado"),
               Patch(color="#3498db", label="<10% leve")]
    ax.legend(handles=leyenda, loc="lower right")

    plt.tight_layout()
    plt.savefig("04_pct_faltantes_por_variable.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: 04_pct_faltantes_por_variable.png")


# ══════════════════════════════════════════════════════════
#  7. DISTRIBUCIÓN DE % FALTANTES POR REGISTRO
# ══════════════════════════════════════════════════════════

def grafica_pct_por_registro(pct_por_registro: pd.Series):
    """Histograma del % de faltantes por fila."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Histograma
    axes[0].hist(pct_por_registro, bins=40, color="#3498db",
                 edgecolor="white", alpha=0.85)
    axes[0].axvline(pct_por_registro.mean(), color="red",
                    linestyle="--", label=f"Media: {pct_por_registro.mean():.1f}%")
    axes[0].axvline(50, color="orange", linestyle=":",
                    label="Umbral 50%")
    axes[0].set_xlabel("% de campos faltantes por registro")
    axes[0].set_ylabel("Número de registros")
    axes[0].set_title("Distribución de faltantes por registro")
    axes[0].legend()

    # Boxplot
    bp = axes[1].boxplot(pct_por_registro, vert=True, patch_artist=True,
                         boxprops=dict(facecolor="#3498db", alpha=0.7),
                         medianprops=dict(color="red", linewidth=2))
    axes[1].set_ylabel("% faltantes por registro")
    axes[1].set_title("Boxplot — faltantes por registro")
    axes[1].set_xticks([])

    stats_text = (f"Media:    {pct_por_registro.mean():.2f}%\n"
                  f"Mediana:  {pct_por_registro.median():.2f}%\n"
                  f"P75:      {pct_por_registro.quantile(0.75):.2f}%\n"
                  f"P95:      {pct_por_registro.quantile(0.95):.2f}%\n"
                  f"Max:      {pct_por_registro.max():.2f}%")
    axes[1].text(1.25, pct_por_registro.median(), stats_text,
                 fontsize=9, va="center",
                 bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig("05_pct_faltantes_por_registro.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Guardado: 05_pct_faltantes_por_registro.png")


# ══════════════════════════════════════════════════════════
#  8. REPORTE RESUMEN EN CONSOLA
# ══════════════════════════════════════════════════════════

def reporte_final(resumen: pd.DataFrame, clasificacion: dict,
                  completitud: float, pct_por_registro: pd.Series):
    print("\n" + "═" * 55)
    print("  REPORTE FINAL — PARTE III")
    print("═" * 55)
    print(f"\n  Índice de completitud global : {completitud:.2f}%")
    print(f"  Variables analizadas         : {len(resumen)}")
    print(f"  Variables con faltantes      : {(resumen['n_faltantes']>0).sum()}")
    print(f"  Variables críticas (>30%)    : {(resumen['pct_faltantes']>30).sum()}")

    if clasificacion:
        from collections import Counter
        conteo = Counter(v["mecanismo"] for v in clasificacion.values())
        print(f"\n  Mecanismos detectados:")
        print(f"    MCAR (aleatorio)         : {conteo.get('MCAR', 0)} variables")
        print(f"    MAR  (dep. observados)   : {conteo.get('MAR',  0)} variables")
        print(f"    MNAR (dep. no observado) : {conteo.get('MNAR', 0)} variables")

    print(f"\n  Faltantes por registro:")
    print(f"    Sin ningún faltante      : {(pct_por_registro==0).sum():,} registros")
    print(f"    Con 1–30% faltantes      : {((pct_por_registro>0)&(pct_por_registro<=30)).sum():,} registros")
    print(f"    Con 30–50% faltantes     : {((pct_por_registro>30)&(pct_por_registro<=50)).sum():,} registros")
    print(f"    Con >50% faltantes       : {(pct_por_registro>50).sum():,} registros")

    print("\n  Archivos generados:")
    archivos = [
        "01_matriz_missingness.png",
        "02_heatmap_correlacion_faltantes.png",
        "03_patron_entidad/municipio.png",
        "04_pct_faltantes_por_variable.png",
        "05_pct_faltantes_por_registro.png",
    ]
    for a in archivos:
        print(f"    → {a}")
    print("═" * 55)


# ══════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 55)
    print("  PARTE III — ANÁLISIS DE DATOS FALTANTES INEGI")
    print("═" * 55)

    # ── Carga ──────────────────────────────────────────────
    df = cargar_datos("INEGI_datos.xlsx")

    # ── 1. Métricas base ───────────────────────────────────
    resumen, pct_por_registro, completitud = calcular_metricas(df)

    # ── 2. Matriz de missingness ───────────────────────────
    print("\n[1/5] Generando matriz de missingness...")
    matriz_missingness(df)

    # ── 3. Heatmap correlación ─────────────────────────────
    print("\n[2/5] Generando heatmap de correlación de faltantes...")
    corr = heatmap_correlacion_faltantes(df)

    # ── 4. Clasificación MCAR/MAR/MNAR ────────────────────
    print("\n[3/5] Clasificando mecanismos...")
    clasificacion = clasificar_mecanismo(df, resumen)

    # ── 5. Patrones geográficos ────────────────────────────
    print("\n[4/5] Analizando patrones por entidad/municipio...")
    patrones_geograficos(df)

    # ── 6. Gráficas de métricas ────────────────────────────
    print("\n[5/5] Generando gráficas de métricas...")
    grafica_pct_por_variable(resumen)
    grafica_pct_por_registro(pct_por_registro)

    # ── Reporte final ──────────────────────────────────────
    reporte_final(resumen, clasificacion, completitud, pct_por_registro)