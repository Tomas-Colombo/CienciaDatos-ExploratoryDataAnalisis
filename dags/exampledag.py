from airflow import DAG
from airflow.decorators import dag, task
from airflow.exceptions import AirflowFailException
from datetime import datetime
from pathlib import Path
import pandas as pd
import numpy as np

# ================== Config & Helpers ==================
REF_DATE = pd.Timestamp("2025-07-16")  # fecha fija para calcular edad

def money_to_eur(s):
    """Convierte '€12.5M' | '€100K' | '€0' | None -> float EUR."""
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    txt = str(s).strip().replace("€", "").replace(",", "").upper()
    if txt == "" or txt == "NAN":
        return np.nan
    mult = 1.0
    if txt.endswith("M"):
        mult = 1_000_000.0
        txt = txt[:-1]
    elif txt.endswith("K"):
        mult = 1_000.0
        txt = txt[:-1]
    try:
        return float(txt) * mult
    except ValueError:
        return np.nan

def simplify_position(pos):
    """Agrupa posiciones específicas a categorías simplificadas."""
    if not isinstance(pos, str) or pos.strip() == "":
        return np.nan
    p = pos.upper().strip()
    if p == "GK":
        return "Arquero"
    if p == "CB":
        return "Defensor central"
    if p in {"RB", "LB", "RWB", "LWB"}:
        return "Lateral"
    if p == "CDM":
        return "Volante defensivo"
    if p in {"CM", "CAM", "RM", "LM"}:
        return "Volante"
    if p in {"ST", "CF"}:
        return "Delantero"
    if p in {"RW", "LW"}:
        return "Extremo"
    return "Otra"

# ================== Funciones del pipeline ==================
def inspeccionar(df: pd.DataFrame) -> pd.DataFrame:
    print("=== Inspección inicial ===")
    print(f"Filas x columnas: {df.shape}")
    print("\nTipos de datos:")
    print(df.dtypes.sort_index())
    nulos = df.isna().sum()
    cols_con_nulos = nulos[nulos > 0].sort_values(ascending=False)
    if len(cols_con_nulos) > 0:
        print("\nColumnas con nulos (top):")
        print(cols_con_nulos.head(40))
    else:
        print("\nNo se detectaron nulos.")
    return df

def calcular_edad(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()                                                              #copia para no modificar el original 
    if "dob" not in df.columns:
        raise KeyError("No se encontró la columna 'dob' (date of birth).")
    df["dob"] = pd.to_datetime(df["dob"], errors="coerce", dayfirst=True)
    df["age"] = ((REF_DATE - df["dob"]).dt.days / 365.25).astype(int)
    return df

def eliminar_columnas_innecesarias(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    cols_drop = [
        "image", "club_logo", "country_flag", "description",
        "specialities", "play_styles",
        "club_kit_number", "country_kit_number",
        "club_league_id", "country_league_id",
        "player_id", "version", "full_name", "body_type", 
        "real_face", "club_id", "club_id", "weak_foot", "skill_moves", 
        "international_reputation", "work_rate", "club_contract_valid_until", 
    ]
    existentes = [c for c in cols_drop if c in df.columns]
    return df.drop(columns=existentes, errors="ignore")

def tratar_nulos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    required = [c for c in ["club_name", "club_league_name", "positions"] if c in df.columns]
    if required:
        df = df.dropna(subset=required)
    if "release_clause" in df.columns:
        df["release_clause"] = df["release_clause"].fillna(0)
    umbral = 0.6                                                            #se establece que una columna es mala si el 60 porciento de ellases Nan
    frac_nan = df.isna().mean()
    cols_malas = frac_nan[frac_nan > umbral].index.tolist()
    if cols_malas:
        df = df.drop(columns=cols_malas)
    return df

def convertir_salarios_y_valores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "value" in df.columns:
        df["value_eur"] = df["value"].apply(money_to_eur)
        df = df.drop(columns=["value"])                             #<----- no quiero la original pq no me sirve para calcular nada 
    if "wage" in df.columns:
        df["wage_eur"] = df["wage"].apply(money_to_eur)
        df = df.drop(columns=["wage"])
    if "release_clause" in df.columns:
        df["release_clause_eur"] = df["release_clause"].apply(money_to_eur)
        df = df.drop(columns=["release_clause"])
    return df

def convertir_tipos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in ["value_eur", "wage_eur", "release_clause_eur", "age", "height_cm", "weight_kg",
              "overall_rating", "potential"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df



def agregar_columnas_utiles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "age" not in df.columns:
        raise KeyError("Falta 'age'. Ejecutar calcular_edad antes.")
    
    bins = [-0.1, 21, 27, 33, 50]
    labels = ["joven", "maduro", "experimentado", "veterano"]
    df["grupo_edad"] = pd.cut(df["age"], bins=bins, labels=labels, right=False, include_lowest=True)
    if "potential" in df.columns:
        df["es_promesa"] = np.where((df["age"] <= 21) & (pd.to_numeric(df["potential"], errors="coerce") >= 80), 1, 0)
    else:
        df["es_promesa"] = 0
    return df

def procesar_posiciones(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "positions" in df.columns:
        df["posicion_principal"] = (
            df["positions"].astype(str).str.split(",").apply(
                lambda xs: xs[0].strip().upper() if isinstance(xs, list) and len(xs) > 0 else np.nan
            )
        )
    elif "club_position" in df.columns:
        df["posicion_principal"] = df["club_position"].astype(str).str.upper()
    else:
        df["posicion_principal"] = np.nan
    df["posicion_simplificada"] = df["posicion_principal"].apply(simplify_position)
    return df

def verificar_final(df: pd.DataFrame) -> pd.DataFrame:
    print("\n=== Verificación final ===")
    print(f"Forma final: {df.shape}")
    nulos = df.isna().sum()
    pendientes = nulos[nulos > 0].sort_values(ascending=False)
    if len(pendientes) > 0:
        print("\nColumnas con nulos aún presentes:")
        print(pendientes.head(40))
    else:
        print("\nNo quedan nulos.")
    return df

def ejecutar_pipeline(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = (
        df_raw
        .pipe(inspeccionar)
        .pipe(calcular_edad)
        .pipe(eliminar_columnas_innecesarias)
        .pipe(tratar_nulos)
        .pipe(convertir_salarios_y_valores)
        .pipe(convertir_tipos)
        .pipe(agregar_columnas_utiles)
        .pipe(procesar_posiciones)
        .pipe(verificar_final)
    )
    return df

# ================== DAG ==================
DATA_DIR = Path("/usr/local/airflow/include/data")
INPUT_CSV = DATA_DIR / "players_raw.csv"      # <- ajustá el nombre si es distinto
OUTPUT_CSV = DATA_DIR / "players_clean.csv"

@dag(
    dag_id="players_eda_pipeline",
    description="ETL de jugadores: limpieza y features para EDA",
    start_date=datetime(2024, 1, 1),
    schedule=None,           # ejecutalo manualmente desde la UI por ahora
    catchup=False,
    tags=["eda", "players"],
)
def pipeline():
    @task
    def check_input_exists() -> str:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not INPUT_CSV.exists():
            raise AirflowFailException(
                f"No existe el archivo de entrada: {INPUT_CSV}. "
                f"Copialo dentro del contenedor o mapeá el volumen en Astro."
            )
        return str(INPUT_CSV)

    @task
    def extract(input_path: str) -> str:
        # Podés ajustar parámetros de lectura acá (sep, encoding, etc.)
        df_raw = pd.read_csv(input_path)
        # Guardamos un snapshot opcional
        snapshot = DATA_DIR / "players_raw_snapshot.csv"
        df_raw.to_csv(snapshot, index=False)
        return str(snapshot)
    @task
    def transform(snapshot_path: str) -> str:
        df_raw = pd.read_csv(snapshot_path, low_memory=False)
        df_clean = ejecutar_pipeline(df_raw)
        df_clean.to_csv(OUTPUT_CSV, index=False)
        return str(OUTPUT_CSV)

    @task
    def report(output_path: str) -> None:
        df = pd.read_csv(output_path, nrows=5)
        print("Muestra de salida (5 filas):")
        print(df)

    # Orquestación
    path = check_input_exists()
    snap = extract(path)
    out = transform(snap)
    report(out)

dag = pipeline()
