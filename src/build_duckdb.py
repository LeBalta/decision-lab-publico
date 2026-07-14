from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DB_PATH = MODELS_DIR / "laboratorio_politicas.duckdb"
ML_DATASET_PATH = MODELS_DIR / "dataset_empregabilidade.csv"

EMPREGABILIDADE_DIR = DATA_DIR / "Empregabilidade"
RENDA_DIR = DATA_DIR / "Renda"
EDUCACAO_DIR = DATA_DIR / "Educação"


def carregar_tabela_generica(caminho, categoria):
    df = pd.read_excel(caminho, header=None)

    registros = []

    for _, row in df.iterrows():
        territorio = row.iloc[0]

        if pd.isna(territorio):
            continue

        territorio = str(territorio).strip()

        if territorio == "" or territorio.lower().startswith("tabela"):
            continue

        for col_idx in range(1, len(row)):
            valor = row.iloc[col_idx]

            if pd.isna(valor):
                continue

            valor_num = pd.to_numeric(valor, errors="coerce")

            if pd.isna(valor_num):
                continue

            registros.append(
                {
                    "categoria": categoria,
                    "arquivo_origem": caminho.name,
                    "territorio": territorio,
                    "coluna_origem": str(col_idx),
                    "valor": float(valor_num),
                }
            )

    return pd.DataFrame(registros)


def carregar_pasta(pasta, categoria):
    tabelas = []

    for arquivo in pasta.glob("*.xls"):
        print(f"Lendo {categoria}: {arquivo.name}")
        try:
            df = carregar_tabela_generica(arquivo, categoria)
            if not df.empty:
                tabelas.append(df)
        except Exception as e:
            print(f"Erro ao ler {arquivo.name}: {e}")

    if tabelas:
        return pd.concat(tabelas, ignore_index=True)

    return pd.DataFrame(
        columns=["categoria", "arquivo_origem", "territorio", "coluna_origem", "valor"]
    )


def main():
    MODELS_DIR.mkdir(exist_ok=True)

    print("Construindo DuckDB...")

    conn = duckdb.connect(str(DB_PATH))

    df_emp = carregar_pasta(EMPREGABILIDADE_DIR, "Empregabilidade")
    df_renda = carregar_pasta(RENDA_DIR, "Renda")
    df_educacao = carregar_pasta(EDUCACAO_DIR, "Educação")

    df_indicadores = pd.concat(
        [df_emp, df_renda, df_educacao],
        ignore_index=True
    )

    conn.execute("DROP TABLE IF EXISTS indicadores_publicos")
    conn.register("df_indicadores", df_indicadores)
    conn.execute("""
        CREATE TABLE indicadores_publicos AS
        SELECT * FROM df_indicadores
    """)

    if ML_DATASET_PATH.exists():
        df_ml = pd.read_csv(ML_DATASET_PATH)

        conn.execute("DROP TABLE IF EXISTS dataset_empregabilidade_ml")
        conn.register("df_ml", df_ml)
        conn.execute("""
            CREATE TABLE dataset_empregabilidade_ml AS
            SELECT * FROM df_ml
        """)

    print("\nBanco criado com sucesso!")
    print(f"Arquivo: {DB_PATH}")

    print("\nTabelas disponíveis:")
    print(conn.execute("SHOW TABLES").fetchdf())

    print("\nAmostra de indicadores_publicos:")
    print(conn.execute("""
        SELECT *
        FROM indicadores_publicos
        LIMIT 10
    """).fetchdf())

    print("\nAmostra de dataset_empregabilidade_ml:")
    print(conn.execute("""
        SELECT *
        FROM dataset_empregabilidade_ml
        LIMIT 10
    """).fetchdf())

    conn.close()


if __name__ == "__main__":
    main()