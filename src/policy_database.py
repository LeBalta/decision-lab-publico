from pathlib import Path
import duckdb
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DB_PATH = BASE_DIR / "models" / "laboratorio_politicas.duckdb"


def get_connection():
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"Banco DuckDB não encontrado em: {DB_PATH}. "
            "Execute primeiro: python src/build_duckdb.py"
        )

    return duckdb.connect(str(DB_PATH), read_only=True)


def listar_tabelas():
    with get_connection() as conn:
        return conn.execute("SHOW TABLES").fetchdf()


def buscar_indicadores_por_territorio(territorio: str) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT
                categoria,
                arquivo_origem,
                territorio,
                coluna_origem,
                valor
            FROM indicadores_publicos
            WHERE lower(territorio) LIKE lower(?)
            ORDER BY categoria, arquivo_origem, coluna_origem
        """

        return conn.execute(query, [f"%{territorio}%"]).fetchdf()


def buscar_dataset_ml_por_territorio(territorio: str) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT *
            FROM dataset_empregabilidade_ml
            WHERE lower(territorio) LIKE lower(?)
            ORDER BY indice_vulnerabilidade_laboral DESC
        """

        return conn.execute(query, [f"%{territorio}%"]).fetchdf()


def ranking_empregabilidade(limite: int = 10) -> pd.DataFrame:
    with get_connection() as conn:
        query = """
            SELECT
                territorio,
                taxa_desocupacao,
                indice_gini,
                taxa_pobreza,
                taxa_analfabetismo,
                indice_vulnerabilidade_laboral,
                risco_baixa_empregabilidade
            FROM dataset_empregabilidade_ml
            ORDER BY indice_vulnerabilidade_laboral DESC
            LIMIT ?
        """

        return conn.execute(query, [limite]).fetchdf()


def resumo_territorio(territorio: str) -> str:
    df = buscar_dataset_ml_por_territorio(territorio)

    if df.empty:
        return f"Nenhum dado consolidado de empregabilidade encontrado para '{territorio}'."

    linha = df.iloc[0]

    risco = (
        "alto risco de baixa empregabilidade"
        if int(linha["risco_baixa_empregabilidade"]) == 1
        else "risco controlado de baixa empregabilidade"
    )

    return (
        f"Território: {linha['territorio']}\n"
        f"Classificação: {risco}\n"
        f"Índice de Vulnerabilidade Laboral: "
        f"{linha['indice_vulnerabilidade_laboral']:.3f}\n"
        f"Taxa de desocupação: {linha['taxa_desocupacao']:.2f}\n"
        f"Índice de Gini: {linha['indice_gini']:.3f}\n"
        f"Taxa de pobreza: {linha['taxa_pobreza']:.2f}\n"
        f"Taxa de analfabetismo: {linha['taxa_analfabetismo']:.2f}"
    )


if __name__ == "__main__":
    print("Tabelas disponíveis:")
    print(listar_tabelas())

    print("\nRanking de risco de baixa empregabilidade:")
    print(ranking_empregabilidade(10))

    print("\nResumo de Pernambuco:")
    print(resumo_territorio("Pernambuco"))