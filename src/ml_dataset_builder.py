from pathlib import Path
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = BASE_DIR / "data"
EMPREGABILIDADE_DIR = DATA_DIR / "Empregabilidade"
RENDA_DIR = DATA_DIR / "Renda"
EDUCACAO_DIR = DATA_DIR / "Educação"

OUTPUT_DIR = BASE_DIR / "models"
OUTPUT_DIR.mkdir(exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "dataset_empregabilidade.csv"


def limpar_nome_territorio(valor):
    if pd.isna(valor):
        return None
    return str(valor).strip()


def carregar_indicador(caminho, nome_coluna, linha_inicio, coluna_territorio=0, coluna_valor=1):
    df = pd.read_excel(caminho, header=None)

    df = df.iloc[linha_inicio:, [coluna_territorio, coluna_valor]].copy()
    df.columns = ["territorio", nome_coluna]

    df["territorio"] = df["territorio"].apply(limpar_nome_territorio)
    df[nome_coluna] = pd.to_numeric(df[nome_coluna], errors="coerce")

    df = df.dropna(subset=["territorio", nome_coluna])
    df = df[df["territorio"] != ""]

    return df


def main():
    arquivos = {
        "desocupacao": EMPREGABILIDADE_DIR / "Tabela 1.25 (Desoc_Geo).xls",
        "gini": RENDA_DIR / "Tabela 2.13 (Indice_Gini_Geo).xls",
        "pobreza": RENDA_DIR / "Tabela 2.18 (Pobr_Geo).xls",
        "analfabetismo": EDUCACAO_DIR / "Tabela 4.14 (TaxaAnalf_Geo).xls",
    }

    desocupacao = carregar_indicador(
        arquivos["desocupacao"],
        "taxa_desocupacao",
        linha_inicio=5,
        coluna_territorio=0,
        coluna_valor=1,
    )

    gini = carregar_indicador(
        arquivos["gini"],
        "indice_gini",
        linha_inicio=7,
        coluna_territorio=0,
        coluna_valor=1,
    )

    pobreza = carregar_indicador(
        arquivos["pobreza"],
        "taxa_pobreza",
        linha_inicio=6,
        coluna_territorio=0,
        coluna_valor=4,
    )

    analfabetismo = carregar_indicador(
        arquivos["analfabetismo"],
        "taxa_analfabetismo",
        linha_inicio=8,
        coluna_territorio=0,
        coluna_valor=1,
    )

    dataset = (
        desocupacao
        .merge(gini, on="territorio", how="inner")
        .merge(pobreza, on="territorio", how="inner")
        .merge(analfabetismo, on="territorio", how="inner")
    )

    features = [
        "taxa_desocupacao",
        "indice_gini",
        "taxa_pobreza",
        "taxa_analfabetismo",
    ]

    scaler = MinMaxScaler()
    valores_normalizados = scaler.fit_transform(dataset[features])

    dataset["indice_vulnerabilidade_laboral"] = valores_normalizados.mean(axis=1)

    mediana = dataset["indice_vulnerabilidade_laboral"].median()

    dataset["risco_baixa_empregabilidade"] = (
        dataset["indice_vulnerabilidade_laboral"] >= mediana
    ).astype(int)

    dataset = dataset.sort_values(
        "indice_vulnerabilidade_laboral",
        ascending=False
    )

    dataset.to_csv(OUTPUT_FILE, index=False, encoding="utf-8-sig")

    print("Dataset gerado com sucesso!")
    print(f"Arquivo salvo em: {OUTPUT_FILE}")
    print(f"Total de registros: {len(dataset)}")
    print("\nAmostra:")
    print(dataset.head(15))


if __name__ == "__main__":
    main()