from pathlib import Path
import joblib
import pandas as pd

try:
    from .policy_database import buscar_dataset_ml_por_territorio
except ImportError:  # execução direta pelo Streamlit
    from policy_database import buscar_dataset_ml_por_territorio

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "modelo_empregabilidade.pkl"


def carregar_modelo():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Modelo não encontrado em: {MODEL_PATH}. "
            "Execute primeiro: python src/train_ml_model.py"
        )

    pacote = joblib.load(MODEL_PATH)
    return pacote["model"], pacote["features"]


def prever_por_indicadores(
    taxa_desocupacao: float,
    indice_gini: float,
    taxa_pobreza: float,
    taxa_analfabetismo: float,
) -> dict:
    model, features = carregar_modelo()

    entrada = pd.DataFrame(
        [
            {
                "taxa_desocupacao": taxa_desocupacao,
                "indice_gini": indice_gini,
                "taxa_pobreza": taxa_pobreza,
                "taxa_analfabetismo": taxa_analfabetismo,
            }
        ]
    )

    entrada = entrada[features]

    classe = int(model.predict(entrada)[0])
    probabilidade = float(model.predict_proba(entrada)[0][1])

    return {
        "classe": classe,
        "probabilidade_alto_risco": probabilidade,
        "interpretacao": (
            "alto risco de baixa empregabilidade"
            if classe == 1
            else "risco controlado de baixa empregabilidade"
        ),
    }


def prever_por_territorio(territorio: str) -> dict:
    df = buscar_dataset_ml_por_territorio(territorio)

    if df.empty:
        return {
            "erro": f"Nenhum dado encontrado para o território '{territorio}'."
        }

    linha = df.iloc[0]

    resultado = prever_por_indicadores(
        taxa_desocupacao=float(linha["taxa_desocupacao"]),
        indice_gini=float(linha["indice_gini"]),
        taxa_pobreza=float(linha["taxa_pobreza"]),
        taxa_analfabetismo=float(linha["taxa_analfabetismo"]),
    )

    resultado.update(
        {
            "territorio": linha["territorio"],
            "taxa_desocupacao": float(linha["taxa_desocupacao"]),
            "indice_gini": float(linha["indice_gini"]),
            "taxa_pobreza": float(linha["taxa_pobreza"]),
            "taxa_analfabetismo": float(linha["taxa_analfabetismo"]),
            "indice_vulnerabilidade_laboral": float(
                linha["indice_vulnerabilidade_laboral"]
            ),
        }
    )

    return resultado


def formatar_previsao(resultado: dict) -> str:
    if "erro" in resultado:
        return resultado["erro"]

    return (
        f"Território analisado: {resultado['territorio']}\n"
        f"Classificação do modelo: {resultado['interpretacao']}\n"
        f"Probabilidade estimada de alto risco: "
        f"{resultado['probabilidade_alto_risco']:.2%}\n\n"
        f"Indicadores utilizados:\n"
        f"- Taxa de desocupação: {resultado['taxa_desocupacao']:.2f}\n"
        f"- Índice de Gini: {resultado['indice_gini']:.3f}\n"
        f"- Taxa de pobreza: {resultado['taxa_pobreza']:.2f}\n"
        f"- Taxa de analfabetismo: {resultado['taxa_analfabetismo']:.2f}\n"
        f"- Índice de Vulnerabilidade Laboral: "
        f"{resultado['indice_vulnerabilidade_laboral']:.3f}"
    )


if __name__ == "__main__":
    resultado = prever_por_territorio("Pernambuco")
    print(formatar_previsao(resultado))