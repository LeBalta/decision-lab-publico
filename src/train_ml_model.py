from pathlib import Path
import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)

BASE_DIR = Path(__file__).resolve().parent.parent

DATASET_PATH = BASE_DIR / "models" / "dataset_empregabilidade.csv"
MODEL_PATH = BASE_DIR / "models" / "modelo_empregabilidade.pkl"
METRICS_PATH = BASE_DIR / "models" / "metricas_modelo_empregabilidade.txt"

FEATURES = [
    "taxa_desocupacao",
    "indice_gini",
    "taxa_pobreza",
    "taxa_analfabetismo",
]

TARGET = "risco_baixa_empregabilidade"


def main():
    df = pd.read_csv(DATASET_PATH)

    X = df[FEATURES]
    y = df[TARGET]

    model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced",
    )

    y_pred = cross_val_predict(model, X, y, cv=5)
    y_proba = cross_val_predict(model, X, y, cv=5, method="predict_proba")[:, 1]

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y, y_proba)

    model.fit(X, y)

    joblib.dump(
        {
            "model": model,
            "features": FEATURES,
        },
        MODEL_PATH,
    )

    relatorio = f"""
Modelo de Classificação de Risco de Baixa Empregabilidade Territorial

Total de registros: {len(df)}
Features utilizadas: {FEATURES}
Variável-alvo: {TARGET}

Métricas com validação cruzada 5-fold:

Accuracy: {accuracy:.4f}
Precision: {precision:.4f}
Recall: {recall:.4f}
F1-score: {f1:.4f}
ROC-AUC: {roc_auc:.4f}

Relatório de classificação:

{classification_report(y, y_pred, zero_division=0)}
"""

    METRICS_PATH.write_text(relatorio, encoding="utf-8")

    print("Modelo treinado com sucesso!")
    print(f"Modelo salvo em: {MODEL_PATH}")
    print(f"Métricas salvas em: {METRICS_PATH}")
    print(relatorio)


if __name__ == "__main__":
    main()