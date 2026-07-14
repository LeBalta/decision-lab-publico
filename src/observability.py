from datetime import datetime
import json
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / "interacoes.jsonl"


def registrar_interacao(
    pergunta,
    territorio,
    resposta,
    agente="multiagente"
):
    registro = {
        "timestamp": datetime.now().isoformat(),
        "territorio": territorio,
        "agente": agente,
        "pergunta": pergunta,
        "resposta": resposta
    }

    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(registro, ensure_ascii=False))
        f.write("\n")