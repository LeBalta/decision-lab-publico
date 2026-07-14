"""Guardrails locais para entrada/saída e anonimização de PII.
Implementação equivalente e auditável, sem depender de serviços externos.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, asdict
from typing import Any

@dataclass(frozen=True)
class SecurityResult:
    allowed: bool
    sanitized_text: str
    reasons: tuple[str, ...]
    pii_found: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

INJECTION_PATTERNS = [
    r"ignore\s+(all\s+)?(previous|prior)\s+instructions",
    r"ignore\s+(todas\s+)?(as\s+)?instru[cç][oõ]es",
    r"revele?\s+(o\s+)?(system prompt|prompt do sistema)",
    r"developer\s+message|mensagem\s+do\s+desenvolvedor",
    r"jailbreak|dan\s+mode|modo\s+dan",
    r"execute\s+(este\s+)?comando|rode\s+(este\s+)?comando",
    r"rm\s+-rf|powershell\s+-enc|cmd\.exe",
]
PROHIBITED_PATTERNS = [
    r"como\s+(invadir|hackear|fraudar)",
    r"roubar\s+(senha|dados|credenciais)",
    r"burlar\s+(a\s+)?seguran[cç]a",
]
PII_PATTERNS = {
    "cpf": re.compile(r"(?<!\d)\d{3}\.?\d{3}\.?\d{3}-?\d{2}(?!\d)"),
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "telefone": re.compile(r"(?<!\d)(?:\+?55\s*)?(?:\(?\d{2}\)?\s*)?9?\d{4}[-\s]?\d{4}(?!\d)"),
}


def anonymize_pii(text: str) -> tuple[str, tuple[str, ...]]:
    sanitized = text
    found: list[str] = []
    for name, pattern in PII_PATTERNS.items():
        if pattern.search(sanitized):
            found.append(name)
            sanitized = pattern.sub(f"[PII_{name.upper()}]", sanitized)
    return sanitized, tuple(found)


def scan_input(text: str) -> SecurityResult:
    raw = (text or "").strip()
    sanitized, pii = anonymize_pii(raw)
    reasons: list[str] = []
    lowered = raw.casefold()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            reasons.append("tentativa de prompt injection/jailbreak")
            break
    for pattern in PROHIBITED_PATTERNS:
        if re.search(pattern, lowered, flags=re.IGNORECASE):
            reasons.append("tópico operacional de segurança proibido")
            break
    if len(raw) > 5000:
        reasons.append("entrada acima do limite de 5.000 caracteres")
    return SecurityResult(not reasons, sanitized, tuple(reasons), pii)


def scan_output(text: str) -> SecurityResult:
    sanitized, pii = anonymize_pii(text or "")
    reasons: list[str] = []
    secret_patterns = [r"sk-[A-Za-z0-9_-]{20,}", r"(?i)(api[_-]?key|secret)\s*[=:]\s*[^\s]+"]
    for pattern in secret_patterns:
        if re.search(pattern, sanitized):
            reasons.append("possível segredo detectado na saída")
            sanitized = re.sub(pattern, "[SEGREDO_BLOQUEADO]", sanitized)
    return SecurityResult(not reasons, sanitized, tuple(reasons), pii)


def safe_refusal(result: SecurityResult) -> str:
    details = "; ".join(result.reasons) or "conteúdo não permitido"
    return (
        "<div class='warning-box'><strong>Solicitação bloqueada pelos guardrails.</strong> "
        f"Motivo: {details}. Reformule a pergunta mantendo o foco em análise legítima de políticas públicas.</div>"
    )
