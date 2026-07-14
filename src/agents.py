try:
    from .policy_database import resumo_territorio, ranking_empregabilidade
    from .ml_empregabilidade import prever_por_territorio, formatar_previsao
except ImportError:  # execução direta pelo Streamlit
    from policy_database import resumo_territorio, ranking_empregabilidade
    from ml_empregabilidade import prever_por_territorio, formatar_previsao


def agente_analista_territorial(territorio: str) -> str:
    """
    Agente responsável por consultar dados estruturados no DuckDB.
    """
    return resumo_territorio(territorio)


def agente_modelo_empregabilidade(territorio: str) -> str:
    """
    Agente responsável por acionar o modelo de Machine Learning.
    """
    resultado = prever_por_territorio(territorio)
    return formatar_previsao(resultado)


def agente_formulador_politicas(territorio: str) -> str:
    """
    Agente responsável por consolidar dados, previsão e recomendações.
    """
    previsao = prever_por_territorio(territorio)

    if "erro" in previsao:
        return previsao["erro"]

    alto_risco = previsao["classe"] == 1

    if alto_risco:
        recomendacoes = [
            "priorizar programas de qualificação profissional vinculados à demanda produtiva local",
            "ampliar ações de intermediação de mão de obra e apoio ao primeiro emprego",
            "integrar políticas de educação de jovens e adultos com inclusão produtiva",
            "monitorar desigualdade de renda e pobreza como fatores associados à baixa empregabilidade",
        ]
    else:
        recomendacoes = [
            "manter ações de monitoramento dos indicadores de trabalho e renda",
            "fortalecer políticas preventivas de qualificação contínua",
            "estimular inovação produtiva e retenção de talentos",
            "acompanhar grupos específicos com maior risco de exclusão laboral",
        ]

    texto_recomendacoes = "\n".join(f"- {item}" for item in recomendacoes)

    return (
        f"Síntese executiva para {previsao['territorio']}:\n\n"
        f"O modelo classificou o território como "
        f"{previsao['interpretacao']} "
        f"com probabilidade estimada de "
        f"{previsao['probabilidade_alto_risco']:.2%}.\n\n"
        f"Principais fatores considerados:\n"
        f"- Taxa de desocupação: {previsao['taxa_desocupacao']:.2f}\n"
        f"- Índice de Gini: {previsao['indice_gini']:.3f}\n"
        f"- Taxa de pobreza: {previsao['taxa_pobreza']:.2f}\n"
        f"- Taxa de analfabetismo: {previsao['taxa_analfabetismo']:.2f}\n\n"
        f"Recomendações de política pública:\n"
        f"{texto_recomendacoes}"
    )


def executar_analise_multiagente(territorio: str) -> str:
    """
    Orquestra os agentes principais do Laboratório de Políticas Públicas.
    """

    resposta_analista = agente_analista_territorial(territorio)
    resposta_modelo = agente_modelo_empregabilidade(territorio)
    resposta_politicas = agente_formulador_politicas(territorio)

    return (
        "# Análise Multiagente de Empregabilidade Territorial\n\n"
        "## 1. Agente Analista Territorial\n"
        f"{resposta_analista}\n\n"
        "## 2. Agente de Machine Learning\n"
        f"{resposta_modelo}\n\n"
        "## 3. Agente Formulador de Políticas Públicas\n"
        f"{resposta_politicas}\n"
    )


def mostrar_ranking_critico(limite: int = 10) -> str:
    df = ranking_empregabilidade(limite)

    linhas = []

    for _, row in df.iterrows():
        linhas.append(
            f"- {row['territorio']}: "
            f"IVL={row['indice_vulnerabilidade_laboral']:.3f}, "
            f"risco={int(row['risco_baixa_empregabilidade'])}"
        )

    return "\n".join(linhas)


if __name__ == "__main__":
    print(executar_analise_multiagente("Pernambuco"))

    print("\nRanking de territórios críticos:")
    print(mostrar_ranking_critico(10))