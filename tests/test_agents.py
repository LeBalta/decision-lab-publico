from src.agents import executar_analise_multiagente

def test_multiagent_output_has_all_sections():
    output = executar_analise_multiagente("Pernambuco")
    assert "Agente Analista Territorial" in output
    assert "Agente de Machine Learning" in output
    assert "Agente Formulador" in output
