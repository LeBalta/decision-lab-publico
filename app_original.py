
import re
import sys
from pathlib import Path
from datetime import datetime
from urllib.parse import quote_plus

import requests
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
import duckdb
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DOCS_DIR = BASE_DIR / "docs"

# ---------- Etapa 2: ML + Agentes ----------
SRC_DIR = BASE_DIR / "src"
if SRC_DIR.exists() and str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from agents import executar_analise_multiagente, mostrar_ranking_critico
    from policy_database import buscar_dataset_ml_por_territorio, ranking_empregabilidade
    AGENTES_ETAPA2_DISPONIVEIS = True
except Exception as e:
    AGENTES_ETAPA2_DISPONIVEIS = False
    ERRO_ETAPA2 = str(e)

try:
    from observability import registrar_interacao
    from security import scan_input, scan_output, safe_refusal
    OBSERVABILIDADE_DISPONIVEL = True
except Exception as e:
    OBSERVABILIDADE_DISPONIVEL = False
    ERRO_OBSERVABILIDADE = str(e)

IBGE_BASE = "https://servicodados.ibge.gov.br/api/v1"
SIDRA_BASE = "https://apisidra.ibge.gov.br/values"
CKAN_TSE = "https://dadosabertos.tse.jus.br/api/3/action/package_search"
CKAN_INEP = "https://dadosabertos.inep.gov.br/api/3/action/package_search"
SICONFI_BASE = "https://apidatalake.tesouro.gov.br/ords/siconfi/tt"
DATASUS_TABNET = "https://datasus.saude.gov.br/informacoes-de-saude-tabnet/"

st.set_page_config(page_title="RE[INOVE] Decision Lab Público", page_icon="🏛️", layout="wide")

st.markdown("""
<style>
.block-container {padding-top: 2rem; max-width: 1200px;}
[data-testid="stSidebar"] {background: linear-gradient(180deg,#f4f6fa,#eceff5); border-right:1px solid rgba(17,24,39,.06);}
[data-testid="stSidebar"] * {color:#1f2937;}
.reinove-hero{padding:2.2rem 2.4rem;border-radius:34px;background:radial-gradient(circle at 8% 18%,rgba(255,102,0,.18),transparent 28%),radial-gradient(circle at 88% 12%,rgba(99,102,241,.13),transparent 30%),linear-gradient(135deg,#fff 0%,#f8fafc 52%,#eef2ff 100%);border:1px solid rgba(17,24,39,.08);box-shadow:0 24px 60px rgba(15,23,42,.09);margin-bottom:1.2rem;}
.reinove-kicker{display:inline-flex;padding:.35rem .75rem;border-radius:999px;background:rgba(255,102,0,.10);color:#D35400;font-weight:600;font-size:.74rem;letter-spacing:.12em;text-transform:uppercase;}
@import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {font-family: 'Montserrat', 'Inter', 'Segoe UI', sans-serif;}
.reinove-title{margin-top:.85rem;font-size:clamp(2rem,4vw,3.35rem);line-height:1.08;font-weight:600;color:#111827;letter-spacing:-.035em;}
.reinove-subtitle{margin-top:1rem;max-width:900px;color:#4b5563;font-size:1.05rem;line-height:1.7;}
.metric-card{background:#fff;border:1px solid rgba(17,24,39,.08);border-radius:24px;padding:1.1rem 1.2rem;box-shadow:0 14px 35px rgba(15,23,42,.06);}
.metric-label{color:#6b7280;text-transform:uppercase;letter-spacing:.12em;font-size:.72rem;font-weight:800;}
.metric-value{color:#111827;font-size:1.35rem;font-weight:600;margin-top:.35rem;}
.answer-card{background:#fff;border:1px solid rgba(17,24,39,.08);border-radius:28px;padding:1.4rem 1.6rem;box-shadow:0 18px 50px rgba(15,23,42,.06);margin:1rem 0;}
.warning-box{border-left:4px solid #FF6600;background:#fff7ed;border-radius:18px;padding:1rem 1.15rem;color:#7c2d12;margin:.8rem 0;}
.badge{display:inline-block;padding:.28rem .65rem;border-radius:999px;font-size:.76rem;font-weight:800;background:rgba(255,102,0,.10);color:#D35400;margin:.12rem .25rem .12rem 0;}
.chat-time{font-size:.74rem;color:#6b7280;margin:.15rem 0 .45rem 0;}
.dynamic-answer{background:#fff;border:1px solid rgba(17,24,39,.08);border-radius:24px;padding:1.15rem 1.25rem;box-shadow:0 14px 38px rgba(15,23,42,.055);margin:.75rem 0;line-height:1.68;color:#1f2937;}
.dynamic-answer h3{margin-top:.4rem;margin-bottom:.35rem;color:#111827;}
.dynamic-answer ul{margin-top:.3rem;}
.etapa2-card{background:#fff;border:1px solid rgba(255,102,0,.22);border-radius:28px;padding:1.35rem 1.55rem;box-shadow:0 18px 50px rgba(15,23,42,.06);margin:1rem 0;line-height:1.68;color:#1f2937;white-space:pre-wrap;}
.etapa2-card h1,.etapa2-card h2,.etapa2-card h3{color:#111827;}
</style>
""", unsafe_allow_html=True)

SYSTEM_PROMPT = """
Você é o RE[INOVE] Decision Lab Público, assessor executivo de decisão para políticas públicas.
Regra máxima: sem evidência, sem recomendação forte.
Responda sempre em camadas:
CAMADA 1 — Síntese Executiva
CAMADA 2 — Ação Imediata
CAMADA 3 — KPIs de Controle
CAMADA 4 — Expansão Técnica
Separe fato, inferência e recomendação. Se a base não retornou dados, diga isso claramente.
"""

# ---------- APIs e conectores ----------
@st.cache_data(ttl=24*60*60)
def ibge_estados():
    try:
        r = requests.get(f"{IBGE_BASE}/localidades/estados?orderBy=nome", timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return [{"id": 50, "sigla": "MS", "nome": "Mato Grosso do Sul", "regiao": {"nome": "Centro-Oeste"}}]

@st.cache_data(ttl=24*60*60)
def ibge_municipios(estado_id):
    try:
        r = requests.get(f"{IBGE_BASE}/localidades/estados/{estado_id}/municipios", timeout=12)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

@st.cache_data(ttl=24*60*60)
def ibge_sidra_populacao(municipio_id=None):
    # Tabela 6579: estimativas de população residente, quando disponível.
    # Em caso de indisponibilidade, retorna aviso sem quebrar o MVP.
    try:
        nivel = f"n6/{municipio_id}" if municipio_id else "n1/all"
        url = f"{SIDRA_BASE}/t/6579/{nivel}/v/9324/p/last?formato=json"
        r = requests.get(url, timeout=16)
        r.raise_for_status()
        data = r.json()
        rows = data[1:] if isinstance(data, list) and len(data) > 1 else []
        return pd.DataFrame(rows), url, "OK"
    except Exception as e:
        return pd.DataFrame(), "SIDRA tabela 6579", f"Não foi possível consultar população no SIDRA: {e}"

@st.cache_data(ttl=24*60*60)
def siconfi_entes():
    try:
        url = f"{SICONFI_BASE}/entes"
        r = requests.get(url, timeout=16)
        r.raise_for_status()
        data = r.json()
        items = data.get("items", data if isinstance(data, list) else [])
        return pd.DataFrame(items), url, "OK"
    except Exception as e:
        return pd.DataFrame(), f"{SICONFI_BASE}/entes", f"Não foi possível consultar SICONFI/entes: {e}"

@st.cache_data(ttl=24*60*60)
def ckan_search(endpoint, termo):
    try:
        r = requests.get(endpoint, params={"q": termo, "rows": 8}, timeout=16)
        r.raise_for_status()
        data = r.json()
        results = data.get("result", {}).get("results", [])
        rows = []
        for item in results:
            rows.append({
                "titulo": item.get("title"),
                "nome": item.get("name"),
                "organizacao": (item.get("organization") or {}).get("title"),
                "url": item.get("url") or item.get("metadata_created"),
                "quantidade_recursos": len(item.get("resources", [])),
            })
        return pd.DataFrame(rows), endpoint, "OK"
    except Exception as e:
        return pd.DataFrame(), endpoint, f"Não foi possível consultar catálogo CKAN: {e}"

def datasus_referencia():
    return pd.DataFrame([{
        "base": "DATASUS / TABNET",
        "status": "Referência oficial incorporada",
        "uso": "Saúde pública, produção, mortalidade, nascidos vivos, internações e outros sistemas.",
        "link": DATASUS_TABNET,
        "observacao": "No MVP, o DATASUS entra como fonte selecionável e link oficial. A extração automática por TABNET exige conector específico por sistema/tabela."
    }]), DATASUS_TABNET, "OK"

def novo_caged_referencia():
    return pd.DataFrame([{
        "base": "Novo CAGED",
        "status": "Referência incorporada",
        "uso": "Admissões, desligamentos, saldo de emprego formal e setores econômicos.",
        "link": "https://www.gov.br/trabalho-e-emprego/pt-br/assuntos/estatisticas-trabalho/novo-caged",
        "observacao": "No MVP, entra como base selecionável. A extração robusta deve baixar microdados/arquivos oficiais por competência e transformar em Parquet."
    }]), "Novo CAGED / MTE", "OK"


def estimar_indicadores_territorio(estado, municipio, municipios):
    """
    Estimativas transparentes para visualização executiva do MVP.
    Em produção:
    população -> IBGE/SIDRA;
    renda per capita -> IBGE/PNAD/PIB per capita;
    IDH -> Atlas Brasil/PNUD/IPEA/FJP.
    """
    if estado.get("id") == 0:
        return {
            "territorio": "Todo o estado do Brasil",
            "populacao": 213421037,
            "renda": 1850.0,
            "idh": 0.760,
        }

    total = max(len(municipios), 1)
    if municipio:
        idx = next((i for i, m in enumerate(municipios) if m.get("id") == municipio.get("id")), 0)
        pop = 14000 + (total - idx) * 9000
        renda = 1180 + idx * 38
        idh = min(0.840, max(0.580, 0.610 + (idx / total) * 0.18))
        return {
            "territorio": f"{municipio.get('nome')}/{estado.get('sigla')}",
            "populacao": int(pop),
            "renda": float(renda),
            "idh": round(idh, 3),
        }

    pop = sum(14000 + (total - i) * 9000 for i, _ in enumerate(municipios))
    renda = 1450 + (estado.get("id", 0) % 20) * 45
    idh = 0.680 + (estado.get("id", 0) % 15) * 0.006
    return {
        "territorio": f"Todo o estado de {estado.get('nome')}",
        "populacao": int(pop) if municipios else 0,
        "renda": float(renda),
        "idh": round(min(0.850, idh), 3),
    }

def format_num(v):
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "-"

def format_money(v):
    try:
        return "R$ " + f"{float(v):,.0f}".replace(",", ".")
    except Exception:
        return "-"

def coletar_bases(bases, estado, municipio):
    tabelas = {}
    evidencias = []
    municipio_id = municipio.get("id") if municipio else None

    if "IBGE Cidades / SIDRA" in bases:
        df, fonte, status = ibge_sidra_populacao(municipio_id)
        tabelas["ibge_sidra"] = df
        evidencias.append({"base": "IBGE Cidades / SIDRA", "fonte": fonte, "status": status, "linhas": len(df)})

    if "FINBRA / SICONFI / Tesouro" in bases:
        df, fonte, status = siconfi_entes()
        if not df.empty and estado:
            # filtro leve, quando colunas existirem
            uf = estado.get("sigla")
            for col in ["co_uf", "sg_uf", "uf"]:
                if col in df.columns:
                    df = df[df[col].astype(str).str.upper().eq(str(uf).upper())]
                    break
        tabelas["siconfi_entes"] = df.head(50)
        evidencias.append({"base": "FINBRA / SICONFI / Tesouro", "fonte": fonte, "status": status, "linhas": len(df)})

    if "INEP" in bases:
        termo = f"educação {estado.get('sigla','')}"
        df, fonte, status = ckan_search(CKAN_INEP, termo)
        tabelas["inep_catalogo"] = df
        evidencias.append({"base": "INEP", "fonte": fonte, "status": status, "linhas": len(df)})

    if "DATASUS" in bases:
        df, fonte, status = datasus_referencia()
        tabelas["datasus_referencia"] = df
        evidencias.append({"base": "DATASUS", "fonte": fonte, "status": status, "linhas": len(df)})

    if "Novo CAGED" in bases:
        df, fonte, status = novo_caged_referencia()
        tabelas["novo_caged_referencia"] = df
        evidencias.append({"base": "Novo CAGED", "fonte": fonte, "status": status, "linhas": len(df)})

    if "TSE" in bases:
        termo = f"{estado.get('sigla','')} eleitorado resultados"
        df, fonte, status = ckan_search(CKAN_TSE, termo)
        tabelas["tse_catalogo"] = df
        evidencias.append({"base": "TSE", "fonte": fonte, "status": status, "linhas": len(df)})

    return tabelas, pd.DataFrame(evidencias)

# ---------- dados internos / RAG ----------
def carregar_dados_base():
    return pd.read_csv(DATA_DIR / "secretarias.csv"), pd.read_csv(DATA_DIR / "indicadores_municipais.csv")

def consulta_duckdb(pergunta, tabelas_bases):
    secretarias, indicadores = carregar_dados_base()
    con = duckdb.connect(database=":memory:")
    con.register("secretarias", secretarias)
    con.register("indicadores", indicadores)
    for nome, df in tabelas_bases.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", nome)
            con.register(safe, df)

    p = pergunta.lower()
    if "orçamento" in p or "orcamento" in p:
        sql = "SELECT secretaria, area, orcamento_anual FROM secretarias ORDER BY orcamento_anual DESC LIMIT 5"
        return sql, con.execute(sql).df()

    if "meta" in p or "indicador" in p or "prioridade" in p:
        sql = """
        SELECT s.secretaria, i.indicador, i.valor, i.meta, i.unidade, i.mes,
               CASE
                 WHEN i.unidade = '%' AND i.valor < i.meta THEN 'abaixo da meta'
                 WHEN i.unidade <> '%' AND i.valor > i.meta THEN 'acima do limite/meta'
                 ELSE 'dentro ou próximo da meta'
               END AS situacao
        FROM indicadores i
        JOIN secretarias s ON s.secretaria_id = i.secretaria_id
        LIMIT 10
        """
        return sql, con.execute(sql).df()

    # fallback: mostra evidências de bases selecionadas se houver
    for nome, df in tabelas_bases.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            return f"SELECT * FROM {nome} LIMIT 10", df.head(10)

    sql = "SELECT * FROM indicadores LIMIT 10"
    return sql, con.execute(sql).df()

def limpar_texto(texto):
    return re.sub(r"\s+", " ", texto or "").strip()

def documentos_base():
    docs = []
    for f in DOCS_DIR.glob("*.txt"):
        docs.append({"fonte": f.name, "texto": f.read_text(encoding="utf-8", errors="ignore")})
    return docs

def ler_upload(uploaded_file):
    name = uploaded_file.name
    data = uploaded_file.getvalue()
    suf = Path(name).suffix.lower()
    if suf in [".txt",".md",".csv",".json",".xml",".html",".htm",".yaml",".yml",".tsv"]:
        return {"fonte": name, "texto": data.decode("utf-8", errors="ignore")[:25000]}
    return {"fonte": name, "texto": f"Arquivo anexado sem extração textual automática neste MVP: {name}. Tamanho: {len(data)} bytes."}

def documentos_de_bases(evidencias, tabelas):
    docs = []
    if evidencias is not None and not evidencias.empty:
        docs.append({"fonte": "bases_publicas_selecionadas", "texto": evidencias.to_markdown(index=False)})
    for nome, df in tabelas.items():
        if isinstance(df, pd.DataFrame) and not df.empty:
            docs.append({"fonte": f"base_{nome}", "texto": df.head(12).to_markdown(index=False)})
    return docs

def documentos_de_links(links):
    return [{"fonte": f"Link informado: {l}", "texto": f"Link contextual informado pelo usuário para verificação: {l}"} for l in links if l.strip()]

def buscar_rag(pergunta, documentos, top_k=5):
    chunks = []
    for doc in documentos:
        partes = re.split(r"(?<=[.!?])\s+|\n\n+", doc["texto"])
        acc, idx = "", 1
        for parte in partes:
            if len(acc) + len(parte) < 900:
                acc += " " + parte
            else:
                if acc.strip():
                    chunks.append({"fonte": doc["fonte"], "chunk": idx, "texto": limpar_texto(acc)})
                    idx += 1
                acc = parte
        if acc.strip():
            chunks.append({"fonte": doc["fonte"], "chunk": idx, "texto": limpar_texto(acc)})
    if not chunks:
        return []
    corpus = [c["texto"] for c in chunks]
    try:
        vec = TfidfVectorizer()
        mat = vec.fit_transform(corpus + [pergunta])
        sims = cosine_similarity(mat[-1], mat[:-1]).flatten()
        idxs = sims.argsort()[::-1][:top_k]
        return [{**chunks[i], "score": float(sims[i])} for i in idxs]
    except Exception:
        return chunks[:top_k]

def listar_modelos(base_url):
    try:
        r = requests.get(f"{base_url.rstrip('/')}/models", timeout=8)
        r.raise_for_status()
        return [m.get("id") for m in r.json().get("data", []) if m.get("id")]
    except Exception:
        return []

def chamar_llm(base_url, modelo, prompt):
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json={
        "model": modelo,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.2,
        "stream": False,
    }, timeout=90)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]

def resposta_local(pergunta, territorio, tema, objetivo, bases, df, contexto):
    return resposta_chat_dinamica(pergunta, territorio, tema, objetivo, bases, df, contexto)

def montar_prompt(pergunta, territorio, tema, objetivo, poder, bases, evidencias, df, contexto):
    return f"""{SYSTEM_PROMPT}

Território: {territorio}
Poder: {poder}
Tema: {tema or 'não informado'}
Objetivo: {objetivo or 'não informado'}
Bases selecionadas: {', '.join(bases) if bases else 'nenhuma'}
Resumo das bases:
{evidencias.to_markdown(index=False) if isinstance(evidencias, pd.DataFrame) and not evidencias.empty else 'Sem evidências de bases.'}

Pergunta:
{pergunta}

Dados estruturados:
{df.to_markdown(index=False) if isinstance(df, pd.DataFrame) and not df.empty else 'Sem dados estruturados.'}

Documentos recuperados:
{chr(10).join([f"Fonte: {c['fonte']} | {c['texto']}" for c in contexto])}

Responda em português brasileiro com as 4 camadas, sem inventar fontes.
"""


# ---------- Tabelas profissionais e resposta dinâmica ----------
def valor_numerico(x):
    if pd.isna(x):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip().replace("R$", "").replace(" ", "")
    if not s:
        return None
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return None


def formatar_real_br(x):
    v = valor_numerico(x)
    if v is None:
        return x
    return f"R$ {v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_numero_br(x):
    v = valor_numerico(x)
    if v is None:
        return x
    if abs(v - int(v)) < 0.000001:
        return f"{int(v):,}".replace(",", ".")
    return f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_percentual_br(x):
    v = valor_numerico(x)
    if v is None:
        return x
    return f"{v:.2f}%".replace(".", ",")


def formatar_tabela_profissional(df):
    if df is None or not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    df_view = df.copy()
    colunas_moeda = ["orcamento", "orçamento", "receita", "despesa", "saldo", "renda", "renda_per_capita", "pib", "valor", "vl_", "vr_"]
    colunas_percentual = ["idh", "taxa", "percentual", "porcentagem", "indice", "índice", "meta"]

    for col in df_view.columns:
        nome_col = str(col).lower()
        if any(x in nome_col for x in colunas_moeda):
            df_view[col] = df_view[col].apply(formatar_real_br)
        elif any(x in nome_col for x in colunas_percentual):
            if "idh" in nome_col:
                df_view[col] = df_view[col].apply(lambda x: f"{valor_numerico(x):.3f}".replace(".", ",") if valor_numerico(x) is not None else x)
            else:
                df_view[col] = df_view[col].apply(formatar_percentual_br)
        else:
            try:
                if pd.api.types.is_numeric_dtype(df_view[col]):
                    df_view[col] = df_view[col].apply(formatar_numero_br)
            except Exception:
                pass
    return df_view


def ordenar_tabela_profissional(df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return df
    prioridade_desc = ["orcamento", "orçamento", "receita", "despesa", "saldo", "renda", "pib", "valor", "populacao", "população", "score", "linhas"]
    prioridade_asc = ["municipio", "município", "secretaria", "base", "titulo", "título", "nome"]
    for palavra in prioridade_desc:
        for col in df.columns:
            if palavra in str(col).lower():
                try:
                    return df.sort_values(by=col, ascending=False)
                except Exception:
                    pass
    for palavra in prioridade_asc:
        for col in df.columns:
            if palavra in str(col).lower():
                try:
                    return df.sort_values(by=col, ascending=True)
                except Exception:
                    pass
    return df


def exibir_tabela_profissional(df, titulo=None, max_linhas=20):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        if titulo:
            st.markdown(f"### {titulo}")
        st.info("Sem dados retornados para exibição.")
        return
    df_ordenado = ordenar_tabela_profissional(df.copy()).head(max_linhas)
    df_view = formatar_tabela_profissional(df_ordenado)
    st.markdown("""
    <style>
    div[data-testid="stDataFrame"] thead tr th {
        background-color: #FF6600 !important;
        color: white !important;
        font-weight: 700 !important;
    }
    div[data-testid="stDataFrame"] {
        border-radius: 18px !important;
        overflow: hidden !important;
    }
    </style>
    """, unsafe_allow_html=True)
    if titulo:
        st.markdown(f"### {titulo}")
    st.dataframe(df_view, use_container_width=True, hide_index=True)


def gerar_insight_por_base(nome, df):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return f"**{nome}:** a base não retornou registros para o recorte selecionado."
    total = len(df)
    colunas = ", ".join([str(c) for c in df.columns[:6]])
    leitura = f"**{nome}:** retornou {total} registro(s). Colunas principais: {colunas}."
    nome_lower = nome.lower()
    if "ibge" in nome_lower or "sidra" in nome_lower:
        leitura += " É a base territorial/demográfica inicial; para decisão final, combine com série histórica e indicadores setoriais."
    elif "siconfi" in nome_lower or "tesouro" in nome_lower:
        leitura += " Ajuda a avaliar capacidade fiscal, receita, despesa e sustentabilidade financeira."
    elif "inep" in nome_lower:
        leitura += " Ajuda a diagnosticar oferta educacional e aderência de cursos ao território."
    elif "datasus" in nome_lower:
        leitura += " Ajuda a identificar pressão assistencial e prioridades sanitárias."
    elif "caged" in nome_lower:
        leitura += " Ajuda a ler vocação econômica, emprego formal e demanda por qualificação."
    elif "tse" in nome_lower:
        leitura += " Ajuda a contextualizar leitura eleitoral e participação territorial."
    return leitura


def carimbo_agora():
    return datetime.now().strftime("%d/%m/%Y %H:%M:%S")


def pergunta_pede_empregabilidade(pergunta):
    p = (pergunta or "").lower()
    termos = [
        "empregabilidade",
        "baixa empregabilidade",
        "vulnerabilidade laboral",
        "risco laboral",
        "risco de emprego",
        "risco de desemprego",
        "desemprego",
        "desocupação",
        "desocupacao",
        "mercado de trabalho",
        "ranking de territórios críticos",
        "ranking de territorios criticos",
        "análise multiagente",
        "analise multiagente",
    ]
    return any(t in p for t in termos)


def territorio_para_empregabilidade(estado, municipio):
    """
    O modelo da Etapa 2 usa recortes agregados do IBGE:
    Brasil, regiões, UFs, capitais e algumas regiões metropolitanas.
    Para análise estadual, usa o nome da UF.
    Para análise municipal, tenta usar o nome do município selecionado.
    """
    if estado.get("id") == 0:
        return "Brasil"

    if municipio:
        return municipio.get("nome")

    return estado.get("nome")


def resposta_etapa2_empregabilidade(pergunta, estado, municipio):
    if not AGENTES_ETAPA2_DISPONIVEIS:
        return (
            "<div class='warning-box'>"
            "<strong>Camada Etapa 2 indisponível.</strong><br>"
            "Verifique se os arquivos em src/ e models/ foram criados. "
            f"Detalhe técnico: {ERRO_ETAPA2}"
            "</div>"
        ), pd.DataFrame(), "Etapa 2 indisponível"

    p = (pergunta or "").lower()

    if "ranking" in p:
        resposta = "# Ranking de territórios críticos em empregabilidade\n\n" + mostrar_ranking_critico(10)
        df = ranking_empregabilidade(10)
        return resposta, df, "SELECT * FROM dataset_empregabilidade_ml ORDER BY indice_vulnerabilidade_laboral DESC LIMIT 10"

    territorio_ml = territorio_para_empregabilidade(estado, municipio)
    resposta = executar_analise_multiagente(territorio_ml)
    df = buscar_dataset_ml_por_territorio(territorio_ml)
    sql = f"SELECT * FROM dataset_empregabilidade_ml WHERE territorio ILIKE '%{territorio_ml}%'"
    return resposta, df, sql


def detectar_intencao(pergunta):
    p = (pergunta or "").lower()
    if any(t in p for t in ["orçamento", "orcamento", "receita", "despesa", "maior orçamento", "maior orcamento"]):
        return "financas"
    if any(t in p for t in ["analfabet", "alfabetização", "alfabetizacao"]):
        return "analfabetismo"
    if any(t in p for t in ["curso", "senai", "graduação", "graduacao", "educação superior", "ensino superior", "mão-de-obra", "mao-de-obra", "qualificação", "qualificacao"]):
        return "educacao_trabalho"
    if any(t in p for t in ["panorama", "diagnóstico", "diagnostico", "situação", "situacao", "perfil", "território", "territorio"]):
        return "panorama"
    if any(t in p for t in ["base", "bases", "evidência", "evidencias", "fonte", "fontes"]):
        return "bases"
    if any(t in p for t in ["saúde", "saude", "datasus", "hospital", "atenção básica", "atencao basica"]):
        return "saude"
    if any(t in p for t in ["emprego", "caged", "indústria", "industria", "desenvolvimento econômico", "desenvolvimento economico"]):
        return "emprego"
    if any(t in p for t in ["detalhe", "detalhar", "plano", "kpi", "projeto", "expandir", "aprofundar"]):
        return "detalhado"
    return "geral"


def tema_efetivo(pergunta, tema):
    if tema and tema.strip():
        return tema.strip()
    intencao = detectar_intencao(pergunta)
    mapa = {
        "financas": "finanças públicas",
        "analfabetismo": "analfabetismo e educação básica",
        "educacao_trabalho": "educação superior, qualificação profissional e trabalho",
        "panorama": "panorama territorial",
        "bases": "evidências e fontes públicas",
        "saude": "saúde pública",
        "emprego": "emprego e desenvolvimento econômico",
        "detalhado": "plano executivo e monitoramento",
    }
    return mapa.get(intencao, "tema inferido pela pergunta")


def achado_principal(df, intencao):
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return "não houve tabela estruturada suficiente para sustentar conclusão objetiva."

    cols = {str(c).lower(): c for c in df.columns}

    if intencao == "financas" and "orcamento_anual" in cols and "secretaria" in cols:
        linha = df.sort_values(cols["orcamento_anual"], ascending=False).iloc[0]
        return f"a maior dotação encontrada é de {linha[cols['secretaria']]}, com {formatar_real_br(linha[cols['orcamento_anual']])}."

    if intencao == "analfabetismo":
        cand = [c for c in df.columns if "analf" in str(c).lower() or "taxa" in str(c).lower()]
        if cand:
            return f"a base retornou coluna relacionada ao tema: {cand[0]}."
        return "as bases consultadas não trouxeram taxa oficial de analfabetismo; isso é uma lacuna, não um resultado."

    if "d2n" in cols and "v" in cols:
        linha = df.iloc[0]
        return f"a base IBGE/SIDRA retornou {linha.get(cols['d2n'], 'indicador')} com valor {formatar_numero_br(linha.get(cols['v']))}."

    if "titulo" in cols:
        titulos = "; ".join(df[cols["titulo"]].astype(str).head(3).tolist())
        return f"a consulta retornou registros de catálogo, incluindo: {titulos}."

    if "indicador" in cols and "situacao" in cols:
        criticos = df[df[cols["situacao"]].astype(str).str.contains("abaixo|acima", case=False, na=False)]
        if not criticos.empty:
            partes = []
            for _, r in criticos.head(3).iterrows():
                secretaria = r.get(cols.get("secretaria", ""), "")
                partes.append(f"{secretaria} — {r[cols['indicador']]} ({r[cols['situacao']]})")
            return "pontos de atenção: " + "; ".join(partes) + "."

    return f"a consulta retornou {len(df)} registro(s), com colunas: {', '.join(map(str, df.columns[:5]))}."


def recomendacao_dinamica(pergunta, intencao, territorio, objetivo):
    obj = objetivo.strip() if objetivo and objetivo.strip() else "objetivo ainda não detalhado"
    if intencao == "financas":
        return "Antes de propor nova política, valide fonte de custeio, margem fiscal e compatibilidade com PPA, LDO e LOA."
    if intencao == "analfabetismo":
        return "Não recomendo ação forte sem trazer uma tabela oficial de analfabetismo por território, ano e faixa etária."
    if intencao == "educacao_trabalho":
        return "Priorize cursos ligados a automação, dados, manutenção industrial, logística, energia e gestão da produção; valide com Novo CAGED, INEP e perfil industrial local."
    if intencao == "panorama":
        return "Monte um panorama por eixos: demografia, fiscal, educação, saúde, emprego e participação eleitoral; depois escolha no máximo três prioridades."
    if intencao == "bases":
        return "Classifique cada base como dado efetivo, catálogo ou referência. Só dado efetivo sustenta recomendação forte."
    if intencao == "saude":
        return "Use DATASUS apenas após escolher tabela/sistema específico. Para decisão, cruze produção, mortalidade, cobertura e demanda reprimida."
    if intencao == "emprego":
        return "Cruze Novo CAGED por setor econômico com oferta educacional e vocação produtiva regional."
    return f"Transforme o objetivo '{obj}' em ação com responsável, prazo, orçamento e KPI."


def resposta_chat_dinamica(pergunta, territorio, tema, objetivo, bases, df, contexto):
    intencao = detectar_intencao(pergunta)
    tema_final = tema_efetivo(pergunta, tema)
    achado = achado_principal(df, intencao)
    recomendacao = recomendacao_dinamica(pergunta, intencao, territorio, objetivo)

    fontes = ", ".join(bases) if bases else "bases internas"
    evidencias = []
    for c in contexto[:4]:
        evidencias.append(f"{c.get('fonte', 'fonte')} (score {c.get('score', 0):.3f})")
    evidencias_txt = "; ".join(evidencias) if evidencias else "sem evidência documental recuperada pelo RAG."

    if intencao == "detalhado":
        return f"""
<div class="dynamic-answer">
<h3>Análise executiva detalhada</h3>
<p><strong>Território:</strong> {territorio}<br>
<strong>Tema:</strong> {tema_final}<br>
<strong>Bases consideradas:</strong> {fontes}</p>
<p><strong>Diagnóstico:</strong> {achado}</p>
<p><strong>Direção recomendada:</strong> {recomendacao}</p>
<ul>
<li><strong>Ação 1:</strong> validar dados oficiais e recorte territorial.</li>
<li><strong>Ação 2:</strong> definir órgão responsável e fonte de custeio.</li>
<li><strong>Ação 3:</strong> escolher 3 KPIs de acompanhamento.</li>
<li><strong>Ação 4:</strong> registrar evidências e limitações metodológicas.</li>
</ul>
<p><strong>Evidências:</strong> {evidencias_txt}</p>
</div>
"""

    return f"""
<div class="dynamic-answer">
<h3>Resposta executiva</h3>
<p><strong>Leitura:</strong> em {territorio}, a pergunta trata de <strong>{tema_final}</strong>.</p>
<p><strong>Achado nos dados:</strong> {achado}</p>
<p><strong>Recomendação:</strong> {recomendacao}</p>
<p><strong>Evidências usadas:</strong> {evidencias_txt}</p>
<p class="small-note">Para receber plano, riscos, responsáveis e KPIs, peça: <em>detalhe em plano executivo</em>.</p>
</div>
"""


def rolar_para_baixo():
    components.html(
        """
        <script>
        function autoScrollChat() {

            const parentDoc = window.parent.document;

            function goBottom() {

                window.parent.scrollTo({
                    top: parentDoc.body.scrollHeight,
                    behavior: "smooth"
                });

                parentDoc.documentElement.scrollTop =
                    parentDoc.documentElement.scrollHeight;

                parentDoc.body.scrollTop =
                    parentDoc.body.scrollHeight;

                const main = parentDoc.querySelector("section.main");
                if(main){
                    main.scrollTop = main.scrollHeight;
                }
            }

            goBottom();
            setTimeout(goBottom, 200);
            setTimeout(goBottom, 600);
            setTimeout(goBottom, 1000);
            setTimeout(goBottom, 1600);
        }

        autoScrollChat();
        window.onload = autoScrollChat;
        </script>
        """,
        height=0
    )



def exibir_contexto_rag(contexto):
    """Exibe documentos recuperados pelo RAG, convertendo tabelas Markdown em DataFrames quando possível."""
    if not contexto:
        st.info("Nenhum documento foi recuperado pelo RAG.")
        return

    for c in contexto:
        st.markdown(
            f"**Fonte:** {c.get('fonte', 'fonte')} · "
            f"**Chunk:** {c.get('chunk', '-')} · "
            f"**Score:** {c.get('score', 0):.3f}"
        )

        texto = str(c.get("texto", "")).strip()

        if "|" not in texto:
            st.write(texto)
            continue

        try:
            linhas = [linha.strip() for linha in texto.splitlines() if linha.strip()]
            linhas = [
                linha for linha in linhas
                if not set(linha.replace("|", "").replace(":", "").replace("-", "").strip()) == set()
            ]

            texto_unico = " ".join(linhas)
            partes = [p.strip() for p in texto_unico.split("|") if p.strip()]

            possiveis_colunas = [
                "titulo", "nome", "organizacao", "organização", "url", "quantidade_recursos",
                "base", "status", "uso", "link", "observacao", "observação",
                "cod_ibge", "ente", "capital", "regiao", "região", "uf", "esfera",
                "exercicio", "exercício", "populacao", "população", "cnpj",
                "secretaria", "indicador", "area", "área", "orcamento_anual"
            ]

            header_idx = None
            for i, parte in enumerate(partes):
                if parte.lower() in possiveis_colunas:
                    header_idx = i
                    break

            if header_idx is not None:
                partes = partes[header_idx:]

            primeiras = [p.lower() for p in partes[:12]]

            if "titulo" in primeiras:
                ncols = 5
            elif "base" in primeiras:
                ncols = 5
            elif "cod_ibge" in primeiras:
                ncols = 9
            elif "secretaria" in primeiras:
                ncols = 6
            else:
                ncols = 5

            colunas = partes[:ncols]
            dados = partes[ncols:]

            linhas_dados = []
            for i in range(0, len(dados), ncols):
                linha = dados[i:i+ncols]
                if len(linha) == ncols:
                    linhas_dados.append(linha)

            if colunas and linhas_dados:
                df_rag = pd.DataFrame(linhas_dados, columns=colunas)
                exibir_tabela_profissional(df_rag, max_linhas=10)
            else:
                st.write(texto)

        except Exception:
            st.write(texto)


# ---------- UI ----------
st.markdown("""
<div class="reinove-hero">
  <div class="reinove-kicker">RE[INOVE]® · Laboratório de Decisão</div>
  <div class="reinove-title">Laboratório Inteligente de Apoio à Decisão Governamental</div>
  <div class="reinove-subtitle">Escolha território, bases públicas e objetivo de seu interesse. Posso ajudar com diagnósticos, prioridades, riscos e propostas de política pública baseadas em evidências.</div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("### Camadas do sistema")
    for item in ["Streamlit", "DuckDB", "RAG local", "Upload", "Links", "IBGE", "Conectores modulares", "ML empregabilidade", "Agentes Etapa 2"]:
        st.write(f"✅ {item}")
    st.divider()

    st.markdown("### Perguntas-modelo")
    st.caption("Use estas perguntas para testar se o sistema está funcionando bem.")

    st.markdown("""
**1. Dados estruturados**
- Qual secretaria possui maior orçamento?
- Quais indicadores estão abaixo da meta?

**2. Território**
- Qual o panorama de Campo Grande/MS?
- Qual o ranking territorial por prioridade?

**3. Bases públicas**
- Quais evidências foram encontradas nas bases selecionadas?
- O que as bases selecionadas indicam sobre educação superior?

**4. Documentos e links**
- O que os documentos recuperados pelo RAG dizem sobre transparência?
- Considere os links adicionados e sugira ações para capacitação de mão de obra.

**5. Política pública**
- Faça sugestões de cursos de graduação superior para capacitar mão de obra no MS.
- Estruture um plano de ação com responsáveis, KPIs e riscos.

**6. Etapa 2 — ML e agentes**
- Analise o risco de baixa empregabilidade de Pernambuco.
- Mostre o ranking de territórios críticos em empregabilidade.
- Faça uma análise multiagente de empregabilidade para Bahia.
    """)
    st.divider()
    st.markdown("### Motor de Inteligência")
    base_url = st.text_input("URL LM Studio", "http://127.0.0.1:1234/v1")
    usar_llm = st.checkbox("Usar LM Studio", value=False)
    modelos = listar_modelos(base_url) if usar_llm else []
    modelo = st.selectbox("Modelo", [""] + modelos)

estados = ibge_estados()

labels = {
    "Todos os Estados (Análise Federal)": {
        "id": 0,
        "sigla": "BR",
        "nome": "Brasil"
    }
}

labels.update({
    f"{e['nome']} ({e['sigla']})": e
    for e in estados
})

c1, c2, c3 = st.columns([1,1,1])
with c1:
    estado_label = st.selectbox("Estado", list(labels.keys()), index=0)
    estado = labels[estado_label]
if estado["id"] == 0:
    municipios = []
    mun_labels = {"Brasil": None}
else:
    municipios = ibge_municipios(estado["id"])
    mun_labels = {"Todo o estado": None, **{m["nome"]: m for m in municipios}}
with c2:
    mun_label = st.selectbox("Município", list(mun_labels.keys()))
    municipio = mun_labels[mun_label]
with c3:
    poder = st.selectbox("Poder responsável", ["Todos", "Executivo", "Legislativo"])

tema = st.text_input("Tema de interesse", placeholder="Ex.: educação superior, saúde, emprego, gestão fiscal")
objetivo = st.text_input("Objetivo da política pública", placeholder="Ex.: criação de cursos técnicos, reduzir filas, ampliar transparência...")

if not tema or not objetivo:
    st.info("Para respostas mais fortes, informe tema e objetivo da política pública.")

bases_disponiveis = [
    "IBGE Cidades / SIDRA",
    "FINBRA / SICONFI / Tesouro",
    "INEP",
    "DATASUS",
    "Novo CAGED",
    "TSE",
]
st.markdown("### Escolha quais bases deseja rodar")
bases = st.multiselect(
    "Bases públicas para consultar",
    bases_disponiveis,
    default=bases_disponiveis,
    help="Selecione uma ou mais bases. O sistema só consulta as bases marcadas aqui."
)

uploaded = st.file_uploader("Anexe arquivos para contexto", type=["txt","md","csv","json","xml","html","htm","yaml","yml","tsv","pdf","docx","xlsx"], accept_multiple_files=True)
if "links_contextuais" not in st.session_state:
    st.session_state.links_contextuais = []

st.markdown("### Links estratégicos")

novo_link = st.text_input(
    "Adicionar link estratégico",
    placeholder="Cole um link e clique em Adicionar"
)

col_link1, col_link2 = st.columns([1, 4])

with col_link1:
    if st.button("Adicionar link"):
        if novo_link.strip():
            if novo_link.strip() not in st.session_state.links_contextuais:
                st.session_state.links_contextuais.append(novo_link.strip())
            st.rerun()

with col_link2:
    if st.button("Limpar links"):
        st.session_state.links_contextuais = []
        st.rerun()

links = st.session_state.links_contextuais

if links:
    st.markdown("**Links adicionados:**")
    for i, link in enumerate(links):
        c1, c2 = st.columns([5, 1])
        with c1:
            st.write(link)
        with c2:
            if st.button("Excluir", key=f"excluir_link_{i}"):
                st.session_state.links_contextuais.pop(i)
                st.rerun()
else:
    st.caption("Nenhum link adicionado ainda.")

territorio = f"{mun_label}/{estado['sigla']}" if municipio else f"Todo o estado de {estado['nome']}"

if st.button("Atualizar evidências", type="primary"):
    st.cache_data.clear()
    st.rerun()

with st.spinner("Consultando bases selecionadas..."):
    tabelas_bases, evidencias_bases = coletar_bases(bases, estado, municipio)

indicadores_card = estimar_indicadores_territorio(estado, municipio, municipios)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Território</div><div class="metric-value">{indicadores_card["territorio"]}</div></div>', unsafe_allow_html=True)
with m2:
    st.markdown(f'<div class="metric-card"><div class="metric-label">População estimada</div><div class="metric-value">{format_num(indicadores_card["populacao"])}</div></div>', unsafe_allow_html=True)
with m3:
    st.markdown(f'<div class="metric-card"><div class="metric-label">Renda per capita estimada</div><div class="metric-value">{format_money(indicadores_card["renda"])}</div></div>', unsafe_allow_html=True)
with m4:
    st.markdown(f'<div class="metric-card"><div class="metric-label">IDH estimado</div><div class="metric-value">{indicadores_card["idh"]}</div></div>', unsafe_allow_html=True)

st.markdown('<div class="warning-box"><strong>Transparência analítica:</strong> os cards de população, renda per capita e IDH são estimativas executivas do MVP. Em produção, devem ser substituídos por IBGE/SIDRA, PNAD/PIB per capita e Atlas Brasil/PNUD. As bases marcadas no seletor são as únicas consultadas.</div>', unsafe_allow_html=True)

with st.expander("Ver evidências das bases públicas selecionadas", expanded=False):
    exibir_tabela_profissional(evidencias_bases, "Resumo das bases consultadas", max_linhas=50)

    for nome, df_base in tabelas_bases.items():
        st.markdown(f"#### {nome}")
        if isinstance(df_base, pd.DataFrame) and not df_base.empty:
            exibir_tabela_profissional(df_base, max_linhas=20)
            st.markdown("**Insight automático:**")
            st.info(gerar_insight_por_base(nome, df_base))
        else:
            st.warning(f"A base {nome} não retornou dados para o recorte selecionado.")

docs = documentos_base()
docs += documentos_de_bases(evidencias_bases, tabelas_bases)
docs += documentos_de_links(links)
if uploaded:
    docs += [ler_upload(up) for up in uploaded]

if "messages" not in st.session_state:
    st.session_state.messages = [{
        "role": "assistant",
        "content": """
<div class='dynamic-answer'>
<h3>Bem-vinda ao RE[INOVE]® Decision Lab Público</h3>
<p>Selecione território, bases públicas e objetivo estratégico.</p>
<p>Posso apoiar diagnósticos, prioridades, riscos, indicadores e propostas governamentais baseadas em evidências.</p>
<p><strong>Exemplo:</strong> Sugira um plano para ampliar educação profissional em Campo Grande/MS.</p>
</div>
""",
        "time": carimbo_agora(),
        "meta": {}
    }]

import re

def exportar_conversa_markdown(messages):
    linhas = ["# RE[INOVE] Decision Lab Público", ""]

    for msg in messages:
        papel = "Usuário" if msg.get("role") == "user" else "Assistente"
        hora = msg.get("time", "")

        linhas.append(f"## {papel} — {hora}")

        texto = re.sub(r"<[^>]+>", "", str(msg.get("content", "")))
        linhas.append(texto.strip())
        linhas.append("")

    return "\n".join(linhas)


def limpar_conversa():
    st.session_state.messages = [{
        "role": "assistant",
        "content": """
<div class='dynamic-answer'>
<h3>Conversa limpa</h3>
<p>Faça uma nova pergunta ou ajuste território, bases públicas e objetivo estratégico.</p>
</div>
""",
        "time": carimbo_agora(),
        "meta": {}
    }]
controle1, controle2, controle3 = st.columns([1, 1, 4])

with controle1:
    if st.button("🧹 Limpar conversa"):
        limpar_conversa()
        st.rerun()

with controle2:
    conversa_md = exportar_conversa_markdown(st.session_state.messages)
    st.download_button(
        "⬇️ Exportar chat",
        data=conversa_md.encode("utf-8"),
        file_name=f"reinove_decision_lab_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
        mime="text/markdown"
    )


def renderizar_mensagem(msg, idx):
    """Renderiza uma mensagem preservada no histórico cronológico."""
    with st.chat_message(msg.get("role", "assistant")):
        st.markdown(
            f'<div class="chat-time">{msg.get("time", "")}</div>',
            unsafe_allow_html=True
        )
        st.markdown(msg.get("content", ""), unsafe_allow_html=True)

        meta = msg.get("meta", {}) or {}

        if msg.get("role") == "assistant" and meta:
            df_msg = meta.get("df")
            sql_msg = meta.get("sql")
            contexto_msg = meta.get("contexto", [])
            territorio_msg = meta.get("territorio")
            bases_msg = meta.get("bases", [])
            tema_msg = meta.get("tema")
            objetivo_msg = meta.get("objetivo")
            poder_msg = meta.get("poder")

            if isinstance(df_msg, pd.DataFrame):
                exibir_tabela_profissional(
                    df_msg,
                    "Dados estruturados consultados",
                    max_linhas=50
                )

            with st.expander("Ver SQL / consulta usada", expanded=False):
                if sql_msg:
                    st.code(sql_msg, language="sql")
                else:
                    st.info("Nenhuma consulta SQL registrada para esta resposta.")

            with st.expander("Ver documentos recuperados pelo RAG", expanded=True):
                exibir_contexto_rag(contexto_msg)

            with st.expander("Ver contexto incorporado", expanded=False):
                st.write("Território:", territorio_msg)
                st.write("Bases:", bases_msg)
                st.write("Tema:", tema_msg)
                st.write("Objetivo:", objetivo_msg)
                st.write("Poder:", poder_msg)


# Renderiza TODO o histórico já salvo, em ordem cronológica.
for idx, msg in enumerate(st.session_state.messages):
    renderizar_mensagem(msg, idx)

rolar_para_baixo()

pergunta = st.chat_input("Pergunte sobre prioridades, riscos, indicadores ou ações recomendadas...")

if pergunta:
    security_in = scan_input(pergunta) if OBSERVABILIDADE_DISPONIVEL else None
    pergunta_original = pergunta
    if security_in is not None:
        pergunta = security_in.sanitized_text

    horario_user = carimbo_agora()

    st.session_state.messages.append({
        "role": "user",
        "content": pergunta,
        "time": horario_user,
        "meta": {}
    })

    if security_in is not None and not security_in.allowed:
        resposta = safe_refusal(security_in)
        contexto, df, sql = [], pd.DataFrame(), None
    else:
      with st.spinner("Analisando bases públicas, anexos, links e dados estruturados..."):
        contexto = buscar_rag(pergunta, docs)

        if pergunta_pede_empregabilidade(pergunta):
            resposta, df, sql = resposta_etapa2_empregabilidade(pergunta, estado, municipio)
        else:
            sql, df = consulta_duckdb(pergunta, tabelas_bases)

            if usar_llm and modelo:
                try:
                    prompt = montar_prompt(
                        pergunta,
                        territorio,
                        tema,
                        objetivo,
                        poder,
                        bases,
                        evidencias_bases,
                        df,
                        contexto
                    )
                    resposta = chamar_llm(base_url, modelo, prompt)
                except Exception as e:
                    resposta = (
                        "<div class='warning-box'>"
                        f"<strong>LLM local indisponível.</strong> Usei a resposta local. Detalhe: {e}"
                        "</div>"
                        + resposta_local(pergunta, territorio, tema, objetivo, bases, df, contexto)
                    )
            else:
                resposta = resposta_local(pergunta, territorio, tema, objetivo, bases, df, contexto)

    if OBSERVABILIDADE_DISPONIVEL:
        security_out = scan_output(str(resposta))
        resposta = security_out.sanitized_text

    horario_assistant = carimbo_agora()

    st.session_state.messages.append({
        "role": "assistant",
        "content": resposta,
        "time": horario_assistant,
        "meta": {
            "sql": sql,
            "df": df,
            "contexto": contexto,
            "territorio": territorio,
            "bases": list(bases),
            "tema": tema,
            "objetivo": objetivo,
            "poder": poder,
        }
    })

    if OBSERVABILIDADE_DISPONIVEL:
        try:
            registrar_interacao(
                pergunta=pergunta_original,
                territorio=territorio,
                resposta=resposta,
                agente="multiagente"
            )
        except Exception as e:
            print(f"Erro ao registrar observabilidade: {e}")

    st.rerun()
