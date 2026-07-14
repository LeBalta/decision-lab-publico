import pandas as pd
from pathlib import Path

arquivo = Path("../data/Renda/Tabela 2.18 (Pobr_Geo).xls")

df = pd.read_excel(arquivo, header=None)

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 300)

print(df.iloc[0:12, :])