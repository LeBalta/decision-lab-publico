import pandas as pd
from pathlib import Path

arquivo = Path("../data/Renda/Tabela 2.18 (Pobr_Geo).xls")

df = pd.read_excel(arquivo, header=None)

print(df.head(20))
print("\n")
print(df.shape)