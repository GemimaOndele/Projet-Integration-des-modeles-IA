import pandas as pd 
data = pd.read_csv("fake.csv", "true.csv")
print(data.head())  # Affichage des premières lignes
