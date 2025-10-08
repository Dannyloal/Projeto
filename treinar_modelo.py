from azureml.core import Workspace, Experimento, Execucao, Modelo
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

espaco_trabalho = Workspace.from_config()
experimento = Experimento(workspace=espaco_trabalho, name="previsao-vendas-sorvete")
execucao = experimento.start_logging()

dados = pd.DataFrame({
    "temperatura": [20, 22, 25, 27, 30, 32, 35],
    "vendas": [100, 120, 150, 170, 200, 220, 250]
})

X = dados[["temperatura"]]
y = dados["vendas"]

X_treino, X_teste, y_treino, y_teste = train_test_split(X, y, test_size=0.2, random_state=42)

modelo = LinearRegression()
modelo.fit(X_treino, y_treino)

previsoes = modelo.predict(X_teste)
erro_medio = mean_squared_error(y_teste, previsoes)

mlflow.log_metric("erro_medio_quadratico", erro_medio)
mlflow.sklearn.log_model(modelo, "modelo_sorvete")

Modelo.register(workspace=espaco_trabalho, model_path="modelo_sorvete", model_name="modelo-sorvete")

execucao.complete()

print("✅ Treinamento concluído e modelo registrado com sucesso!")