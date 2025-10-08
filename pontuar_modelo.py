import json
import numpy as np
import joblib
from azureml.core.model import Model

def init():
    global modelo
    caminho_modelo = Model.get_model_path("modelo-sorvete")
    modelo = joblib.load(caminho_modelo)

def run(dados_recebidos):
    dados = json.loads(dados_recebidos)
    temperaturas = np.array(dados["temperatura"]).reshape(-1, 1)
    previsao = modelo.predict(temperaturas)
    return json.dumps({"previsao_vendas": previsao.tolist()})