from azureml.pipeline.core import Pipeline
from azureml.pipeline.steps import PythonScriptStep
from azureml.core import Workspace, Experimento, Ambiente
from azureml.core.runconfig import RunConfiguration

espaco_trabalho = Workspace.from_config()
ambiente = Ambiente.from_conda_specification(name="ambiente-azureml", file_path="ambiente.yml")
config_execucao = RunConfiguration()
config_execucao.environment = ambiente

etapa_treinamento = PythonScriptStep(
    name="Treinar Modelo de Sorvete",
    script_name="treinar_modelo.py",
    compute_target="local",
    source_directory=".",
    runconfig=config_execucao
)

pipeline = Pipeline(workspace=espaco_trabalho, steps=[etapa_treinamento])
experimento = Experimento(espaco_trabalho, "pipeline-treinamento-sorvete")
execucao_pipeline = experimento.submit(pipeline)
execucao_pipeline.wait_for_completion(show_output=True)