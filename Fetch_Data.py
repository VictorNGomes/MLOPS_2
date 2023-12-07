
from pipeline import depends_on

def fetch_data():   
    import wandb
    import requests

    def baixar_arquivo(url, caminho_destino):
        """
        Baixa um arquivo da URL especificada e o salva no caminho de destino.

        Parameters:
        - url (str): URL do arquivo a ser baixado.
        - caminho_destino (str): Caminho local onde o arquivo será salvo.
        """
        # Baixa o arquivo da URL
        response = requests.get(url)

        # Verifica se a requisição foi bem-sucedida (código 200)
        if response.status_code == 200:
            # Salva o conteúdo do arquivo no caminho especificado
            with open(caminho_destino, 'wb') as file:
                file.write(response.content)
            print(f"O arquivo foi baixado e salvo em {caminho_destino}")
        else:
            print(f"A requisição falhou com o código de status: {response.status_code}")

    url_do_arquivo = "https://dsserver-prod-resources-1.s3.amazonaws.com/nlp/train.csv"
    file_path = "train.csv"

    baixar_arquivo(url_do_arquivo, file_path)



    # Inicialize o WandB
    wandb.init(project='sentiment_analysis', job_type='generate_vocab')

    # Defina os parâmetros do artefato
    artifact_name = "disaster_tweets"
    artifact_type = "RawData"
    artifact_description = "Tweets about real disasters or not"

    # Crie um novo artefato
    artifact = wandb.Artifact(name=artifact_name, type=artifact_type, description=artifact_description)

    # Adicione os arquivos ao artefato
    artifact.add_file(file_path)

    # Finalize o artefato
    wandb.log_artifact(artifact)
