import requests
from src.benchmarks.VFNLUTests.run_voiceflow_benchmark import read_intents_from_nlu_format


def train_nlu(project_id,api_key):
    """

    Args:
        project_id:
        api_key:

    Returns:

    """
    url = f"https://general-service.voiceflow.com/train/{project_id}"
    response = requests.post(url,
        json={},
        headers={"Authorization": api_key},
    )

    # Log the response
    r = response.json()
    print(r)


def get_job_status(project_id,tag,api_key):
    """

    Args:
        project_id:
        tag:
        api_key:

    Returns:

    """
    url = f"https://general-service.voiceflow.com/train/{project_id}/status?tag={tag}"
    response = requests.get(url,
                             json={},
                             headers={"Authorization": api_key},
                             )

    # Log the response
    r = response.json()
    print(r)


def inference(query,project_id,version_id,api_key):
    url = f"https://general-runtime.voiceflow.com/nlu/project/{project_id}/version/{version_id}/inference?query={query}"

    response = requests.get(url,
                             json={},
                             headers={"Authorization": api_key},
                             )

    # Log the response
    r = response.json()
    return r


def run_benchmark(benchmark_folder="../benchmarks/VFNLUTests/data/hwu64_10/test"):
    intents = read_intents_from_nlu_format(benchmark_folder)
    counter = 0
    correct = 0
    for intent, utterances in intents.items():
        for utterance in utterances:
            r = inference(utterance)
            a =  r["payload"]["intent"]["name"]
            if a == intent:
                correct +=1
            counter +=1


def publish(project_id,api_key):
    url = f"https://general-service.voiceflow.com/publish/{project_id}?tag=PRODUCTION"
    response = requests.post(url,
                             json={},
                             headers={"Authorization": api_key},
                             )

    # Log the response
    r = response.json()
    print(r)

