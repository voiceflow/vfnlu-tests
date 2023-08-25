import os
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv
import yaml
import requests
from uuid import uuid4

load_dotenv()
banking_api_key = os.getenv("BANKING_API_KEY")
hwu_api_key = os.getenv("HWU_API_KEY")
clinc_api_key = os.getenv("CLINC_API_KEY")
curekart_api_key = os.getenv("CUREKART_API_KEY")
keys = [hwu_api_key, clinc_api_key, banking_api_key, curekart_api_key]

def read_intents_from_rasa_test_file(file_path:str):
    """
    Reads data from rasa format
    Args:
        file_path:

    Returns:

    """
    intent_dict =defaultdict(list)
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

        for item in data['nlu']:
            utterances = [i.strip('\n')  for i in item['examples'].split('- ') if i != '']
            intent_dict[item['intent']] = utterances

    return intent_dict


def read_intents_from_nlu_format(folder_path:str):
    """
    Expected structure, seq.in and labels.txt
    """
    intent_dict =defaultdict(list)
    folder_path = Path(folder_path)
    with open(folder_path/"seq.in", 'r') as f_seq:
        with open(folder_path/"label", 'r') as f_labels:
            for utterance, intent in zip(f_seq.readlines(), f_labels.readlines()):
                intent_dict[intent.strip()].append(utterance.strip())

    return intent_dict


def run_vf_intent_test(intents_dict: dict[str,list[str]],api_key:str):
    """
    Run throught all the intents and utterances in the intents_dict and return the accuracy and f1
    Args:
        intents_dict:
        api_key:

    Returns:
        accuracy, f1

    """
    correct = 0
    total_utterances = 0
    predicted_intents = []
    actual_intents = []
    for intent, utterances in intents_dict.items():
        for utterance in utterances:
            # uuid for each user
            user_id = uuid4()
            user_input = utterance  # User's message to your Voiceflow assistant
            total_utterances +=1
            body = {"action": {"type": "text", "payload": user_input}}

            # Start a conversation
            response = requests.post(
                f"https://general-runtime.voiceflow.com/state/user/{user_id}/interact?verbose=true",
                json=body,
                headers={"Authorization": api_key},
            )

            # Log the response
            r = response.json()
            matched_intent = r['request']["payload"]["intent"]["name"]
            predicted_intents.append(matched_intent)
            actual_intents.append(intent)
            if intent == matched_intent:
                correct += 1
            try:
                print("real intent: " + intent," predicted intent: " +matched_intent, r['request']["payload"]["confidence"])

            except Exception as e:
                print(e)

    #f1 score
    from sklearn.metrics import f1_score
    return correct / total_utterances, f1_score(actual_intents, predicted_intents, average='macro')


def run_vf_intent_test_rasa_file(file_path):
    intents = read_intents_from_rasa_test_file(file_path)
    run_vf_intent_test(intents)


def run_vf_intent_test_nlu_format(file_path,api_key):
    intents = read_intents_from_nlu_format(file_path)
    return run_vf_intent_test(intents,api_key)


def run_benchmark_vfnlu(datasets,local_result_load=False,none_tests=True):
    if not none_tests:
        f1_scores = [0, 0, 0,0]  # 0.3752362810038506
        accuracy_scores = [0.941, 0.863, 0.863,0.806]  # 0.5317860746720484
        return f1_scores, accuracy_scores
    if local_result_load:
        f1_scores = [0.9022871900713476, 0.8150565317645434, 0.8067784206093752,  0.5624212155271786] #0.3752362810038506
        accuracy_scores = [0.9052044609665427, 0.8150565317645434, 0.8133116883116883,0.7145969498910676 ] # 0.5317860746720484
        return f1_scores, accuracy_scores
    f1_scores = []
    accuracy_scores = []
    base_path = Path(__file__).parent / "data"
    for dataset, key in zip(datasets,keys):
        accuracy,f1 = run_vf_intent_test_nlu_format(str(base_path/ dataset/ "test"),key)
        print("Accuracy", accuracy)
        print("F1 Score: ",f1)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    return f1_scores, accuracy_scores