import os
from collections import defaultdict
from pathlib import Path
from dotenv import load_dotenv

import requests

# View our quick start guide to get your API key:
# https://www.voiceflow.com/api/dialog-manager#section/Quick-Start

load_dotenv()
banking_api_key = os.getenv("BANKING_API_KEY")
hwu_api_key = os.getenv("HWU_API_KEY")
clinc_api_key = os.getenv("CLINC_API_KEY")

import yaml
def read_intents_from_rasa_test_file(file_path):
    intent_dict =defaultdict(list)
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

        for item in data['nlu']:
            utterances = [i.strip('\n')  for i in item['examples'].split('- ') if i != '']
            intent_dict[item['intent']] = utterances

    return intent_dict

def read_intents_from_nlu_format(folder_path):
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

#TODO add F1 score per intent
def run_vf_intent_test(intents_dict,api_key):
    correct = 0
    total_utterances = 0
    predicted_intents = []
    actual_intents = []
    for intent, utterances in intents_dict.items():
        for utterance in utterances:
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
            print("real intent: " + intent," predicted intent: " +matched_intent, r['request']["payload"]["confidence"])

    #f1 score
    from sklearn.metrics import f1_score
    return correct / total_utterances, f1_score(actual_intents, predicted_intents, average='macro')


import aiohttp
import asyncio
from uuid import uuid4

def run_vf_intent_test_rasa_file(file_path):
    intents = read_intents_from_rasa_test_file(file_path)
    run_vf_intent_test(intents)


def run_vf_intent_test_nlu_format(file_path,api_key):
    intents = read_intents_from_nlu_format(file_path)
    return run_vf_intent_test(intents,api_key)


def run_benchmark_vfnlu(datasets,local_result_load=True):
    if local_result_load:
        f1_scores = [0.9022871900713476, 0.8150565317645434, 0.8067784206093752]
        accuracy_scores = [0.9052044609665427, 0.8150565317645434, 0.8133116883116883]
        return f1_scores, accuracy_scores
    f1_scores = []
    accuracy_scores = []
    for dataset, key in zip(datasets, [hwu_api_key,clinc_api_key,banking_api_key]):
        accuracy,f1 = run_vf_intent_test_nlu_format(f"./VFNLUTests/data/{dataset}/test",key)
        print("Accuracy", accuracy)
        print("F1 Score: ",f1)
        f1_scores.append(f1)
        accuracy_scores.append(accuracy)

    return f1_scores, accuracy_scores

