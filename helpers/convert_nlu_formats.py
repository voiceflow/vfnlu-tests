import os
from pathlib import Path

import yaml
def convert_dataset_to_rasa(datasets=["banking77_10", "hwu64_10", "clinc150_10"],none_intent_training_utterances=None):
    parent_dir = Path(__file__).parent
    for dataset in datasets:
        #os.mkdir(f"{dataset}_rasa")
        for split in ["train","test"]:
            with open(parent_dir/f"VFNLUTests/data/{dataset}/{split}/label") as f_label:
                with open(parent_dir/f"VFNLUTests/data/{dataset}/{split}/seq.in") as f_utterances:
                    labels = f_label.read().split('\n')
                    utterances = f_utterances.read().split('\n')
                    data = {}
                    for utterance, label in zip(utterances, labels):
                        if label not in data:
                            data[label] = []
                        data[label].append(utterance)
                    if split == "train" and none_intent_training_utterances is not None:
                            data["None"] = none_intent_training_utterances

                    # output the training data in Rasa YAML format
                    suffix = ""
                    if none_intent_training_utterances:
                        suffix = "_none"

                    # check if the directory exists
                    if not os.path.exists(parent_dir/f"RasaTest/data/{dataset}_rasa{suffix}"):
                        os.mkdir(parent_dir/f"RasaTest/data/{dataset}_rasa{suffix}")

                    with open(parent_dir/f"RasaTest/data/{dataset}_rasa{suffix}/{split}.yml","w") as f_rasa:
                        f_rasa.write("version: \"3.1\"""\n\n")
                        f_rasa.write("nlu:\n")
                        for intent, examples in data.items():
                            f_rasa.write("- intent: " + intent + "\n")
                            f_rasa.write("  examples: |\n")
                            for example in examples:
                                f_rasa.write("    - " + example + "\n")

                    with open(parent_dir/f"RasaTest/data/{dataset}_rasa{suffix}/domain.yml","w") as f_domain:
                        f_domain.write("version: \"3.1\"""\n\n")
                        f_domain.write("intents:\n")
                        for l in data.keys():
                            if l != "":
                                f_domain.write(f"- {l}\n")
                        if none_intent_training_utterances is not None:
                            f_domain.write(f"- None\n")



none_utterances = ["I want pizza", "Order a burger", "My name is Ronald", "1234",
                              "What are your store hours?", "aabiusdf89", "905 922 3231","Australia",
                              "What notable sites would you recommend?", "How far is the airport from the city center?",'I want to grab universe twelve', 'And a feeding for James the Great', 'get water pressure tips', 'view tasks for upcoming', 'are there any open recalls on this car', 'show me the video', 'No that is everything', 'Why was I charged for SMS on my mobility bill', 'unusual scents', "Let's go with market validation", 'we have Mental status changes', 'what is showing', 'i want to hear handel', 'Egg Free Adult', 'What bird is that on the left?', 'Can you tell me about Commonwealth Bank of Australia', 'supply chain and logistics', 'Does the event have wires', 'the website', 'Air freshener snap', 'Leo DiCaprio', 'remove the large tree', 'how far can a radar see?', 'i want to raise a issue please', 'Since last May', 'read me a Seagull story', 'remind Doctors Task',
                    'update owner name', 'What was the total cost of the last week by Warehouse', 'Italian sounds good', 'i want duck liver', 'How can I help', 'give me a new question', 'I am cocoa', 'please deliver to 124 fake st Australia Square 1003', 'great place to retire', 'help me calculate my mortgage payments', 'today I applying insecticide on Field 24', 'Where is the butter chicken', 'add meeting on Thursday from six to eight pm on the calendar']

convert_dataset_to_rasa(none_intent_training_utterances=none_utterances)

def convert_dataset_to_dialogflow_cx(datasets=["banking77_10", "hwu64_10", "clinc150_10"]):
    parent_dir = Path(__file__).parent
    for dataset in datasets:
        #os.mkdir(f"{dataset}_rasa")
        for split in ["train","test"]:
            with open(parent_dir/f"VFNLUTests/data/{dataset}/{split}/label") as f_label:
                with open(parent_dir/f"VFNLUTests/data/{dataset}/{split}/seq.in") as f_utterances:
                    labels = f_label.read().split('\n')
                    utterances = f_utterances.read().split('\n')
                    data = {}
                    for utterance, label in zip(utterances, labels):
                        if label not in data:
                            data[label] = []
                        data[label].append(utterance)

                    # output the training data in Rasa YAML format
                    suffix = ""

                    # check if the directory exists
                    if not os.path.exists(parent_dir/f"DialogflowCXTest/data/{dataset}_dialogflow_cx"):
                        os.mkdir(parent_dir/f"DialogflowCXTest/data/{dataset}_dialogflow_cx")

                    with open(parent_dir/f"DialogflowCXTest/data/{dataset}_dialogflow_cx/{split}.json","w") as f_dialogflow:
                        f_dialogflow.write("Intent Display Name,Language,Phrase\n")
                        for intent, examples in data.items():
                            for example in examples:
                                f_dialogflow.write(f"{intent},en,{example}\n")
