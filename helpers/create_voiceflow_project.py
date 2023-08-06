from collections import defaultdict
from pathlib import Path


def convert_rasa_train_to_vf(rasa_train_path, vf_output_path):
    pass

def convert_nlu_train_to_vf(nlu_train_path, vf_output_path, dataset_name):
    import json
    import uuid
    from collections import OrderedDict

    def load_data(file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        return [line.strip() for line in lines]

    def create_intent(name, utterances):
        intent_key = str(uuid.uuid4())[:8].replace("-", "")
        return intent_key,{
            "key": intent_key,
            "name": name,
            "inputs": [{"text": text, "slots": []} for text in utterances],
            "slots": []
        }

    def create_intent_node(node_key, position,intent_id):
        dict_1 = {
                "type": "intent",
                "data": {
                    "name": "",
                    "intent": intent_id,
                    "mappings": [],
                    "availability": "GLOBAL",
                    "portsV2": {
                        "byKey": {},
                        "builtIn": {
                            "next": {
                                "type": "next",
                                "target": None,
                                "id": f"{node_key[:-1]}d"
                            }
                        },
                        "dynamic": []
                    }
                },
                "nodeID": f"{node_key[:-1]}a"
            }


        node_key_block = f"{node_key[:-1]}e"
        dict_2 = {
                "type": "block",
                "data": {
                    "name": "",
                    "steps": [
                        node_key
                    ]
                },
                "nodeID": node_key_block,
                "coords": position
            }

        return f"{node_key[:-1]}a",node_key_block, dict_1, dict_2


    label_data = load_data(nlu_train_path/'label')
    seq_data = load_data(nlu_train_path/'seq.in')
    intent_utterances_dict = defaultdict(list)
    for label, seq in zip(label_data, seq_data):
        intent_utterances_dict[label].append(seq)

    with open('../VFNLUTests/vf_files/vf_mockfile.vf', 'r') as vf_file:
        vf_data = json.load(vf_file, object_pairs_hook=OrderedDict)


    i = 0
    intent_node_key = [{
          "type": "NODE",
          "sourceID": "start00000000000000000000"
        }]
    vf_data['project']['name'] = dataset_name
    for intent, utterances in intent_utterances_dict.items():
        intent_id, intent_dict = create_intent(intent, utterances)
        vf_data['version']['platformData']['intents'].append(intent_dict)

        width = 150
        height = 100
        margin = 300
        per_row = 10
        x = (i % per_row) * (width + margin)
        y = (i // per_row) * (height + margin)
        position = [x, y]
        # shorten to 24 char
        node_key = f"{str(uuid.uuid4()).replace('-', '')}"[:-8]
        node_key,node_key_block, dict_1, dict_2 = create_intent_node(node_key, position,intent_id)
        vf_data['diagrams']['64c97fde09d04078840fe435']['nodes'][node_key] = dict_1
        vf_data['diagrams']['64c97fde09d04078840fe435']['nodes'][node_key_block] = dict_2
        intent_node_key.append({
          "type": "NODE",
          "sourceID": f"{node_key}"
        })
        i += 1

    vf_data['diagrams']['64c97fde09d04078840fe435']['menuItems'] = intent_node_key

    with open(vf_output_path, 'w') as new_vf_file:
        json.dump(vf_data, new_vf_file, indent=2)

for dataset in ["banking77_10","clinc150_10","hwu64_10"]:
    convert_nlu_train_to_vf(Path(f"VFNLUTests/data/{dataset}/train"), Path(f"VFNLUTests/vf_files/{dataset}.vf"),f"{dataset}")


