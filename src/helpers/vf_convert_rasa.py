import json

def convert_vf_to_rasa(vf_file_path,dataset_name, rasa_file_data='RasaTest'):
    # Load the Voiceflow file
    with open(vf_file_path, 'r') as f:
        vf_data = json.load(f)

    # Extract the intents from the Voiceflow file
    vf_intents = vf_data['version']['platformData']['intents']


    # Convert to Rasa format
    rasa_data = {'nlu': []}
    for intent in vf_intents:
        rasa_intent = {}
        rasa_intent['intent'] = intent['name']
        rasa_intent['examples'] = [input['text'] for input in intent['inputs']]
        rasa_data['nlu'].append(rasa_intent)

    # Save as Rasa file
    with open(f'{rasa_file_data}/data/{dataset_name}/train.yml', 'w') as f_rasa:
        f_rasa.write("version: \"3.1\"""\n\n")
        f_rasa.write("nlu:\n")
        for rasa_intent in rasa_data['nlu']:
            f_rasa.write("- intent: " + rasa_intent['intent'] + "\n")
            f_rasa.write("  examples: |\n")
            for example in rasa_intent['examples']:
                f_rasa.write("    - " + example + "\n")


    with open(f'{rasa_file_data}/data/{dataset_name}/domain.yml', 'w') as f_domain:
        f_domain.write("version: \"3.1\"""\n\n")
        f_domain.write("intents:\n")
        for intent in vf_intents:
            if intent['name'] != "":
                f_domain.write(f"- {intent['name']}\n")
