import tempfile
import zipfile
import boto3
import json
import os

def extract_zip(zip_path):
    extracted_files = []
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        for name in zip_ref.namelist():
            if name.endswith('.zip'):
                # Extracting nested ZIP file
                temp_dir = tempfile.mkdtemp()
                nested_zip_path = os.path.join(temp_dir, name)
                with zip_ref.open(name) as nested_zip_file:
                    with open(nested_zip_path, 'wb') as f:
                        f.write(nested_zip_file.read())
                extracted_files.extend(extract_zip(nested_zip_path))
            elif name.endswith('.json'):
                temp_file_path = os.path.join(tempfile.mkdtemp(), name)
                with zip_ref.open(name) as json_file:
                    with open(temp_file_path, 'wb') as f:
                        f.write(json_file.read())
                extracted_files.append(temp_file_path)
    return extracted_files








import boto3

def create_lexv2_intent(botId, botVersion, localeId, intentName, description, sampleUtterances,
                        parentIntentSignature=None, dialogCodeHook=None, fulfillmentCodeHook=None,
                        intentConfirmationSetting=None, intentClosingSetting=None, inputContexts=None,
                        outputContexts=None, kendraConfiguration=None, initialResponseSetting=None):

    client = boto3.client('lexv2-models')

    intent_definition = {
        'botId': botId,
        'botVersion': botVersion,
        'localeId': localeId,
        'intentName': intentName,
        'description': description,
        'sampleUtterances': [{'utterance': utterance} for utterance in sampleUtterances]
    }

    if parentIntentSignature:
        intent_definition['parentIntentSignature'] = parentIntentSignature

    if dialogCodeHook:
        intent_definition['dialogCodeHook'] = dialogCodeHook

    if fulfillmentCodeHook:
        intent_definition['fulfillmentCodeHook'] = fulfillmentCodeHook

    if intentConfirmationSetting:
        intent_definition['intentConfirmationSetting'] = intentConfirmationSetting

    if intentClosingSetting:
        intent_definition['intentClosingSetting'] = intentClosingSetting

    if inputContexts:
        intent_definition['inputContexts'] = inputContexts

    if outputContexts:
        intent_definition['outputContexts'] = outputContexts

    if kendraConfiguration:
        intent_definition['kendraConfiguration'] = kendraConfiguration

    if initialResponseSetting:
        intent_definition['initialResponseSetting'] = initialResponseSetting

    response = client.create_intent(**intent_definition)

    return response

# Example usage:
botVersion=  "DRAFT"
localeId = "en_US"


def create_training_for_dataset(dataset, botId):
    zip_path = f"data/{dataset}-lexv1-model.zip"
    intent_files = extract_zip(zip_path)
    for intent_file in intent_files:
        with open(intent_file, 'r') as intent_file_fp:
            intent_definition = json.load(intent_file_fp)
            response = create_lexv2_intent(botId, botVersion, localeId, intent_definition['resource']['name'], intent_definition['resource']['description'],sampleUtterances=intent_definition['resource']['sampleUtterances'])
            print(response)

datasets = ["banking77_10","hwu64_10","clinc150_10","curekart"]
bot_ids = ["N6XLYC6MEA","SQ8M15QC3V","LKAWF92WBT","ODJLMICUPR"]
i = 0
for dataset,bot_id in zip(datasets,bot_ids):
    create_training_for_dataset(dataset,bot_id)