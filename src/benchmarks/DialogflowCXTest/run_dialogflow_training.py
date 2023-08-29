import csv
import os
import time
from collections import defaultdict
from time import sleep

from dotenv import load_dotenv
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from google.protobuf import field_mask_pb2

from src.benchmarks.config import read_project_config

DEFUALT_FLOW_ID = "00000000-0000-0000-0000-000000000000"
DEFAULT_NONE_INTENT_ID = "00000000-0000-0000-0000-000000000001"
dialogflow_config_data = read_project_config()["dialogflow"]
load_dotenv()
project_id = os.getenv("DF_PROJECT_ID")
location_id = os.getenv("DF_LOCATION_ID")


def create_intent(project_id, agent_id, location_id, intent_display_name, training_phrases_parts,update=False):
    # Create a client
    client = dialogflow.IntentsClient(client_options={"api_endpoint": f"{location_id}-dialogflow.googleapis.com"})

    # Define the parent in the form of the agent
    parent = f"projects/{project_id}/locations/{location_id}/agents/{agent_id}"

    # Define the training phrases

    if update:
        intent_path = client.intent_path(project_id, location_id, agent_id, intent_display_name)
        intent = client.get_intent(name=intent_path)
        intent.training_phrases.clear()
        intent.training_phrases = training_phrases_parts
        response = client.update_intent(intent=intent)
        print(f"Intent updated: {response.name}")
    else:
        # Define the intent
        intent = dialogflow.Intent(
            display_name=intent_display_name,
            training_phrases=training_phrases_parts
        )

        # Create the intent
        response = client.create_intent(
            request={"parent": parent, "intent": intent}
        )

        print(f"Intent created: {response.name}")
    return response


def update_start_page_with_intent(project_id, agent_id, location, flow_id,target_page, intent_name):
    client = dialogflow.PagesClient(client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"})
    flow_client = dialogflow.FlowsClient(client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"})
    target_page_path = client.page_path(project_id, location, agent_id, flow_id, target_page)

    flow_path = flow_client.flow_path(project_id, location, agent_id, flow_id)
    flow = flow_client.get_flow(name=flow_path)

    # Add a transition that uses the specified intent without target page
    new_transition = dialogflow.TransitionRoute(
        intent=intent_name,
        condition='true',
        target_page=target_page_path
        # No target page specified
    )
    flow.transition_routes.append(new_transition)
    update_mask = field_mask_pb2.FieldMask(paths=['transition_routes'])

    update_flow_request = dialogflow.UpdateFlowRequest(
        flow=flow,
        update_mask=update_mask
    )
    response = flow_client.update_flow(update_flow_request)

    print(f'Page updated: {response.name}')


def train_assistant(agent_id,flow_id):
    # Create a client
    flow_client = dialogflow.FlowsClient(client_options={"api_endpoint": f"{location_id}-dialogflow.googleapis.com"})

    flow_path = flow_client.flow_path(project_id, location_id, agent_id, flow_id)
    request = dialogflow.TrainFlowRequest(
        name=flow_path,
    )
    operation = flow_client.train_flow(request=request)

    print("Waiting for operation to complete...")

    response = operation.result()

    # Handle the response
    print(response)


def create_training_for_dataset(dataset_name,agent_id,target_page,is_none=False):
    if is_none:
        suffix = "_none"
    else:
        suffix = ""

    with open(f'data/{dataset_name}_dialogflow_cx{suffix}/train.csv', newline='') as lines:
        csv_reader =  csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
        next(csv_reader)
        intent_dict = defaultdict(list)
        for l in csv_reader:
            intent_dict[l[0]].append(l[2])

    for intent, utterances in intent_dict.items():
        training_phrases_parts = [
            dialogflow.Intent.TrainingPhrase(
                parts=[dialogflow.Intent.TrainingPhrase.Part(text=phrase)],
                repeat_count=1
            ) for phrase in utterances]
        if intent == "None":
            # get fallback intent
            intent_object = create_intent(project_id, agent_id, location_id, DEFAULT_NONE_INTENT_ID, training_phrases_parts,update=True)
        else:
            training_phrases_parts = [
                dialogflow.Intent.TrainingPhrase(
                    parts=[dialogflow.Intent.TrainingPhrase.Part(text=phrase)],
                    repeat_count=1
                ) for phrase in utterances]
            try:
                intent_object = create_intent(project_id, agent_id, location_id, intent, training_phrases_parts,update=False)
                update_start_page_with_intent(project_id, agent_id, location_id, flow_id=DEFUALT_FLOW_ID, intent_name=intent_object.name,target_page=target_page)

                sleep(1)
            except Exception as e:
                print(e)
    start_time = time.time()
    train_assistant(agent_id,DEFUALT_FLOW_ID)
    print(f"Training time: {time.time() - start_time}")


def create_datasets(datasets, agents_ids, target_pages,is_none=False):
    i = 0
    for dataset, agent_id, target_page in zip(datasets, agents_ids, target_pages):
        create_training_for_dataset(dataset, agent_id, target_page,is_none=is_none)


#create_datasets(dialogflow_config_data["datasets"], dialogflow_config_data["agents_ids"], dialogflow_config_data["target_pages"])
create_datasets(dialogflow_config_data["datasets"], dialogflow_config_data["agents_ids_none"], dialogflow_config_data["target_pages_none"],is_none=True)