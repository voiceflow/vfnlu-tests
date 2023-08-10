import csv
import time
from collections import defaultdict
from time import sleep

from google.cloud import dialogflowcx_v3beta1 as dialogflow
from google.protobuf import field_mask_pb2


def create_intent(project_id, agent_id, location, intent_display_name, training_phrases_parts):
    # Create a client
    client = dialogflow.IntentsClient(client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"})

    # Define the parent in the form of the agent
    parent = f"projects/{project_id}/locations/{location}/agents/{agent_id}"

    # Define the training phrases


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
    # Fetch the flow to get the start page
    # start_page_path = client.page_path(project_id, location, agent_id, flow_id, "START_PAGE")
    # start_page = client.get_page(name=start_page_path)
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




project_id = "denys-staging-1111"
location = "us-central1" # example: "us-central1"
flow_id = "00000000-0000-0000-0000-000000000000"
# read data from train.csv
def train_assistant(agent_id):
    # Create a client
    flow_client = dialogflow.FlowsClient(client_options={"api_endpoint": f"{location}-dialogflow.googleapis.com"})

    flow_path = flow_client.flow_path(project_id, location, agent_id, flow_id)
    flow = flow_client.get_flow(name=flow_path)
    request = dialogflow.TrainFlowRequest(
        name=flow_path,
    )
    operation = flow_client.train_flow(request=request)

    print("Waiting for operation to complete...")

    response = operation.result()

    # Handle the response
    print(response)




def create_training_for_dataset(dataset_name,agent_id,target_page):
    with open(f'data/{dataset_name}_dialogflow_cx/train.csv', newline='') as lines:
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
        try:
            intent_object = create_intent(project_id, agent_id, location, intent, training_phrases_parts)
            update_start_page_with_intent(project_id, agent_id, location, flow_id=flow_id, intent_name=intent_object.name,target_page=target_page)

            sleep(1)
        except Exception as e:
            print(e)
    start_time = time.time()
    train_assistant(agent_id)
    print(f"Training time: {time.time() - start_time}")

datasets = ["banking77_10","hwu64_10","clinc150_10","curekart"]
agents_ids = ["0394f14a-83e0-467f-af48-92eedfa48d42","ca4266e9-52f6-43fa-8d00-d25c8fb63e94","00d8ca3c-eae8-4420-8cbc-72ac5207cb77","222d90cc-78b5-41ff-a593-f9e6505bd19d"]
target_pages = ["1b51e650-64ff-49c5-a1aa-ba0dc63d44e9","7e8b7c17-41ad-4dc7-a7e2-71abc4fb7bc8","7ef0ba07-9b4a-4dc7-a4c2-ee9dee722261","457a4037-f6bc-4b5d-b5a5-f466666808e1"]
i = 0
for dataset,agent_id, target_page in zip(datasets,agents_ids,target_pages):
    create_training_for_dataset(dataset,agent_id,target_page)