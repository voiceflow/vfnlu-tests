import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from src.benchmarks.config import read_project_config
from sklearn.metrics import f1_score, accuracy_score

dialogflow_config_data = read_project_config()["dialogflow"]
load_dotenv()
project_id = os.getenv("DF_PROJECT_ID")
location_id = os.getenv("DF_LOCATION_ID")


def detect_intent(session_path, session_client, text):
	# Create a session client


	# Create the text input
	text_input = dialogflow.TextInput(text=text)

	# Create the query input
	query_input = dialogflow.QueryInput(text=text_input, language_code="en-US")

	# Create the request
	request = dialogflow.DetectIntentRequest(
		session=session_path,
		query_input=query_input,
	)

	# Send the request and receive the response
	response = session_client.detect_intent(request=request)

	# Print the matched intent and confidence score
	print("Matched Intent:", response.query_result.intent.display_name, "Confidence:", response.query_result.intent_detection_confidence)
	return response.query_result.intent.display_name


def run_dataset_eval_dialogflow_cx(dataset, agent_id):
	# Replace these variables with the appropriate values for your setup
	session_id = "1"
	session_client = dialogflow.services.sessions.SessionsClient(client_options={"api_endpoint": f"{location_id}-dialogflow.googleapis.com"})
	session_path = session_client.session_path(project_id, location_id, agent_id, session_id)

	with open(dataset, newline='') as lines:
		csv_reader =  csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		next(csv_reader)
		predicted_intents = []
		actual_intents = []
		none_counter = 0
		for l in csv_reader:
			utterance = l[2]
			predicted_intent = detect_intent(session_path,session_client, utterance)
			actual_intents.append(l[0])
			if predicted_intent == "":
				predicted_intent = "None"
				none_counter += 1
			predicted_intents.append(predicted_intent)
		print("None Counter", none_counter)


	return accuracy_score(actual_intents,predicted_intents), f1_score(actual_intents, predicted_intents, average='macro')


def run_benchmark_dialogflow_cx(datasets=None,agent_ids=None, local_result_load=False,is_none=False):
	if not agent_ids:
		if is_none:
			agent_ids = dialogflow_config_data["agents_ids"]
		else:
			agent_ids = dialogflow_config_data["agents_ids_none"]


	if local_result_load and is_none:
		f1_scores = [0.4079240250458726, 0.8089818636087039, 0.74597812696408, 0.6149963914224815]
		accuracy_scores = [0.4141132776230269, 0.7817777777777778, 0.7386363636363636, 0.7777777777777778]
		return f1_scores, accuracy_scores
	elif local_result_load:
		f1_scores = [0.6885960813594082, 0.8421294611426569, 0.7714419376222091,0.6328873741110872]
		accuracy_scores = [0.6555246053853296, 0.821373028215952, 0.7474025974025974,0.7995642701525054]
		return f1_scores, accuracy_scores

	f1_scores = []
	accuracy_scores = []
	for dataset,agent_id in zip(datasets,agent_ids):
		if is_none:
			suffix = "_none"
		else:
			suffix = ""
		base_path = Path(__file__).parent / "data"
		accuracy,f1 = run_dataset_eval_dialogflow_cx(base_path/f"{dataset}_dialogflow_cx{suffix}"/"test.csv",agent_id)
		print("Accuracy", accuracy)
		print("F1 Score: ",f1)
		f1_scores.append(f1)
		accuracy_scores.append(accuracy)

	return f1_scores, accuracy_scores
