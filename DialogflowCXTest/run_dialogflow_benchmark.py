import csv
import os
from pathlib import Path
from dotenv import load_dotenv
from google.cloud import dialogflowcx_v3beta1 as dialogflow
from config import read_project_config
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
		f1_scores = [0.4038897971948421, 0.7987145065110172, 0.7662692956454572, 0.5989294749178822]
		accuracy_scores = [0.41225626740947074, 0.7762719395689847, 0.7461038961038962, 0.7712418300653595]
		return f1_scores, accuracy_scores
	elif local_result_load:
		f1_scores = [0.7114360174991043, 0.8292905394131249, 0.8114345728752027,0.6552839455156556]
		accuracy_scores = [0.680594243268338, 0.805820928682515, 0.7935064935064935,0.7930283224400871 ]
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
