import csv

import numpy as np
from google.cloud import dialogflowcx_v3beta1 as dialogflow


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
	project_id = "denys-staging-1111"
	location_id = "us-central1"
	session_id = "1"
	session_client = dialogflow.services.sessions.SessionsClient(client_options={"api_endpoint": f"{location_id}-dialogflow.googleapis.com"})
	session_path = session_client.session_path(project_id, location_id, agent_id, session_id)

	with open(dataset, newline='') as lines:
		csv_reader =  csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		next(csv_reader)
		predicted_intents = []
		actual_intents = []
		for l in csv_reader:
			utterance = l[2]
			predicted_intent = detect_intent(session_path,session_client, utterance)
			actual_intents.append(l[0])
			if predicted_intent == "":
				predicted_intent = "None_Intent"
			predicted_intents.append(predicted_intent)


	from sklearn.metrics import f1_score, accuracy_score

	return accuracy_score(actual_intents,predicted_intents), f1_score(actual_intents, predicted_intents, average='macro')

agents_ids = ["ca4266e9-52f6-43fa-8d00-d25c8fb63e94","00d8ca3c-eae8-4420-8cbc-72ac5207cb77","0394f14a-83e0-467f-af48-92eedfa48d42","222d90cc-78b5-41ff-a593-f9e6505bd19d"]
intent_ids = []

def run_benchmark_dialogflow_cx(datasets,agent_ids=agents_ids, local_result_load=False):
	if local_result_load:
		f1_scores = [0.7114360174991043, 0.8292905394131249, 0.8114345728752027,0.6552839455156556] #  0.09374454531706725
		accuracy_scores = [0.680594243268338, 0.805820928682515, 0.7935064935064935,0.7930283224400871 ] # 0.5417369308600337
		return f1_scores, accuracy_scores
	f1_scores = []
	accuracy_scores = []
	i = 0
	#agent_ids = agent_ids[:len(datasets)]
	for dataset,agent_id in zip(datasets,agent_ids):
		accuracy,f1 = run_dataset_eval_dialogflow_cx(f"DialogflowCXTest/data/{dataset}_dialogflow_cx/test.csv",agent_id)
		print("Accuracy", accuracy)
		print("F1 Score: ",f1)
		f1_scores.append(f1)
		accuracy_scores.append(accuracy)

	return f1_scores, accuracy_scores

#datasets = ["banking77_10","hwu64_10","clinc150_10","curekart"]

# run_benchmark_dialogflow_cx(datasets,agents_ids)