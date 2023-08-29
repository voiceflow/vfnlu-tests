import csv

import boto3

def run_test(client,bot_id,bot_alias, utterance):
	# Run inference
	response = client.recognize_text(
		botId=bot_id,
		botAliasId=bot_alias,
		localeId='en_US',
		sessionId='UserID1',  # You can specify the user ID as needed
		text=utterance
	)

	# Print the bot's response
	confidence = 0
	intent_name = response["sessionState"]['intent']['name']
	if 'nluConfidence' in response['interpretations'][0]:
		confidence =  response['interpretations'][0]['nluConfidence']['score']
	print('Bot response: ', intent_name, confidence)
	return intent_name


def run_dataset_eval_lex(dataset, bot_id,bot_alias):
	client = boto3.client('lexv2-runtime')
	with open(dataset, newline='') as lines:
		csv_reader = csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		next(csv_reader)
		predicted_intents = []
		actual_intents = []
		for l in csv_reader:
			utterance = l[3]
			predicted_intent = run_test(client,bot_id,bot_alias, utterance)
			actual_intents.append(l[4])
			if predicted_intent == "":
				predicted_intent = "FallbackIntent"
			predicted_intents.append(predicted_intent)

	from sklearn.metrics import f1_score, accuracy_score

	return accuracy_score(actual_intents, predicted_intents), f1_score(actual_intents, predicted_intents,
																	   average='macro')


def run_benchmark_lex(datasets, bot_ids,bot_aliases, local_result_load=False):
	if local_result_load:
		f1_scores = [0.6241867003288128,0.6669900181234607,0.755719049261228,0.6241867003288128]
		accuracy_scores = [0.7952069716775599,0.5668888888888889,0.7100649350649351,0.7952069716775599]
		return f1_scores, accuracy_scores
	f1_scores = []
	accuracy_scores = []
	for dataset, bot_id, bot_alias in zip(datasets, bot_ids, bot_aliases):
		accuracy, f1 = run_dataset_eval_lex(f"LexTest/data/{dataset}_lex/test.csv",bot_id,bot_alias)
		print("Accuracy", accuracy)
		print("F1 Score: ", f1)
		f1_scores.append(f1)
		accuracy_scores.append(accuracy)
	return f1_scores, accuracy_scores

