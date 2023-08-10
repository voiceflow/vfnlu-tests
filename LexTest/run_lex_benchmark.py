import csv

import boto3

client = boto3.client('lex-runtime')


def run_test(bot_name, utterance):
	bot_alias = 'TestLocal'
	# Run inference
	response = client.post_text(
		botName=bot_name,
		botAlias=bot_alias,
		userId='UserID1',  # You can specify the user ID as needed
		inputText=utterance
	)

	# Print the bot's response
	intent_name, confidence = response['intentName'], response['nluIntentConfidence']['score']
	print('Bot response: ', intent_name, confidence)
	return intent_name


datasets = ["hwu64_10","clinc150_10","banking77_10","curekart"]
bot_names = ["Hwu64", "Clinc150", "Banking77", "CureKart"]
bot_ids = ["SQ8M15QC3V","LKAWF92WBT","N6XLYC6MEA","ODJLMICUPR"]

def run_dataset_eval_lex(dataset, bot_name):
	with open(dataset, newline='') as lines:
		csv_reader = csv.reader(lines, quotechar='"', delimiter=',', quoting=csv.QUOTE_ALL, skipinitialspace=True)
		next(csv_reader)
		predicted_intents = []
		actual_intents = []
		for l in csv_reader:
			utterance = l[3]
			predicted_intent = run_test(bot_name, utterance)
			actual_intents.append(l[4])
			if predicted_intent == "":
				predicted_intent = "FallbackIntent"
			predicted_intents.append(predicted_intent)

	from sklearn.metrics import f1_score, accuracy_score

	return accuracy_score(actual_intents, predicted_intents), f1_score(actual_intents, predicted_intents,
																	   average='macro')


intent_ids = []


def run_benchmark_lex(datasets, bot_names=bot_ids, local_result_load=False):
	if local_result_load:
		f1_scores = [0,0,0,0]
		accuracy_scores = []
		return f1_scores, accuracy_scores
	f1_scores = []
	accuracy_scores = []
	for dataset, bot_name in zip(datasets, bot_names):
		accuracy, f1 = run_dataset_eval_lex(f"LexTest/data/{dataset}_lex/test.csv",bot_name)
		print("Accuracy", accuracy)
		print("F1 Score: ", f1)
		f1_scores.append(f1)
		accuracy_scores.append(accuracy)
	return f1_scores, accuracy_scores

# [0.61, 0.79]
