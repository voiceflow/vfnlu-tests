import datetime
import json
from typing import Any
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
import pandas as pd
from src.benchmarks.VFNLUTests.run_voiceflow_benchmark import read_intents_from_nlu_format
from src.regression_testing.general_voiceflow_helpers import inference


def create_confusion_matrix(y_true, y_pred, benchmark_name):
	"""
	Create a confusion matrix plot and save it to the benchmark folder
	Args:
		y_true: True intents
		y_pred: Predicted intents
		benchmark_name: path to results

	Returns:
		None

	"""
	# Get unique class names
	classes = np.union1d(np.unique(y_true), np.unique(y_pred))

	# Create a confusion matrix
	cm = confusion_matrix(y_true, y_pred, labels=classes)

	# Use seaborn to plot the confusion matrix
	plt.figure(figsize=(8, 6))
	sns.set(font_scale=1.2)
	heatmap = sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=classes, yticklabels=classes)
	heatmap.set(xlabel='Predicted Labels', ylabel='Test Labels')
	plt.title(benchmark_name)

	# Show the plot
	plt.savefig(benchmark_name + "_confusion_matrix.png")
	plt.show()


def run_regression_test(project_id:str, project_id2:str, version_id1:str, version_id2:str, api_key1,api_key2=None, benchmark_folder="../benchmarks/VFNLUTests/data/hwu64_10/test",confidence_threshold=0.9) -> dict[str, dict[str, float] | dict[str, int | list[dict[str, Any]]] | dict[str, int | list[dict[str, Any]]]]:
	"""
	Runs a regression test on two versions of a project. The test is run on the benchmark_folder, which contains a set of test utterances.
	Assumes both NLU versions exist and are trained.
	Args:
		project_id: which project you're testing against
		project_id: if None, then will assume you run against the same project
		version_id1: original version of the project to test against, usually a production version
		version_id2: second version of the project to test against, is the newest one, usually a development version
		api_key: DM API key to use
		benchmark_folder: where the benchmarks live
		confidence_threshold: confidence below this threshold are set to None

	Returns:
		regression test results, JSON and pandas dataframe

	"""

	intents = read_intents_from_nlu_format(benchmark_folder)
	actual_intents = []
	predicted_intents_version1 = []
	predicted_intents_version2 = []
	regressions = []
	improvements = []
	incorrect = []
	both_incorrect = []
	all_utterances = []
	confidence_1 = []
	confidence_2 = []
	if project_id2 is None:
		print("No second project id provided, assuming you're testing against the same project")
		project_id2 = project_id
		api_key2 = api_key1

	for intent, utterances in intents.items():
		for utterance in utterances:
			r1 = inference(utterance, project_id, version_id1, api_key1)
			r2 = inference(utterance, project_id2, version_id2, api_key2)
			a1 = r1["payload"]["intent"]["name"]
			a2 = r2["payload"]["intent"]["name"]
			if confidence_threshold is not None:
				c1 = r1["payload"]["confidence"]
				c2 = r2["payload"]["confidence"]
				if c1 < confidence_threshold:
					a1 = "None"
					c1 = 1 - c1
				if c2 < confidence_threshold:
					a2 = "None"
					c2 = 1 - c2

			confidence_1.append(c1)
			confidence_2.append(c2)

			actual_intents.append(intent)
			all_utterances.append(utterance)
			predicted_intents_version1.append(a1)
			predicted_intents_version2.append(a2)
			if a1 == intent and a1 != a2:
				regressions.append({"old_predicted_intent": a1,"correct_intent":intent, "utterance": utterance})
			if a2 == intent and a1 != a2:
				improvements.append({"new_predicted_intent": a2,"correct_intent":intent, "utterance": utterance})

			if a1 != intent or a2 != intent:
				incorrect.append({"old_predicted_intent": a1, "new_predicted_intent": a2, "correct_intent": intent, "utterance": utterance})
			if a1 != intent and a2 != intent:
				both_incorrect.append({"old_predicted_intent": a1, "new_predicted_intent": a2, "correct_intent": intent, "utterance": utterance})



	results = dict()
	results[version_id1] = {
			"accuracy": accuracy_score(actual_intents, predicted_intents_version1),
			"f1": f1_score(actual_intents, predicted_intents_version1,average='weighted'),
			"precision": precision_score(actual_intents, predicted_intents_version1,average='weighted'),
			"recall": recall_score(actual_intents, predicted_intents_version1,average='weighted')
		}

	results[version_id2] = {
			"accuracy": accuracy_score(actual_intents, predicted_intents_version2),
			"f1": f1_score(actual_intents, predicted_intents_version2,average='weighted'),
			"precision": precision_score(actual_intents, predicted_intents_version2,average='weighted'),
			"recall": recall_score(actual_intents, predicted_intents_version2,average='weighted')
		}
	results["regressions"] = {"count": len(regressions), "values": regressions}
	results["improvements"] = {"count": len(improvements), "values": improvements}
	results["both_incorrect"] = {"count": len(improvements), "values": incorrect}
	results["any_incorrect"] = {"count": len(improvements), "values": incorrect}
	results["raw"] = {"utterances": all_utterances,"actual": actual_intents, "predicted_version1": predicted_intents_version1, "predicted_version2": predicted_intents_version2, "confidence_1": confidence_1, "confidence_2": confidence_2}
	df = pd.DataFrame(results["raw"])
	df["correct1"] = df["actual"] == df["predicted_version1"]
	df["correct2"] = df["actual"] == df["predicted_version2"]

	benchmark_name = benchmark_folder.split("/")[-1] + "_" + str(datetime.datetime.now())

	with open(f"{benchmark_name}.json", "w") as f:
		json.dump(results, f)
	df.to_csv(f"{benchmark_name}.csv", index=False)
	create_confusion_matrix(df["actual"], df["predicted_version1"],benchmark_name + "_version1")
	create_confusion_matrix(df["actual"], df["predicted_version2"],benchmark_name + "_version2")
	return results, df

