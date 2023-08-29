from typing import Any

from sklearn.metrics import accuracy_score, f1_score

from src.benchmarks.VFNLUTests.run_voiceflow_benchmark import read_intents_from_nlu_format
from src.regression_testing.general_voiceflow_helpers import inference


def run_regression_test(project_id:str, version_id1:str, version_id2:str, api_key, benchmark_folder="../benchmarks/VFNLUTests/data/hwu64_10/test") -> dict[str, dict[str, float] | dict[str, int | list[dict[str, Any]]] | dict[str, int | list[dict[str, Any]]]]:
	"""
	Runs a regression test on two versions of a project. The test is run on the benchmark_folder, which contains a set of test utterances.
	Assumes both NLU versions exist and are trained.
	Args:
		project_id: which projects you're testing against
		version_id1: original version of the project to test against, usually a production version
		version_id2: second version of the project to test against, is the newest one, usually a development version
		api_key: DM API key to use
		benchmark_folder: where the benchmarks live

	Returns:
		regression test results

	"""

	intents = read_intents_from_nlu_format(benchmark_folder)
	actual_intents = []
	predicted_intents_version1 = []
	predicted_intents_version2 = []
	regressions = []
	improvements = []
	for intent, utterances in intents.items():
		for utterance in utterances:
			r1 = inference(utterance, project_id, version_id1, api_key)
			r2 = inference(utterance, project_id, version_id2, api_key)
			a1 = r1["payload"]["intent"]["name"]
			a2 = r2["payload"]["intent"]["name"]
			actual_intents.append(intent)
			predicted_intents_version1.append(a1)
			predicted_intents_version2.append(a2)
			if a1 == intent and a1!= a2:
				regressions.append({"intent": intent, "utterance": utterance})
			if a2 == intent and a1 != a2:
				improvements.append({"intent": intent, "utterance": utterance})

	results = dict()
	results[version_id1] = {
			"accuracy": accuracy_score(actual_intents, predicted_intents_version1),
			"f1": f1_score(actual_intents, predicted_intents_version1, average='macro')
		}

	results[version_id2] = {
			"accuracy": accuracy_score(actual_intents, predicted_intents_version2),
			"f1": f1_score(actual_intents, predicted_intents_version2, average='macro')
		}
	results["regressions"] = {"count": len(regressions), "values": regressions}
	results["improvements"] = {"count": len(improvements), "values": improvements}

	return results
