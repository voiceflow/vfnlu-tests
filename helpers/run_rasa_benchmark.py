import asyncio
import json
from pathlib import Path

import rasa
from rasa.model_testing import test_nlu

def run_rasa_train_test(bench_mark_name:str,retrain=False,local_result_load=True,none_tests=True):
	"""
	Run rasa benchmark on a folder containing rasa files
	"""
	this_path = Path(__file__).parent.parent
	rasa_root = this_path/ "RasaTest"
	if none_tests:
		bench_mark_path = rasa_root/"data"/ (bench_mark_name+"_rasa_none")
		suffix = "_none"
	else:
		bench_mark_path = rasa_root/"data"/ (bench_mark_name+"_rasa")
		suffix = ""
	if not local_result_load:
		if retrain:
			print("starting to train rasa benchmark on " + bench_mark_name)
			rasa.train(domain=bench_mark_path/"domain.yml", config=rasa_root/"config.yml", training_files=str(bench_mark_path/"train.yml"),output=rasa_root/"models",fixed_model_name=bench_mark_name+suffix)
			print("finished training rasa benchmark on "+bench_mark_name)
		print("starting to run rasa benchmark on " + bench_mark_name)
		asyncio.run(test_nlu(model=str(rasa_root/"models"/(bench_mark_name+suffix+".tar.gz")), nlu_data=str(bench_mark_path/"test.yml"), output_directory=rasa_root/"results"/(bench_mark_name+suffix),additional_arguments={}) )
		print("finished running rasa benchmark on "+bench_mark_name)

	with open(str(rasa_root/"results"/(bench_mark_name+suffix)/"intent_report.json")) as f:
		result = json.load(f)
	f1, accuracy = result["macro avg"]["f1-score"], result["accuracy"]
	return f1, accuracy


def run_rasa_benchmark(datasets,retrain=False,local_result_load=False,none_tests=True):
	f1_scores, accuracy_scores = [], []
	for dataset in datasets:
		f1, accuracy = run_rasa_train_test(dataset,retrain,local_result_load,none_tests)
		f1_scores.append(f1)
		accuracy_scores.append(accuracy)

	return f1_scores, accuracy_scores
