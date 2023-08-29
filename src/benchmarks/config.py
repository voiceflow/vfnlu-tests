import json


def read_project_config():
	with open("config.json", "r") as f:
		data_dict = json.load(f)
	return data_dict