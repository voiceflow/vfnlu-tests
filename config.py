def read_project_config():
	data_dict = {
		"lex": {
			"datasets": ["hwu64_10", "clinc150_10", "banking77_10", "curekart"],
			"bot_names": ["Hwu64", "Clinc150", "Banking77", "CureKart"],
			"bot_aliases": ["TSTALIASID", "TSTALIASID", "TSTALIASID", "TSTALIASID"],
			"bot_ids": ["SQ8M15QC3V", "LKAWF92WBT", "N6XLYC6MEA", "ODJLMICUPR"]
		},
		"dialogflow": {
			"datasets": ["hwu64_10", "clinc150_10", "banking77_10", "curekart"],
			"agents_ids": ["ca4266e9-52f6-43fa-8d00-d25c8fb63e94","00d8ca3c-eae8-4420-8cbc-72ac5207cb77",
						   "0394f14a-83e0-467f-af48-92eedfa48d42" "222d90cc-78b5-41ff-a593-f9e6505bd19d"],
			"target_pages": ["7e8b7c17-41ad-4dc7-a7e2-71abc4fb7bc8","7ef0ba07-9b4a-4dc7-a4c2-ee9dee722261",
							 "1b51e650-64ff-49c5-a1aa-ba0dc63d44e9", "457a4037-f6bc-4b5d-b5a5-f466666808e1"],
			"agents_ids_none": ["7da5a9c1-2839-49fd-8aef-8afa4cf6743b","61bd91c7-ddf1-4300-b751-bea13f578304",
								"09ad8cd3-b3b0-4447-8355-ca9709f028cf","97f8a767-3e64-43eb-851e-a97f56768ba5"],
			"target_pages_none": ["1a9fffa8-4dcf-4071-ad4b-05c00e6bd0c5","ac9644e5-622b-41f3-ab3d-510886bcb92a",
								  "29f5f24e-0ffd-4540-870b-ab8fabd0e9fe", "68db0e3d-bdcc-41d5-8a9a-e4b70fd4320c"]
		}
	}
	return data_dict




