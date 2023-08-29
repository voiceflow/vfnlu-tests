import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from src.benchmarks.DialogflowCXTest.run_dialogflow_benchmark import run_benchmark_dialogflow_cx
from src.benchmarks.LexTest.run_lex_benchmark import run_benchmark_lex
from src.benchmarks.RasaTest.run_rasa_benchmark import run_rasa_benchmark
from src.benchmarks.VFNLUTests.run_voiceflow_benchmark import run_benchmark_vfnlu
from config import read_project_config

lex_config = read_project_config()["lex"]
df_config = read_project_config()["dialogflow"]
datasets = lex_config["datasets"]
lex_bot_ids = lex_config["bot_ids"]
lex_bot_aliases = lex_config["bot_aliases"]

df_agent_ids = df_config["agents_ids"]
df_agent_none_ids = df_config["agents_ids_none"]


lex_f1_scores, lex_accuracy_scores = run_benchmark_lex(datasets,bot_ids=lex_bot_ids, bot_aliases=lex_bot_aliases,local_result_load=True)
rasa_f1_scores, rasa_accuracy_scores = run_rasa_benchmark(datasets,none_tests=False,local_result_load=True,retrain=False)
rasa_f1_scores_none, rasa_accuracy_scores_none = run_rasa_benchmark(datasets,none_tests=True,local_result_load=True, retrain=False)
vf_f1_scores_none, vf_accuracy_scores_none = run_benchmark_vfnlu(datasets,local_result_load=True,none_tests=True)
vf_f1_scores, vf_accuracy_scores = run_benchmark_vfnlu(datasets,local_result_load=True,none_tests=False)
df_f1_scores, df_accuracy_scores = run_benchmark_dialogflow_cx(datasets,agent_ids=df_agent_ids,local_result_load=False)
df_f1_scores_none, df_accuracy_scores_none = run_benchmark_dialogflow_cx(datasets,agent_ids=df_agent_none_ids,local_result_load=True,is_none=True)



# Combine the F1 scores
f1_scores = rasa_f1_scores+  df_f1_scores + lex_f1_scores + vf_f1_scores
accuracy_scores = rasa_accuracy_scores + df_accuracy_scores + lex_accuracy_scores +  vf_accuracy_scores


# Create a list of model labels
models = ['Rasa'] * len(rasa_f1_scores) +  ['DFCX'] * len(df_f1_scores) +  ['LexV2'] * len(df_f1_scores)+ ['VFNLU'] * len(vf_f1_scores)

datasets_regular = datasets * 4
# Create a DataFrame
df_results = pd.DataFrame({
    'F1 Score': f1_scores,
    'Accuracy': accuracy_scores,
    'Model': models,
    'Dataset': datasets_regular
}).round(3)




plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Dataset', y='Accuracy', hue='Model', data=df_results)
plt.title('Comparison of Accuracy Scores between VFNLU and other NLUs')
for i in ax.containers:
    ax.bar_label(i,)
plt.savefig("nlu_accuracy.png",dpi=300)
plt.show()


for scores in [{"name":"Rasa", "accuracy":[rasa_accuracy_scores,rasa_accuracy_scores_none ],"f1":[rasa_f1_scores,rasa_f1_scores_none]},
               {"name":"DFCX","accuracy": [df_accuracy_scores, df_accuracy_scores_none],"f1": [df_f1_scores,df_f1_scores_none]},
                {"name":"VF","accuracy": [vf_accuracy_scores, vf_accuracy_scores_none],"f1": [vf_f1_scores,vf_f1_scores_none]}
               ]:
    models_none = [scores["name"]] * len(datasets) + [scores["name"] + " None"] * len(datasets)
    datasets_none = datasets * 2
    df_none_results = pd.DataFrame({
        'F1 Score': scores["f1"][0] + scores["f1"][1],
        'Accuracy': scores["accuracy"][0] + scores["accuracy"][1],
        'Model': models_none,
        'Dataset': datasets_none
    }).round(3)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Dataset', y='Accuracy', hue='Model', data=df_none_results)
    plt.title('Comparison of Accuracy Scores with a None intent')
    for i in ax.containers:
        ax.bar_label(i,)
    plt.savefig(scores["name"] + "_accuracy.png", dpi=300)
    plt.show()


    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x='Dataset', y='F1 Score', hue='Model', data=df_none_results)
    plt.title('Comparison of F1 Scores with a None intent')
    for i in ax.containers:
        ax.bar_label(i, )
    plt.savefig(scores["name"] + "_f1.png", dpi=300)
    plt.show()




