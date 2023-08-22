import seaborn as sns
import matplotlib.pyplot as plt

from DialogflowCXTest.run_dialogflow_benchmark import run_benchmark_dialogflow_cx
from LexTest.run_lex_benchmark import run_benchmark_lex
from RasaTest.run_rasa_benchmark import run_rasa_benchmark
from VFNLUTests.run_voiceflow_benchmark import run_benchmark_vfnlu

datasets = ["hwu64_10","clinc150_10","banking77_10","curekart"]

lex_f1_scores, lex_accuracy_scores = run_benchmark_lex(datasets,local_result_load=True)
rasa_f1_scores, rasa_accuracy_scores = run_rasa_benchmark(datasets,none_tests=False,local_result_load=True,retrain=False)
rasa_f1_scores_none, rasa_accuracy_scores_none = run_rasa_benchmark(datasets,none_tests=True,local_result_load=True, retrain=False)
vf_f1_scores_none, vf_accuracy_scores_none = run_benchmark_vfnlu(datasets,local_result_load=True,none_tests=True)
vf_f1_scores, vf_accuracy_scores = run_benchmark_vfnlu(datasets,local_result_load=True,none_tests=False)
df_f1_scores, df_accuracy_scores = run_benchmark_dialogflow_cx(datasets,local_result_load=True)
df_f1_scores_none, df_accuracy_scores_none = run_benchmark_dialogflow_cx(datasets,local_result_load=True,is_none=True)


import pandas as pd

# Combine the F1 scores
f1_scores = rasa_f1_scores+ rasa_f1_scores_none +  df_f1_scores + lex_f1_scores + vf_f1_scores + vf_f1_scores_none
accuracy_scores = rasa_accuracy_scores+rasa_accuracy_scores_none + df_accuracy_scores + lex_accuracy_scores +  vf_accuracy_scores + vf_accuracy_scores_none

# Create a list of model labels
models = ['Rasa'] * len(rasa_f1_scores) +['RasaNone'] * len(rasa_f1_scores) +  ['DFCX'] * len(df_f1_scores) +  ['LexV2'] * len(df_f1_scores)+ ['VFNLU'] * len(vf_f1_scores) + ['VFNLUNone'] * len(vf_f1_scores)

datasets = datasets * 6
# Create a DataFrame
df = pd.DataFrame({
    'F1 Score': f1_scores,
    'Accuracy': accuracy_scores,
    'Model': models,
    'Dataset': datasets
}).round(3)


plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Dataset', y='Accuracy', hue='Model', data=df)
plt.title('Comparison of Accuracy Scores between VFNLU and other NLUs')
for i in ax.containers:
    ax.bar_label(i,)
plt.show()


