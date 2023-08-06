import seaborn as sns
import matplotlib.pyplot as plt
from helpers.run_rasa_benchmark import run_rasa_benchmark
from helpers.run_voiceflow_benchmark import run_benchmark_vfnlu

datasets = ["hwu64_10","clinc150_10","banking77_10"]
rasa_f1_scores, rasa_accuracy_scores = run_rasa_benchmark(datasets,none_tests=False,local_result_load=True)
#rasa_f1_scores_none, rasa_accuracy_scores_none = run_rasa_benchmark(datasets,none_tests=True)
vf_f1_scores, vf_accuracy_scores = run_benchmark_vfnlu(datasets)


import pandas as pd

# Combine the F1 scores
f1_scores = rasa_f1_scores + vf_f1_scores

# Create a list of model labels
models = ['Rasa'] * len(rasa_f1_scores) + ['VFNLU'] * len(vf_f1_scores)

datasets = datasets * 2
# Create a DataFrame
df = pd.DataFrame({
    'F1 Score': f1_scores,
    'Model': models,
    'Dataset': datasets
})

plt.figure(figsize=(10, 6))
ax = sns.barplot(x='Dataset', y='F1 Score', hue='Model', data=df)
plt.title('Comparison of F1 Scores between Rasa and VFNLU')
for i in ax.containers:
    ax.bar_label(i,)
plt.show()


