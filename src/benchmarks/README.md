# VFNLU Benchmarks

This library contains a set of benchmarks for evaluating the performance of the Voiceflow, VFNLU model on intent classification. A comparison point is also provided compared to baseline Rasa NLU model.
The benchmarks chose are the popular intent classification benchmarks of HWU64, Banking77, CLINC150 and CureKart. The first three are the 10-shot variants.

## Installation guide
`pip install -r requirements.txt`

## Testing the VFNLU
1. Create a free Voiceflow account.
2. Upload the four benchmarks vf_files (.vf)*
3. After uploading the files, head to the integrations tag on the left side of the screen and copy the DM key.
4. Paste the DM key in a .env file under the corresponding project key name.
5. Train the model either through the VFNLU or calling the train NLU function in `run_voiceflow_benchmark.py`
6. Run the evaluation by running `compare_benchmarks.py`

*the free version of Voiceflow only allows 2 projects at a time, so you will have to delete a benchmark to upload the next one.

## Extending the benchmarks
To extend the benchmarks, you can do the following steps:
1. Add addition benchmarks to VFNLU/data folder. They should have a seq.in file and a label file and have a test and train sub folder.
2. Run the Rasa, DF CX or Lex converters under nlu_rasa.py
3. Run the VF file created under `helpers/create_voiceflow_project.py`
4. Open a pull request with the new datafiles and the new VF files.

## Results
VFNLU outperforms DialogFlow CX (DFCX), Rasa and Lex v2  on the four benchmarks included.
![VFNLU vs Other NLUs](figures/nlu_accuracy.png)



### Impact of the None intent
The None intent in practice helps avoid false positives for unrelated domains. Due to the introduction of this new intent, it reduces the accuracy in benchmarks since they assume no unrelated utterances.
The VFNLU is quite senstive to the introduction of this new intent, so it is important to understand its impact.
When deployed as a hosted option, the VFNLU always has None as its largest intent, impacting true positive rate of some requests.

### Comparison against other VFNLU published benchmarks
The additional difference from the two results, apart from the None intent can be attributed to variations in hyperparameters such as batch size and learning rate.
Below is a comparison between both sets of benchmarks for % accuracy.

![VFNLUNone](figures/VF_accuracy.png)

| Dataset              | HWU64 | CLINC150 | Banking77 | CureKart |
|----------------------|-------|----------|-----------|----------|
| VFNLU Benchmark      | 94.1% | 86.3%    | 86.3%     | 80.6%    |
| VFNLU Prod           | 90.5% | 81.5%    | 81.3%     | 71.4%    |
| Difference           | 3.6%  | 4.8%     | 5.0%      | 9.2%     |
| None intent %        | 2.8%  | 2.5%     | 0.7%      | 19.9%    |
| Unexplained Variance | 0.8%  | 2.3%     | 4.3%      | -10.5%   |

### Comparison Rasa with None intent
Rasa shows minimal changes in accuracy between the base dataset and one with explicit None examples. Rasa in one case performs better with a None intent which is quite strange.
![RasaNone](figures/Rasa_accuracy.png)

| Dataset              | HWU64 | CLINC150 | Banking77 | CureKart |
|----------------------|-------|----------|-----------|----------|
| Rasa Base            | 63.7% | 72.4%    | 67.1%     | 81.6%    |
| Rasa None            | 62.4% | 72.2%    | 69.8%     | 79.8%    |
| Difference           | 1.3%  | 0.2%     | -2.7%     | 1.8%     |
| None intent %        | 1.6%  | 0.4%     | 0.4%      | 3.9%     |
| Unexplained Variance | -0.3% | -0.2%    | - 3.1%    | -2.1%    |

### Comparison DFCX with Enriched None (Fallback) intent 
Dialogflow CX is quite sensitive when enriching their built in None Intent (called Fallback). For DFCX many of the mismatches seem to be redirected to the None intent which is an interesting pattern.
![DFCXNone](figures/DFCX_accuracy.png)

| Dataset              | HWU64  | CLINC150 | Banking77 | CureKart |
|----------------------|--------|--------|-----------|----------|
| DFCX Base            | 68.1%  | 80.6%  | 79.4%     | 79.3%    |
| DFCX None            | 41.4%  | 77.6%  | 74.6%     | 77.1%    |
| Difference           | 26.7%  | 3.0%   | 4.8%      | 2.1%     |
| None intent %        | 40.3%  | 16.9%  | 8.6%      | 8.9%     |
| Unexplained Variance | -13.4% | -13.9% | -3.8%     | -6.8%    |