label_file = 'label'
seq_in_file = 'seq.in'

# Read the content of both files
with open(f"test_none/{label_file}", 'r') as f1, open(f"test_none/{seq_in_file}", 'r') as f2:
    labels = f1.readlines()
    sequences = f2.readlines()

# Prepare new content without "None_Intent"
new_labels = []
new_sequences = []
for label, sequence in zip(labels, sequences):
    if label.strip() != "None_Intent":
        new_labels.append(label)
        new_sequences.append(sequence)

# Write the new content back to the files
with open(f"test/{label_file}", 'w') as f1, open(f"test/{seq_in_file}", 'w') as f2:
    f1.writelines(new_labels)
    f2.writelines(new_sequences)

print("Lines with 'None_Intent' removed from both files.")