def read_dataset(file_path):
    dataset = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                dataset.append(line)
    return dataset

def predict_tags(dataset, emission_params):
    predictions = []
    for line in dataset:
        word = line.strip()
        if word in emission_params:
            tag = max(emission_params[word], key=emission_params[word].get)
        else:
            # If the word is not in the training set, use the #UNK# token
            tag = max(emission_params['#UNK#'], key=emission_params['#UNK#'].get)
        predictions.append(f"{word} {tag}")
    return predictions

# Load the training data and estimate emission parameters for RU dataset
ru_training_set = read_dataset('RU_train.txt')
ru_emission_params = estimate_emission_parameters(ru_training_set, k=1)

# Load the development data for RU dataset
ru_dev_data = read_dataset('RU_dev.in')

# Predict tags for RU dataset
ru_predictions = predict_tags(ru_dev_data, ru_emission_params)

# Write output to dev.p1.out for RU dataset
with open('RU_dev.p1.out', 'w', encoding='utf-8') as file:
    file.write('\n'.join(ru_predictions))

# Load the training data and estimate emission parameters for ES dataset
