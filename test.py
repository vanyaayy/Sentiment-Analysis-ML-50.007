def read_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file: #open 
        for line in file:
            line = line.strip()
            if line:
                word, tag = line.split()
                dataset.append({'word': word, 'tag': tag})
    return dataset
    

file_path = 'Data/ES/train'
training_set = read_dataset(file_path)

def estimate_emission_parameters(training_set, k=1):
    # Initialize dictionaries to store the counts of (x, y) and individual y's
    emission_counts = {}
    tag_counts = {}

    # Count occurrences of each (x, y) pair and each individual y
    for data in training_set:
        x, y = data['word'], data['tag']
        if y not in emission_counts:
            emission_counts[y] = {}
        if x not in emission_counts[y]:
            emission_counts[y][x] = 0
        emission_counts[y][x] += 1

        if y not in tag_counts:
            tag_counts[y] = 0
        tag_counts[y] += 1
    
    # Add the #UNK# token to emission counts for each tag
    for y in emission_counts:
        emission_counts[y]["#UNK#"] = k
        tag_counts[y] += k
    # print("emission", emission_counts)
    #print("tag",tag_counts)
    # Calculate the emission parameters e(x|y) using MLE
    emission_parameters = {}
    for y, word_counts in emission_counts.items():
        emission_parameters[y] = {}
        for x, count in word_counts.items():
            emission_parameters[y][x] = count / tag_counts[y]
        
    return emission_parameters


emission_params = estimate_emission_parameters(training_set)

def read_unlabelled_dataset(file_path):
    dataset = []
    with open(file_path, 'r') as file:
        for line in file:
            word = line.strip()
            if word:
                dataset.append(word)
    return dataset

file_path1 = 'Data/ES/dev.in'
test_set = read_unlabelled_dataset(file_path1)
#print(test_set)
def predict_tags(listt, emission_params, tag_counts):
    keys = emission_params.keys()
    for word in listt:
        max_prob = 0
        max_tag = ''
        for x in keys:
            if word in emission_params[x]:
                prob = emission_params[x][word] * tag_counts[x] / len(training_set)
            else:
                prob = emission_params[x]["#UNK#"] * tag_counts[x] / len(training_set)
            if prob > max_prob:
                max_prob = prob
                max_tag = x
        print(f"{word} {max_tag}")

# Before calling the predict_tags() function, calculate the tag_counts from the training set.
tag_counts = {}
for data in training_set:
    y = data['tag']
    if y not in tag_counts:
        tag_counts[y] = 0
    tag_counts[y] += 1

# Call the predict_tags() function with the updated parameters.
predict_tags(test_set, emission_params, tag_counts)
