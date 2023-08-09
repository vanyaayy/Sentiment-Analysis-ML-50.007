from collections import defaultdict
from queue import PriorityQueue

def estimate_emission_parameters(train_path, k=1):
    token_tag_count = defaultdict(int)
    tag_count = defaultdict(int)
    with open(train_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token, tag = line.rsplit(' ', 1)
                token_tag_count[(token, tag)] += 1
                tag_count[tag] += 1
    emission_parameters = defaultdict(lambda: defaultdict(float))
    for (token, tag), count in token_tag_count.items():
        emission_parameters[token][tag] = count / (tag_count[tag] + k)
        emission_parameters['#UNK#'][tag] = k / (tag_count[tag] + k)
    return emission_parameters

def estimate_transition_parameters(train_path):
    transition_count = defaultdict(int)
    tag_count = defaultdict(int)
    START = 'START'
    STOP = 'STOP'
    with open(train_path, 'r', encoding='utf-8') as file:
        prev_tag = START
        for line in file:
            line = line.strip()
            if line:
                _, tag = line.rsplit(' ', 1)
                transition_count[(prev_tag, tag)] += 1
                tag_count[prev_tag] += 1
                prev_tag = tag
            else:
                transition_count[(prev_tag, STOP)] += 1
                tag_count[prev_tag] += 1
                prev_tag = START
    tag_count[STOP] = 0
    transition_parameters = defaultdict(lambda: defaultdict(float))
    for (tag1, tag2), count in transition_count.items():
        transition_parameters[tag1][tag2] = count / tag_count[tag1]
    return transition_parameters


def k_viterbi(emission_parameters, transition_parameters, sentence, k):
    words = sentence.split()
    n = len(words)
    
    viterbi_matrix = defaultdict(lambda: defaultdict(list))
    
    viterbi_matrix[0]['START'] = [(1.0, ['START'])]
    
    for t in range(1, n+1):
        word = words[t-1]
        if word not in emission_parameters:
            word = '#UNK#'
        for v in emission_parameters[word]:
            paths_to_v = []
            for u in viterbi_matrix[t-1]:
                for score, path in viterbi_matrix[t-1][u]:
                    new_score = score * transition_parameters[u][v] * emission_parameters[word][v]
                    new_path = path + [v]
                    paths_to_v.append((new_score, new_path))
            viterbi_matrix[t][v] = sorted(paths_to_v, reverse=True)[:k]
                    
    opt_paths = []
    for tag in viterbi_matrix[n]:
        for score, path in viterbi_matrix[n][tag]:
            opt_paths.append((score, path[1:]))
            
    sorted_paths = sorted(opt_paths, reverse=True)
    if k <= len(sorted_paths):
        return sorted_paths[k-1][1]
    else:
        default = []
        for i in range (0,n) :
            default.append('O')
        return default

###ES, k=2#####
train_path = "Data/ES/train"
dev_in_path = "Data/ES/dev.in"
dev_predicted_path = "Data/ES/dev.p3.2nd.out"
    
emission_params = estimate_emission_parameters(train_path)
transition_params = estimate_transition_parameters(train_path)

sentences = ""   
with open(dev_in_path, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path, 'w', encoding='utf-8') as file_write:
        for line in file:
            word = line.strip()
            if word:
                sentences+= word + " "
            else:
                predicted_tags = k_viterbi(emission_params, transition_params, sentences, 2)
                for word, tag in zip(sentences.split(), predicted_tags):
                    file_write.write(f"{word} {tag}\n")
                file_write.write("\n")
                sentences=""

###ES, k=8#####
train_path = "Data/ES/train"
dev_in_path = "Data/ES/dev.in"
dev_predicted_path = "Data/ES/dev.p3.8th.out"
    
emission_params = estimate_emission_parameters(train_path)
transition_params = estimate_transition_parameters(train_path)

sentences = ""   
with open(dev_in_path, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path, 'w', encoding='utf-8') as file_write:
        for line in file:
            word = line.strip()
            if word:
                sentences+= word + " "
            else:
                predicted_tags = k_viterbi(emission_params, transition_params, sentences, 8)
                for word, tag in zip(sentences.split(), predicted_tags):
                    file_write.write(f"{word} {tag}\n")
                file_write.write("\n")
                sentences=""


###RU, k=2#####
train_path = "Data/RU/train"
dev_in_path = "Data/RU/dev.in"
dev_predicted_path = "Data/RU/dev.p3.2nd.out"
    
emission_params = estimate_emission_parameters(train_path)
transition_params = estimate_transition_parameters(train_path)

sentences = ""   
with open(dev_in_path, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path, 'w', encoding='utf-8') as file_write:
        for line in file:
            word = line.strip()
            if word:
                sentences+= word + " "
            else:
                predicted_tags = k_viterbi(emission_params, transition_params, sentences, 2)
                for word, tag in zip(sentences.split(), predicted_tags):
                    file_write.write(f"{word} {tag}\n")
                file_write.write("\n")
                sentences=""

###RU, k=8#####
train_path = "Data/RU/train"
dev_in_path = "Data/RU/dev.in"
dev_predicted_path = "Data/RU/dev.p3.8th.out"
    
emission_params = estimate_emission_parameters(train_path)
transition_params = estimate_transition_parameters(train_path)

sentences = ""   
with open(dev_in_path, 'r', encoding='utf-8') as file:
    with open(dev_predicted_path, 'w', encoding='utf-8') as file_write:
        for line in file:
            word = line.strip()
            if word:
                sentences+= word + " "
            else:
                predicted_tags = k_viterbi(emission_params, transition_params, sentences, 8)
                for word, tag in zip(sentences.split(), predicted_tags):
                    file_write.write(f"{word} {tag}\n")
                file_write.write("\n")
                sentences=""