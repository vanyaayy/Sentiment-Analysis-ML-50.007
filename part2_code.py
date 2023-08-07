from collections import defaultdict

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

def viterbi_algorithm(emission_parameters, transition_parameters, dev_in_path, dev_out_path):
    output_lines = []
    sentence_tokens = []
    with open(dev_in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                sentence_tokens.append(line)
            else:
                n = len(sentence_tokens)
                pi = defaultdict(lambda: defaultdict(float))
                backpointer = defaultdict(lambda: defaultdict(str))
                pi[0]['START'] = 1
                for k in range(1, n + 1):
                    token = sentence_tokens[k - 1]
                    if token not in emission_parameters:
                        token = '#UNK#'
                    for v in emission_parameters[token]:
                        max_prob = 0
                        max_u = ''
                        for u in transition_parameters:
                            prob = pi[k - 1][u] * transition_parameters[u][v] * emission_parameters[token][v]
                            if prob > max_prob:
                                max_prob = prob
                                max_u = u
                        pi[k][v] = max_prob
                        backpointer[k][v] = max_u
                max_prob = 0
                max_u = ''
                for u in transition_parameters:
                    prob = pi[n][u] * transition_parameters[u]['STOP']
                    if prob > max_prob:
                        max_prob = prob
                        max_u = u
                backpointer[n + 1]['STOP'] = max_u
                best_tags = ['STOP']
                for k in range(n + 1, 1, -1):
                    best_tags.append(backpointer[k][best_tags[-1]])
                best_tags = best_tags[::-1][1:]
                for token, tag in zip(sentence_tokens, best_tags):
                    output_lines.append(f"{token} {tag}")
                output_lines.append("")
                sentence_tokens = []
    with open(dev_out_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))

###ES####
emission_params_ES = estimate_emission_parameters("Data/ES/train")
transition_params_ES = estimate_transition_parameters("Data/ES/train")
viterbi_algorithm(emission_params_ES, transition_params_ES, "Data/ES/dev.in", "Data/ES/dev.p2.out")

###RU####
emission_params_RU = estimate_emission_parameters("Data/RU/train")
transition_params_RU = estimate_transition_parameters("Data/RU/train")
viterbi_algorithm(emission_params_RU, transition_params_RU, "Data/RU/dev.in", "Data/RU/dev.p2.out")
