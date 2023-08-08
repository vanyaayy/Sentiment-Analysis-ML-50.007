from collections import defaultdict
import heapq    
import math

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

def k_best_viterbi_algorithm(k, emission_parameters, transition_parameters, dev_in_path, dev_out_path):
    # Small value to avoid taking the logarithm of zero
    EPSILON = 1e-10
    
    output_lines = []
    sentence_tokens = []
    with open(dev_in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                sentence_tokens.append(line)
            else:
                n = len(sentence_tokens)
                pi = defaultdict(lambda: defaultdict(list))
                pi[0]['START'].append((0, [])) # log probability, path
                for v in emission_parameters['#UNK#']:
                    pi[0][v].extend([(float('-inf'), []) for _ in range(k)])
                
                for i in range(1, n + 1):
                    token = sentence_tokens[i - 1]
                    if token not in emission_parameters:
                        token = '#UNK#'
                    for v in emission_parameters[token]:
                        max_k_probs = []
                        for u in transition_parameters:
                            for w in pi[i - 1][u]:
                                # Adding checks to handle zero probabilities
                                trans_prob = transition_parameters[u][v] + EPSILON
                                emiss_prob = emission_parameters[token][v] + EPSILON
                                prob = w[0] + math.log(trans_prob) + math.log(emiss_prob)
                                path = w[1] + [u]
                                heapq.heappush(max_k_probs, (prob, path))
                        pi[i][v] = heapq.nlargest(k, max_k_probs)
                
                max_k_probs = []
                for u in transition_parameters:
                    for v in pi[n][u]:
                        # Adding checks to handle zero probabilities
                        trans_prob = transition_parameters[u]['STOP'] + EPSILON
                        prob = v[0] + math.log(trans_prob)
                        path = v[1] + [u]
                        heapq.heappush(max_k_probs, (prob, path))
                best_path = heapq.nlargest(k, max_k_probs)[k - 1][1]

                for token, tag in zip(sentence_tokens, best_path):
                    output_lines.append(f"{token} {tag}")
                output_lines.append("")
                sentence_tokens = []

    with open(dev_out_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))

###ES####
emission_params_ES = estimate_emission_parameters("Data/ES/train")
transition_params_ES = estimate_transition_parameters("Data/ES/train")
k_best_viterbi_algorithm(2, emission_params_ES, transition_params_ES, "Data/ES/dev.in", "Data/ES/dev.p3.2nd.out")
k_best_viterbi_algorithm(8, emission_params_ES, transition_params_ES, "Data/ES/dev.in", "Data/ES/dev.p3.8th.out")

###RU####
emission_params_RU = estimate_emission_parameters("Data/RU/train")
transition_params_RU = estimate_transition_parameters("Data/RU/train")
k_best_viterbi_algorithm(2, emission_params_RU, transition_params_RU, "Data/RU/dev.in", "Data/RU/dev.p3.2nd.out")
k_best_viterbi_algorithm(8, emission_params_RU, transition_params_RU, "Data/RU/dev.in", "Data/RU/dev.p3.8th.out")
