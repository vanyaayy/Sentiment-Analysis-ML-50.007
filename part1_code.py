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

def simple_sentiment_analysis(emission_parameters, dev_in_path, dev_out_path):
    output_lines = []
    with open(dev_in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token = line
                if token not in emission_parameters:
                    token = '#UNK#'
                predicted_tag = max(emission_parameters[token], key=emission_parameters[token].get)
                output_lines.append(f"{line} {predicted_tag}")
            else:
                output_lines.append("")
    with open(dev_out_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))


emission_params_ES = estimate_emission_parameters("Data/ES/train")
simple_sentiment_analysis(emission_params_ES, "Data/ES/dev.in", "Data/ES/dev.p1.out")