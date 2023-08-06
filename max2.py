emission_params = {
    'tag1': {'word1': 0.000000000073823802380112, "#UNK#": 3},
    'tag2': {'word1': 0.000000000067381798317013095, 'word2': 12, "#UNK#": 5}
}
max_vals = {}
listt = ["word1", "word2", "word3"]
keys = emission_params.keys()

for word in listt:
    max_val = 0
    max_tag = ''
    for x in keys:
        if word in emission_params[x]:
            if emission_params[x][word] > max_val:
                max_val = emission_params[x][word]
                max_tag = x
        else:
            if emission_params[x]["#UNK#"] > max_val:
                max_val = emission_params[x]["#UNK#"]
                max_tag = x
    max_vals[word] = max_val
    print(f"{word} {max_tag}")
