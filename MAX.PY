emission_params = {
    'tag1': {'word1': 12, "#UNK#": 3},
    'tag2': {'word1': 5, 'word2': 12, "#UNK#": 5}
}
max_vals = {}
listt = ["word1", "word2", "word3"]
for word in listt:      
    keys = emission_params.keys()
    for x in keys:
        # Iterate through each word in the emission_params
        if word in emission_params[x]:
            if word not in max_vals.keys():
                max_vals[word] = emission_params[x][word]
            else:
                if max_vals[word] < emission_params[x][word]:
                    max_vals[word] = emission_params[x][word]
        else:
            if word not in max_vals.keys():
                max_vals[word] = emission_params[x]["#UNK#"]
            else:
                if max_vals[word] < emission_params[x]["#UNK#"]:
                    max_vals[word] = emission_params[x]["#UNK#"]
print(max_vals)  

for word in max_vals:
  finalval = max_vals[word]
  keys = emission_params.keys()
  for x in keys:
    if word in emission_params[x]: 
      if emission_params[x][word]==finalval:
        print(f"{word} {x}")
        break
    else:
      if emission_params[x]["#UNK#"]==finalval:
        print(f"{word} {x}")
        break
