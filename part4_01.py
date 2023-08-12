from collections import defaultdict

class NaiveBayesClassifier:
    def __init__(self, k=1):
        self.k = k
        self.token_tag_count = defaultdict(int)
        self.tag_count = defaultdict(int)
        self.emission_parameters = defaultdict(lambda: defaultdict(float))

    def train(self, train_path):
        with open(train_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if line:
                    token, tag = line.rsplit(' ', 1)
                    self.token_tag_count[(token, tag)] += 1
                    self.tag_count[tag] += 1
        for (token, tag), count in self.token_tag_count.items():
            self.emission_parameters[token][tag] = count / (self.tag_count[tag] + self.k)
            self.emission_parameters['#UNK#'][tag] = self.k / (self.tag_count[tag] + self.k)

    def predict(self, token):
        if token not in self.emission_parameters:
            token = '#UNK#'
        predicted_tag = max(self.emission_parameters[token], key=self.emission_parameters[token].get)
        return predicted_tag

def simple_sentiment_analysis(classifier, dev_in_path, dev_out_path):
    output_lines = []
    with open(dev_in_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if line:
                token = line
                predicted_tag = classifier.predict(token)
                output_lines.append(f"{line} {predicted_tag}")
            else:
                output_lines.append("")
    with open(dev_out_path, 'w', encoding='utf-8') as file:
        file.write('\n'.join(output_lines))

# Create and train classifiers for different languages
emission_params_ES = NaiveBayesClassifier()
emission_params_ES.train("Data/ES/train")
simple_sentiment_analysis(emission_params_ES, "Data/ES/dev.in", "Data/ES/train.in")

emission_params_RU = NaiveBayesClassifier()
emission_params_RU.train("Data/RU/train")
simple_sentiment_analysis(emission_params_RU, "Data/RU/dev.in", "Data/RU/train.in")
