import sys

import numpy as np
import pandas as pd
import pythainlp
import nltk

class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)

    def compute_probability(self, text_string):
        tokens = nltk.tokenize.wordpunct_tokenize(text_string)

        labels = self.get_all_possible_labels()
        features = self.get_all_possible_features()

        scores = {label: 0 for label in labels}

        for feature in tokens:
            if feature not in features:
                continue

            for label in labels:
                scores[label] = self.model_params.loc[feature][label]

        # Softmax scores to probabilities
        total = sum([np.exp(i) for i in scores.values()])
        probabilities = {k: float(np.exp(v)) for k, v in scores.items()}
        return probabilities

    def get_all_possible_features(self):
        return self.model_params.index

    def get_all_possible_labels(self):
        return self.model_params.columns

    def classify(self, text_string):
        probabilities = self.compute_probability(text_string)
        current = 0
        result = None

        for k, v in probabilities.items():
            if v > current:
                result = k
                current = v

        return result



if __name__ == '__main__':
    if (len(sys.argv) != 2):
        print('usage:\tpython logistic_regression.py <model_file>')
        sys.exit(0)
    model_file_name = sys.argv[1]
    model = TextClassifier(model_file_name)
    
    text_string = input()
    print(model.compute_probability(text_string))
    print(model.classify(text_string))
