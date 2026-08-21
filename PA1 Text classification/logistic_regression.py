import re
import sys

import numpy as np
import pandas as pd
import nltk

nltk.download('punkt')
nltk.download('stopwords')

class TextClassifier:

    def __init__(self, csv_file_name):
        self.model_params = pd.read_csv(csv_file_name, index_col=0)

    def _clean_text(self, text_string):
        # Lowercase
        text_string = text_string.lower()

        # Remove quotes
        text_string = text_string.replace('"', '')

        # Remove links
        text_string = re.sub(r'https?://\S+', '', text_string)

        # Remove non-alphabetic characters
        text_string = re.sub(r'[^a-z\s]', '', text_string)

        # Normalize whitespace after removals
        text_string = re.sub(r'\s+', ' ', text_string).strip()

        # Remove stopwords
        stopwords = set(nltk.corpus.stopwords.words('english'))
        text_string = ' '.join(word for word in text_string.split() if word not in stopwords)

        return text_string

    def compute_probability(self, text_string):
        text_string = self._clean_text(text_string)
        tokens = nltk.tokenize.wordpunct_tokenize(text_string)

        labels = self.get_all_possible_labels()
        features = self.get_all_possible_features()

        scores = {label: 0 for label in labels}

        for feature in tokens:
            if feature not in features:
                continue

            for label in labels:
                scores[label] += self.model_params.loc[feature][label]

        # Softmax scores to probabilities
        total = sum([np.exp(i) for i in scores.values()])
        probabilities = {k: float(np.exp(v)) / total for k, v in scores.items()}
        return probabilities

    def get_all_possible_features(self):
        return self.model_params.index.tolist()

    def get_all_possible_labels(self):
        return self.model_params.columns.tolist()

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
