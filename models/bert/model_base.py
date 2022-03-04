from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
)

import tensorflow as tf


class BertBase:
    def __init__(self):

        self.tokenizer = AutoTokenizer.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased-finetuned-sst-2-english"
        )

    def tokenize(self, sentence):
        return self.tokenizer(
            sentence,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=50,
            add_special_tokens=True,
        )

    def predict(self, tokenized_sentence):
        model_return = self.model(**tokenized_sentence)
        scores = model_return[0][0].detach().numpy()
        scores = tf.nn.softmax(scores)
        #  positive
        return scores[1].numpy()

    def tokenize_and_predict(self, sentence):
        return self.predict(self.tokenize(sentence))


if __name__ == "__main__":
    bert = BertBase()
    print(bert.tokenize_and_predict("Apple is a bad company"))
