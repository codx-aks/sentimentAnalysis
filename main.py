from transformers import TFAutoModel , AutoTokenizer
from datasets import load_dataset
import tensorflow as tf
import pandas as pd
import numpy as np

model = TFAutoModel.from_pretrained("bert-base-uncased")
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
# input=tokenizer(["dHexagon","SIH2023"],padding=True,truncation=True,return_tensors='tf')
# print(input)
# output=model(input)
# print(output)
emotions = load_dataset('dair-ai/emotion')
print(emotions)

def tokenize(batch):
    return tokenizer(batch["text"], padding=True, truncation=True)

emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)
print(emotions_encoded)

emotions_encoded.set_format('tf',
                            columns=['input_ids', 'attention_mask', 'token_type_ids', 'label'])

BATCH_SIZE = 32

def order(inp):
    data = list(inp.values())
    return {
        'input_ids': data[1],
        'attention_mask': data[2],
        'token_type_ids': data[3]
    }, data[0]

train_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['train'][:10])
train_dataset = train_dataset.batch(BATCH_SIZE).shuffle(10)
train_dataset = train_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)

test_dataset = tf.data.Dataset.from_tensor_slices(emotions_encoded['test'][:5])
test_dataset = test_dataset.batch(BATCH_SIZE)
test_dataset = test_dataset.map(order, num_parallel_calls=tf.data.AUTOTUNE)


class BERTForClassification(tf.keras.Model):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        self.fc2 = tf.keras.layers.Dense(num_classes, activation='softmax')

    def call(self, inputs):
        x = self.bert(inputs)[1]
        x = self.fc1(x)
        return self.fc2(x)

classifier = BERTForClassification(model, num_classes=6)

classifier.compile(
    optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)
history = classifier.fit(
    train_dataset,
    epochs=1
)

custom_text = "what terrible job have u done , im fucking angry at you"

tokenized_text = tokenizer(custom_text, padding=True, truncation=True, return_tensors='tf')
predictions = classifier.predict(tokenized_text.data)
predicted_class = tf.argmax(predictions, axis=-1).numpy()

if np.isscalar(predicted_class):
    predicted_class = np.array([predicted_class])

if predicted_class.size > 0:
    predicted_class = predicted_class[0]
    print("Predicted class:", predicted_class)
else:
    print("No predictions available.")

class_labels = ["sad", "joy", "love", "anger", "fear", "surprise"]
predicted_label = class_labels[predicted_class]

print("Custom Text:", custom_text)
print("Predicted Label:", predicted_label)


classifier.save('my_model.keras')
eval_result = classifier.evaluate(test_dataset)
print("Evaluation Loss:", eval_result[0])
print("Evaluation Accuracy:", eval_result[1])





