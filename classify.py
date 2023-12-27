import pandas as pd
import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification
from sklearn.preprocessing import LabelEncoder

file_path = 'issue_dataset.csv'
df = pd.read_csv(file_path)

train_texts = df['Text'][:7000].tolist()
train_labels = df['Label'][:7000].tolist()

test_texts = df['Text'][7000:8500].tolist()
test_labels = df['Label'][7000:8500].tolist()

validation_texts = df['Text'][8500:10000].tolist()
validation_labels = df['Label'][8500:10000].tolist()


label_encoder = LabelEncoder()
train_labels = label_encoder.fit_transform(train_labels)

test_labels = label_encoder.fit_transform(test_labels)

validation_labels = label_encoder.fit_transform(validation_labels)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = TFBertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=6)

tokenized_train_inputs = tokenizer(train_texts, padding=True, truncation=True, return_tensors='tf')

tokenized_test_inputs = tokenizer(test_texts, padding=True, truncation=True, return_tensors='tf')


train_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_train_inputs), train_labels))

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
metric = tf.keras.metrics.SparseCategoricalAccuracy('accuracy')
model.compile(optimizer=optimizer, loss=loss, metrics=[metric])
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=3, restore_best_weights=True
)
test_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_test_inputs), test_labels))

# model.fit(train_dataset.shuffle(36000).batch(32), epochs=6)
history = model.fit(
    train_dataset.shuffle(1000).batch(32),
    epochs=3,
    validation_data=test_dataset.batch(32),
    callbacks=[early_stopping]
)

test_dataset = tf.data.Dataset.from_tensor_slices((dict(tokenized_test_inputs), test_labels))
eval_result = model.evaluate(test_dataset.batch(32))
print("Evaluation Loss:", eval_result[0])
print("Evaluation Accuracy:", eval_result[1])


model.save_pretrained('issue_classifier')
