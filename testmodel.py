import tensorflow as tf
from transformers import BertTokenizer, TFBertForSequenceClassification

loaded_model = TFBertForSequenceClassification.from_pretrained('issue_classifier', num_labels=6)

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
class_list=["Billing Issues", "Delivery Issues", "Product Availability", "Quality Issues", "Service Issues", "Technical Issues"]
new_texts = ["i really want to buy the product but its not available"]
new_tokenized_inputs = tokenizer(new_texts, padding=True, truncation=True, return_tensors='tf')
predictions = loaded_model.predict(dict(new_tokenized_inputs))
probabilities = tf.nn.softmax(predictions.logits, axis=-1)

predicted_classes = tf.argmax(probabilities, axis=-1).numpy()
i=predicted_classes[0]
print("Predicted Classes:", class_list[i])



