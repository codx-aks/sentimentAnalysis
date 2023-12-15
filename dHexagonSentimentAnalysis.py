import librosa
import numpy as np
import os
from pydub import AudioSegment
from langcodes import Language
import matplotlib.pyplot as plt
from transformers import pipeline
import whisper
from keras.models import model_from_json

class result:
    def __init__(self, coordinates,emotions,pos_percent,neg_percent,rating,language,duration,gender,transcript):
        self.coordinates = coordinates
        self.emotions=emotions
        self.pos_percent=pos_percent
        self.neg_percent=neg_percent
        self.rating=rating
        self.language=language
        self.duration=duration
        self.gender=gender
        self.transcript=transcript
def process_audio(audio_path, chunk_duration=10):

    y, sr = librosa.load(audio_path, sr=None)
    chunk_size = int(chunk_duration * sr)
    duration = librosa.get_duration(y=y, sr=sr)
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]
    features = [librosa.feature.mfcc(y=chunk, sr=sr) for chunk in chunks]
    return features,duration

def preprocess_audio(audio_path):
    X, sample_rate = librosa.load(audio_path, res_type='kaiser_fast', duration=2.5, sr=22050*2, offset=0.5)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13), axis=0)
    return np.expand_dims(mfccs, axis=0)

def classify_audio(audio_file_path):
    model_architecture_path = 'model.json'
    json_file = open(model_architecture_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()

    model_weights_path = 'Emotion_Voice_Detection_Model.h5'
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(model_weights_path)

    input_data = preprocess_audio(audio_file_path)

    predictions = loaded_model.predict(input_data)
    high1 = max(predictions[0])
    high1Index = 0
    high2index = 0
    high2 = 0
    weights = [-0.8, 0.2, -0.3, 1.0, -0.6, -0.8, 0.2, -0.3, 1.0, -0.6]
    score = 0
    for i in range(0, len(predictions[0])):

        score += weights[i] * float(predictions[0][i])

        if (predictions[0][i] < high1 and predictions[0][i] > high2):
            high2 = predictions[0][i]
            high2index = i

        elif (predictions[0][i] == high1):
            high1Index = i

    label = ["female_angry",
             "female_calm",
             "female_fearful",
             "female_happy",
             "female_sad",
             "male_angry",
             "male_calm",
             "male_fearful",
             "male_happy",
             "male_sad"]

    result = [label[high1Index], label[high2index]]
    return result,score
def audio_to_text(audio_chunk):
    model = whisper.load_model("base")
    result = model.transcribe(audio_chunk)
    return result["text"]


def classify_mood(text,emotions):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    input=[text]
    model_outputs = classifier(input)
    emotions.append(model_outputs[0][0]['label'])
    return model_outputs[0][:5]

def combine_results(speech_score, text_classification, emotion_weights,i):
    weighted_scores = {}
    for emotion in text_classification:
        if(i==len(text_classification)):
            if(emotion['label']=='gratitude'):
                weighted_scores[emotion['label']] = emotion['score'] * 0.4
            else:
                weighted_scores[emotion['label']] = emotion['score'] * emotion_weights.get(emotion['label'], 0.0)
        else:
            weighted_scores[emotion['label']] = emotion['score'] * emotion_weights.get(emotion['label'], 0.0)

    combined_result = speech_score + 4*sum(weighted_scores.values())
    return combined_result,sum(weighted_scores.values())

def code_to_language_name(language_code):
    try:
        language = Language.get(language_code)
        return language.display_name().title()
    except ValueError:
        return "Unknown"
def plot_results(results):
    plt.plot(results)
    plt.xlabel('Time/10')
    plt.ylabel('Emotion Value')
    plt.title('Mood swings over time')
    plt.show()

def plot_text(results):
    plt.plot(results)
    plt.xlabel('Time/10')
    plt.ylabel('time emotion Value')
    plt.title('text Mood swings over time')
    plt.show()

def normalize_data(data):
    min_val = min(data)
    max_val = max(data)
    if(min_val!=max_val):
        normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    else:
        normalized_data = [(x - 0) / (max_val - 0) for x in data]
    return normalized_data

def dHexagonAnalysis(audio_path):

    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    language_code=max(probs, key=probs.get)
    language_name = code_to_language_name(language_code)

    audio_features,duration = process_audio(audio_path)
    results = []
    textResults=[]
    emotions=[]
    i=0

    emotion_weights = {
        'neutral': 0.0,
        'approval': 0.3,
        'annoyance': -0.9,
        'admiration': 0.6,
        'realization': 0.1,
        'excitement': 0.6,
        'disappointment': -0.7,
        'disapproval': -0.8,
        'disgust': -0.9,
        'anger': -1.0,
        'joy': 0.7,
        'love': 0.7,
        'confusion': -0.3,
        'amusement': 0.3,
        'sadness': -0.8,
        'optimism': 0.4,
        'curiosity': 0.1,
        'fear': -0.9,
        'desire': 0.7,
        'surprise': 0.1,
        'gratitude': 0.7,
        'caring': 0.2,
        'embarrassment': -0.4,
        'pride': 0.2,
        'grief': -0.9,
        'relief': 0.7,
        'remorse': -0.7,
        'nervousness': -0.2
    }
    for j, audio_chunk in enumerate(audio_features):

        audio = AudioSegment.from_wav(audio_path)
        output_directory = "/Users/akshayv/Desktop/SIH2023"
        os.makedirs(output_directory, exist_ok=True)

        start_time = i * 10000
        end_time = (i + 1) * 10000
        chunk = audio[start_time:end_time]

        temp_filename = f"{output_directory}/chunk_{i}.wav"
        chunk.export(temp_filename, format="wav")

        gen,speech_score=classify_audio(temp_filename)
        text = audio_to_text(temp_filename)
        text_classification = classify_mood(text,emotions)
        os.remove(temp_filename)
        combined_result,text_result = combine_results(speech_score, text_classification, emotion_weights,j)
        textResults.append(text_result)
        results.append(combined_result)
        i=i+1

    normalized_combined_results = normalize_data(results)
    pos_rating_var = 0
    neg_rating_var = 0

    for k in range(1, len(normalized_combined_results)):
        if (normalized_combined_results[k] > normalized_combined_results[k-1] ) :
            pos_rating_var += (normalized_combined_results[k]-normalized_combined_results[k-1])
        elif (normalized_combined_results[k] < normalized_combined_results[k-1]):
            neg_rating_var += (normalized_combined_results[k] - normalized_combined_results[k - 1])



    if(len(normalized_combined_results)<=1):
        pos_percentage = 50 + textResults[0]*50
        neg_percentage = 100-pos_percentage
        rating_var = (textResults[0]+1)*2.5
    else:
        rating_var = ((((1.0 + pos_rating_var ** 2.0 - neg_rating_var ** 2.0) * 12.5) ** 0.5))
        pos_percentage = (pos_rating_var * 5 / (pos_rating_var - neg_rating_var)) * 20
        neg_percentage = (neg_rating_var * -5 / (pos_rating_var - neg_rating_var)) * 20


    # print(f"emotion swings- {emotions}")
    # print(f"overall call rating -> {rating_var}")
    # plot_text(textResults)
    # plot_results(normalized_combined_results)

    if(gen[0][0] == 'm'):
        gender = "male"
    else:
        gender = "female"

    transcript=audio_to_text(audio_path)
    model_result = result(normalized_combined_results,
                          emotions,
                          pos_percentage,
                          neg_percentage,
                          rating_var,
                          language_name,
                          duration,
                          gender,
                          transcript)
    return model_result


