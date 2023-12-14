import librosa
import numpy as np
import os
from pydub import AudioSegment
from langcodes import Language
import matplotlib.pyplot as plt
from transformers import pipeline,RobertaTokenizer
import whisper
def process_audio(audio_path, chunk_duration=10):

    y, sr = librosa.load(audio_path, sr=None)
    chunk_size = int(chunk_duration * sr)
    chunks = [y[i:i+chunk_size] for i in range(0, len(y), chunk_size)]
    features = [librosa.feature.mfcc(y=chunk, sr=sr) for chunk in chunks]
    return features

def audio_to_text(audio_chunk):
    model = whisper.load_model("base")
    result = model.transcribe(audio_chunk)
    return result["text"]


def classify_mood(text,emotions):
    classifier = pipeline(task="text-classification", model="SamLowe/roberta-base-go_emotions", top_k=None)
    input=[text]
    model_outputs = classifier(input)
    emotions.append(model_outputs[0][0]['label'])
    print(model_outputs[0][:5])
    return model_outputs[0][:5]

def combine_results(audio_features, text_classification, emotion_weights,i):
    weighted_scores = {}
    for emotion in text_classification:
        if(i==len(text_classification)):
            if(emotion['label']=='gratitude'):
                weighted_scores[emotion['label']] = emotion['score'] * 0.4
            else:
                weighted_scores[emotion['label']] = emotion['score'] * emotion_weights.get(emotion['label'], 0.0)
        else:
            weighted_scores[emotion['label']] = emotion['score'] * emotion_weights.get(emotion['label'], 0.0)

    combined_result = np.mean(audio_features) + 5*sum(weighted_scores.values())
    return combined_result

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

def plot_mfcc(results):
    plt.plot(results)
    plt.xlabel('Time/10')
    plt.ylabel('mfcc emotion value')
    plt.title('Mfcc mood swings over time')
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
    normalized_data = [(x - min_val) / (max_val - min_val) for x in data]
    return normalized_data

def dHexagonAnalysis(audio_path):

    model = whisper.load_model("base")
    audio = whisper.load_audio(audio_path)
    audio = whisper.pad_or_trim(audio)

    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    _, probs = model.detect_language(mel)
    language_code=max(probs, key=probs.get)
    language_name = code_to_language_name(language_code)
    print(f"Detected language: {language_name}")


    audio_features = process_audio(audio_path)
    results = []
    mfccResults=[]
    textResults=[]
    emotions=[]
    i=0
    tokenizer = RobertaTokenizer.from_pretrained("SamLowe/roberta-base-go_emotions")

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

        text = audio_to_text(temp_filename)
        print(text)
        text_classification = classify_mood(text,emotions)
        os.remove(temp_filename)

        combined_result = combine_results(audio_chunk, text_classification, emotion_weights,j)
        mfccResults.append(np.mean(audio_chunk))
        textResults.append(combined_result-np.mean(audio_chunk))
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

    rating_var = ((((1.0+pos_rating_var**2.0 - neg_rating_var**2.0)*12.5)**0.5))

    print(f"emotion swings- {emotions}")

    print(f"positiveness rating -> {pos_rating_var*5/(pos_rating_var-neg_rating_var)}")
    print(f"negativeness rating -> {neg_rating_var*-5/(pos_rating_var-neg_rating_var)}")
    print(f"overall call rating -> {rating_var}")

    plot_mfcc(mfccResults)
    plot_text(textResults)
    plot_results(normalized_combined_results)
