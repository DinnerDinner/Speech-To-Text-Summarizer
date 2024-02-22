from deepgram import Deepgram
import json
import numpy as np
import sys
from scipy.io.wavfile import write
import sounddevice as sd
import random
import os
from summa import summarizer
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import keyboard
import time
from tqdm import tqdm
import random
from nltk.tokenize import sent_tokenize, word_tokenize
import asyncio
from threading import Thread, Event
from concurrent.futures import ThreadPoolExecutor
import threading
from multiprocessing import Process
from language_tool_python import LanguageTool
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
import subprocess
from fpdf import FPDF
import yt_dlp
from tkinter import filedialog, Tk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from wordcloud import WordCloud
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.cluster import KMeans
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Tokenizer, GPT2LMHeadModel, AdamW
from tqdm import tqdm
import nltk
import random



# Function to get user's preferred language
def choose_language():
    while True:
        try:
            lang_choice = input("Choose your preferred language( 1 , 2 , 3):\n1 English\n2 Français\n3 Español\n")
            lang_choice = int(lang_choice)
            if lang_choice in [1, 2, 3]:
                return lang_choice
            else:
                raise ValueError
        except ValueError:
            print("Invalid input. Please enter a number: 1 , 2 , 3")




# Function to display messages based on language choice
def display_message(lang_choice, message_dict):
    language = {1: "English", 2: "Français", 3: "Español"}[lang_choice]
    return message_dict.get(language, message_dict[1])  # Default to English if translation not available



# Startup
def startup(lang_choice):
    iterations = 10

    for _ in tqdm(range(iterations), desc=display_message(lang_choice, {"English": "Starting up", "Français": "Démarrage", "Español": "Iniciando"}), unit=""):
        time.sleep(random.uniform(0.1, 0.3))  # Adjust the sleep time as needed



# #Audio Recorder
def record_audio(lang_choice):
    print(display_message(lang_choice, {"English": "Recording... Press any key to stop.",
                                         "Français": "Enregistrement... Appuyez n'importe quelle touche pour arrêter.",
                                         "Español": "Grabando... Presiona cualquier tecla para detener."}))

    try:
        audio_data = []

        while True:
            recording_chunk = sd.rec(44100, channels=2, dtype=np.int16)
            sd.wait()
            audio_data.append(recording_chunk)

            if keyboard.read_event().event_type == keyboard.KEY_DOWN:
                raise KeyboardInterrupt

    except KeyboardInterrupt:
        print(display_message(lang_choice, {"English": "Audio recording complete",
                                             "Français": "Enregistrement audio terminé",
                                             "Español": "Grabación de audio completa"}))
        return np.concatenate(audio_data)



# Save audio file
def save_audio_file(audio_data, file_path):
    write(file_path, 44100, audio_data)

def download_youtube_video(url, output_path, lang_choice):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': output_path,
        'quiet': True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])



def browse_audio_file(lang_choice):
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=display_message(lang_choice, {"English": "Select an audio file", "Français": "Sélectionnez un fichier audio", "Español": "Selecciona un archivo de audio"}),
                                           filetypes=[("Audio files", "*.wav;*.mp3")])
    return file_path


def grammar_check(summary, lang_choice):
    tool = LanguageTool(lang_choice_codes[lang_choice])
    matches = tool.check(summary)
    corrected_text = tool.correct(summary)
    return corrected_text



r = random.randint(1, 10000)
output_file = "audio" + str(r) + '.wav'


if __name__ == "__main__":
    lang_choice = choose_language()
    lang_choice_codes = {1: 'en-US', 2: 'fr', 3: 'es'}

    while True:
        try:
            choice = input(display_message(lang_choice, {"English": "Choose an option:\n1. Use Microphone\n2. Enter YouTube URL\n3. Browse and pick an audio file\n",
                                                         "Français": "Choisissez une option:\n1. Utiliser le microphone\n2. Entrer l'URL de YouTube\n3. Parcourir et choisir un fichier audio\n",
                                                         "Español": "Elige una opción:\n1. Usar el micrófono\n2. Ingresar la URL de YouTube\n3. Navegar y elegir un archivo de audio\n"}))


            if choice == "1":
                recorded_data = record_audio(lang_choice)
                save_audio_file(recorded_data, output_file)
                DEEPGRAM_API_KEY = '44d9c053698084b3e1d426024059efe751bb8569'
                PATH_TO_FILE = output_file
                MIMETYPE = 'audio/wav'
                break


            elif choice == "2":
                youtube_url = input(display_message(lang_choice, {"English": "Input a YouTube video URL (Ctrl+Shift+V): ",
                                                                   "Français": "Entrez l'URL d'une vidéo YouTube (Ctrl+Shift+V): ",
                                                                   "Español": "Ingresa la URL de un video de YouTube (Ctrl+Shift+V): "}))
                download_youtube_video(youtube_url, output_file)
                DEEPGRAM_API_KEY = '44d9c053698084b3e1d426024059efe751bb8569'
                PATH_TO_FILE = output_file + '.mp3'
                MIMETYPE = 'audio/wav'
                break


            elif choice == "3":
                audio_file_path = browse_audio_file(lang_choice)
                DEEPGRAM_API_KEY = '44d9c053698084b3e1d426024059efe751bb8569'
                PATH_TO_FILE = audio_file_path
                MIMETYPE = 'audio/wav'
                break


            else:
                print(display_message(lang_choice, {"English": "Invalid choice. Please enter a valid option.",
                                                     "Français": "Choix non valide. Veuillez entrer une option valide.",
                                                     "Español": "Opción no válida. Por favor, ingresa una opción válida."}))



        except Exception as e:
            print(display_message(lang_choice, {"English": f"Error: {e}",
                                                 "Français": f"Erreur : {e}",
                                                 "Español": f"Error: {e}"}))
            continue






#Manual loading animation
def loading_animation(lang_choice):
    total_iterations = 10

    for _ in tqdm(range(total_iterations),
                  desc=display_message(lang_choice, {"English": "Processing", "Français": "Traitement", "Español": "Procesando"}),
                  unit=""):
        time.sleep(random.uniform(0.8, 1.1))  # Adjust the sleep time as needed




#DeepGram audio processing
def main(lang_choice):
    try:
        # Ask the user to choose the language for audio processing with error handling
        audio_language = None
        while audio_language is None:
            try:
                audio_lang_choice = input(display_message(lang_choice, {"English": "Choose the language of the audio:\n1. English\n2. Français\n3. Español\n"}))
                audio_lang_choice = int(audio_lang_choice)
                audio_language = {1: "en-US", 2: "fr-FR", 3: "es-ES"}[audio_lang_choice]
            except (ValueError, KeyError):
                print("Invalid choice. Please enter a valid option.")

        dg_client = Deepgram(DEEPGRAM_API_KEY)
        with open(PATH_TO_FILE, 'rb') as audio:
            source = {'buffer': audio, 'mimetype': MIMETYPE}
            options = {"smart_format": True, "model": "nova-2", "language": audio_language}

            loading_process = Process(target=loading_animation)
            loading_process.start()
            print(display_message(lang_choice, {"English": "\nProcessing...",
                                                 "Français": "\nTraitement en cours...",
                                                 "Español": "\nProcesando..."}))

            response = dg_client.transcription.sync_prerecorded(source, options)
            json_initial_response = (json.dumps(response, indent=4))
            json_text_only = response['results']['channels'][0]['alternatives'][0]['transcript']
            loading_process.join()

        print(display_message(lang_choice, {"English": '\nProcessing complete. Please wait for the summary.\n',
                                             "Français": '\nTraitement terminé. Veuillez patienter pour le résumé.\n',
                                             "Español": '\nProcesamiento completo. Por favor, espere el resumen.\n'}))


    # print(json_text_only)


        def pipeline_summarization():
            
            def chunk_size(text):
                length = len(word_tokenize(text))

                if length > 500 and length < 1000:
                    words_per_chunk_threshold = 200
                    
                elif length <= 500 and length > 200: 
                    words_per_chunk_threshold = 100

                elif length <= 50:
                    words_per_chunk_threshold = length

                elif length > 50 and length <= 200:
                    words_per_chunk_threshold = 50

                elif length >= 1000 and length<3000: 
                    words_per_chunk_threshold = 400

                elif length >= 3000:
                    words_per_chunk_threshold = 1000
                    
                else: 
                    words_per_chunk_threshold = 200
                
                return words_per_chunk_threshold



            def get_chunks_sentences(text):
                try:
                    sentences = sent_tokenize(text)
                except Exception as e:
                    print(f"Error during sentence tokenization: {e}")
                    return []



                words_per_chunk_threshold = chunk_size(text)
                # print(f"Calculated words_per_chunk_threshold: {words_per_chunk_threshold}")
                current_chunk = ''
                current_chunk_word_count = 0
                chunks = []


                for sentence in sentences:
                    try:
                        words = word_tokenize(sentence)
                    except Exception as e:
                        print(f"Error during word tokenization: {e}")
                        continue
                    

                    word_count = len(words)

                    if current_chunk_word_count + word_count <= words_per_chunk_threshold:
                        current_chunk += ' ' + sentence
                        current_chunk_word_count += word_count
                    
                    else:
                        if current_chunk_word_count > 0 and current_chunk_word_count + word_count <= words_per_chunk_threshold + 20:
                            current_chunk += ' ' + sentence
                            current_chunk_word_count += word_count

                        else:
                            chunks.append(current_chunk.strip())
                            current_chunk = sentence
                            current_chunk_word_count = word_count


                if current_chunk.strip():
                    chunks.append(current_chunk.strip())

                return chunks



            def generate_summary(chunk, max_length_generate):
                inputs = tokenizer(chunk, return_tensors="pt", truncation=True)
                summary_ids = model.generate(
                            inputs["input_ids"],
                            max_length=max_length_generate, 
                            num_beams=4,
                            length_penalty=2.0,
                            early_stopping=True
            )
                
                summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
                return summary




            def create_pdf(file_path, text, chunk_summaries, lang_choice):            
                class CustomPDF(FPDF):

                    pdf = FPDF()
                    pdf.set_auto_page_break(auto=True, margin=15)
                    def footer(self):
                        self.set_y(-15)
                        self.set_font('Arial', 'I', 8)
                        self.cell(0, 10, 'page %s' % self.page_no(), 0, 0, 'C')


                pdf = CustomPDF()
                pdf.add_page()


                # Title for Summaries
                pdf.set_font("Arial", style='B', size=16)
                pdf.set_text_color(30, 144, 255)
                pdf.cell(200, 10, txt=display_message(lang_choice, {'English': 'Summaries', 'Français': 'Résumés', 'Español': 'Resúmenes'}), ln=True, align='C')


                # Content for Summaries
                pdf.set_font("Arial", size=12)
                pdf.set_text_color(0, 0, 0)
                for i, chunk_summary in enumerate(chunk_summaries, 1):
                    pdf.multi_cell(0, 10, txt=f"\n{i}. {chunk_summary}")


                # Title for Transcript
                pdf.ln(10)  # Add some space before the next title
                pdf.set_font('Arial', size=16)
                pdf.set_text_color(255, 69, 0)
                pdf.cell(200, 10, txt=display_message(lang_choice, {'English': 'Transcription', 'Français': 'Transcription', 'Español': 'Transcripción'}), ln=True, align='C')


                # Content for Transcript
                pdf.set_font("Arial", size=12)
                pdf.set_text_color(0, 0, 0)
                pdf.multi_cell(0, 10, txt=text)


                # Save the PDF
                pdf.output(file_path)


                # Open the PDF file after creation
                if os.name == "nt":  # For Windows
                    os.startfile(file_path)
                else:  # For Linux and macOS
                    opener = "open" if sys.platform == "darwin" else "xdg-open"
                    subprocess.call([opener, file_path])




            text = json_text_only
            chunks = get_chunks_sentences(text)

            model_name = "knkarthick/MEETING_SUMMARY"
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            chunk_summaries = []

            for i, chunk in enumerate(chunks, 1):
                max_length_generate = 200
                summary = generate_summary(chunk, max_length_generate)
                print(f"\n{display_message(lang_choice, {'English': 'Chunk', 'Français': 'Fragment', 'Español': 'Fragmento'})} {i}:\n{chunk}")
                print(f"\n{display_message(lang_choice, {'English': 'Summary for chunk', 'Français': 'Résumé pour le fragment', 'Español': 'Resumen del fragmento'})} {i}:\n", summary)
                corrected_summary = grammar_check(summary)
                chunk_summaries.append(corrected_summary)

            create_pdf("summary.pdf", text, chunk_summaries, lang_choice)
                    
                # full_summary = " |  ".join(summaries)
                # print(full_summary)

            pipeline_summarization()
                
        def train_text_generation_model(texts, epochs=3, batch_size=8):

            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenized_texts = [tokenize_text(text, tokenizer) for text in texts]


            dataset = TensorDataset(*tokenized_texts)
            dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


            model = GPT2LMHeadModel.from_pretrained("gpt2")


            criterion = nn.CrossEntropyLoss()
            optimizer = AdamW(model.parameters(), lr=5e-5)

            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0

                for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch"):
                    optimizer.zero_grad()

                    input_ids = batch[0].to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
                    labels = input_ids.clone()

                    outputs = model(input_ids, labels=labels)
                    loss = outputs.loss
                    total_loss += loss.item()

                    loss.backward()
                    optimizer.step()


                average_loss = total_loss / len(dataloader)
                print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {average_loss}")

                # Save a checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': average_loss,
                }, f'checkpoint_epoch_{epoch + 1}.pt')

            print("Training complete.")

        # Function to visualize loss over epochs
        def plot_loss_curve(losses):
            plt.plot(losses, label="Training Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss Curve")
            plt.legend()
            plt.show()


        class SentimentAnalyzer:
            def __init__(self):
                self.sia = SentimentIntensityAnalyzer()

            def analyze_sentiment(self, text):
                sentiment_scores = self.sia.polarity_scores(text)
                return sentiment_scores


        class WordFrequencyAnalyzer:
            def __init__(self):
                self.vectorizer = CountVectorizer()

            def analyze_word_frequency(self, texts):
                word_matrix = self.vectorizer.fit_transform(texts)
                word_frequencies = np.array(word_matrix.sum(axis=0))[0]
                feature_names = np.array(self.vectorizer.get_feature_names_out())
                sorted_indices = np.argsort(word_frequencies)[::-1]
                return feature_names[sorted_indices], word_frequencies[sorted_indices]


        class ExtractiveSummarizer:
            def __init__(self):
                self.kmeans = KMeans(n_clusters=5, random_state=42)

            def summarize_extractive(self, texts):

                text_vectors = self.vectorize_texts(texts)
                cluster_labels = self.kmeans.fit_predict(text_vectors)


                summaries = []
                for cluster_id in range(self.kmeans.n_clusters):
                    cluster_indices = np.where(cluster_labels == cluster_id)[0]
                    cluster_texts = [texts[i] for i in cluster_indices]
                    representative_sentence = self.get_representative_sentence(cluster_texts)
                    summaries.append(representative_sentence)

                return summaries

            def vectorize_texts(self, texts):


                vectorizer = CountVectorizer()
                return vectorizer.fit_transform(texts).toarray()

            def get_representative_sentence(self, cluster_texts):


                word_frequencies = self.get_word_frequencies(cluster_texts)
                max_word_index = np.argmax(word_frequencies)
                return cluster_texts[max_word_index]

            def get_word_frequencies(self, texts):
                vectorized_texts = self.vectorize_texts(texts)
                return np.sum(vectorized_texts, axis=0)

        # Class for Linear Regression on Word Frequency
        class WordFrequencyRegression:
            def __init__(self):
                self.regressor = LinearRegression()

            def train_regression_model(self, feature_names, word_frequencies):
                X = np.arange(len(feature_names)).reshape(-1, 1)
                y = word_frequencies.reshape(-1, 1)

                self.regressor.fit(X, y)

            def plot_regression(self, feature_names, word_frequencies):
                X = np.arange(len(feature_names)).reshape(-1, 1)
                predictions = self.regressor.predict(X)

                plt.figure(figsize=(10, 6))
                plt.scatter(X, word_frequencies, label="Actual Word Frequencies")
                plt.plot(X, predictions, color='red', label="Regression Line")
                plt.xlabel("Word Index")
                plt.ylabel("Word Frequency")
                plt.title("Linear Regression on Word Frequency")
                plt.legend()
                plt.show()


        def part_of_speech_tagging(text):
            tokens = word_tokenize(text)
            pos_tags = nltk.pos_tag(tokens)
            return pos_tags

        def named_entity_classification(text):
            entities = nltk.chunk.ne_chunk(nltk.pos_tag(word_tokenize(text)))
            return entities


        dummy_texts_analysis = [
                                        ]


                # # Sentiment Analysis
                # sentiment_analyzer = SentimentAnalyzer()
                # sentiment_scores = [sentiment_analyzer.analyze_sentiment(text) for text in dummy_texts_analysis]
                # print("Sentiment Scores:", sentiment_scores)

                # Word Frequency Analysis
        word_frequency_analyzer = WordFrequencyAnalyzer()
        feature_names, word_frequencies = word_frequency_analyzer.analyze_word_frequency(dummy_texts_analysis)
        print("Word Frequencies:", dict(zip(feature_names, word_frequencies)))

                # Extractive Summarization
        extractive_summarizer = ExtractiveSummarizer()
        extractive_summaries = extractive_summarizer.summarize_extractive(dummy_texts_analysis)
        print("Extractive Summaries:", extractive_summaries)

                # Linear Regression on Word Frequency
        word_frequency_regression = WordFrequencyRegression()
        word_frequency_regression.train_regression_model(feature_names, word_frequencies)
        word_frequency_regression.plot_regression(feature_names, word_frequencies)


        dummy_text_nlp = "Natural language processing is a subfield of artificial intelligence."


        pos_tags_result = part_of_speech_tagging(dummy_text_nlp)
        print("Part-of-Speech Tags:", pos_tags_result)


        named_entity_result = named_entity_classification(dummy_text_nlp)
        print("Named Entities:", named_entity_result)



            
    except Exception as e:
        print(display_message(lang_choice, {"English": f"Error: {e}",
                                             "Français": f"Erreur : {e}",
                                             "Español": f"Error: {e}"}))




if __name__ == "__main__":
    lang_choice = choose_language()
    startup(lang_choice)
    main(lang_choice)
# os.remove(output_file)def part_of_speech_tagging(text):
    
    
