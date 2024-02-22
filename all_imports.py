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

