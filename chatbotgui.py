import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import pandas as pd
import numpy as np
from pandas import *
 
data = read_csv("data/movies.csv")  
title = data['title'].tolist()

from hybrid import get_recommendations_based_on_genres, hybrid_content_svd_model

from keras.models import load_model
model = load_model('chatbot_model.h5')
import json
import random
import pyttsx3

engine = pyttsx3.init()

intents = json.loads(open('data/intents.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence, model):
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res

# ---------- GUI ----------
import tkinter
from tkinter import *
import customtkinter

customtkinter.set_appearance_mode("dark")
customtkinter.set_default_color_theme("blue")

import speech_recognition as sr
r = sr.Recognizer()

def voice():
    with sr.Microphone() as source:
        print("Talk")
        audio_text = r.listen(source)
        print("Time over, thanks")
        try:
            userSay = r.recognize_google(audio_text)
        except sr.UnknownValueError:
            userSay = ""
        except sr.RequestError:
            userSay = ""
        
        if userSay != '':
            ChatLog.config(state=NORMAL)
            ChatLog.insert(END, "You: " + userSay + '\n\n')
            ChatLog.config(foreground="#442265", font=("Verdana", 10))
            res = chatbot_response(userSay)
            ChatLog.insert(END, "Covenant: " + res + '\n\n')
            ChatLog.config(state=DISABLED)
            ChatLog.yview(END)
            
            voices = engine.getProperty("voices")
            engine.setProperty("rate", 115)
            engine.setProperty("voice", voices[1].id)
            engine.say(res)
            engine.runAndWait()

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)
    if msg == '':
        return
    
    ChatLog.config(state=NORMAL)
    ChatLog.insert(END, "You: " + msg + '\n\n')
    ChatLog.config(foreground="#442265", font=("Verdana", 10))
    
    if msg in title:
        tryy = get_recommendations_based_on_genres(msg)
        ChatLog.insert(END, "Covenant: \n" + str(tryy) + '\n\n')
    
    elif msg.isdigit() and 1 <= int(msg) <= 700:
        tryy = hybrid_content_svd_model(int(msg))
        ChatLog.insert(END, "Covenant: \n" + str(tryy) + '\n\n')
    
    else:
        res = chatbot_response(msg)
        ChatLog.insert(END, "Covenant: " + res + '\n\n')
    
    ChatLog.config(state=DISABLED)
    ChatLog.yview(END)

from PIL import Image, ImageTk  
import os
PATH = os.path.dirname(os.path.realpath(__file__))

image_size = 18

base = customtkinter.CTk()
base.title("Covenant")
base.geometry("500x500")          # 🔹 WIDTH REDUCED to 500px
base.resizable(width=FALSE, height=FALSE)

try:
    mic = ImageTk.PhotoImage(Image.open("test_images/mic.png").resize((image_size, image_size), Image.Resampling.LANCZOS))
except:
    mic = None

ChatLog = Text(base, bd=0, bg="white", height="6", width="40", font=("Arial", 10))
ChatLog.config(state=NORMAL)
ChatLog.config(state=DISABLED)

scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
ChatLog['yscrollcommand'] = scrollbar.set

SendButton = customtkinter.CTkButton(base, text="Send", width=60, height=35,
                 corner_radius=6, command=send)

if mic:
    MICButton = customtkinter.CTkButton(base, text="", width=35, height=35, image=mic,
                     corner_radius=6, command=voice)
else:
    MICButton = customtkinter.CTkButton(base, text="🎤", width=35, height=35,
                     corner_radius=6, command=voice)

EntryBox = Text(base, bg="white", width=25, height=3, font=("Arial", 10))

# ----- Compact placement for 500px width -----
# Scrollbar on the right
scrollbar.place(x=475, y=6, height=440)          # x = window_width - 25

# Chat log fills the space left
ChatLog.place(x=6, y=6, height=440, width=463)   # width = 500 - 6 - 25 - 6 = 463

# Input row – entry box, then Send, then Mic, tightly packed
entry_width = 373                                 # 500 - 12 - 60 - 35 - 20 = 373
gap = 10

EntryBox.place(x=6, y=450, height=40, width=entry_width)
SendButton.place(x=6 + entry_width + gap, y=450)          # x = 389
MICButton.place(x=6 + entry_width + gap + 60 + gap, y=450) # x = 459

base.attributes('-topmost', True)
base.focus_force()
base.attributes('-topmost', False)

base.mainloop()