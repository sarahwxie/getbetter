# import numpy as np
import pandas as pd
# import time
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer
# import pickle
# from nltk.probability import ProbabilisticMixIn
from flask import Flask, render_template, request
# import requests

app = Flask(__name__)

class TextAnalysis():
    def analyse_journal(self, entry, lower= True):
        '''
        Pulls out words
        '''
        stop_words = set(stopwords.words('english'))  # Common stop words
        other_words = set(['.', ","])
        if lower:
            word_tokens = word_tokenize(entry.lower())  # Tokenise
            filtered_sentence = [
                w for w in word_tokens if not (w in stop_words or w in other_words)]  # Filters
        else:
            word_tokens2 = word_tokenize(entry)
            word_tokens = word_tokenize(entry.lower())
            filtered_sentence = []
            for w in range(len(word_tokens)):
                if not (word_tokens[w] in stop_words or word_tokens[w] in other_words):
                    filtered_sentence.append(word_tokens2[w])
        return filtered_sentence

    def add_user(self, username, password):
        '''
        Adds a user
        '''
        filename = "users/"+username+".csv"
        with open('data.csv', 'a') as f:
            if username in pd.read_csv('data.csv').iloc[:, 0].values:
                print("ERROR")
            else:
                f.write(username+ ","+ password+","+ filename + "\n")
                with open(filename, "a") as f:
                    f.write("WORD, SENTIMENT, TIMES\n")

    def sentiment(self, text):
        '''
        Really rough Sentimental Analysis.
        '''
        text = self.analyse_journal(text)
        print(text)
        posnum = 0
        negnum = 0
        neunum = 0
        with open("negative-words.txt", "r") as f: neg = set(f.read().split("\n"))
        with open("positive-words.txt", "r") as f: pos = set(f.read().split("\n"))
        for i in text:
            if i in neg: negnum += 1
            elif i in pos: posnum += 1
            else: neunum += 1
        print(posnum, negnum, neunum)
        print(posnum/(negnum+posnum) * 2 - 1)
        return posnum/(negnum+posnum) * 2 - 1 
        
    def get_key_words(self, text):
        '''
        Puts together the key words and the sentiment
        '''
        words = self.analyse_journal(text, False)
        tagged = nltk.pos_tag(words)
        print(tagged)
        word_types_wanted = set(
            ["NN", "NNS", "NNP", "NNPS", "VB", "VBP", "VBD", "VBN"])
        val = self.sentiment(text) / len(words) # Sentiment divided by the number of major words
        new_words_and_val = []
        for word, wt in tagged:
            if wt in word_types_wanted:
                new_words_and_val.append([word, val])
        return new_words_and_val

    def update_words(self, user, text):
        '''
        Updates the CSV
        '''
        new_set = self.get_key_words(text)
        print(new_set)
        data = pd.read_csv("users/" + user + ".csv")
        old_words = list(data.iloc[:, 0])
        old_vals = list(data.iloc[:, 1])
        old_num = list(data.iloc[:, 2])
        for new_word in new_set:
            found = False
            for j in range(len(old_words)):
                print(new_word)
                if old_words[j] == new_word[0]:
                    old_vals[j]+=new_word[1]
                    old_num[j]+=1
                    found = True
                    break 
            if not found:
                old_words.append(new_word[0])
                old_vals.append(new_word[1])
                old_num.append(1)
        dct = {'WORD': old_words, 'SENTIMENT': old_vals, 'TIMES': old_num}
        df = pd.DataFrame(dct)
        df.to_csv('users/'+user+'.csv', index=False)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/login.html')
def login():
    return render_template('login.html')


@app.route('/SignUp.html')
def signup():
    return render_template('SignUp.html')


@app.route('/About-Us.html')
def aboutus():
    return render_template('About-Us.html')


@app.route('/Information.html')
def Information():
    return render_template('Information.html')


@app.route('/Predictions.html')
def Predictions():
    return render_template('Predictions.html')


@app.route('/Journals.html')
def Journals():
    return render_template('Journals.html')


@app.route('/Journal.html/recieve_data', methods=["POST"])
def Journals1():
    # add_user("Hello", "abc")
    # update_words("Hello", "I hate life.")

    username = request.form['please1']
    passName = request.form['please2']
    print(username, passName)
    journal = TextAnalysis()
    journal.update_words("Hello", username)
    journal.update_words("Hello", passName)

    return render_template('Journals.html')
    



if __name__ == '__main__':
  app.run(debug=True)


THRESHOLD = 0.2
nltk.download('stopwords')
# nltk.download('punkt')

def analyse_journal(entry, lower = True):
    '''
    Pulls out words
    '''
    stop_words = set(stopwords.words('english'))  # Common stop words
    other_words = set(['.', ","])
    if lower:
        word_tokens = word_tokenize(entry.lower())  # Tokenise
        filtered_sentence = [
            w for w in word_tokens if not (w in stop_words or w in other_words)]  # Filters
    else: 
        word_tokens2 = word_tokenize(entry)
        word_tokens = word_tokenize(entry.lower())
        filtered_sentence = []
        for w in range(len(word_tokens)):
            if not (word_tokens[w] in stop_words or word_tokens[w] in other_words):
                filtered_sentence.append(word_tokens2[w])
    return filtered_sentence


def add_user(username, password):
    '''
    Adds a user
    '''
    filename = "users/"+username+".csv"
    with open('data.csv', 'a') as f:
        if username in pd.read_csv('data.csv').iloc[:, 0].values:
            print("ERROR")
        else:
            f.write(username+ ","+ password+","+ filename + "\n")
            with open(filename, "a") as f:
                f.write("WORD, SENTIMENT, TIMES\n")

def sentiment(text):
    '''
    Really rough Sentimental Analysis.
    '''
    text = analyse_journal(text)
    print(text)
    posnum = 0
    negnum = 0
    neunum = 0
    with open("negative-words.txt", "r") as f: neg = set(f.read().split("\n"))
    with open("positive-words.txt", "r") as f: pos = set(f.read().split("\n"))
    for i in text:
        if i in neg: negnum += 1
        elif i in pos: posnum += 1
        else: neunum += 1
    print(posnum, negnum, neunum)
    print(posnum/(negnum+posnum) * 2 - 1)
    return posnum/(negnum+posnum) * 2 - 1 
    
def get_key_words(text):
    '''
    Puts together the key words and the sentiment
    '''
    words = analyse_journal(text, False)
    tagged = nltk.pos_tag(words)
    print(tagged)
    word_types_wanted = set(
        ["NN", "NNS", "NNP", "NNPS", "VB", "VBP", "VBD", "VBN"])
    val = sentiment(text) / len(words) # Sentiment divided by the number of major words
    new_words_and_val = []
    for word, wt in tagged:
        if wt in word_types_wanted:
            new_words_and_val.append([word, val])
    return new_words_and_val

def update_words(user, text):
    '''
    Updates the CSV
    '''
    new_set = get_key_words(text)
    print(new_set)
    data = pd.read_csv("users/" + user + ".csv")
    old_words = list(data.iloc[:, 0])
    old_vals = list(data.iloc[:, 1])
    old_num = list(data.iloc[:, 2])
    for new_word in new_set:
        found = False
        for j in range(len(old_words)):
            print(new_word)
            if old_words[j] == new_word[0]:
                old_vals[j]+=new_word[1]
                old_num[j]+=1
                found = True
                break 
        if not found:
            old_words.append(new_word[0])
            old_vals.append(new_word[1])
            old_num.append(1)
    dct = {'WORD': old_words, 'SENTIMENT': old_vals, 'TIMES': old_num}
    df = pd.DataFrame(dct)
    df.to_csv('users/'+user+'.csv', index=False)


add_user("Hello", "abc")
update_words("Hello", "I hate life.")
