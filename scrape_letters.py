#Code adapted from: https://hackernoon.com/how-to-scrape-google-with-python-bo7d2tal
import requests
from bs4 import BeautifulSoup
import re
import time as time
import pickle
import numpy as np
from datetime import datetime

# Create a web scraper to obtain English translations of Seneca's letters to his friend Lucilius
class Scraper:

    def __init__(self, index:int):
        self.letter_index = index
        self.url = f"https://en.wikisource.org/wiki/Moral_letters_to_Lucilius/Letter_{self.letter_index}"
        self.USER_AGENT_HEADERS = {"user-agent":"Mozilla/5.0 (X11; Ubuntu; Linux i686; rv:24.0) Gecko/20100101 Firefox/24.0"}
        self.letter = ""
        
    def scrape_letter(self):
        # GET request
        self.response = requests.get(self.url, headers=self.USER_AGENT_HEADERS)
        # if response status code is 200
        if self.response:
          try:
              #Parse HTML response
              soup = BeautifulSoup(self.response.text, 'html.parser')
              raw_text = soup.findAll("p")
              just_text = [t.text for t in raw_text]
              for paragraph in just_text:
                  self.letter += paragraph
          except AttributeError:
              print("Something went wrong (Attribute Error)")
        else:
            print(f"Webpage response status code did not return 200 (STATUS: {response.status_code})")


#Get all 124 letters to Lucilius
letters = []
for l in range(0, 124):
    l = Scraper(index=l+1)
    l.scrape_letter()
    letters.append(l.letter)
    time.sleep(0.1)

# Create functions to process text for the model
def REPLACE_NEW_LINE(x):
    return re.compile("\\n").sub(" ", x)
    
def REPLACE_FOOTNOTE(x):
    return re.compile("[\\.]\\[\\d+\\]").sub(" ", x)
    
def REPLACE_FORMAT(x):
    return re.compile("\\'").sub("'", x)

def REPLACE_QUOTATION(x):
    return re.compile('"').sub("", x)

def REPLACE_CHARACTER(x):
    # return re.compile("[\\–;:\\-,\[\]\/]").sub(" ", x)
    return re.compile("[\[\]\/]").sub(" ", x)

def REPLACE_NUMBER(x):
    return re.compile("\\d+\\.").sub("", x)

def REPLACE_DIGIT(x):
    return re.compile("\\d+").sub("", x)

def REPLACE_CLOSING(x):
    return re.compile("\\sFarewell.*").sub("", x)

def REPLACE_BLANK(x):
    return re.compile("\\s{2,}").sub(" ", x)

def REPLACE_PADDING(x):
    return re.compile("\\s+$|^\\s+").sub("", x)
    
def REPLACE_EMPTY_SENTENCE(x):
    return re.compile("\\.\\s+\\.").sub(".", x)
    
def pre_process_text(letter):
    letter = REPLACE_NEW_LINE(letter)
    letter = REPLACE_FOOTNOTE(letter)
    letter = REPLACE_CHARACTER(letter)
    letter = REPLACE_FORMAT(letter)
    # letter = REPLACE_QUOTATION(letter)
    letter = REPLACE_NUMBER(letter)
    letter = REPLACE_DIGIT(letter)
    letter = REPLACE_EMPTY_SENTENCE(letter)
    letter = REPLACE_BLANK(letter)
    letter = REPLACE_CLOSING(letter)
    letter = REPLACE_PADDING(letter)
    return letter

# Process the text in all letters
processed_letters = [pre_process_text(letter) for letter in letters]
char_length = [len(l) for l in processed_letters]
print(f"N = {len(letters)} Letters; Min. Character Length: {np.min(char_length)}; Max. Character Length: {np.max(char_length)}")

# Save the output to hard disk
fname = "data/letters_punct.pkl"
with open(fname, 'wb') as f:
    pickle.dump(processed_letters, f)

# Save the output to hard disk
fname = 'data/corpus_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())
with open(fname, 'w') as f:
    f.writelines(processed_letters)

