import tweepy
import pickle
import gpt_2_simple as gpt2
from datetime import datetime
import tweepy
import json
import pandas as pd
import random as rand
import time
import numpy as np
import schedule
import re

# Authenticate to Twitter
with open('tokens.json') as f:
  keys = json.load(f)
  
auth = tweepy.OAuthHandler(keys["CONSUMER_KEY"], keys["CONSUMER_SECRET"])
auth.set_access_token(keys["ACCESS_TOKEN"], keys["ACCESS_TOKEN_SECRET"])

# Create API object
api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)


# # Get trending topics
# USA_WOE_ID = 23424977
#  
# usa_trends = api.trends_place(USA_WOE_ID)
#  
# trends = json.loads(json.dumps(usa_trends, indent=1))
# 
# for trend in trends[0]["trends"]:
# 	print (trend["name"])

# Load scraped text
fname = "data/letters_punct.pkl"
with open(fname, 'rb') as f:
    processed_letters = pickle.load(f)

#Get some sentence starters
phrase_starters = []
for letter in processed_letters:
    sentences = re.split("[\\.!?]", letter)
    for sentence in sentences:
        words = sentence.split(" ")
        phrase = " ".join(words[0:4])
        phrase = re.compile("\\s+$|^\\s+").sub("", phrase)
        phrase = re.compile('^\\"').sub("", phrase)
        phrase = re.compile("^\\'").sub("", phrase)
        phrase = re.compile('^"\\s\\"').sub("", phrase)
        phrase = re.compile("^\\'\\s\\'").sub("", phrase)
        phrase_starters.append(phrase)


#Load pre-trained model from storage
sess = gpt2.start_tf_sess()
gpt2.load_gpt2(sess)

#Create function that generates new text (with parameters) and updates Twitter status
def new_status(sess, word_count: int=500, temperature: float=0.7):
  #output file name
  prefix = phrase_starters[rand.randint(0, len(phrase_starters))]
  output_file = 'text/gpt2_gentext_{:%Y%m%d_%H%M%S}.txt'.format(datetime.utcnow())
  gpt2.generate_to_file(
                  sess,
                  destination_path = output_file,
                  length=np.floor(word_count*2.5),
                  temperature=temperature,
                  prefix=prefix,
                  # truncate=".",
                  nsamples=1,
                  batch_size=1
                  )
  # read text from generated file
  with open(output_file, "r") as f:
    generated_text = f.readlines()
  #split text into a list of sentences
  # generated_sentences = re.split(["[\\.!?]"], str(generated_text[0]))#.replace("\n"," ")
  generated_sentences = generated_text[0].split(".")
  #calculate the length of each sentence
  sentence_lengths = [len(sentence)+1 for sentence in generated_sentences]
  #cumulative sum of sentence lengths
  cumulative_length = np.cumsum(sentence_lengths)
  #find those sentences that can fit under the specified word count (default=500)
  sentence_index = np.where(cumulative_length<word_count)[0]
  #isolate only those initial sentences
  sentences = [generated_sentences[i] for i in sentence_index]
  #join sentences into a single string
  sentences_text = ".".join(sentences)+"."
  #split string by word
  words = sentences_text.split(" ")
  #containers
  status_text = []
  text = ""
  #sequentially adds words to a string until they exceed word limit
  while words:
    if (len(text) + len(words[0]) + 1) <  276:
      text = text + " " + words[0]
      del words[0]
    else:
      status_text.append(text)
      text = ""
  status_text.append(text)
  #how many tweets to spread text over?
  num_tweets = len(status_text)
  for tweet in range(0, num_tweets):
    if num_tweets > 1:
      text = status_text[tweet][1:]
      # Create a status update (with sequence formatting)
      status = text + " {}/{}".format(tweet+1, num_tweets)
      if tweet == 0:
        #Status update to timeline
        api.update_status(status)
        tweetId = api.get_user("SenecaGPT2").status.id
      else:
        #Create status update as a reply in thread
        api.update_status(status, in_reply_to_status_id = tweetId)
        tweetId = api.get_user("SenecaGPT2").status.id
    else:
      # Create a status update (without sequence formatting)
      status = status_text[tweet][1:]
      api.update_status(status)
    time.sleep(1)


# Get conversation prefix(es)
opts = {"sess":sess, "word_count":600, "temperature":0.73}
new_status(**opts)
schedule.every(3).to(12).hours.do(new_status(**opts))
while True:
  schedule.run_pending()
  time.sleep(10)


