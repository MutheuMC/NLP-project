import math
import random
import torch.nn as nn
from text_preprocessing import preprocess_sentence
import tweepy
import tensorflow
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import BertTokenizer
from transformers import BertModel
import transformers
import torch
from flask import Flask, render_template, flash, request, url_for, redirect, session
# Libraries for general purpose
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Text cleaning
import re
import string
import emoji
import nltk
from nltk.stem import WordNetLemmatizer, PorterStemmer
# import nltk
from nltk.corpus import stopwords
import tokenizers
nltk.download('stopwords')
stopword = stopwords.words('english')


# Transformers library for BERT
# from transformers import BertModel


api_key = "zZafvJzRpee6Kl2ZQcfUjPBxg"
api_secret_key = "saHJbaIoR7CWXvYGP4sAuJgKSIanaUqhGh2OVEECDktyKkYF8k"
access_token = "1502212195895152641-XdNr0lrPcnT7YPGpQQ21IwlUdypVAP"
access_token_secret = "Fr8LNePuszvrkgWTiwQxVpxI0e7dSyYPyRU0yLx6evjdr"
# Tweepy authentication
# with open('C:/Users/user/Desktop/javascript/project/aggressive-tweet-analyzer-main/web_app/text_preprocessing.py') as data_file:
# credentials = json.load(data_file , strict=False)
auth = tweepy.OAuthHandler(api_key, api_secret_key)
auth.set_access_token(access_token, access_token_secret)
api = tweepy.API(auth, wait_on_rate_limit=True)

# device = cuda

Max_LEN = 512

app = Flask(__name__)

device = torch.device('cpu')


class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        # Specify hidden size of BERT, hidden size of the classifier, and number of labels
        n_input = 768
        n_hidden = 50
        n_output = 5
        # Instantiate BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Add dense layers to perform the classification
        self.classifier = nn.Sequential(
            nn.Linear(n_input,  n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        # Add possibility to freeze the BERT model
        # to avoid fine tuning BERT params (usually leads to worse results)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        # Feed input data to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)

        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits


model = torch.load('modelBert1.pth', torch.device('cpu'))
model.eval()
tokenizer = pd.read_pickle('tokenizer.pickle')


@app.route('/')
def home():
    return render_template("home.html")


def bert_tokenizer(data, text_tokinizer):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = text_tokinizer.encode_plus(
            text=sent,
            # Add `[CLS]` and `[SEP]` special tokens
            add_special_tokens=True,
            max_length=235,             # Choose max length to truncate/pad
            pad_to_max_length=True,         # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
        )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks
# tweet="black is a curse i fucking hate black people"


@app.route('/prediction', methods=['POST', "GET"])
def sent_anly_prediction():
    if request.method == 'POST':
        text = request.form['tweet']
        tweet = str(text)
        tweet = " ".join(tweet.split())
        # label_name = ""

        a, b = bert_tokenizer([tweet], tokenizer)

        x = model(a.to(device), b.to(device))

        s = torch.nn.Softmax(dim=1)
        pred_labels = s(x).argmax()

        if pred_labels == 0:
            label_name = "Cyberbullying in form of religion"
        elif pred_labels == 1:
            label_name = "Cyberbullying in form of age"
        elif pred_labels == 2:
            label_name = "Cyberbullying in form of ethnicity"
        elif pred_labels == 3:
            label_name = "Cyberbullying in form of gender"
        elif pred_labels == 4:
            label_name = "Not_cyberbullying"

    return render_template('result.html', prediction = label_name, tweets= tweet)



@app.route('/result', methods=['POST'])
def result():
    # Get the Twitter handle from the user input
   # Get the Twitter handle from the user input
    if request.method == 'POST':
      
        user = request.form['user_handle']

        # Remove @ if the handle is inputed with it
        if '@' in user:
            user = user.split('@')[1]

    input_ids = []
    attention_masks = []    

        # Extract tweets
    tweet_dict = [{'tweet': tweet.full_text,  # Full text
                       'created_at': tweet.created_at,  # Date
                       'username': user,  # Username
                       'headshot_url': tweet.user.profile_image_url,  # Profile image URL
                       # Tweet URL
                       'url': f'https://twitter.com/user/status/{tweet.id}'
                       } for tweet in tweepy.Cursor(api.user_timeline,
                                                    screen_name=user,
                                                    exclude_replies=False,  # Include replies
                                                    include_rts=False,  # Exclude retweets
                                                    tweet_mode="extended"  # Include full tweet text
                                                    ).items(10)]  # Extract 100 items
    #     # Save only the tweet texts in a list
    tweet_text_list = [tweet['tweet'] for tweet in tweet_dict]

    #    # Preprocess tweets
    tweets_clean = list(map(preprocess_sentence, tweet_text_list))

    # tweet = random.choice(tweets_clean)
    # print(tweet)

    
    # a, b = bert_tokenizer([tweet], tokenizer)

    # x = model(a.to(device), b.to(device))

    # s = torch.nn.Softmax(dim=1)
    # pred_labels = s(x).argmax()

    

        #  tweet_dict['label'] = ''
        #  tweet_dict['label'].append(pred_labels)

    # for tweet, pred in zip(tweet_dict, pred_labels):
    #         tweet['label'] = pred

    # 
    
 
   
    for tweet in tweets_clean:
            encoded_sent = tokenizer.encode_plus(
                text=tweet,
                add_special_tokens=True,        # Add `[CLS]` and `[SEP]` special tokens
                max_length=235,             # Choose max length to truncate/pad
                pad_to_max_length=True,         # Pad sentence to max length 
                return_attention_mask=True      # Return attention mask
                )
            input_ids.append(encoded_sent['input_ids'])
            attention_masks.append(encoded_sent['attention_mask'])

        # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)
   
    preds = model(input_ids ,attention_masks)
    

    s = torch.nn.Softmax()
    # for i, pred in enumerate(preds):
    # #     print(pred)
    #      pred_labels= s(pred).argmax()
    #     #  print(pred_labels)
    pred_labels = [s(pred).argmax() for i, pred in enumerate(preds)]


        # Create a new variable 'label' for each tweet in tweet_dict
    for tweet, pred in zip(tweet_dict, pred_labels):
        if pred == 0:
            label_name = "religion"
        elif pred == 1:
                label_name = "age"
        elif pred == 2:
                label_name = "ethnicity"
        elif pred == 3:
                label_name = "gender"
        elif pred == 4:
                label_name = "not_cyberbullying"
        tweet['label'] = label_name
        # print(tweet_dict)

        # Define output variables
        # Filter to keep the aggressive tweets only

    religion = [
              tweet for tweet in tweet_dict if tweet['label'] == 'religion']
    age = [
              tweet for tweet in tweet_dict if tweet['label'] == 'age']
    ethnicity = [
              tweet for tweet in tweet_dict if tweet['label'] == 'ethnicity']
    gender = [
              tweet for tweet in tweet_dict if tweet['label'] == 'gender']
    not_cyberbullying = [
              tweet for tweet in tweet_dict if tweet['label'] == 'not_cyberbullying']
    

    return render_template('tweet_result.html', tweets=religion, tweets1=age, tweets2=ethnicity, tweets3=gender, tweets4=not_cyberbullying  )


if __name__ == "__main__":
    app.run(debug=True)
