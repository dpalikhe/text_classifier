import pandas as pd 
import text_classifier as tc
from flask import Flask, request

#prevent flask from printing
import os
from os import path
import sys
f = open(os.devnull, 'w')
sys.stdout = f

# hatespeech
hs = tc.text_classifier("hs_data.csv")
hs_pipeline = hs.svc()
hs.fit(hs_pipeline)

# sentiment
sentiment = tc.text_classifier("sentiment_data.csv")
sentiment_pipeline = sentiment.rf()
sentiment.fit(sentiment_pipeline)


 #flask app
app = Flask(__name__)
@app.route("/")
def index():
    return """
    # <style>
    #   body {
    #     display: flex;
    #     justify-content: center;
    #     align-items: center;
    #     min-height: 100vh;
    #     margin: 0;
    #     font-family: sans-serif;
    #   }
    #   .container {
    #     text-align: center;
    #   }
    # </style>
    <div class="container">
    <form action="/submit_data" method = "POST">
    <br><br>
        <p><center><h3>Enter your text below </h3><input type = "text" name = "Text Box" /></p>
        <p><input type = "submit" value = "Submit" /></center></p>
    </form>
    """

@app.route("/submit_data", methods = ['POST'])
def get_label():  
  try:
    # to get labels of the new text
    text = request.form['Text Box']
    hs_predict = hs.predict(hs_pipeline, text)
    sentiment_predict = sentiment.predict(sentiment_pipeline, text)

    hs_dict = {0: "Hate Speech", 1: "Offensive Language",  2: "Neutral"}
    sentiment_dict = {0: "Negative", 1: "Neutral", 2: "Positive" }

 
    return f"""
      <center> <br> <br>
      Hate Speech Classification: {hs_dict[hs_predict[0]]} <br><br>
      Sentiment Classification: {sentiment_dict[sentiment_predict[0]]} <br><br>
      <form action="/" method="GET">
          <input type="submit" value="Go Back">
      </form>
      </center>
      """ 
  except ValueError:
    return "Invalid Input"

if __name__ == "__main__":
    app.run(host='0.0.0.0')




