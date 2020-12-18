from flask import Flask, render_template, request
import sklearn
import pickle
import pandas as pd 
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import spacy
import matplotlib 
matplotlib.use('Agg') 
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

with open('./models/model_lr.p', "rb") as f1:
        trained_lr = pickle.load(f1)

with open('./models/rf_model.p', "rb") as f2:
        trained_rf = pickle.load(f2)

with open('./models/dt_model.p', "rb") as f3:
        trained_dt = pickle.load(f3)

with open('./models/tv_model.p', "rb") as f4:
        tv = pickle.load(f4)

with open('./models/model_nb.p', "rb") as f5:
        trained_nb = pickle.load(f5)

nlp = spacy.load('en_core_web_md')

def predict(input, trained_model):
    list1 = [input]
    test_corpus = tv.transform(list1)  
    final_input= pd.DataFrame(test_corpus.todense().round(2), columns=tv.get_feature_names())
    prediction = trained_model.predict_proba(final_input)
    prediction=prediction[0][0]

    return (prediction *100).round(2)


app = Flask(__name__)  # instantiating a flask application, "__name__" is a reference to the current script

# HTTP request (e.g. clicking link) triggers route function, which renders HTML
@app.route('/')
def hello():
    return render_template('index.html')
    # automatically looking for templates/index.html

@app.route('/results')
def results():
    news_input = request.args.get("news")
    with open('./news_input.p', 'wb') as f6:
        pickle.dump(news_input, f6)

    model_input = request.values.get('ModelInput')
    print('ModelInput', model_input)
    
    if model_input=="Option 1":
        result = predict(news_input, trained_nb)
        print('hello')
        print(result)

    elif model_input=="Option 2":
        result = predict(news_input, trained_rf)
    
    elif model_input=="Option 3":
        result=predict(news_input, trained_dt)

    elif model_input=="Option 4":
        result=predict(news_input, trained_lr)

    
    s = SentimentIntensityAnalyzer()
    sentiment = s.polarity_scores(news_input)['compound']

    word_cloud()

    return render_template('results.html',
                            result=result,
                            sentiment=sentiment)

# @app.route('/word_cloud')
def word_cloud(): 

    with open('./news_input.p', "rb") as f7:
        news = pickle.load(f7)

    new_doc = []
    doc = nlp(news)
    for word in doc:
        if not word.is_stop and word.is_alpha:
            new_doc.append(word.lemma_.lower())
    
    word_in_red1= 'coronavirus'
    word_in_red2='facebook'
    word_in_red3='man'

    def color_word(word, *args, **kwargs):
        if (word == word_in_red1) or (word == word_in_red2) or (word == word_in_red3):
            color = '#ff0000' # red
        else:
            color = '#000000' # black
        return color

    wordcloud2 = WordCloud(height=1000, color_func=color_word, width=1500, 
                        background_color='white').generate(" ".join(new_doc))

    plt.figure(figsize = (15,15), facecolor = None)
    plt.imshow(wordcloud2)
    plt.axis('off')
    plt.tight_layout(pad = 0)
    plt.savefig('./static/img/wordcloud.png')

    return render_template('results.html')

if __name__ == '__main__':
    app.run(debug=True)
