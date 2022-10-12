from flask import Blueprint, render_template, request, flash
from joblib import dump, load

import pandas as pd
import string
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.tokenize import WhitespaceTokenizer
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')




#Clean review (1)
def get_wordnet_pos(pos_tag):
    if pos_tag.startswith('J'):
        return wordnet.ADJ
    elif pos_tag.startswith('V'):
        return wordnet.VERB
    elif pos_tag.startswith('N'):
        return wordnet.NOUN
    elif pos_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN

def clean_text(text):
    # lower text
    text = text.lower()
    # tokenize text and remove puncutation
    text = [word.strip(string.punctuation) for word in text.split(" ")]
    # remove words that contain numbers
    text = [word for word in text if not any(c.isdigit() for c in word)]
    # remove stop words
    stop = stopwords.words('english')
    text = [x for x in text if x not in stop]
    # remove empty tokens
    text = [t for t in text if len(t) > 0]
    # pos tag text
    pos_tags = pos_tag(text)
    # lemmatize text
    text = [WordNetLemmatizer().lemmatize(t[0], get_wordnet_pos(t[1])) for t in pos_tags]
    # remove words with only one letter
    text = [t for t in text if len(t) > 1]
    # join all
    text = " ".join(text)
    return(text)


#Add sentiment score (2)
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')
def sentiment(review_df):
    sid = SentimentIntensityAnalyzer()
    review_df = review_df
    review_df["sentiments"] = review_df["review"].apply(lambda x: sid.polarity_scores(x))
    return pd.concat([review_df.drop(['sentiments'], axis=1), review_df['sentiments'].apply(pd.Series)], axis=1)


#Convert review into vector data (3)
import gensim
from gensim.test.utils import common_texts, get_tmpfile
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

def vector(review_df):
    model = Doc2Vec.load("website/doc2vec_trained.model")

    review_df = review_df
    doc2vec_df = review_df["review_clean"].apply(lambda x: model.infer_vector(x.split(" "))).apply(pd.Series)
    doc2vec_df.columns = ["doc2vec_vector_" + str(x) for x in doc2vec_df.columns]
    return pd.concat([review_df, doc2vec_df], axis=1)


from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
import pickle

def TFIDF(review_df):
    review_df = review_df
    #Load tfidf_transformer
    cv = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("website/feature.pkl", "rb")))

    #Compute IDF score
    tfidf_transformer=TfidfTransformer()
    tfidf = tfidf_transformer.fit_transform(cv.fit_transform(review_df["review_clean"])).toarray()

    #Concat tfidf score with original review df
    tfidf_df = pd.DataFrame(tfidf, columns = cv.get_feature_names())
    tfidf_df.columns = ["word_" + str(x) for x in tfidf_df.columns]
    tfidf_df.index = review_df.index
    return pd.concat([review_df, tfidf_df], axis=1)

def convert(review):
    data = {"review": [review]}
    review_df = pd.DataFrame(data)

    #Clean review (1)
    review_df["review_clean"] = review_df["review"].apply(lambda x: clean_text(x))

    #Add sentiment score (2)
    review_df = sentiment(review_df)

    # add number of characters column
    review_df["nb_chars"] = review_df["review"].apply(lambda x: len(x))

    # add number of words column
    review_df["nb_words"] = review_df["review"].apply(lambda x: len(x.split(" ")))

    #Convert review into vector data (3)
    review_df = vector(review_df)

    #Add TFIDF score
    review_df = TFIDF(review_df)
    print(review_df)
    del review_df["review"]
    del review_df["review_clean"]
    return review_df.values.tolist()
    
    



views = Blueprint('views', __name__)


model = load("website/review_prediction.joblib")

@views.route('/', methods=['GET', 'POST'])
def home():
    if request.method == 'POST':
        review = request.form.get('review')

        if len(review) == 0:
            flash('Please input review', category='error')
        else:
            review_to_predict = convert(review)
            score = model.predict(review_to_predict)
            if score == 0:
                flash("This is a GOOD review", category='success')
            else:
                flash("This is a BAD review", category='success')
            

    return render_template("home.html")