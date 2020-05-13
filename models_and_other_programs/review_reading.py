import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

#Reading the data_frame
df = pd.read_csv('train.csv')

df_bad_reviews = df[df['points'] < 90]

df_good_reviews = df[df['points'] >= 98]

TOKENS_ALPHANUMERIC = '[A-Za-z0-9]+(?=\\s+)'
from nltk.tokenize import word_tokenize
import re

def remove_noise(text, stop_words = ["i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your", "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she", "her", "hers", "herself", "it", "its", "itself", "they", "them", "their", "theirs", "themselves", "what", "which", "who", "whom", "this", "that", "these", "those", "am", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an", "the", "and", "but", "if", "or", "because", "as", "until", "while", "of", "at", "by", "for", "with", "about", "against", "between", "into", "through", "during", "before", "after", "above", "below", "to", "from", "up", "down", "in", "out", "on", "off", "over", "under", "again", "further", "then", "once", "here", "there", "when", "where", "why", "how", "all", "any", "both", "each", "few", "more", "most", "other", "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than", "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"]):
    tokens = word_tokenize(text)

    cleaned_tokens = []
    for token in tokens:
        token = re.sub(TOKENS_ALPHANUMERIC, '', token)
        if len(token) > 2 and token.lower() not in stop_words:
            cleaned_tokens.append(token.lower())

    return cleaned_tokens

#Model to get top labels in the review_description
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer


def top_labels(df):


    tokenizer = TfidfVectorizer(max_df= 0.9, max_features= 100, min_df= 0.2, tokenizer= remove_noise, ngram_range= (1,2))
    wfa = tokenizer.fit_transform(df['review_description'])

    nmf = NMF(n_components= 15)
    nmf_features = nmf.fit_transform(wfa)
    for i in range(15):
        nmf_comp_max = np.argmax(nmf.components_[i])


        print(tokenizer.get_feature_names()[nmf_comp_max])




"""print('\n\n\n')
top_labels(df_bad_reviews)
print('\n\n\n')
top_labels(df_good_reviews)"""


#Clustring of users based on their review_description
#We'll use cosine similarity rule to find it out

from sklearn.preprocessing import normalize

def similar_user(user_index):
    """print list of 10 users having similar review_description to
        that of the user at user_index"""

    tokenizer = TfidfVectorizer(max_df= 0.8, max_features= 100, min_df= 0.2, tokenizer= remove_noise)
    wfa = tokenizer.fit_transform(df['review_description'])
    df['user_and_review'] = df['user_name'] + "--" + df['review_description']
    nmf = NMF(n_components= 30)
    nmf_features = nmf.fit_transform(wfa)
    norm_features = normalize(nmf_features)

    df_features = pd.DataFrame(norm_features, index= df['user_and_review'], columns= None)

    user_at_index = df_features.iloc[user_index, :]

    similarities = df_features.dot(user_at_index)

    print(similarities.nlargest(5))



similar_user(5)
#Similarly, Clustring of review_titles based on their review_description
#We'll use cosine similarity rule to find it out

def similar_review_title(review_index):
    """print list of 10 review_titles having similar review_description to
        that of the review_title at review_index"""

    tokenizer = TfidfVectorizer(max_df= 0.8, max_features= 100, min_df= 0.2, tokenizer= remove_noise)
    wfa = tokenizer.fit_transform(df['review_description'])

    nmf = NMF(n_components= 20)
    nmf_features = nmf.fit_transform(wfa)
    norm_features = normalize(nmf_features)

    df_features = pd.DataFrame(norm_features, index= df['review_title'], columns= None)

    user_at_index = df_features.iloc[review_index, :]

    similarities = df_features.dot(user_at_index)

    print(similarities.nlargest(5))
