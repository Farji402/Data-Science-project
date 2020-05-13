import pandas as pd
import numpy as np


# Defining combine_text_columns()
def combine_text_columns(data_frame):
    """ converts all text in each row of data_frame to single vector """
    to_drop = ['user_name', 'review_title', 'review_description', 'price', 'points', 'variety']
    # Drop non-text columns that are in the df
    to_drop = set(to_drop) & set(data_frame.columns.tolist())
    text_data = data_frame.drop(to_drop, axis= 'columns')

    # Replace nans with blanks
    text_data.fillna('', inplace= True)

    # Join all text items in a row that have a space in between
    return text_data.apply(lambda x: " ".join(x), axis=1)




#Reading the train data
df = pd.read_csv('train.csv')


#Changing variety type from string to Category

df['variety'] = df['variety'].astype('category')

#Importing all scikit-learn packages required here
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer




#replacing missing vales in numeric data with the mean
imp = SimpleImputer(missing_values=np.nan, strategy='mean')



get_text_data = FunctionTransformer(combine_text_columns, validate= False)
get_numeric_data = FunctionTransformer(lambda x: x[['price', 'points']], validate= False)


#making a pipeline
pl = Pipeline([
            ('union', FeatureUnion([
            ('numeric_features', Pipeline([
                ('selector', get_numeric_data),
                ('imputer', SimpleImputer())
            ])),
            ('text_features', Pipeline([
                ('selector', get_text_data),
                ('vectorizer', TfidfVectorizer()),
            ]))
            ])

            ),
            ('Scale', MaxAbsScaler()),
            ('clf', DecisionTreeClassifier())
]
)

#spliting data with 20% data given for train accuracy
X_train, X_test, y_train, y_test = train_test_split(df, pd.get_dummies(df['variety']),test_size= 0.2,  random_state= 20)

#fitting the model
pl.fit(X_train, y_train)

#predicting values
y_pred = pl.predict(X_test)

#Accuracy score on the train data
print(accuracy_score(y_test, y_pred))

test_data = pd.read_csv('test.csv')

test_pred = pl.predict(test_data)

test_data_pred = pd.concat([test_data, pd.DataFrame(test_pred)], axis=0)

"""test_data_pred.to_csv('test_data_pred.csv')"""
