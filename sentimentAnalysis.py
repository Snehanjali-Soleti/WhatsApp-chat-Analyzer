import re
import pickle
import nltk
from nltk.corpus import movie_reviews,stopwords
from nltk.tokenize import word_tokenize
# import nltk.classify.util
import pandas as pd
from nltk.stem import PorterStemmer


nltk.download('movie_reviews')
nltk.download('punkt')

documents = [(fileid, category)
             for category in movie_reviews.categories()
             for fileid in movie_reviews.fileids(category)]

# Create a DataFrame
df = pd.DataFrame(documents, columns=['fileid', 'category'])

# Add a column for the review text
df['text'] = df['fileid'].apply(lambda x: ' '.join(movie_reviews.words(x)))

# Display the DataFrame
print(df.head())


print(df.columns)

df['category'].replace({'pos':1,'neg':0},inplace=True)

print(df.head())

def convert_lower(text):
    return text.lower()

df['text']=df['text'].apply(convert_lower)


#removing special characters

def removespecial(text):
    x=''
    for i in text:
        if i.isalnum():
            x=x+i
        else:
            x=x+' '
    return x


df['text'] = df['text'].apply(removespecial)
print(df.head())


#remove stopwords followed by stemming

porter = PorterStemmer()

def remove_stopwords_and_stem(text):
    stop_words = set(stopwords.words('english'))
    words = nltk.word_tokenize(text)  # Tokenize text into words
    filtered_words = [word for word in words if word.lower() not in stop_words]
    stemmed_words = [porter.stem(word) for word in filtered_words]  # Stem each word
    return ' '.join(stemmed_words)

# Apply the remove_stopwords_and_stem function to the 'text' column
df['text'] = df['text'].apply(remove_stopwords_and_stem)

# Display the first few rows of the DataFrame to see the result
print(df.head())

print(df)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=2000)

x=cv.fit_transform(df['text']).toarray()
y = df['category'].values
print(x.shape)
print(y.shape)

from sklearn.model_selection import train_test_split

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)

from sklearn.naive_bayes import MultinomialNB,GaussianNB, BernoulliNB

clf2=MultinomialNB()

clf2.fit(x_train,y_train)


# with open('model.pkl', 'wb') as f:
#     pickle.dump(clf2, f)

y_pred2=clf2.predict(x_test)

from sklearn.metrics import accuracy_score

print("Multinomial:",accuracy_score(y_test,y_pred2))

# Perform sentiment analysis on chat content
def perform_sentiment_analysis(chat_content):
    # Load the trained classifier
    with open('model.pkl', 'rb') as f:
        classifier = pickle.load(f)

    output = ""
    pos_count = 0
    neg_count = 0
    opinion = {}
    
    pattern = re.compile(r'(\d{2}/\d{2}/\d{2}), (\d{2}:\d{2}) - ([^:]+): (.+)')
    
    for match in pattern.finditer(chat_content):
        date, time, name, chat = match.groups()
        chat_vectorized = cv.transform([chat]).toarray()
        res = classifier.predict(chat_vectorized)[0]
        sentiment_color = 'green' if res == 1 else 'red'
        bsentiment_color = '#e6ffe8' if res == 1 else '#ffe8e6'
        output += f"<p><span style='color:brown; font-size:20px; font-weight:bold; display:inline-block; width:80px;'>{name} :</span> <span style='border: 1px solid {sentiment_color}; padding: 8px; margin: 2px; border-radius: 5px; background-color: {bsentiment_color};display:inline-block; width:600px;'>{chat}</span></p>\n"
        
        if name not in opinion:
            opinion[name] = [0, 0]
        if res == 1:
            pos_count += 1
            opinion[name][0] += 1
        else:
            neg_count += 1
            opinion[name][1] += 1

    # Generate bar chart
    
    return output.strip(), pos_count, neg_count ,opinion
