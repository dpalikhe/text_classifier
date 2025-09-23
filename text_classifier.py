import pandas as pd
import numpy as np 
import re
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

class text_classifier(object):		  	   		 	 	 			  		 			 	 	 		 		 			  	   		 	 	 			  		 			 	 	 		 		 	
    def __init__(self, train_data_path:str, test_data = None,n_splits=5, shuffle=True, random_state=819):
        self.train_data_path = train_data_path 
        self.train_data = self.load_train_data()
        self.X,self.y = self.x_y()
        
        self.test_data = test_data         
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
        
    # loading the data and renaming columns
    def load_train_data(self):
        if self.train_data_path == "hs_data.csv":
            df = pd.read_csv(self.train_data_path)
            df = df[["tweet", "class"]].rename(columns={"tweet": "text"})
            # Dropping any NAs and duplicates
            df = df.dropna().drop_duplicates(subset=['text', 'class'])
            # Hate speech class label: 0 - hate speech, 1 - offensive language & 2 - neither
            df['class'] = df['class'].astype('category')
            
        elif self.train_data_path == "sentiment_data.csv":
            df = pd.read_csv(self.train_data_path, encoding="cp1252")
            df = df[["text", "sentiment"]].rename(columns={"sentiment": "class"})
            df['class'] = df['class'].replace({'negative': 0,'neutral': 1,'positive': 2})
            df = df.dropna().drop_duplicates(subset=['text', 'class'])
            # Sentiment label: 0 — Negative, 1 — Neutral & 2 — Positive
            df['class'] = df['class'].astype('category')
            
        else:
            raise ValueError("Training data path must be 'hs_data.csv' or 'sentiment_data.csv'.")
        return df

    ## EDA ##
    def eda(self):
        self.train_data.info()
        print(self.train_data.describe())
        
        class_counts = self.train_data['class'].value_counts().reset_index()

        if self.train_data_path == "hs_data.csv":
            class_counts['class'].replace({0: "hate speech", 1:"offensive", 2:"neither"}, inplace = True)
        elif self.train_data_path == "sentiment_data.csv":
            class_counts['class'].replace({0: "Negative", 1:"Neutral", 2:"Positive"}, inplace = True)
        
        class_counts.plot(x='class', y='count',kind='bar', legend = False)
        plt.title('Class Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.xticks(rotation=0)
        plt.show()
    
    def x_y(self): 
        X = self.train_data['text']
        y = self.train_data['class']
        return X, y

    def clean_text(self,text):
        cleaned = re.sub(r'@\w+\s*', '', text) # remove usernames and the trailing space
        # cleaned = re.sub(r'&#\d+;', '', cleaned) # remove emojis
        cleaned = re.sub(r'http[s]?://\S+|www\.\S+', '', cleaned) #remove urls
        cleaned = re.sub("&amp;", '&', cleaned) # change &amp; to &
        return cleaned.strip().lower() # remove leading and trailing spaces

    ## Logistic Regression ##
    def lr(self):
        lr_pipeline = Pipeline([('cleaner', FunctionTransformer(lambda x: x.apply(self.clean_text))),
                        ('tfidf', TfidfVectorizer(stop_words='english')), # converts text to numerical features and ignores words like the , a, and, etc
                        ('clf', LogisticRegression(max_iter=500))])
        return lr_pipeline

    ## Using 5 fold cross validation for model evaluation ##
    def cv(self,pipeline):
        model = pipeline.steps[-1][1]
        if isinstance(model, LogisticRegression):
            print("Logistic Regression")
        elif isinstance(model, SVC):
            print("Support Vector Machine")
        elif isinstance(model, KNeighborsClassifier):
            print("KNN")
        elif isinstance(model, RandomForestClassifier):
            print("Random Forest")
            
        skf = StratifiedKFold(n_splits = self.n_splits, shuffle = self.shuffle, random_state = self.random_state)
        accuracy_scores = cross_val_score(pipeline, self.X, self.y, cv = skf)
        print(f'CV scores: {accuracy_scores}')
        print(f'Average 5-fold accuracy: {np.mean(accuracy_scores):.4f}')

    ## SVM ##
    def svc(self):
        svc_pipeline = Pipeline([('cleaner', FunctionTransformer(lambda x: x.apply(self.clean_text))),
                            ('tfidf', TfidfVectorizer(stop_words='english')),
                            ('clf', SVC())])
        return svc_pipeline

    ## KNN ##
    def knn(self):
        knn_pipeline = Pipeline([('cleaner', FunctionTransformer(lambda x: x.apply(self.clean_text))),
                            ('tfidf', TfidfVectorizer(stop_words='english')),
                            ('clf', KNeighborsClassifier())])
        return knn_pipeline

    ## Random Forest ##
    def rf(self):
        rf_pipeline = Pipeline([('cleaner', FunctionTransformer(lambda x: x.apply(self.clean_text))),
                        ('tfidf', TfidfVectorizer(stop_words='english')),
                        ('clf', RandomForestClassifier())])
        return rf_pipeline
    
    def fit(self, pipeline):
        pipeline.fit(self.X,self.y)
        self.is_fitted = True
        
    def predict(self, pipeline, new_text = None):
        if not self.is_fitted:
            raise RuntimeError("Need to fit the model (text_classifier.fit()) before predicting!")
        elif self.test_data is not None and new_text is not None: 
            raise ValueError(" Can only predict either a dataframe or a single text, not both at once.")
        elif self.test_data is not None:
            return pipeline.predict(self.test_data) # Predicts entire df
        elif new_text is not None: 
            df = pd.DataFrame({'text': [new_text]})
            return pipeline.predict(df['text']) # Predicts only one text
        else:
            raise ValueError("Load test data or provide new text before calling text_classifier.predict().")

if __name__ == "__main__": 
    # hate speech 		  	   		 	 	 			  		 			 	 	 		 		 	
    hs = text_classifier("hs_data.csv")
    hs.eda()
    hs.cv(hs.lr())
    hs.cv(hs.svc()) # high accuracy rate
    hs.cv(hs.knn())
    hs.cv(hs.rf())
    
    # sentiment analysis
    sentiment = text_classifier("sentiment_data.csv")
    sentiment.eda()
    sentiment.cv(sentiment.lr())
    sentiment.cv(sentiment.svc())
    sentiment.cv(sentiment.knn())
    sentiment.cv(sentiment.rf()) # high accuracy rate
    
