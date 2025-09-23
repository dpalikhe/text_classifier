import pandas as pd 
import text_classifier as tc
import scipy.stats as stats


df = pd.read_csv("dataset.csv", encoding="cp1252", header = None)

df = df.rename(columns={
    df.columns[0]: "target",
    df.columns[5]: "text"})

df = df[["text", "target"]].dropna().drop_duplicates(subset=['text'])
df.info()

# target is polarity of the tweet; 0 = negative, 2 = neutral, 4 = positive) 
# the dataset does not have any neutrals.
print(df['target'].value_counts())

# hatespeech
hs = tc.text_classifier("hs_data.csv", df.iloc[:, 0].astype(str))
hs_pipeline = hs.svc()
hs.fit(hs_pipeline)
hs_predicts = hs.predict(hs_pipeline)

# sentiment
sentiment = tc.text_classifier("sentiment_data.csv", df.iloc[:, 0].astype(str))
sentiment_pipeline = sentiment.rf()
sentiment.fit(sentiment_pipeline)
sentiment_predicts = sentiment.predict(sentiment_pipeline)

df["hs_class"] = hs_predicts # hate speech class
df["sentiment_class"] = sentiment_predicts # sentiment class

# Pearson's Chi-Square Test
# Do the average sentiment_class differ between different hs_data groups?

contingency_table1 = pd.crosstab(df['hs_class'], df['sentiment_class'])
print(contingency_table1)

stat, p, dof, expected = stats.chi2_contingency(contingency_table1)

alpha = 0.05
print("p value = " + str(p))

if p <= alpha:
    print('reject H0')
else:
    print('accept H0')


contingency_table2 = pd.crosstab(df['target'], df['sentiment_class'])
print(contingency_table2)

stat, p, dof, expected = stats.chi2_contingency(contingency_table2)

alpha = 0.05
print("p value = " + str(p))

if p <= alpha:
    print('reject H0')
else:
    print('accept H0')





