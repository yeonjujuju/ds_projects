import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import linear_model as lm
from sklearn.linear_model import LogisticRegression


df = pd.read_csv("Spotify_Youtube.csv")
df = df.dropna()


check_features = df[["Danceability", "Energy", "Loudness", "Comments", "Tempo", "Speechiness", "Liveness", "Instrumentalness", "Key", "Acousticness", "Valence"]]
df_corr = check_features.corr()
# print(df_corr)

dff = df.copy()
dff["dance_able"] = (dff["Danceability"] >= 0.65).astype(int)


def feature_engineering(data):

    data["high_energy"] = (data["Energy"] >= 0.6).astype(int)
    data["dance_bpm"] = (data["Tempo"] <= 140).astype(int)
    data["loud_enough"] = (data["Loudness"] <= -10).astype(int)
    data["more_livey"] = (data["Liveness"] <= 0.33).astype(int)
    data["more_positive"] = (data["Valence"] >= 0.55).astype(int)
    data["isit_accoustic"] = (data["Acousticness"] <= -0.5).astype(int)

    featured_data = data[["high_energy", "dance_bpm", "loud_enough", "more_livey", "more_positive", "isit_accoustic"]]
    return featured_data

train, val = train_test_split(dff, test_size = 0.2, random_state = 42)
train = train.reset_index(drop = True)

x_train = feature_engineering(train)    
y_train = train["dance_able"]
  
x_val = feature_engineering(val)
y_val = val["dance_able"]

my_model = LogisticRegression(fit_intercept=True)
my_model.fit(x_train, y_train)

train_predictions = my_model.predict(x_train)
val_predictions = my_model.predict(x_val)

training_accuracy = np.mean(train_predictions == train["dance_able"])
val_accuracy = np.mean(val_predictions == val["dance_able"])

print(f"training accracy is {training_accuracy} and val_accracy is {val_accuracy}.")








