# LIBRARIES
# Visuals and data wrangling
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Machine Learning
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC

# balancing data
import imblearn
from imblearn import under_sampling, over_sampling
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.over_sampling import RandomOverSampler

# _________________________________________________________________________________

###### This algoritm has the purpose of classify early warnings, which is news' description
###### that it's supposed to be defined as incident or not manually by a Team.
###### The main idea is using Machine learning to determine automatically if an alert is considered
###### critical incident or not based on the data already gathered manually by the Early Warning and Monitoring team.

# FUNCTIONS


# Function 1 - encoding data in 0 and 1
def convert_status(status):
    if status == "Triage/Closed":
        return 0
    if status == "Incident Created":
        return 1
    else:
        return 0

        ### Extracting data with pandas features


# _____________________________________________________________________________________

# STEPS

# -----------Load the data and get desired columns on data --- Incident Report----------

data_incident = pd.read_csv(r"incident_report1.csv", encoding="ISO-8859-1")
df_inc = pd.DataFrame(data_incident)
df_inc = df_inc.dropna(subset=["Description"])
df_inc["is_incident"] = 1
df_inc = df_inc[["Description", "is_incident"]]

# -----------Load the data and get desired columns --- Early Warning Report-------------

data_ew = pd.read_csv(r"earlywarning_report1.csv", encoding="UTF-8")
df_ew = pd.DataFrame(data_ew)
df_ew = df_ew.dropna(subset=["Description"])
df_ew["is_incident"] = df_ew["Action status"].apply(convert_status)
df_ew = df_ew[["Description", "is_incident"]]

df = pd.concat([df_inc, df_ew], ignore_index=True)

incident_created = df[df["is_incident"] == 1]
triage_closed = df[df["is_incident"] == 0]

print("incident porcentage:\n", (len(incident_created) / len(df)) * 100, "%")
print("Not incident porcentage:\n", (len(triage_closed) / len(df)) * 100, "%")

X_imb = df.drop(["is_incident"], axis=1)
y_imb = df["is_incident"]


# --------------Spliting data in  - 80% for training &&&&& 20% test size-----------------

x_train, x_test, y_train, y_test = train_test_split(
    X_imb, y_imb, train_size=0.8, test_size=0.2, random_state=4
)

# ----------------------Balancing train data using random oversampling-----------------

# ros = RandomOverSampler(sampling_strategy=1) # Float
ros = RandomOverSampler(sampling_strategy="not majority")  # String
x_train_ros, y_train_ros = ros.fit_resample(x_train, y_train)

# ---------------Chart to check balanced data before and after ROS---------------------

countsy_res = y_train_ros.value_counts()
fig1, (ax1, ax2) = plt.subplots(1, 2)

countsy = y_imb.value_counts()
explode = (0.2, 0)
# fig1, ax1 = plt.subplots()
ax1.pie(
    countsy,
    colors=["tab:blue", "tab:orange"],
    autopct="%1.1f%%",
    labels=countsy.index,
    startangle=90,
    shadow=False,
    explode=explode,
)

ax1.set_title("Incident balance before ROS", fontsize=10, fontweight="bold")

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"

ax2.pie(
    countsy_res,
    colors=["tab:blue", "tab:orange"],
    autopct="%1.1f%%",
    labels=countsy_res.index,
    startangle=90,
    shadow=False,
)
ax2.set_title("Incident balance After ROS", fontsize=10, fontweight="bold")

plt.rcParams["font.size"] = 12
plt.rcParams["font.family"] = "Arial"
plt.show()


# -----------------Using TFIDF VECTORIZER to preprocess the data-------------------------
# -------(feature extraction, conversion to lower case and removal of stop words)--------

# I needed to change the type from Dataframe to Series before applying TFIDF Vectorizer using the function squeeze
x_train_ros_series = x_train_ros.squeeze()
x_test_series = x_test.squeeze()


tfvec = TfidfVectorizer(min_df=1, stop_words="english", lowercase=True)
x_trainFeat = tfvec.fit_transform(x_train_ros_series)
x_testFeat = tfvec.transform(x_test_series)

# -------------------------------SVM is used to model-------------------------------------

y_trainSvm = y_train_ros.astype("int")
classifierModel = LinearSVC()
classifierModel.fit(x_trainFeat, y_trainSvm)
predResult = classifierModel.predict(x_testFeat)

# GNB is used to model
y_trainGnb = y_train_ros.astype("int")
classifierModel2 = MultinomialNB()
classifierModel2.fit(x_trainFeat, y_trainGnb)
predResult2 = classifierModel2.predict(x_testFeat)

# Calc accuracy,converting to int - solves - cant handle mix of unknown and binary
y_test = y_test.astype("int")
actual_Y = y_test.values

print("~~~~~~~~~~SVM RESULTS~~~~~~~~~~")

# Accuracy score using SVM
print(
    "Accuracy Score using SVM: {0:.4f}".format(
        accuracy_score(actual_Y, predResult) * 100
    )
)
# FScore MACRO using SVM
print(
    "F Score using SVM: {0: .4f}".format(
        f1_score(actual_Y, predResult, average="macro") * 100
    )
)
cmSVM = confusion_matrix(actual_Y, predResult)
# "[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using SVM:")
print(cmSVM)


print("~~~~~~~~~~MNB RESULTS~~~~~~~~~~")
# Accuracy score using MNB
print(
    "Accuracy Score using MNB: {0:.4f}".format(
        accuracy_score(actual_Y, predResult2) * 100
    )
)
print(" Error rate: ", (100 - accuracy_score(actual_Y, predResult2) * 100))
# FScore MACRO using MNB
print(
    "F Score using MNB:{0: .4f}".format(
        f1_score(actual_Y, predResult2, average="macro") * 100
    )
)
cmMNb = confusion_matrix(actual_Y, predResult2)
# "[True negative  False Positive\nFalse Negative True Positive]"
print("Confusion matrix using MNB:")
print(cmMNb)

# final processing print
# fff
print("Final preprocessing")
