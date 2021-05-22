import numpy as np
import pandas as pd
import sklearn.model_selection
from sklearn import linear_model, preprocessing
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("../star.csv")

# print(df.head())

# print(df.info())

# print(df['Star color'].unique())


#  PREPROCESSIG DATA
# As we can see we have categorical data and the format for most value in series of Star color is different even thought
# they are referring to the same category, So lets make the data consistent first.

color_mapping = {
    "Red": "Red",
    "Blue white": "Blue White",
    "White": "White",
    "Yellowish White": "Yellow White",
    "Pale yellow orange": "Pale Yellow Orange",
    "Blue": "Blue",
    "Blue-white": "Blue White",
    "Whitish": "White",
    "yellow-white": "Yellow White",
    "Orange": "Orange",
    "White-Yellow": "Yellow White",
    "white": "White",
    "Blue": "Blue",
    "yellowish": "Yellow",
    "Yellowish": "Yellow",
    "Orange-Red": "Orange Red",
    "Blue white": "Blue White",
    "Blue-White": "Blue White",
    "Blue white ": "Blue White"
}

df['Star color'] = df['Star color'].replace(color_mapping)


# le = LabelEncoder()
# df['Spectral Class'] = le.fit_transform(list(df['Spectral Class']))
# df['Star color'] = le.fit_transform(list(df['Star color']))
# print(df['Spectral Class'])
# print(df['Star color'])

# x = df.drop('Star type', axis=1)
# y = df['Star type']

# Lets split the data into training and testing set:
# X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.3, shuffle=False,
# random_state=101)


# print(df['Star color'].unique())
# Method 2: We will use dummy encoding to classify the categorical data

# I like to define the function at the begging.
def dummy_encoder(df, column, prefix):
    dummies = pd.get_dummies(df[column], prefix=prefix)  # adding dummies
    df = pd.concat([df, dummies], axis=1)  # contacting dummies
    df = df.drop(column, axis=1)  # removing categories with no numerical data
    return df


# Lets Dummy the Star Color and Spectral Class  the reason why we do this because if allow us to hold the variable
# category in place as numerical representation to keep the data consistent.
# https://www.statisticssolutions.com/dummy-coding-the-how-and-why/

df = dummy_encoder(df, 'Star color', prefix='color')
df = dummy_encoder(df, 'Spectral Class', prefix='spectral')

# Now lets create the label data to supervised the training model. We will use "Star Type" as labels

y = df['Star type']
x = df.drop('Star type', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, train_size=0.3, shuffle=True, random_state=101)

# Now we will scale our data by using StandardScalar. We standardize our dataframe
# to reduce training time and to save computational power, standardize limited the data between -1 and 1 it help the
# distance formula to  easily compute the distance.

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

# scalar = StandardScaler()  # Initiating the scalar

# X_train = scalar.fit_transform(X_train)  # Standardized the X_train
# X_test = scalar.fit_transform(X_test)  # Standardized the X_test

# print(X_train)
# print(y_train)
# print(X_test)
# print(y_test)


# So far so good we have process data,
# Lets Use Logistic Regression

from sklearn.linear_model import LogisticRegression

LGR = LogisticRegression()  # instantiating the model

LGR.fit(X_train, y_train)
prediction = LGR.predict(X_test)
accuracy = accuracy_score(y_test, prediction)
# print("Accuracy Score: ", accuracy*100)             # 89% accuracy

from sklearn.neighbors import KNeighborsRegressor

KNR = KNeighborsRegressor(n_neighbors=5)  # Instantiating the model
KNR.fit(X_train, y_train)
# print(KNR.score(X_test, y_test))  # accuracy of 83% with 5_neighbour parameter

# Lets Use Support Vector Classifier

from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC

model = SVC(kernel="linear", C=0.1)

# model.fit(X_train, y_train)
# prediction = model.predict(X_test)

# score = accuracy_score(y_test, prediction)
# print("accuracy score: ", score * 100)


# With support vector classifier we have accuracy score of 99%

from sklearn.ensemble import RandomForestRegressor

RFR = RandomForestRegressor()
RFR.fit(X_train, y_train)
print(RFR.score(X_test, y_test))

# It appear RandomForrest is the best model fot classifying start with 99% accuracy
