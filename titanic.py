# > Dependencies
import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
from sklearn import tree, model_selection, linear_model

# > Load all datasets into df's
gender_submission = pd.read_csv("titanic/gender_submission.csv")
test_df = pd.read_csv("titanic/test.csv")
train_df = pd.read_csv("titanic/train.csv")

# > Preview df's
# print(f"gender_submission df:")
# print(tabulate(gender_submission.head(), headers="keys", tablefmt="psql"))
# print()
# print(f"test df:")
# print(tabulate(test_df.head(), headers="keys", tablefmt="psql"))
# print()
print(f"train df: \n"
      f">> 0 - Died \n"
      f">> 1 - Survived")
print(tabulate(train_df.head(), headers="keys", tablefmt="psql"))
print()

'''

The training set should be used to build your machine learning models. For the training set, we provide the outcome 
(also known as the “ground truth”) for each passenger. Your model will be based on “features” like passengers’ gender 
and class. You can also use feature engineering to create new features.

The test set should be used to see how well your model performs on unseen data. For the test set, we do not provide 
the ground truth for each passenger. It is your job to predict these outcomes. For each passenger in the test set, 
use the model you trained to predict whether or not they survived the sinking of the Titanic.

We also include gender_submission.csv, a set of predictions that assume all and only female passengers survive, 
as an example of what a submission file should look like.

'''

# > Data splicing
# Gender
sex_data = train_df["Sex"].value_counts()
print("Gender: ")
print(sex_data)
print()

# Age
age_data = train_df["Age"].value_counts()
age_range = train_df["Age"].max() - train_df["Age"].min()
print("Age: ")
print(age_data.head())
print()
print(f"The range in the age is {age_range}")
print()

# Siblings & Spouses aboard
sibsp_data = train_df["SibSp"].value_counts()
print("Siblings & Spouses: ")
print(sibsp_data)
print()

# Parents & Children aboard
parch_data = train_df["Parch"].value_counts()
print("Parents & Children: ")
print(parch_data)
print()

# Ticket class
pclass_data = train_df["Pclass"].value_counts()
pclasses = list(train_df["Pclass"].value_counts().index)
print("Ticket Class: ")
print(pclass_data)
print(f"The different classes are: {pclasses}")
print()

# Ticket cost
fare_data = train_df["Fare"].value_counts()
different_tickets = len(train_df["Fare"].value_counts().index)
ticket_max = train_df["Fare"].max()
ticket_min = train_df["Fare"].min()
ticket_range = ticket_max - ticket_min
print("Ticket Cost: ")
print(fare_data.head())
print()
print(f"There were {different_tickets} different types of tickets to buy, \n"
      f">>> The most expensive: ${ticket_max}\n"
      f">>> The least expensive: ${ticket_min}\n"
      f">> The range: {ticket_range}")
print()

# Cabin info
cabin_data = train_df["Cabin"].value_counts()
different_cabins = len(train_df["Cabin"].value_counts().index)
print("Cabin Info: ")
print(cabin_data.head())
print(f"There are {different_cabins} number of cabins.")
print()

# > New df with relevant columns
features = ["PassengerId", "Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Cabin"]
titanic_df = train_df[features]
print("Simplified df: ")
print(titanic_df.head())
print()
print(titanic_df.count())

# > Visualizing data
# plt.subplots(2, 2, constrained_layout=True)
plt.subplot2grid((2, 3), (0, 0))
train_df["Survived"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Survived")

plt.subplot2grid((2, 3), (0, 1))
plt.scatter(train_df["Survived"], train_df["Age"], alpha=0.1)
plt.title("Survived vs. Age")

plt.subplot2grid((2, 3), (0, 2))
train_df["Pclass"].value_counts(normalize=True).plot(kind="bar", alpha=0.5)
plt.title("Class")

# plt.subplot2grid((2, 3), (1, 0), colspan=2)
# for x in [1, 2, 3]:
#       train_df[["Age", "Pclass"]].groupby("Pclass").plot(kind="kde")
# plt.title("Class vs. Age")

plt.subplot2grid((2, 3), (1, 0))
plt.scatter(train_df["Survived"], train_df["Pclass"], alpha=0.1)
plt.title("Survived vs. Class")

plt.subplot2grid((2, 3), (1, 1))
plt.scatter(train_df["Age"], train_df["Pclass"], alpha=0.1)
plt.title("Age vs. Class")

plt.subplot2grid((2, 3), (1, 2))
plt.scatter(train_df["Survived"], train_df["SibSp"], alpha=0.1)
plt.title("Survived vs. Num Siblings/Spouses")


plt.tight_layout()
plt.show()

# > Replacing non-numeric variables with numeric values
cabin_fill = titanic_df["Cabin"].values


def clean_data(data):
      data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

