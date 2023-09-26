# 0.76315
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

train_path = "./train.csv"
test_path = "./test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

## Processing the cabin from C123, D32, ...  -> 3.0, 4.0, ... (take the first char, then turn it into float)
# convert the cabin into numerical value
def to_numerical(char):
    # use if incase of nan value
    if pd.notna(char):
        return ord(char)-ord('A')+1
    return np.nan

# take the first char in cabin, then change it into numerical using to_numerical function
def fix_cabin(df):
    df["Cabin"] = df["Cabin"].str[0]
    df["Cabin"] = df["Cabin"].apply(lambda x: to_numerical(x))
    return df

# use train dataset to find filler
# filler is used to fill the nan value (filler is based on the pclass)
def find_filler(df):
    # find the average age per class
    average_age_class = df.groupby('Pclass')['Age'].mean()
    # find the mode embarked per class
    mode_embarked_class = df.groupby('Pclass')['Embarked'].agg(lambda x: x.mode().iloc[0])
    # find the mode of cabin
    mode_cabin_class = df.groupby('Pclass')['Cabin'].agg(lambda x: x.mode().iloc[0])

    average_age_class = list(average_age_class)
    mode_embarked_class = list(mode_embarked_class)
    mode_cabin_class = list(mode_cabin_class)

    return average_age_class+mode_cabin_class+mode_embarked_class

# nan value is filled based on the Pclass, hence why the long repetitive code
# fill nan value of age, cabin, embarked (connected to fill_na function)
def fill(df, age, cabin, embark):
  replace = {"Age":age, "Cabin":cabin, "Embarked":embark}
  for col, value in replace.items():
    df.loc[df[col].isna(), col] = value
  return df

# fill the nan value (connected to preprocessing function)
## fill na -> split the dataset into 3 based on the class, then fill nan of each class, lastly combine the dataset (if train dataset = shuffle, if test dataset = sort based on id)
def fill_na(df, prep_type):

    # split into 3 based on class
    df1 = df.loc[df["Pclass"]==1]
    df2 = df.loc[df["Pclass"]==2]
    df3 = df.loc[df["Pclass"]==3]
    df_list = [df1, df2, df3]
    
    # shape probably like this [age1, age2, age3, mode1, mode2, mode3, ...]
    filler = find_filler(train_df)
    
    print(filler)
    # fill the nan value in df1, df2, df33
    for idx, x in enumerate(df_list):
        # what is id, 3+idx, 6+idx?
        # basically the filler shape is (age(class1), age(class2), age(class3), cabin(class1), ..., embarked(class3))
        df_list[idx] = fill(x, int(filler[idx]), filler[3+idx], filler[6+idx])

    # concat all the separated df
    final_df = pd.concat([df_list[0], df_list[1], df_list[2]])
    
    # if train df, shuffle the dataset
    if prep_type == "Train":
        final_df = final_df.sample(frac=1, random_state=64)
    # if test df, sort the dataset based on passenger id (to match the correct answer)
    else:
        final_df.sort_values(by=["PassengerId"], inplace=True)
    
    return final_df

## this is basically combining the fix cabin, fill na, and dropping some column (also changing sex into numerical)
def preprocess(df, type):
    
    # in test we dont drop the passenger id to sort the answer
    if type == "Train":
        # drop passenger id
        df.drop("PassengerId", axis=1, inplace=True)

    # dropping name, fare, ticket
    df.drop(columns=["Name", "Fare", "Ticket"], inplace=True)
    
    # changing sex  into numerical
    df["Sex"].replace("male", 1, inplace=True)
    df["Sex"].replace("female", 0, inplace=True)

    # fix the cabin
    df = fix_cabin(df)

    # input fillna here
    df = fill_na(df, type)

    # change embarked into numerical
    df["Embarked"].replace('C', 1, inplace=True)
    df["Embarked"].replace('Q', 2, inplace=True)
    df["Embarked"].replace('S', 3, inplace=True)

    # split x and y
    # case: 
    # Train: x -> feature, y -> survived
    # Test: x -> feature, y -> PassengerID
    labels = list(df.columns)
    x = df[labels[1:]]
    y = df[labels[0]]

    return x, y

x_train, y_train = preprocess(train_df, "Train")
x_test, test_PassengerId = preprocess(test_df, "Test")

# scaling
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#### modelling
model = LogisticRegression()
model.fit(x_train, y_train)

Survived = model.predict(x_test)

## export
passenger_id = list(test_PassengerId)
Survived = list(Survived)
datas = {
    "PassengerId" : passenger_id,
    "Survived" : Survived
}
ans = pd.DataFrame(datas)

ans.to_csv("prediction.csv", index=False)
print("done")