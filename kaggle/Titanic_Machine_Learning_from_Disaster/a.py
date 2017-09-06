# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def main():
    #######################################################
    ## Load Dataset
    ## read train dataset
    train = pd.read_csv("train.csv", index_col=["PassengerId"])
    ## read test dataset
    test = pd.read_csv("test.csv", index_col=["PassengerId"])

    #######################################################
    ## Preprocessing
    ## merge train dataset and test dataset
    combi = pd.concat([train, test])
    # ## encode gender
    # combi["Gender_encode"] = (combi["Gender"] == "male").astype(int)
    ## Fill out missing fare
    mean_fare = train["Fare"].mean()
    combi["Fare_fillout"] = combi["Fare"]
    combi.loc[pd.isnull(combi["Fare"]), "Fare_fillout"] = mean_fare
    ## encode embarked
    embarked = pd.get_dummies(combi["Embarked"], prefix="Embarked").astype(np.bool)
    combi = pd.concat([combi, embarked], axis=1)
    ## Add Family
    combi["Family"] = combi["SibSp"] + combi["Parch"]
    ## split dataset into train and test
    train = combi[pd.notnull(combi["Survived"])]
    test = combi[pd.isnull(combi["Survived"])]

    ## train
    # feature_names = ["Pclass", "Gender_encode", "Fare_fillout"]
    feature_names = ["Pclass", "Fare_fillout"]
    feature_names = feature_names + list(embarked.columns)
    label_name = "Survived"
    X_train = train[feature_names]
    y_train = train[label_name]
    seed = 37
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(max_depth=5, random_state=seed)

    ## score
    X_test = test[feature_names]
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    ## Submit
    submission = pd.read_csv("gender_submission.csv", index_col="PassengerId")
    submission["Survived"] = prediction.astype(np.int32)
    submission.to_csv("decision-tree-pclass-fare-max-depth-5.csv")  

if __name__ == "__main__":
    main()
