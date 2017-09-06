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
    print(combi.head())
    

if __name__ == "__main__":
    main()
