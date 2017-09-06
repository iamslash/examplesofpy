# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def main():
    train = pd.read_csv("train.csv", index_col=["PassengerId"])
    print(train.shape)
    print(train.head())

if __name__ == "__main__":
    main()
