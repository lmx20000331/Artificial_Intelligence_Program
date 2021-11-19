from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
import pandas as pd
from icecream import ic
import numpy as np
from sklearn.metrics import r2_score

x = load_boston()['data']
y = load_boston()['target']

dataframe = pd.DataFrame(x, columns=load_boston()['feature_names'])

print(dataframe)

x_train, x_test, y_train, y_test = train_test_split(dataframe, y, test_size=0.4)

regressor = DecisionTreeRegressor()

regressor.fit(x_train, y_train)

ic(regressor.score(x_train, y_train))
ic(regressor.score(x_test, y_test))


def random_select(frame, y, drop=2):
    columns = np.random.choice(list(frame.columns), size=len(frame.columns) - drop)
    indices = np.random.choice(range(len(y)), size=len(y) - drop)

    return frame.iloc[indices][columns], y[indices]


predicts = []

for i in range(10):
    sample_x, sample_y = random_select(x_train, y_train)
    regressor = DecisionTreeRegressor()
    regressor.fit(sample_x, sample_y)
    train_score = regressor.score(sample_x, sample_y)
    test_score = regressor.score(x_test[sample_x.columns], y_test)
    ic(train_score)
    ic(test_score)

    predicts.append(regressor.predict(x_test[sample_x.columns]))

forest_predict = np.mean(predicts, axis=0)

ic(r2_score(y_test, forest_predict))
