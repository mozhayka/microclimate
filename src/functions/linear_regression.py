import pandas as pd
from scipy import stats
from sklearn.linear_model import LinearRegression
import itertools
from functions.table import *


def get_XY(df, point, shift, max_shift):
    if shift > max_shift:
        return None
    X = point_temperature_2d(df, point, shift, shift - max_shift)
    Y = outer_temperature(df, 0, -max_shift)
    return X, Y
    
    
def find_best_shift(df_train, point, max_shift):
    best_shift = None
    best_stat = 0
    for shift in range(0, max_shift + 1):
        X = point_temperature_1d(df_train, point, shift, shift - max_shift)
        Y = outer_temperature(df_train, 0, -max_shift)
        stat = stats.pearsonr(X, Y).statistic
        if best_stat < stat:
            best_shift = shift
            best_stat = stat
    return best_shift


def get_regression(df_train, point, shift, max_shift):
    X, Y = get_XY(df_train, point, shift, max_shift)
    regressor = LinearRegression()
    regressor.fit(X, Y)
    return regressor


def get_predict(df_test, point, shift, max_shift, regressor):
    X, Y = get_XY(df_test, point, shift, max_shift)
    return regressor.predict(X), regressor.score(X, Y)
    



class Result:
    def __init__(self, point, shift, score, predict, regressor):
        self.point = point
        self.shift = shift
        self.score = score
        self.predict = predict
        self.regressor = regressor


def calc_all_points(df_train, df_test, max_shift):
    ans = []
    for point in get_points(df_train):
        shift = find_best_shift(df_train, point, max_shift)
        regressor = get_regression(df_train, point, shift, max_shift)
        predict, score = get_predict(df_test, point, shift, max_shift, regressor)
        
        ans.append(Result(point, shift, score, predict, regressor))
    
    return sorted(ans, key=(lambda x: x.score))


def print_results(df, result, max_shift):
    print(f"Лучшая точка: {point_coordinates(df, result.point)}, отсуп времени: {result.shift * 5} минут")
    print("Коэффициент детерминации (R^2):", result.score, '\n')

    compare = pd.DataFrame({
        "Время": time_column(df, 0, -max_shift),
        "Предсказание": pd.Series(result.predict),
        "Реальность": outer_temperature(df, 0, -max_shift)
    })
    print(compare)
