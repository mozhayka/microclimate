import pandas as pd
from sklearn.linear_model import LinearRegression
import itertools
from functions.table import *


def approximate(df, point, shift = 36, max_shift = 60):
    if shift > max_shift:
        return None
    X = point_temperature_2d(df, point, shift, shift - max_shift)
    Y = outer_temperature(df, 0, -max_shift)
        
    regressor = LinearRegression()
    regressor.fit(X, Y)
    r_squared = regressor.score(X, Y)
    y_pred = regressor.predict(X)
    
    return regressor, y_pred, r_squared


def find_best_point_and_shift(df, max_shift = 60):
    cur_r_squared = 0
    best_point = None
    best_shift = None
    for point, shift in itertools.product(range(1, len(df.columns) - 2), range(0, max_shift + 1)):
        regressor, y_pred, r_squared = approximate(df, point, shift, max_shift)
        if cur_r_squared < r_squared:
            cur_r_squared = r_squared
            best_point = point
            best_shift = shift
    return best_point, best_shift


def calc_all_points_and_shifts(df, max_shift = 60):
    ans = []
    for point, shift in itertools.product(range(1, len(df.columns) - 2), range(0, max_shift + 1)):
        regressor, y_pred, r_squared = approximate(df, point, shift, max_shift)
        ans.append((r_squared, point, shift, y_pred, regressor))
    return ans


def print_results(df, best_point, best_shift, y_pred, r_squared, max_shift):
    print(f"Лучшая точка: {point_coordinates(df, best_point)}, отсуп времени: {best_shift * 5} минут")
    print("Коэффициент детерминации (R^2):", r_squared, '\n')

    compare = pd.DataFrame({
        "Время": time_column(df, 0, -max_shift),
        "Предсказание": pd.Series(y_pred),
        "Реальность": outer_temperature(df, 0, -max_shift)
    })
    print(compare)
