def time_column(df, cut_left = 0, cut_right = 0):
    if cut_right == 0:
        cut_right = None
    return df.iloc[:, 0].values[cut_left:cut_right]

def outer_temperature(df, cut_left = 0, cut_right = 0):
    if cut_right == 0:
        cut_right = None
    return df.iloc[:, -1].values[cut_left:cut_right]

def point_temperature_2d(df, point, cut_left = 0, cut_right = 0):
    if cut_right == 0:
        cut_right = None
    return df[[df.columns[point]]].values[cut_left:cut_right]

def point_temperature_1d(df, point, cut_left = 0, cut_right = 0):
    if cut_right == 0:
        cut_right = None
    return df.iloc[:, point].values[cut_left:cut_right]


def point_coordinates(df, number):
    return df.columns[number][20:]