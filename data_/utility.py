import math
import numpy as np
import pandas as pd

def split_sequence(start, end, num_parts):
    """
    Split a sequence into approximately equal parts.

    :param start: The start of the sequence.
    :param end: The end of the sequence.
    :param num_parts: The number of parts to split the sequence into.
    :return: A list of split points in the sequence.
    """
    result = []

    if num_parts == 1:
        return [start, end + 1]

    if num_parts > end - start + 1:
        return list(range(start, end + 2))

    length = end - start + 1
    part_length_short = length // num_parts
    part_length_long = math.ceil(length / num_parts)
    cut_pos = length % num_parts

    for i in range(start, start + cut_pos * part_length_long, part_length_long):
        result.append(i)

    for i in range(start + cut_pos * part_length_long, end + 2, part_length_short):
        result.append(i)

    return result

def equal_doubles(first, second, epsilon=1e-7):
    """
    Check if two floating point numbers are equal within a given tolerance.

    :param first: The first floating point number.
    :param second: The second floating point number.
    :param epsilon: The tolerance for equality.
    :return: True if the numbers are considered equal, False otherwise.
    """
    if math.isnan(first):
        return math.isnan(second)
    return abs(first - second) < epsilon

def load_data(file_name):
    """
    Load data from a file into a numpy array.

    :param file_name: The name of the file to load data from.
    :return: A tuple containing the data as a numpy array and its dimensions.
    """
    with open(file_name, 'r',  encoding='utf-8') as file:
        lines = file.readlines()

    if not lines:
        raise RuntimeError("Could not open input file.")

    num_rows = len(lines)
    num_cols = len(lines[0].split())

    # 用pd.read_excel读取Excel文件
    # df = pd.read_excel(file_name)
    #
    # # 获取 DataFrame 的行数和列数
    # num_rows, num_cols = df.shape
    #
    # # 打印行数和列数
    # print("行数:", num_rows)
    # print("列数:", num_cols)

    # # 将 DataFrame 转换为行列表
    # lines = df.values.tolist()

    storage = np.zeros((num_rows, num_cols))

    for row, line in enumerate(lines):
        values = line.split()
        if len(values) != num_cols:
            raise RuntimeError("Inconsistent number of columns.")
        storage[row] = np.array([float(value) for value in values])

    return storage, (num_rows, num_cols)

def set_data(data, row, col, value):
    """
    Set a value in a 2D data array.

    :param data: The data array.
    :param row: The row index.
    :param col: The column index.
    :param value: The value to set.
    """
    data[row, col] = value
