def equal_doubles(first, second, epsilon):
    if first is None or second is None:
        return first is None and second is None
    return abs(first - second) < epsilon
