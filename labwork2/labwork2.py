import csv

def means(values):
    return sum(values) / float(len(values))

def variance(values, mean):
    return sum([(x - mean)**2 for x in values])

def covariance(x_values, y_values, x_mean, y_mean):
    
    covar = 0.0
    for i in range(len(x_values)):
        covar += (x_values[i] - x_mean) * (y_values[i] - y_mean)

    return covar

def coefficients(data):
    x_values = [row[0] for row in data]
    y_values = [row[1] for row in data]

    x_mean = means(x_values)
    y_mean = means(y_values)

    b1 = covariance(x_values, y_values, x_mean, y_mean) / variance(x_values, x_mean)
    b0 = y_mean - b1 * x_mean

    return [b0, b1]

def linear_regression(train, test):
    
    predictions = list()
    b0, b1 = coefficients(train)

    for row in test:
        # yha = y_pred
        yhat = b0 + b1 * row[0]
        predictions.append(yhat)

    return predictions


def testing():
    with open('winequality-white.csv', 'r') as f:
        winewhites = list(csv.reader(f, delimiter=';'))

    for i in range(len(winewhites[0])):
        for row in winewhites:
            row[i] = float(row[i].strip())

    train_size = int(0.8 * len(winewhites))
    train, test = winewhites[:train_size], winewhites[train_size:]

    x = [row[0] for row in train]
    y = [row[1] for row in train]

    train = list(zip(x, y))
    test = list(zip(x, y))

    predictions = linear_regression(train, test)