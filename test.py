'''import yfinance as yf

ticker = yf.Ticker("AAPL")

data = ticker.history(period="60d", interval="5m")
data.to_csv("data.csv")
'''

import csv

values = []

with open('data.csv', mode='r') as csv_file:
    csv_reader = csv.DictReader(csv_file)
    line_count = 0
    for row in csv_reader:
        if line_count == 0:
            print(f'Column names are {", ".join(row)}')
            line_count += 1
        for key in row:
            values.append(row[key])
        line_count += 1
    print(f'Processed {line_count} lines.')

print(values)