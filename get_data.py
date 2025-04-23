import csv

def get_data(file="data.csv", label="Adj Close"):
    values = []

    with open(file, mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            try:
                values.append(float(row[label]))
            except:
                pass
            line_count += 1

    return values

get_data("etfs/USI.csv")