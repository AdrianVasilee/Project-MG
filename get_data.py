import csv

def get_data():
    values = []

    with open('data.csv', mode='r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
            for key in row:
                values.append(row[key])
            line_count += 1

    return values