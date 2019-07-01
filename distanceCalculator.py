import csv
import itertools

csv_path = "Results/PositionDataset.csv"
i, j = 1, 10
with open(csv_path, "rt") as f_obj:
    reader = list(csv.reader(f_obj))
    # print(len(reader))
    for i in range(1, len(reader)):
        row = reader[i]