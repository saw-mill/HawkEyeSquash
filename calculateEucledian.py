import csv
import math
with open('PositionDataset.csv') as csv_file:
	csv_reader = csv.reader(csv_file, delimiter=',')

	previousRow = list()
	for row in csv_reader:
		if not previousRow:
			previousRow=row.copy()
			continue
		else:
			if not row:
				previousRow=row.copy()
				continue
			else:
				eucDist = math.sqrt(math.pow((int(row[0]) - int(previousRow[0])), 2) + math.pow(int(row[1]) - int(previousRow[1]), 2))
				print(eucDist)
