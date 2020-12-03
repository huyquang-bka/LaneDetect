import csv

f = open('rois.csv','r',newline='')
roi = csv.reader(f)
roi = list(roi)

rois = [[int(float(j)) for j in i] for i in roi]
print(rois)
