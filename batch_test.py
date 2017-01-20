import os
import csv
import sys
import time

import parions_scanner

entries = []
with open('test_subjects.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        entries.append(row)

if len(sys.argv) > 2:
    startid = int(sys.argv[1])
    endid = int(sys.argv[2])
    test_subjects = [e for e in entries if (int(e["id"]) >= startid and int(e["id"]) <= endid)]
    print 'z'
elif len(sys.argv) > 1:
    testid = sys.argv[1]
    test_subjects = [e for e in entries if e["id"] == testid]
    print 'f'
else:
    print 'c'
    test_subjects = entries
    


print "Processing " + str(len(test_subjects)) + " images..."
count = 0
for test in test_subjects:
    print "Processing {}/{}\t\t{}:{}".format(count+1, len(test_subjects), test["id"], test["image"])
    count += 1

    starttime = time.time()
    jobdata = parions_scanner.getBetsFromImage(test["image"], "test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv")
    print("Time: {}s".format(time.time() - starttime))
    resultstring = " ".join(jobdata["betdata"][2])
    if not resultstring == test["result"]:
        print("FAILED {}; expected {}; actual {}".format(test["image"], test["result"], resultstring))
