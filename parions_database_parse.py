import csv
from collections import defaultdict


def getParionsData(csvfile):
    with open(csvfile) as csvfile:
        reader = csv.DictReader(csvfile)
        entries = []
        for row in reader:
            entries.append(row)
        return entries


def csv_to_list(csvfile):
    with open(csvfile) as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        entries = []
        for row in reader:
            entries.append(row)
        return entries


def getParionsLotoFootData(csvfile):
    # This is obviously just a sample, this function needs to be implemented
    # properly when we get the loto foot data
    data = defaultdict(lambda: [])
    for i in csv_to_list(csvfile):
        data[i[0]].append(i[1])
    return dict(data)


# Converts the parions database to a dictionary
# keyed by the event Name and maps to a list of all
# markets corresponding to that name
def getNameToMarketMapping(parionsdata):
    nameSet = set([e['Name'] for e in parionsdata])
    mapping = {}
    for name in nameSet:
        mapping[name] = [e for e in parionsdata if e['Name'] == name]
    return mapping
