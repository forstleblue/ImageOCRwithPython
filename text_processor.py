# This Python file uses the following encoding: utf-8
import re
from datetime import datetime
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
from unidecode import unidecode

# tesseract text_region.jpg text parions

re_date = re.compile(r'(\d+\/\d+\/\d+)')

gametypes = [
    'jeu simple',
    'jeu combine',
    'loto foot',
    'jeu multiple',
]

# blacklist chars (to be filled with other chars)
blacklist = [
    "'",
    '~',
    "‘",
    "î",
    "Î",
    ":",
    "<<",
    ".",
    ";"]

# Find the index of the best match of matchtext in text
def findBestMatch(matchtext, text):
    assert len(text) > len(matchtext), "%s > %s" % (matchtext, text)
    matchlen = len(matchtext)
    substrings = [text[i:i+matchlen] for i in range(len(text) - matchlen + 1)]
    matches = map(lambda i: (i, fuzz.ratio(matchtext, substrings[i])),range(len(substrings)))
    maxmatch = max(matches, key=lambda t: t[1])
    return maxmatch[0]

# Scan ahead a certain number of characters searching for something
def scanAhead(ismatch, matchresult, startindex, maxdist, nummatches=1):
    matches = []
    i = startindex
    while len(matches) < nummatches and (i-startindex) < maxdist:
        if ismatch(i):
            matches.append(matchresult(i))
        i+=1
    return matches

def get_gametype(textlines):
    text = " ".join(textlines)
    gameranks = map(lambda g: fuzz.partial_ratio(text, g), gametypes)

    # If there were no good matches, return nothing
    if max(gameranks) < 60:
        return (None, None)

    game = gametypes[gameranks.index(max(gameranks))]
    extras = None
    index = findBestMatch(game, text)
    if game == 'jeu multiple':
        extras = scanAhead(lambda i: text[i].isdigit(), lambda i: text[i], index+len(game), 15, 2)
        extras = map(int, extras)
        if len(extras) != 2 or extras[0] > extras[1] or extras[1] >= 7:
            extras = None
    elif game == 'jeu combine':
        extras = scanAhead(lambda i: text[i].isdigit(), lambda i: text[i], index+len(game), 10, 1)
        extras = map(int, extras)
        if len(extras) != 1 or extras[0] > 7:
            extras = None
    elif game == 'loto foot':
        test = lambda i: all(map(lambda c: c.isdigit(), text[i:i+3]))
        extras = scanAhead(test, lambda i: text[i:i+3], index+len(game), 10, 1)
        extras = map(int, extras)
    return (game, extras)

def preprocess(text):

    if isinstance(text, str):
        text = unicode(text)
    # convert unicode chars to closest ascii
    text = map(unidecode, text)

    # make everything lower case
    text = map(lambda s: s.lower(), text)

    # remove all the whitespace
    text = [l.strip() for l in text if l.strip()]

    # remove all char in the blacklist
    for ch in blacklist:
       text = [t.replace(ch, '') for t in text]

    return text

# Find the loto foot selections for the given event
def getLotoFootSelections(event, text):
    line, _ = process.extractOne(event, text)
    # index = findLotoBestMatch(event, line)
    # import ipdb; ipdb.set_trace()
    index = text.index(line)
    for i in ['1', 'n', '2']:
        if i in line.replace('-', '').replace('>', '').strip().split(' ').pop():
            return ([i], index)
    return (['x'], index)
    # selections = scanAhead(
    #     lambda i: line[i] in ['1', 'n', '2'],
    #     lambda i: line[i],
    #     (len(line) - 3) - (len(line.split(' ').pop())-1), len(line.split(' ').pop()) + 1, 8)
    # return selections

def segmentBets(textlines):
    betstarthints = ["football" , "f00tball", "f0otball", "fo0tball", "tennis" , "rugby" , "basketball"]

    # Rank the lines by how likely they are to be the start of
    # a bet (i.e. have a starting hint in them)
    betstartranking = []
    for (i, l) in enumerate(textlines):
        betstartranking.append(0)
        for hint in betstarthints:
            if len(l) > len(hint):
                match = fuzz.partial_ratio(l, hint)
                if match > betstartranking[-1]:
                    betstartranking[-1] = match

    # Get the indices most likely to be bet starts
    betstartindices = []
    for (i, rank) in enumerate(betstartranking):
        if rank > 85:
            betstartindices.append(i)

    # Split the bets
    betstrings = []
    betstringlines = []
    for (i, beti) in enumerate(betstartindices):
        if i + 1 < len(betstartindices):
            betj = betstartindices[i+1]
        else:
            betj = len(textlines)
        betstrings.append(' '.join(textlines[beti:betj]))
        betstringlines.append(textlines[beti:betj])

    return betstrings

# Find the most likely event in parionsdata to match the betstring
def findMostProbableEvent(parionsdata, betstring, top=1):

    # To find a match between a single event and the database
    # build a confidence value for each possible entry
    confidences = []
    for event in parionsdata:
        # confidences.append((event, fuzz.ratio(event.lower(), betstring)))
        confidences.append((event, fuzz.partial_ratio(event.lower(), betstring)))

    if top == 1:
        return max(confidences, key=lambda i: i[1])
    else:
        return [i for i in reversed([confidences[j] for j in sorted(range(len(confidences)), key=lambda i: confidences[i][1])[-top:]])]

    # first =
    # del confidences[confidences.index(first)]
    # second = max(confidences, key=lambda i: i[1])
    # return [first, second]


def getBestLotoFootMatch(lotofootdata, gamecode):
    k, _ = process.extractOne(gamecode, lotofootdata.keys())
    return k


def get_date_by_key_phrase(text, phrase):
    confidences = []
    for item in text:
        confidences.append((item, fuzz.ratio(item.lower(), phrase)))
    best_match = max(confidences, key=lambda i: i[1])
    dt = re_date.findall(best_match[0]).pop().split('/')

    month = int(dt[1])
    day = int(dt[0])

    if month in [0, 99]:
        month = 9
    if day in [0, 99]:
        day = 9

    return datetime(
        year=int(dt[2].replace('8', '0')),
        month=month,
        day=day,
    ).date()


def get_first_match_date(text):
    for phrase in [
        'date premier match / / / ',
        'date du match / / / '
    ]:
        try:
            date = get_date_by_key_phrase(text, phrase)
            return date
        except Exception:
            pass
    return None
