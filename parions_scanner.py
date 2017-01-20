import os
from subprocess import call, STDOUT
import codecs
import uuid
from datetime import datetime, timedelta

import parions_database_parse
import text_processor
from fuzzywuzzy import fuzz
# from fuzzywuzzy import process
# from unidecode import unidecode
from receiptimg.receipt import readAndSift
from receiptimg.receipt_old import readAndSift as readAndSift_old


class MatchesDateNotSupported(Exception):
    pass


def getTextFromTesseract(imagefile):
    command = ["tesseract", "-psm", "6", "-l", "eng", imagefile, "text"]
    call(command, stdout=open(os.devnull, 'w'), stderr=STDOUT)
    with codecs.open("text.txt", "r", encoding='utf-8') as f:
        text = f.readlines()
    return text


def getBets(typeimgfile, betimgfile, dates_image_file, csvfile, csvfileloto, check_date):
    # get dates
    date = None
    try:
        text = getTextFromTesseract(dates_image_file)
        text = text_processor.preprocess(text)
        date = text_processor.get_first_match_date(text)
    except Exception as e:
        pass

    if check_date and date and date < (datetime.now() - timedelta(days=3)).date():
        raise MatchesDateNotSupported

    # Detect game type
    text = getTextFromTesseract(typeimgfile)
    text = text_processor.preprocess(text)
    gtype = text_processor.get_gametype(text)

    # Get text and preprocess
    text = getTextFromTesseract(betimgfile)
    text = text_processor.preprocess(text)

    # Detect bets
    if gtype[0] == 'loto foot':

        # Get loto foot data and find most likely entry
        lotofootdata = parions_database_parse.getParionsLotoFootData(csvfileloto)
        lotofootkey = text_processor.getBestLotoFootMatch(lotofootdata, str(gtype[1]))
        gtype = ('loto foot', [int(lotofootkey)])
        lotofootnames = lotofootdata[lotofootkey]

        # Remove any very short strings, these return spurious fuzzywuzzy matches
        # minlen = min(map(len, lotofootnames))
        # text = filter(lambda t: len(t) > minlen, text)

        selections = [None] * len(lotofootnames)
        for event in lotofootnames:
            selection, index = text_processor.getLotoFootSelections(event, text)
            selections[index] = (event, selection)
        return (gtype, selections)

    else:
        # Read and process parions database
        parionsdata = parions_database_parse.getParionsData(csvfile)
        nameToMarketsMap = parions_database_parse.getNameToMarketMapping(parionsdata)

        # Find most likely bets
        betstrings = text_processor.segmentBets(text)
        probableevents = []
        probableevents2 = [text_processor.findMostProbableEvent(nameToMarketsMap.keys(), b, 10) for b in betstrings]
        ret = []
        for index, bet in enumerate(betstrings):
            numFound = 0
            probableMatch = []
            event = probableevents2[index][0]
            for prob in nameToMarketsMap[event[0]]:
                match = fuzz.partial_ratio(prob['Market'], bet)
                probableMatch.append([match, prob['Market'], prob['Number']])
                sorted(probableMatch, key=lambda x: x[0])
            for probableevent in probableevents2[index]:
                if numFound != 0:
                    continue
                for prob in nameToMarketsMap[probableevent[0]]:
                    # find best match with 'Market'
                    for step in range(1, 4):
                        test = bet
                        if step == 2:
                            test = bet.replace('o', '0')
                        if step == 3:
                            test = bet.replace('e', '6')
                        if len([i for i in test.split(' ') if prob['Number'] in i]) == 0:
                            continue
                        event = probableevent
                        idIdx = test.find(prob['Number'])
                        eventIndex = test[idIdx:].split()[0]
                        try:
                            eventPoint = test[idIdx:].split()[1]
                        except:
                            continue
                        # check for disamb.
                        if eventPoint in ["l"]:
                            eventPoint = "1"
                        if eventPoint in ['h']:
                            eventPoint = "n"
                        if eventPoint in ["1", "n", "2"]:
                            ret.append(str(eventIndex) + " " + str(eventPoint))
                        else:
                            ret.append(str(prob['Number']) + ' x')
                        numFound = 1
                        break
            probableevents.append([event])
            if numFound == 0:
                indexMatch = text_processor.findBestMatch(probableMatch[0][1], bet)
                matchNum1 = fuzz.partial_ratio(text_processor.preprocess(probableMatch[0][1] + " " + probableMatch[0][2] + " 1"), text_processor.preprocess(bet[indexMatch:]))
                matchNumN = fuzz.partial_ratio(text_processor.preprocess(probableMatch[0][1] + " " + probableMatch[0][2] + " n"), text_processor.preprocess(bet[indexMatch:]))
                matchNum2 = fuzz.partial_ratio(text_processor.preprocess(probableMatch[0][1] + " " + probableMatch[0][2] + " 2"), text_processor.preprocess(bet[indexMatch:]))
                # Find best match
                matchMax = [matchNum1, matchNumN, matchNum2].index(max([matchNum1, matchNumN, matchNum2]))
                if matchNum1 == matchNumN and matchNum1 == matchNum2:
                    ret.append(str(probableMatch[0][2]) + " x")
                else:
                    if matchMax == 0:
                        ret.append(str(probableMatch[0][2]) + " 1")
                    elif matchMax == 1:
                        ret.append(str(probableMatch[0][2]) + " n")
                    elif matchMax == 2:
                        ret.append(str(probableMatch[0][2]) + " 2")
            else:
                numFound = 0
        return (gtype, probableevents, ret, date)


def getBetsFromImage(imagefile, csvfile, csvfileloto, check_date):
    jobdata = {}
    jobdata["id"] = str(uuid.uuid4())
    typeimagefile = ""
    betimagefile = ""

    try:
        try:
            bettype = readAndSift(imagefile, jobdata["id"])
        except IndexError:
            try:
                bettype = readAndSift_old(imagefile, jobdata["id"])
            except IndexError:
                raise Exception('Probably your image is at wrong orientation')
        try:
            bettype['error']
        except Exception:
            pass
        else:
            return bettype['error']
        typeimagefile = jobdata["id"] + bettype + "Type.jpg"
        betimagefile = jobdata["id"] + bettype + "Bet.jpg"
        dates_image_file = jobdata["id"] + bettype + "Dates.jpg"

        jobdata["betdata"] = getBets(typeimagefile, betimagefile, dates_image_file, csvfile, csvfileloto, check_date)
    finally:
        # Remove the intermediate images
        for f in (typeimagefile, betimagefile, dates_image_file):
            try:
                os.remove(f)
            except OSError:
                pass

    return jobdata
