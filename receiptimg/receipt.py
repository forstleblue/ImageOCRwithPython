import numpy as np
import sys
import cv2
import math
import utils as ut
# from PIL import Image
import os
import os.path
import csv
# import cPickle
import time

img1 = None
img2 = None
img3 = None
img4 = None
img5 = None
img6 = None

kp1 = None
kp2 = None
kp3 = None
kp4 = None
kp5 = None
kp6 = None

des1 = None
des2 = None
des3 = None
des4 = None
des5 = None
des6 = None

good = None
goodMontant = None
goodMontanTotal = None

vertex = None

firstExec = True


def readAndSift(imageFile, outImageFile):
    splitted = imageFile.split('\\')
    nameFile = splitted[len(splitted) - 1].split('.')[0]
    print nameFile
    package_directory = os.path.dirname(os.path.abspath(__file__))

    global img1
    global img2
    global img3
    global img4
    global img5
    global img6

    global kp1
    global kp2
    global kp3
    global kp4
    global kp5
    global kp6

    global des1
    global des2
    global des3
    global des4
    global des5
    global des6

    global good
    global goodMontant
    global goodMontanTotal

    global vertex

    global firstExec

    if(firstExec):
        # read detected and computed SIFT descriptors and keypoints (first set
        # of receipts)
        kp1 = ut.read_keyps_from_file("kp1.txt")
        des1 = ut.read_desc_from_file("des1.desc.npy")

        firstExec = False

    # logo Image
    img1 = cv2.imread(os.path.join(package_directory, "logo.jpg"), 0)
    # test image
    imgTest = cv2.imread(imageFile, 0)
    # preserve original image
    imgTestBig = imgTest.copy()
    # scale down image
    imgTest = cv2.pyrDown(imgTest)

    # find logo corrispondences in the test image
    kp2t, des2t = ut.sift.detectAndCompute(imgTest, None)
    imgTestP = imgTest.copy()

    # Find out horizontal lines
    # Calculate Y-gradient
    sobely = cv2.Sobel(imgTest, cv2.CV_8U, 0, 1, imgTest, 3, 5, 128)
    # Threshold gradient image
    ret, sobel_8u = cv2.threshold(sobely, 200, 255, cv2.THRESH_BINARY)

    # Perform erosion and dilation
    elementerode = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
    elementdilate = cv2.getStructuringElement(cv2.MORPH_RECT, (60, 10))
    # kernel = np.ones((2, 2), np.uint8)
    erosion = cv2.erode(sobel_8u, elementerode, iterations=1)
    dilation = cv2.dilate(erosion, elementdilate, iterations=1)

    # Find resulting countours
    im2, contours, hierarchy = cv2.findContours(
        dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    contourSorted = []

    # Filter contours based on its width
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        if ((rect[1][0] >= 200 or rect[1][1] >= 200) and (rect[1][0] <= 500 or rect[1][1] <= 500)):
            contourSorted.append(cnt)

    # Sort contours starting from the top
    contourSorted = reversed(contourSorted)
    all_contourSorted = reversed(contours)

    vertex = []
    vertex_all = []

    # Calculate rectangle approximation of contours
    for cnt in contourSorted:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        boxSorted = sorted(box, key=lambda x: (x[0], x[1]))
        vertex.append(boxSorted)

    for cnt in all_contourSorted:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        boxSorted = sorted(box, key=lambda x: (x[0], x[1]))
        vertex_all.append(boxSorted)

    imgTest = imgTestP.copy()

    # Find logo
    matchesLogot = ut.flann.knnMatch(des1, des2t, k=2)

    # Store logo good matches
    goodt = []
    for m, n in matchesLogot:
        if m.distance < 0.7 * n.distance:
            goodt.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in goodt]).reshape(-1, 1, 2)
    dst_pts = np.float32(
        [kp2t[m.trainIdx].pt for m in goodt]).reshape(-1, 1, 2)

    # calculate rotation matrix able to match points position
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    # take dimensions
    h, w = img1.shape
    hOrig, wOrig = imgTest.shape

    # make dimensions of new images

    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1],
                      [w - 1, 0]]).reshape(-1, 1, 2)

    # warp images
    dst = cv2.perspectiveTransform(pts, M)

    typeLine = []

    # Find the horizontal line under receipt type
    for vert in vertex:
        if vert[0][1] > dst[1][0][1] + 30:
            if vert[0][1] > vert[1][1]:
                temp = vert[0]
                vert[0] = vert[1]
                vert[1] = temp
            if vert[2][1] < vert[3][1]:
                temp = vert[2]
                vert[2] = vert[3]
                vert[3] = temp
            typeLine = vert
            break

    # check if it's a "loto" or a "standard" receipt

    if((not typeLine) or (abs(typeLine[0][1] - dst[1][0][1]) > (dst[1][0][1] - dst[0][0][1]) * 3)):
        # print "LOTO receipt..."
        raise IndexError('loto')

        # select cropping points
        pts = np.array([[dst[1][0][0], dst[1][0][1] + 30], [dst[1][0][0], hOrig],
                        [dst[2][0][0], hOrig], [dst[2][0][0] + 0, dst[2][0][1] + 20]], dtype=np.int32)

        # create cropping mask
        mask = np.zeros(imgTest.shape, dtype=np.uint8)

        cv2.fillPoly(mask, np.int32([pts]), (255))

        elementerode = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 2))
        elementdilate = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))

        # Crop both the small and big image
        masked_image = imgTest[dst[1][0][1] + 30:hOrig,
                               dst[1][0][0] - 0:dst[2][0][0] + 0]
        masked_imageBig = imgTestBig[
            dst[1][0][1] * 2 + 30:hOrig * 2, (dst[1][0][0]) * 2 - 0:(dst[2][0][0] * 2) + 0]

        # Take small images dimensions
        hm, wm = masked_image.shape

        # Calculate line limit for "Montant Total"
        limit = wm * 3 / 4

        # Perform erosion and dilation of cropped small image
        erosion = cv2.erode(masked_image, elementerode, iterations=3)
        masked_image_dilated = cv2.dilate(
            erosion, elementdilate, iterations=10)
        ret, masked_image_dilated = cv2.threshold(
            masked_image_dilated, 110, 255, cv2.THRESH_BINARY_INV)

        # Find resulting contours
        im2, contours, hierarchy = cv2.findContours(
            masked_image_dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        contourSorted = []
        contours = reversed(contours)
        firstRectLoto = []
        secondRectLoto = []
        firstFound = 0

        # Find rectangle containing type and "montant total"
        for cnt in contours:
            rect = cv2.minAreaRect(cnt)
            box = cv2.boxPoints(rect)
            boxSorted = sorted(box, key=lambda x: (x[0], x[1]))
            if(boxSorted[0][1] > boxSorted[1][1]):
                temp = boxSorted[0][1]
                boxSorted[0][1] = boxSorted[1][1]
                boxSorted[1][1] = temp
            if(boxSorted[3][1] > boxSorted[2][1]):
                temp = boxSorted[3][1]
                boxSorted[3][1] = boxSorted[2][1]
                boxSorted[2][1] = temp
            box = boxSorted
            if(box[0][0] < wm / 4):
                if(firstFound == 0):
                    if(box[0][1] < hm / 4):
                        if(box[2][0] > wm / 2):
                            contourSorted.append(box)
                            firstRectLoto.append(box)
                            firstFound = 1
                else:
                    if(box[0][0] < wm / 8):
                        if(box[2][0] <= limit):
                            secondRectLoto.append(box)
                            break

        # Crop type
        lotoType = masked_imageBig[50: (firstRectLoto[0][1][1] + (firstRectLoto[0][1][1] - firstRectLoto[
                                        0][0][1])) * 2, (firstRectLoto[0][0][0]) * 2:firstRectLoto[0][2][0] * 2]

        # Crop bets
        lotoBet = masked_imageBig[(firstRectLoto[0][1][1] + (firstRectLoto[0][1][1] - firstRectLoto[
                                   0][0][1])) * 2:secondRectLoto[0][0][1] * 2 + 0, 0:firstRectLoto[0][2][0] * 2]

        # Post-processing
        lotoType = jeuReceiptTypeNew(lotoType)

        # Rotate odds images 90 degree counter-clock
        lotoBet = cv2.transpose(lotoBet)
        lotoBet = cv2.flip(lotoBet, 0)

        # Post-processing
        lotoBet = lotoReceiptBetNew(lotoBet)

        cv2.imwrite(outImageFile + "LotoType.jpg", lotoType)
        cv2.imwrite(outImageFile + "LotoBet.jpg", lotoBet)

        return "Loto"

    else:
        # take cropping points
        # print "Standard receipt"

        # Prevent lines to be shorter than they must be
        if(abs(typeLine[3][0] - dst[2][0][0]) > 10):
            typeLine[3][0] = dst[2][0][0]

        if(abs(typeLine[0][0] - dst[0][0][0]) > 10):
            typeLine[0][0] = dst[0][0][0]

        # Calculate warping points
        pts = np.array([[dst[1][0][0] * 2 - 30, dst[1][0][1] * 2 + 30], [typeLine[0][0] * 2, typeLine[0][1] * 2 - 40],
                        [typeLine[3][0] * 2, typeLine[3][1] * 2 - 40], [dst[2][0][0] * 2 + 0, dst[2][0][1] * 2 + 20]], dtype=np.int32)

        # Warp image
        masked_image = warpImage(imgTestBig, pts)
        temph, tempw = masked_image.shape

        # Crop image
        masked_image = masked_image[0: temph - (temph / 3), 0:tempw]

        betLine = []

        # Find the line delimiting bets at bottom
        for vert in vertex:
            lineLength = math.sqrt(math.pow(
                (vert[3][0] - vert[0][0]), 2) + math.pow((vert[3][1] - vert[0][1]), 2))
            lineTypeLength = math.sqrt(math.pow(
                (typeLine[3][0] - typeLine[0][0]), 2) + math.pow((typeLine[3][1] - typeLine[0][1]), 2))

            if vert[0][1] > typeLine[0][1]:
                if lineLength >= lineTypeLength * 3 / 4:
                    if vert[0][1] > vert[1][1]:
                        temp = vert[0]
                        vert[0] = vert[1]
                        vert[1] = temp
                    if vert[2][1] < vert[3][1]:
                        temp = vert[2]
                        vert[2] = vert[3]
                        vert[3] = temp

                    betLine = vert
                    break

        line_date_top = []
        line_date_botom = []

        lineBetLength = math.sqrt(math.pow(
            (betLine[3][0] - betLine[0][0]), 2) + math.pow((betLine[3][1] - betLine[0][1]), 2))

        # Find the line delimiting dates at top
        line_date_top = betLine

        # Find the last line
        # first_line_found = []
        for vert in vertex_all:
            lineLength = math.sqrt(math.pow(
                (vert[3][0] - vert[0][0]), 2) + math.pow((vert[3][1] - vert[0][1]), 2))
            if vert[0][1] > line_date_top[0][1]:
                line_date_botom = np.array([
                    np.array([line_date_top[0][0], vert[0][1]]),
                    np.array([line_date_top[1][0], vert[1][1]]),
                    np.array([line_date_top[2][0], vert[2][1]]),
                    np.array([line_date_top[3][0], vert[3][1]]),
                ])

        # Calculate lines length
        lineTypeLength = math.sqrt(math.pow(
            (typeLine[3][0] - typeLine[0][0]), 2) + math.pow((typeLine[3][1] - typeLine[0][1]), 2))

        # Prevent betline to be shorter than it must be
        if(lineBetLength < lineTypeLength - 20):
            betLine[3][0] = typeLine[3][0] + 20

        # shifting threshold
        shiftZeroZero = 5

        # Set warping points
        pts = np.array([[typeLine[1][0] * 2 + shiftZeroZero, (typeLine[1][1]) * 2], [betLine[1][0] * 2 + shiftZeroZero, (betLine[1][1]) * 2 - 10], [(betLine[2][0] - 10) * 2, betLine[2][1] * 2 - 10], [(typeLine[2][0] - 10) * 2, typeLine[2][1] * 2]], dtype=np.int32)

        masked_imageBet = warpImage(imgTestBig, pts)

        # Get warped image dimensions
        temph, tempw = masked_imageBet.shape

        # Crop bet image

        masked_imageBet = masked_imageBet[0:temph, 0:tempw]

        # Post-process type image
        masked_image = jeuReceiptTypeNew(masked_image)
        cv2.imwrite(outImageFile + "JeuType.jpg", masked_image)

        # Post-process bet image
        masked_imageBet = jeuReceiptBetNew(masked_imageBet)
        cv2.imwrite(outImageFile + "JeuBet.jpg", masked_imageBet)

        date_pts = np.array(
            [
                [
                    line_date_top[1][0] * 2 + shiftZeroZero,
                    (line_date_top[1][1]) * 2 - 20
                ],
                [
                    line_date_botom[1][0] * 2 + shiftZeroZero,
                    (line_date_botom[1][1]) * 2
                ],
                [
                    (line_date_botom[2][0] - 10) * 2,
                    line_date_botom[2][1] * 2 + 30
                ],
                [
                    (line_date_top[2][0] - 10) * 2,
                    line_date_top[2][1] * 2 - 20
                ]
            ], dtype=np.int32
        )

        # Warp image
        masked_image_date = warpImage(imgTestBig, date_pts)

        # Get warped image dimensions
        temph, tempw = masked_image_date.shape

        # Crop bet image
        masked_image_date = masked_image_date[0:temph, 0:tempw]
        masked_image_date = jeuReceiptDates(masked_image_date)

        cv2.imwrite(outImageFile + "JeuDates.jpg", masked_image_date)
        return "Jeu"


def show_image(name, image):
    cv2.imshow(name, image)
    cv2.waitKey(0)


def warpImage(toWarp, pts):
        # take every single point
    (tl, bl, br, tr) = pts

    # width of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # height of the new image
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    dstWarp = np.array([
        [0, 0],
        [0, maxHeight - 1],
        [maxWidth - 1, maxHeight - 1],
        [maxWidth - 1, 0]
    ], dtype="float32")

    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(np.float32([pts]), dstWarp)
    warped = cv2.warpPerspective(toWarp, M, (maxWidth, maxHeight))

    return warped


def lotoReceiptBetNew(crop_imgBet):
    # preserve bet zone
    preBet = crop_imgBet.copy()

    # convert to float
    preBet.astype("float")

    # sharpen bet
    crop_imgBetBlur = cv2.GaussianBlur(preBet, (213, 213), 332150)
    crop_imgBet = cv2.addWeighted(preBet, 5.2, crop_imgBetBlur, -5.5, 0)

    w, h = crop_imgBet.shape

    if (ut.thresh == 1):
        for wn in range(0, w):
            for hn in range(0, h):
                if(crop_imgBet[wn, hn] > 170):
                    crop_imgBet[wn, hn] = 255
                else:
                    crop_imgBet[wn, hn] = 0

    # Denoise Type image
    crop_imgBet = cv2.fastNlMeansDenoising(crop_imgBet, None, 80, 2, 60)

    # Invert image
    crop_imgBetBlack = cv2.bitwise_not(crop_imgBet)

    # Find lines
    lines = cv2.HoughLines(crop_imgBetBlack, 0.9, np.pi / 180, 100)

    if ut.opencv3:
        numlines = len(lines)
    else:
        numlines = len(lines[0])

    thetaTotal = 0
    numRightLines = 0
    totalLines = []

    if numlines > 1:
        for line in lines:
            if ut.opencv3:
                rho, theta = line[0]
            else:
                rho, theta = line

            # choose only horizontal lines
            if theta > 1.53 and theta < 1.57:
                totalLines.append(rho)
                thetaTotal += theta
                numRightLines += 1

        # Calculate mean theta
        thetaTotal /= numRightLines

        # Calculate rotation angle
        thetaTotal = 90 - (thetaTotal * 180 / np.pi)

    # Sort horizontal lines
    totalLines.sort()

    # Check if threshold is greater than 0
    shift = 80
    while totalLines[0] - shift < 0:
        shift -= 1

    # Trim white space
    crop_imgBetBlack = crop_imgBetBlack[
        int(totalLines[0]) - shift:(int(totalLines[len(totalLines) - 1]) + 60), 0:]

    # Rotate inverted image
    crop_imgBetBlack = ut.rotateImage(crop_imgBetBlack, -thetaTotal)

    # Invert image
    crop_imgBetBlack = cv2.bitwise_not(crop_imgBetBlack)

    # remove scratches
    # crop_imgBetBlack = ut.deleteBlobs(crop_imgBetBlack)

    # Save bet type and odds
    return crop_imgBetBlack


def jeuReceiptTypeNew(crop_imgJeuType):
    # preserve image
    pre = crop_imgJeuType.copy()
    # convert to float
    pre.astype("float")

    crop_imgJeuTypeBlur = cv2.GaussianBlur(pre, (213, 213), 332150)
    crop_imgJeuType = cv2.addWeighted(pre, 5.2, crop_imgJeuTypeBlur, -5.5, 0)

    w, h = crop_imgJeuType.shape

    # Threshold images
    if (ut.thresh == 1):
        for wn in range(0, w):
            for hn in range(0, h):
                if(crop_imgJeuType[wn, hn] > 150):
                    crop_imgJeuType[wn, hn] = 255
                else:
                    crop_imgJeuType[wn, hn] = 0

    # Denoise original image
    # crop_imgJeuType = cv2.fastNlMeansDenoising(crop_imgJeuType,None,60,2,60)

    # remove scratches
    # crop_imgJeuType = ut.deleteBlobs(crop_imgJeuType)

    return crop_imgJeuType


def jeuReceiptBetNew(crop_imgBet):

    w, h = crop_imgBet.shape

    # preserve image
    pre = crop_imgBet.copy()

    # convert to float
    pre.astype("float")

    crop_imgBetBlur = cv2.GaussianBlur(pre, (213, 213), 332150)
    crop_imgBet = cv2.addWeighted(pre, 5.2, crop_imgBetBlur, -5.5, 0)

    # Threshold images
    if (ut.thresh == 1):
        for wn in range(0, w):
            for hn in range(0, h):
                if(crop_imgBet[wn, hn] > 170):
                    crop_imgBet[wn, hn] = 255
                else:
                    crop_imgBet[wn, hn] = 0

    # Denoise original image
    crop_imgBet = cv2.fastNlMeansDenoising(crop_imgBet, None, 80, 2, 60)

    # remove scratches
    # crop_imgBet = ut.deleteBlobs(crop_imgBet)

    # crop_imgBet = cv2.GaussianBlur(crop_imgBet,(5,5),0)

    # Save odds
    return crop_imgBet


def jeuReceiptDates(crop_img):
    # preserve image
    pre = crop_img.copy()

    # convert to float
    pre.astype("float")
    crop_imgBlur = cv2.GaussianBlur(pre, (213, 213), 332150)
    crop_img = cv2.addWeighted(pre, 5.2, crop_imgBlur, -5.5, 0)

    # Threshold images
    thresholded = threshold(crop_img, 170, 255)

    # Denoise original image
    crop_img2 = cv2.fastNlMeansDenoising(thresholded, None, 80, 2, 60)
    # remove scratches
    # crop_img = ut.deleteBlobs(crop_img)

    # crop_img = cv2.GaussianBlur(crop_img,(5,5),0)

    # Save odds
    return crop_img2


def threshold(image, from_color, to_color):
    w, h = image.shape
    img_copy = image.copy()
    if (ut.thresh == 1):
        for wn in range(0, w):
            for hn in range(0, h):
                if(img_copy[wn, hn] > 170):
                    img_copy[wn, hn] = 255
                else:
                    img_copy[wn, hn] = 0
    return img_copy


if __name__ == "__main__":
    if sys.argv[1] == "-1":
        fileList = []
        dirs = []
        # All paths
        # dirs = ["test\\2Missingbets"]
        # dirs = ["test\\5X1match"]
        # dirs = ["test\\5X2matches"]
        # dirs = ["test\\9X6matches"]
        # dirs = ["test\\10x4matches"]
        # dirs = ["test\\Loto7"]
        # dirs = ["test\\Loto15"]
        # dirs = ["test\\old"]
        # dirs = ["test\\oldest"]

        # Batch detection
        for dir in dirs:
            for file in os.listdir(dir):
                if file.endswith(".jpg"):
                    fileList.append(os.path.abspath(dir + "\\" + file))
        print "Processing " + str(len(fileList)) + " images..."
        count = 0
        for file in fileList:
            print "Processing " + str(count + 1) + "/" + str(len(fileList))
            count += 1
            start = time.time()
            type = readAndSift(file, sys.argv[2])
            print "Elapsed time: " + str(time.time() - start)
            # Resample image
            # im = Image.open(sys.argv[2]+type + "Bet.jpg")
            # nx, ny = im.size
            # im2 = im.resize((int(nx*8), int(ny*8)), Image.BICUBIC)
            # im2.save(sys.argv[2]+type + "Bet.jpg",dpi=(600,600))

            # Save results
            # outfile = open(dir+"\\"+str(count)+".csv", "wb")
            # writer = csv.writer(outfile)
            # Call receipt parser
            # writer.writerow(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))
    else:
        start = time.time()
        type = readAndSift("test\\9X6matches\\55.jpg", "out")
        print "Elapsed time: " + str(time.time() - start)
        # Resample image
        # im = Image.open(sys.argv[2]+type + "Bet.jpg")
        # nx, ny = im.size
        # im2 = im.resize((int(nx*8), int(ny*8)), Image.BICUBIC)
        # im2.save(sys.argv[2]+type + "Bet.jpg",dpi=(600,600))
        #
        # Print results
        # print(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/Sports_Report_20160830_113207.csv'))
        # print(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))
        #
        # Save results
        outfile = open("./results.csv", "wb")
        writer = csv.writer(outfile)

        # Call receipt parser
        # writer.writerow(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))
