import numpy as np
import sys
import cv2
import math
import utils as ut
from PIL import Image
import os
import os.path
import csv
import cPickle
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

firstExec = True

def readAndSift(imageFile, outImageFile):
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

	global firstExec

	# logo Image
	img1 = cv2.imread(os.path.join(package_directory, "logo.jpg"),0)
	# test image
	img2 = cv2.imread(imageFile,0)
	# Bottom montant Image
	img3 = cv2.imread(os.path.join(package_directory, "gain.jpg"),0)
	# "Jeu" montant Image
	img4 = cv2.imread(os.path.join(package_directory, "1n2.jpg"),0)

	# second type of receipt
	img5 = cv2.imread(os.path.join(package_directory, "montanTotal.jpg"),0)
	img6 = cv2.imread(os.path.join(package_directory, "lotoFoot.jpg"),0)

	if(firstExec):
		# read detected and computed SIFT descriptors and keypoints (first set of receipts)
		kp1 = ut.read_keyps_from_file("kp1.txt")
		des1 = ut.read_desc_from_file("des1.desc.npy")

		kp3 = ut.read_keyps_from_file("kp3.txt")
		des3 = ut.read_desc_from_file("des3.desc.npy")

		kp4 = ut.read_keyps_from_file("kp4.txt")
		des4 = ut.read_desc_from_file("des4.desc.npy")
		# read detected and computed SIFT descriptors and keypoints (second set of receipts)
		kp5 = ut.read_keyps_from_file("kp5.txt")
		des5 = ut.read_desc_from_file("des5.desc.npy")

		kp6 = ut.read_keyps_from_file("kp6.txt")
		des6 = ut.read_desc_from_file("des6.desc.npy")
		firstExec = False

	#kp1, des1 = ut.sift.detectAndCompute(img1,None)
	kp2, des2 = ut.sift.detectAndCompute(img2,None)
	#kp3, des3 = ut.sift.detectAndCompute(img3,None)
	#kp4, des4 = ut.sift.detectAndCompute(img4,None)
	# detect and compute SIFT descriptors (second set of receipts)
	#kp5, des5 = ut.sift.detectAndCompute(img5,None)
	#kp6, des6 = ut.sift.detectAndCompute(img6,None)

	# Find logo
	matchesLogo = ut.flann.knnMatch(des1,des2,k=2)
	# Find "Gain"
	matchesMontant = ut.flann.knnMatch(des3,des2,k=2)
	# Find "Montant"
	matchesMontanTotal = ut.flann.knnMatch(des5,des2,k=2)

	# Store logo good matches
	good = []
	for m,n in matchesLogo:
		if m.distance < 0.7*n.distance:
			good.append(m)

	# store "Gain" good matches
	goodMontant = []
	for m,n in matchesMontant:
		if m.distance < 0.7*n.distance:
			goodMontant.append(m)

	# store "Montant" the good matches
	goodMontanTotal = []
	for m,n in matchesMontanTotal:
		if m.distance < 0.7*n.distance:
			goodMontanTotal.append(m)

	if len(good)>ut.MIN_MATCH_COUNT:
		if len(goodMontant)> len(goodMontanTotal):
			# First type of receipt
			type, warped, dstJeu = jeuReceiptType()
			# save bet type
			if type is not None:
				# Write type image
				cv2.imwrite(outImageFile + "JeuType.jpg", type)

				# Find bet zone
				betZone = jeuReceiptBet(warped, dstJeu)

				if betZone is not None:
					# Write bet zone
					cv2.imwrite(outImageFile + "JeuBet.jpg", betZone)
					return "Jeu"
				else:
					return {'error': "No bets have been found!"}
			else:
				return {'error': "No type has been found!"}

		else:
			# Second type of receipt
			type, warped, dstLoot = lotoReceiptType()

			# save bet type
			if type is not None:
				# Write type image
				cv2.imwrite(outImageFile + "LotoType.jpg", type)

				# Find bet zone
				betZone = lotoReceiptBet(warped, dstLoot)

				if betZone is not None:
					# Write bet zone
					cv2.imwrite(outImageFile + "LotoBet.jpg", betZone)
					return "Loto"
				else:
					return {"error": "No bets have been found!"}
			else:
				return {"error": "No type has been found!"}

	else:
		return {"error": "This is not a valid receipt! No logo found."}


def jeuReceiptType():
		# take only good matches (both train and test)
		src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
		src_ptsMontant = np.float32([ kp3[m.queryIdx].pt for m in goodMontant ]).reshape(-1,1,2)

		dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
		dst_ptsMontant = np.float32([ kp2[m.trainIdx].pt for m in goodMontant ]).reshape(-1,1,2)

		# calculate rotation matrix able to match points position
		M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
		MMontant, maskMontant = cv2.findHomography(src_ptsMontant, dst_ptsMontant, cv2.RANSAC,5.0)

		# take dimensions
		h,w = img1.shape
		hMontant,wMontant = img3.shape
		hOrig,wOrig = img2.shape

		# make dimensions of new images
		pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
		ptsMontant = np.float32([ [0,0],[0,hMontant-1],[wMontant-1,hMontant-1],[wMontant-1,0] ]).reshape(-1,1,2)

		# warp images
		dst = cv2.perspectiveTransform(pts,M)
		dstMontant = cv2.perspectiveTransform(ptsMontant,MMontant)

		# take cropping points
		pts = np.array([[dst[1][0][0]-30, dst[1][0][1]], [dstMontant[0][0][0]-0, dstMontant[0][0][1] + 20], [dstMontant[3][0][0], dstMontant[3][0][1]], [dst[2][0][0]+10, dst[2][0][1]  - 20 ]], dtype=np.int32)

		# take every single point
		(tl, bl, br, tr) = pts

		# create cropping mask
		mask = np.zeros(img2.shape, dtype=np.uint8)

		cv2.fillPoly(mask, np.int32([pts]), (255))

		# apply the mask
		masked_image = cv2.bitwise_and(img2, mask)

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
		], dtype = "float32")

		# compute the perspective transform matrix and then apply it
		M = cv2.getPerspectiveTransform(np.float32([pts]), dstWarp)
		warped = cv2.warpPerspective(masked_image, M, (maxWidth, maxHeight))

		kpJeu, desJeu = ut.sift.detectAndCompute(warped,None)
		matchesJeuMontant = ut.flann.knnMatch(des4,desJeu,k=2)

		goodJeuMontant = []
		for m,n in matchesJeuMontant:
			if m.distance < 0.7*n.distance:
				goodJeuMontant.append(m)
		dstJeu = None

		if len(goodJeuMontant)>ut.MIN_MATCH_COUNTJeu:
			src_ptsJeuMontant = np.float32([ kp4[m.queryIdx].pt for m in goodJeuMontant ]).reshape(-1,1,2)

			dst_ptsJeuMontant = np.float32([ kpJeu[m.trainIdx].pt for m in goodJeuMontant ]).reshape(-1,1,2)

			MJeu, maskJeu = cv2.findHomography(src_ptsJeuMontant, dst_ptsJeuMontant, cv2.RANSAC,5.0)

			hJeu,wJeu = img4.shape
			hOrig,wOrig = warped.shape

			ptsJeu = np.float32([ [0,0],[0,hJeu-1],[wJeu-1,hJeu-1],[wJeu-1,0] ]).reshape(-1,1,2)

			dstJeu = cv2.perspectiveTransform(ptsJeu,MJeu)

			ww, hw = warped.shape

			crop_imgJeuType = warped[int(dstJeu[0][0][1]):int(dstJeu[1][0][1]), 1:hw-1]

			# preserve image
			pre = crop_imgJeuType.copy()

			# convert to float
			pre.astype("float")

			crop_imgJeuTypeBlur = cv2.GaussianBlur(pre,(213,213),332150)
			crop_imgJeuType = cv2.addWeighted(pre, 5.2, crop_imgJeuTypeBlur, -5.5, 0);

			w, h = crop_imgJeuType.shape

			# Threshold images
			if (ut.thresh == 1):
				for wn in range(0, w):
					for hn in range(0, h):
						if(crop_imgJeuType[wn,hn] > 150):
							crop_imgJeuType[wn,hn] = 255
						else:
							crop_imgJeuType[wn,hn] = 0

			#Denoise original image
			#crop_imgJeuType = cv2.fastNlMeansDenoising(crop_imgJeuType,None,60,2,60)

			# remove scratches
			crop_imgJeuType = ut.deleteBlobs(crop_imgJeuType)

			return crop_imgJeuType, warped, dstJeu
		else:
			print "No matches have been found for first type receipt!"
			return None, None, None

def jeuReceiptBet(warped, dstJeu):
	#Find horizontal lines
	adapt = cv2.adaptiveThreshold(warped, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, -2)
	hadapt, wadapt = adapt.shape
	horizontalsize = wadapt / 30
	horizontalStructure = cv2.getStructuringElement(cv2.MORPH_RECT,(horizontalsize,2))
	erosion = None
	dilation = None

	erosion = cv2.erode(adapt,horizontalStructure, erosion, (-1,-1))
	dilation = cv2.dilate(erosion,horizontalStructure, dilation, (-1,-1))

	#Find canny
	edges = cv2.Canny(dilation,50,150,apertureSize = 3)
	lines = cv2.HoughLines(edges,0.9,np.pi/180,100)
	if ut.opencv3:
		numlines = len(lines)
	else:
		numlines = len(lines[0])

	lastRho = []
	lastTheta = []
	firstEx = True
	write = True

	# select only suitable lines
	if numlines > 1:
		for line in lines:
			if ut.opencv3:
				rho,theta = line[0]
			else:
				rho,theta = line
			if firstEx==True:
				lastRho.append(rho)
				lastTheta.append(theta)
				firstEx = False
			else:
				for rhos in lastRho:
					if abs(rhos - rho) < 50:
						write = False
				if write == True:
					lastRho.append(rho)
					lastTheta.append(theta)
				else:
					write = True

		thetaRho = zip(lastRho, lastTheta)
		thetaRho.sort();
		lastRho.sort();

		#select first line index
		firstLineIndex = None

		for idx, rhos in enumerate(lastRho):
			if rhos >  dstJeu[1][0][1]:
				firstLineIndex = idx
				break

		maskBet = np.zeros(warped.shape, dtype=np.uint8)
		heightMask, widthMask = maskBet.shape

		# calculate line start and end points

		# shifting last line
		shift = 1

		lastRho = [lastRho for lastRho, lastTheta in thetaRho]

		# crop odds
		ptsBet = np.array([ [ 0, lastRho[firstLineIndex]], [0, lastRho[len(lastRho)-shift]], [widthMask, lastRho[len(lastRho)-shift]], [widthMask, lastRho[firstLineIndex]]], dtype=np.int32)

		# create mask
		cv2.fillPoly(maskBet, np.int32([ptsBet]), (255))

		# apply mask
		maskedBet_image = cv2.bitwise_and(warped, maskBet)

		w,h = maskedBet_image.shape
		crop_imgBet = maskedBet_image[int(ptsBet[0][1]):int(ptsBet[1][1]), 1:h-1]

		w, h = crop_imgBet.shape

		# preserve image
		pre = crop_imgBet.copy()

		# convert to float
		pre.astype("float")

		crop_imgBetBlur = cv2.GaussianBlur(pre,(213,213),332150)
		crop_imgBet = cv2.addWeighted(pre, 5.2, crop_imgBetBlur, -5.5, 0);

		# Threshold images
		if (ut.thresh == 1):
			for wn in range(0, w):
				for hn in range(0, h):
					if(crop_imgBet[wn,hn] > 170):
						crop_imgBet[wn,hn] = 255
					else:
						crop_imgBet[wn,hn] = 0



		#Denoise original image
		crop_imgBet = cv2.fastNlMeansDenoising(crop_imgBet,None,80,2,60)


		# remove scratches
		crop_imgBet = ut.deleteBlobs(crop_imgBet)

		#crop_imgBet = cv2.GaussianBlur(crop_imgBet,(5,5),0)

		#Save odds
		return crop_imgBet

	else:
		print "No odds have been founds!"
		return None

def lotoReceiptType():
	src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
	src_ptsMontanTotal = np.float32([ kp5[m.queryIdx].pt for m in goodMontanTotal ]).reshape(-1,1,2)

	dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
	dst_ptsMontanTotal = np.float32([ kp2[m.trainIdx].pt for m in goodMontanTotal ]).reshape(-1,1,2)

	M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
	MMontanTotal, maskMontanTotal = cv2.findHomography(src_ptsMontanTotal, dst_ptsMontanTotal, cv2.RANSAC,5.0)

	h,w = img1.shape
	hMontanTotal,wMontanTotal = img5.shape

	pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
	ptsMontanTotal = np.float32([ [0,0],[0,hMontanTotal-1],[wMontanTotal-1,hMontanTotal-1],[wMontanTotal-1,0] ]).reshape(-1,1,2)

	dst = cv2.perspectiveTransform(pts,M)
	dstMontantTotal = cv2.perspectiveTransform(ptsMontanTotal,MMontanTotal)

	pts = np.array([[dst[1][0][0]-30, dst[1][0][1] + 0], [dstMontantTotal[0][0][0]-15, dstMontantTotal[0][0][1] + 20], [dst[3][0][0], dstMontantTotal[3][0][1]], [dst[2][0][0]+15, dst[2][0][1]  - 20 ]], dtype=np.int32)

	(tl, bl, br, tr) = pts

	mask = np.zeros(img2.shape, dtype=np.uint8)

	cv2.fillPoly(mask, np.int32([pts]), (255))

	# apply the mask
	masked_image = cv2.bitwise_and(img2, mask)

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
	], dtype = "float32")

	# compute the perspective transform matrix and then apply it
	M = cv2.getPerspectiveTransform(np.float32([pts]), dstWarp)
	warped = cv2.warpPerspective(masked_image, M, (maxWidth, maxHeight))

	kpW, desW = ut.sift.detectAndCompute(warped,None)

	matchesLoto = ut.flann.knnMatch(des6,desW,k=2)

	# store "Loto" good matches
	goodLoto = []
	for m,n in matchesLoto:
		if m.distance < 0.8*n.distance:
			goodLoto.append(m)

	if len(goodLoto)> ut.MIN_MATCH_COUNTLoto:
		src_ptsLoot = np.float32([ kp6[m.queryIdx].pt for m in goodLoto ]).reshape(-1,1,2)
		dst_ptsLoot = np.float32([ kpW[m.trainIdx].pt for m in goodLoto ]).reshape(-1,1,2)

		MLoot, maskLoot = cv2.findHomography(src_ptsLoot, dst_ptsLoot, cv2.RANSAC,5.0)

		hLoot,wLoot = img6.shape

		ptsLoot = np.float32([ [0,0],[0,hLoot-1],[wLoot-1,hLoot-1],[wLoot-1,0] ]).reshape(-1,1,2)

		dstLoot = cv2.perspectiveTransform(ptsLoot,MLoot)

		ww, hw = warped.shape

		crop_img = warped[int(dstLoot[0][0][1]):int(dstLoot[1][0][1]), 1:hw-1]
		# preserve image
		preType = crop_img.copy()
		# convert to float
		preType.astype("float")
		# sharpen type
		crop_imgTypeBlur = cv2.GaussianBlur(preType,(213,213),332150)
		crop_img = cv2.addWeighted(preType, 5.2, crop_imgTypeBlur, -5.5, 0);

		w, h = crop_img.shape

		# Threshold images
		if (ut.thresh == 1):
			for wn in range(0, w):
				for hn in range(0, h):
					if(crop_img[wn,hn] > 150):
						crop_img[wn,hn] = 255
					else:
						crop_img[wn,hn] = 0

		#Denoise Type image
		#crop_img = cv2.fastNlMeansDenoising(crop_img,None,80,2,60)

		# remove scratches
		crop_img = ut.deleteBlobs(crop_img)

		return crop_img, warped, dstLoot
	else:
		return None, None, None

def lotoReceiptBet(warped, dstLoot):
	# Take image dims
	ww, hw = warped.shape

	# Calculate "loto" length in pixel
	lotoLength = math.sqrt(math.pow((dstLoot[0][0][0] - dstLoot[1][0][0]),2) + math.pow((dstLoot[0][0][1] - dstLoot[1][0][1]),2))

	# crop bet zone
	crop_imgBet = warped[int(dstLoot[1][0][1])+ int(lotoLength):ww-int(lotoLength), 1:hw-1]

	# Rotate odds images 90 degree counter-clock
	crop_imgBet = cv2.transpose(crop_imgBet)
	crop_imgBet = cv2.flip(crop_imgBet,0)


	#preserve bet zone
	preBet = crop_imgBet.copy()

	# convert to float
	preBet.astype("float")

	# sharpen bet
	crop_imgBetBlur = cv2.GaussianBlur(preBet,(213,213),332150)
	crop_imgBet = cv2.addWeighted(preBet, 5.2, crop_imgBetBlur, -5.5, 0);

	w, h = crop_imgBet.shape

	if (ut.thresh == 1):
		for wn in range(0, w):
			for hn in range(0, h):
				if(crop_imgBet[wn,hn] > 170):
					crop_imgBet[wn,hn] = 255
				else:
					crop_imgBet[wn,hn] = 0

	#Denoise Type image
	crop_imgBet = cv2.fastNlMeansDenoising(crop_imgBet,None,80,2,60)

	# Invert image
	crop_imgBetBlack = cv2.bitwise_not(crop_imgBet);

	# Find lines
	lines = cv2.HoughLines(crop_imgBetBlack,0.9,np.pi/180,100)

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
				rho,theta = line[0]
			else:
				rho,theta = line

			# choose only horizontal lines
			if theta > 1.53 and theta < 1.57:
				totalLines.append(rho)
				thetaTotal += theta
				numRightLines += 1

		# Calculate mean theta
		thetaTotal /= numRightLines;

		# Calculate rotation angle
		thetaTotal = 90 - (thetaTotal * 180 / np.pi)

	# Sort horizontal lines
	totalLines.sort()

	# Check if threshold is greater than 0
	shift = 50
	while totalLines[0]-shift <0:
		shift -= 1

	# Trim white space
	crop_imgBetBlack = crop_imgBetBlack[int(totalLines[0])-shift:(int(totalLines[len(totalLines)-1]) + 20), 0:]

	# Rotate inverted image
	crop_imgBetBlack = ut.rotateImage(crop_imgBetBlack, -thetaTotal)

	# Invert image
	crop_imgBetBlack = cv2.bitwise_not(crop_imgBetBlack)

	# remove scratches
	crop_imgBetBlack = ut.deleteBlobs(crop_imgBetBlack)

	# Save bet type and odds
	return crop_imgBetBlack


if __name__ == "__main__":

	if sys.argv[1] == "-1":
		fileList = []

		# All paths
		#dirs = ["test\\old","test\\nr\\5x1","test\\nr\\5x2","test\\nr\\10x4","test\\nr\\11x6"]

		dirs = ["test"]

		#dirs = ["test\\old"]

		#dirs = ["test\\nr\\5x1"]

		#dirs = ["test\\nr\\5x2"]

		#dirs = ["test\\nr\\10x4"]

		#dirs = ["test\\nr\\11x6"]

		# Batch detection
		for dir in dirs:
			for file in os.listdir(dir):
				if file.endswith(".jpg"):
					fileList.append(os.path.abspath(dir+"\\"+file))
		print "Processing " + str(len(fileList)) + " images..."
		count = 0
		for file in fileList:
			print "Processing " + str(count+1) + "/"+str(len(fileList))
			count += 1
			start = time.time()
			type = readAndSift(file,sys.argv[2])
			print "Elapsed time: " + str(time.time()-start)
			# Resample image
			#im = Image.open(sys.argv[2]+type + "Bet.jpg")
			#nx, ny = im.size
			#im2 = im.resize((int(nx*8), int(ny*8)), Image.BICUBIC)
			#im2.save(sys.argv[2]+type + "Bet.jpg",dpi=(600,600))

			# Save results
			outfile = open(dir+"\\"+str(count)+".csv", "wb")
			writer = csv.writer(outfile)
			# Call receipt parser
			#writer.writerow(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))
	else:
		#type = readAndSift("test\\nr\\5x1\\"+sys.argv[1]+".jpg",sys.argv[2])

		#type = readAndSift("test\\nr\\5x2\\"+sys.argv[1]+".jpg",sys.argv[2])

		#type = readAndSift("test\\nr\\10x4\\"+sys.argv[1]+".jpg",sys.argv[2])

		#type = readAndSift("test\\nr\\11x6\\"+sys.argv[1]+".jpg",sys.argv[2])
		start = time.time()
		type = readAndSift("test\\old\\"+sys.argv[1]+".jpg",sys.argv[2])
		print "Elapsed time: " + str(time.time()-start)
		# Resample image
		#im = Image.open(sys.argv[2]+type + "Bet.jpg")
		#nx, ny = im.size
		#im2 = im.resize((int(nx*8), int(ny*8)), Image.BICUBIC)
		#im2.save(sys.argv[2]+type + "Bet.jpg",dpi=(600,600))

		# Print results
		#print(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/Sports_Report_20160830_113207.csv'))
		#print(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))

		# Save results
		outfile = open("./results.csv", "wb")
		writer = csv.writer(outfile)

		# Call receipt parser
		#writer.writerow(ps.getBets('./'+sys.argv[2]+type+'Type.jpg', './'+sys.argv[2]+type+'Bet.jpg', 'test/csv/20160830/12_13_14_15_16_17_18_19_0916.csv'))
