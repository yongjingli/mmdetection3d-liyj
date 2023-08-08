import sys
import os

width = 1920
height = 1208
topMetaRows = 24
bottomMetaRows = 4
bytesPerPixel = 2

if len(sys.argv) == 2 :
	sourceFile = sys.argv[1]
	frameCount = 10
elif len(sys.argv) == 3 :
	sourceFile = sys.argv[1]
	frameCount = int(sys.argv[2])
else :
	print "usage: python frame-extractor.py [sourceFileName] [frameCount]"
	sys.exit(2)

dstFile = "{}-{}frame.raw".format(os.path.splitext(sourceFile)[0], frameCount)

fo = open(sourceFile, "rb")
fw = open(dstFile, "wb")
try:
	junk = fo.read(88)
	for t in range(frameCount):
		print "frame ",t
		sumBytes = 0
		
		for i in range(topMetaRows):
			junk = fo.read(width*bytesPerPixel)

		for i in range(height):
			goodBytes = fo.read(width*bytesPerPixel)
			fw.write(goodBytes)
		
		junk = fo.read(4*bytesPerPixel)

		for i in range(bottomMetaRows):
			junk = fo.read(width*bytesPerPixel)
	
		for i in range(1236-height):
			byteList2 = [0] * (width*bytesPerPixel)
			fw.write(bytearray(byteList2))
finally:
	fo.close()
	fw.close()

