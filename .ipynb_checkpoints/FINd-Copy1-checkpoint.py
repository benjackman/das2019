#!/usr/bin/env python

import math

from PIL import Image

from matrix import MatrixUtil #Ensure that matrix.py is in the same directory as this file!
from imagehash import ImageHash
import numpy as np


class FINDHasher:

	#  From Wikipedia: standard RGB to luminance (the 'Y' in 'YUV').
	LUMA_FROM_R_COEFF = float(0.299)
	LUMA_FROM_G_COEFF = float(0.587)
	LUMA_FROM_B_COEFF = float(0.114)

	#  Since FINd uses 64x64 blocks, 1/64th of the image height/width
	#  respectively is a full block.
    
    # This is just used as the dimensions of the 
	FIND_WINDOW_SIZE_DIVISOR = 64

	def compute_dct_matrix(self):
		matrix_scale_factor = math.sqrt(2.0 / 64.0)
		d = [0] * 16
		for i in range(0, 16):
			di = [0] * 64
			for j in range(0, 64):
				di[j] = math.cos((math.pi / 2 / 64.0) * (i + 1) * (2 * j + 1))
			d[i] = di
		return d

	def __init__(self):
		"""See also comments on dct64To16. Input is (0..63)x(0..63); output is
		(1..16)x(1..16) with the latter indexed as (0..15)x(0..15).
		Returns 16x64 matrix."""
		self.DCT_matrix = self.compute_dct_matrix()

	def fromFile(self, filepath):
		img = None
		try:
			img = Image.open(filepath)
		except IOError as e:
			raise e
		return self.fromImage(img)

	def fromImage(self,img):
		try:
			# resizing the image proportionally to max 512px width and max 512px height
			img=img.copy()
			img.thumbnail((512, 512)) #https://pillow.readthedocs.io/en/3.1.x/reference/Image.html#PIL.Image.Image.thumbnail
		except IOError as e:
			raise e
		numCols, numRows = img.size
		
        buffer1 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer2 = MatrixUtil.allocateMatrixAsRowMajorArray(numRows, numCols)
		buffer64x64 = MatrixUtil.allocateMatrix(64, 64)
		buffer16x64 = MatrixUtil.allocateMatrix(16, 64)
		buffer16x16 = MatrixUtil.allocateMatrix(16, 16)
		numCols, numRows = img.size
		
        # Note that buffer1 here is luma (since fillfloatLumefrombufferimage requires the input "luma")
        # So we give 
        self.fillFloatLumaFromBufferImage(img, buffer1)
		return self.findHash256FromFloatLuma(
			buffer1, buffer2, numRows, numCols, buffer64x64, buffer16x64, buffer16x16
		)

	def fillFloatLumaFromBufferImage(self, img, luma): # he passes buffer1 as luma in the line above (in the fromImage object)
		
        numCols, numRows = img.size
		rgb_image = img.convert("RGB")
		
        # This is line is run twice (here and bove for some reason)
        numCols, numRows = img.size
        
		for i in range(numRows):
			for j in range(numCols):
				r, g, b = rgb_image.getpixel((j, i))
				# Turning to grayscale by changing the luminence 
                # This index positioning in the list creates an enormous list where each position is the grayscale number 
                luma[i * numCols + j] = (
					self.LUMA_FROM_R_COEFF * r
					+ self.LUMA_FROM_G_COEFF * g
					+ self.LUMA_FROM_B_COEFF * b
				)

	def findHash256FromFloatLuma(
		self,
		fullBuffer1,
		fullBuffer2,
		numRows,
		numCols,
		buffer64x64,
		buffer16x64,
		buffer16x16,
	):
		windowSizeAlongRows = self.computeBoxFilterWindowSize(numCols)
		windowSizeAlongCols = self.computeBoxFilterWindowSize(numRows)
		
		# Here fullbuffer1 and fullbuffer2 are just buffer1 and buffer2 (just renamed)
        self.boxFilter(fullBuffer1,fullBuffer2,numRows,numCols,windowSizeAlongRows,windowSizeAlongCols)
		fullBuffer1=fullBuffer2

		self.decimateFloat(fullBuffer1, numRows, numCols, buffer64x64)
		self.dct64To16(buffer64x64, buffer16x64, buffer16x16)
		hash = self.dctOutput2hash(buffer16x16)
		return hash

	@classmethod
	def decimateFloat(
		cls, in_, inNumRows, inNumCols, out  # numRows x numCols in row-major order
	):
		for i in range(64):
			ini = int(((i + 0.5) * inNumRows) / 64)
			for j in range(64):
				inj = int(((j + 0.5) * inNumCols) / 64)
				out[i][j] = in_[ini * inNumCols + inj]

	def dct64To16(self, A, T, B):
		""" Full 64x64 to 64x64 can be optimized e.g. the Lee algorithm.
		But here we only want slots (1-16)x(1-16) of the full 64x64 output.
		Careful experiments showed that using Lee along all 64 slots in one
		dimension, then Lee along 16 slots in the second, followed by
		extracting slots 1-16 of the output, was actually slower than the
		current implementation which is completely non-clever/non-Lee but
		computes only what is needed."""
		D = self.DCT_matrix

		# B = D A Dt
		# B = (D A) Dt ; T = D A
		# T is 16x64;

		# T = D A
		# Tij = sum {k} Dik Akj

		T = [0] * 16
		for i in range(0, 16):
			ti = [0] * 64
			for j in range(0, 64):
				tij = 0.0
				for k in range(0, 64):
					tij += D[i][k] * A[k][j]
				ti[j] = tij
			T[i] = ti

		# B = T Dt
		# Bij = sum {k} Tik Djk
		for i in range(16):
			for j in range(16):
				sumk = float(0.0)
				for k in range(64):
					sumk += T[i][k] * D[j][k]
				B[i][j] = sumk

	def dctOutput2hash(self, dctOutput16x16):
		"""
		Each bit of the 16x16 output hash is for whether the given frequency
		component is greater than the median frequency component or not.
		"""
		hash = np.zeros((16,16),dtype="int")
		dctMedian = MatrixUtil.torben(dctOutput16x16, 16, 16)
		for i in range(16):
			for j in range(16):
				if dctOutput16x16[i][j] > dctMedian:
					hash[15-i,15-j]=1
		return ImageHash(hash.reshape((256,)))

	@classmethod
	# We think this might tell us what sizse the windo is
    # Note that we call cls when we make a method, and when we create a function we call self
    
    # I do not understand the intuition behind producing the window size this way 
    
    # Note that above Scott passes num_rows and num_columns as dimensions 
    def computeBoxFilterWindowSize(cls, dimension):
		""" Round up."""
		# This avoides the typical off by one error
        # But the int function rounds DOWN we think - I have sent him an email  
        return int(
			(dimension + cls.FIND_WINDOW_SIZE_DIVISOR - 1)
			/ cls.FIND_WINDOW_SIZE_DIVISOR
		)

	@classmethod
	def boxFilter(cls,input,output,rows,cols,rowWin,colWin):
		
        # 
        
        # Not sure why he's doing weird division here 
        # We think that he is creating the window size or something 
        
        halfColWin = int((colWin + 2) / 2)  # 7->4, 8->5
		halfRowWin = int((rowWin + 2) / 2) 
		
        # This is ridiculous chunk of 4 nested 4 loops 
                
        for i in range(0,rows):
			for j in range(0,cols):
				s=0
				
                # I think this is cropping the window? So we only pull information from actual pixels (otherwise we try to index outside the range)
                xmin=max(0,i-halfRowWin)
				xmax=min(rows,i+halfRowWin)
				ymin=max(0,j-halfColWin)
				ymax=min(cols,j+halfColWin)
				for k in range(xmin,xmax):
					for l in range(ymin,ymax):
						# += is just a shortuct for s = s + ....
                        s+=input[k*rows+l]
				output[i*rows+j]=s/((xmax-xmin)*(ymax-ymin))

	@classmethod
	def prettyHash(cls,hash):
		#Hashes are 16x16. Print in this format
		if len(hash.hash)!=256:
			print("This function only works with 256-bit hashes.")
			return
		return np.array(hash.hash).astype(int).reshape((16,16))
		#return "{}".format(np.array2string(np.array(hash.hash).astype(int).reshape((16,16)),separator=""))


if __name__ == "__main__":
	import sys
	find=FINDHasher()
	for filename in sys.argv[1:]:
		h=find.fromFile(filename)
		print("{},{}".format(h,filename))
		print(find.prettyHash(h))