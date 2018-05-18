#!/usr/bin/env python3

import csv,operator,sys,os
import numpy as np

sys.path.append('../')
from fitModels import fitModels
def readFile(path):
	X = []
	Y = []

	f = open(path,'r')

	for row in f:
		entries = row.strip("\n").split(",")
		
		# 
		Y.append(int(entries[0])-1)
		x = [int(e) for e in entries[1:]]
		X.append(x)

	return np.array(X).astype(dtype=np.float32),np.array(Y)

def main(argv):
	XTrain,YTrain = readFile("train.csv")
	XTest,YTest = readFile("test.csv")

	fitModels(XTrain,YTrain,XTest,YTest)

if __name__ == "__main__":
   main(sys.argv[1:])