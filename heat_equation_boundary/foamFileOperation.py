# Python function to manipulate OpenFOAM files
# Developer: Jian-Xun Wang (jwang33@nd.edu)
###############################################################################
# system import
import numpy as np
import re
global unitTest 
unitTest = False

def readVectorFromFile(UFile):
	""" 
	Arg: 
	tauFile: The directory path of OpenFOAM vector file (e.g., velocity)

	Regurn: 
	vector: Matrix of vector    
	"""
	resMid = extractVector(UFile)
	with open('Utemp', 'w') as file:
		glob_pattern = resMid.group()
		glob_pattern = re.sub(r'\(', '', glob_pattern)
		glob_pattern = re.sub(r'\)', '', glob_pattern)
		file.write(glob_pattern)
	vector = np.loadtxt('Utemp')
	return vector

	
def readScalarFromFile(fileName):    
	""" 

	Arg: 
	fileName: The file name of OpenFOAM scalar field

	Return:
	a vector of scalar field    
	"""
	resMid = extractScalar(fileName)
	with open('temp', 'w') as file:
		glob_patternx = resMid.group()
		glob_patternx = re.sub(r'\(', '', glob_patternx)
		glob_patternx = re.sub(r'\)', '', glob_patternx)
		file.write(glob_patternx)
	scalarVec = np.loadtxt('temp')
	return scalarVec


################################################ Regular Expression ##################################################### 


def extractVector(vectorFile):
	""" Function is using regular expression select Vector value out
	
	Args:
	UFile: The directory path of file: U

	Returns:
	resMid: the U as (Ux1,Uy1,Uz1);(Ux2,Uy2,Uz2);........
	"""

	fin = open(vectorFile, 'r')  # need consider directory
	line = fin.read() # line is U file to read
	fin.close()
	### select U as (X X X)pattern (Using regular expression)
	patternMid = re.compile(r"""
	(
	\(                                                   # match(
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	(\ )                                                 # match space
	[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
	\)                                                   # match )
	\n                                                   # match next line
	)+                                                   # search greedly
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)
	return resMid    
	
def extractScalar(scalarFile):
	""" subFunction of readTurbStressFromFile
		Using regular expression to select scalar value out 
	
	Args:
	scalarFile: The directory path of file of scalar

	Returns:
	resMid: scalar selected;
			you need use resMid.group() to see the content.
	"""
	with open(scalarFile, 'r') as f:
		line=f.read()
	### select k as ()pattern (Using regular expression)
	patternMid = re.compile(r"""
		\(                                                   # match"("
		\n                                                   # match next line
		(
		[\+\-]?[\d]+([\.][\d]*)?([Ee][+-]?[\d]+)?            # match figures
		\n                                                   # match next line
		)+                                                   # search greedly
		\)                                                   # match")"
	""",re.DOTALL | re.VERBOSE)
	resMid = patternMid.search(line)
	return resMid

