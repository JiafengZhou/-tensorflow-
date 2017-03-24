f = open("1.txt","r+")
for eachLine in f:
    print(eachLine)

filename = input('Enter file name: ')
f = open(filename, 'r')
allLines = f.readlines(),
f.close()
for eachLine in allLines:
    print (eachLine) # suppress print’s NEWLINE

import sys
print ('you entered', len(sys.argv), 'arguments...')
print ('they were:', str(sys.argv))

import os
print(os.path.expanduser('~/py'))




