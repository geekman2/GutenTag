#  -------------------------------------------------------------------------------
# Name:         module1
# Purpose:
# Author:       Devon Muraoka
# Created:      
# Copyright:   (c) Devon Muraoka, Bharat Ramanathan 
#  -------------------------------------------------------------------------------
import os
import shutil
import sys
# if desired, change the sub dir's name body below
namebody = "chunk_"

dr = os.path.join(os.getcwd(),'test_files')
size = 10000
files = [f for f in os.listdir(dr) if os.path.isfile(dr+"/"+f)]

n = max(1, size)
chunks = [files[i:i + size] for i in xrange(0, len(files), size)]
for i, item in enumerate(chunks):
    subfolder = os.path.join(dr, namebody+str(i+1))
    if not os.path.exists(subfolder):
        os.makedirs(subfolder)
    for f in chunks[i]:
        shutil.move(dr+"/"+f, subfolder+"/"+f)