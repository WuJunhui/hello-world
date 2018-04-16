import os
rootdir='train'
dstf=open('train.txt','w')
parts=['babies',  'data0', 'part_katong']
for partdir in parts:
    print partdir
    subdirs = os.listdir(partdir)
    for subdir in subdirs:
        files=os.listdir(os.path.join(partdir,subdir))
        for afile in files:
            filepath=os.path.join(rootdir,partdir,subdir,afile)
            dstf.write(filepath+'\n')
