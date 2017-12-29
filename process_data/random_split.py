import random

ftv=open('train.txt','w')
ftest=open('val.txt','w')
with open('list.txt','r') as f:
    lines=f.readlines()

random.shuffle(lines)
linenum=len(lines)
for i in range(linenum):
    if i <linenum*0.8:
        ftv.write(lines[i])
    else:
        ftest.write(lines[i])
