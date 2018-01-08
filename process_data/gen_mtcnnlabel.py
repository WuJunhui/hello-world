
import xml.etree.ElementTree as ET
dstf=open('mtcnnlabel.txt','w')
with open('trainval.txt','r') as f:
    lines = f.readlines()
for line in lines:
    dstf.write(line.strip()+'.jpg')
    filename='XML/'+line.strip()+'.xml'
    tree = ET.parse(filename)
    for obj in tree.findall('object'):
        bbox = obj.find('bndbox')
        xmin=bbox.find('xmin').text
        ymin=bbox.find('ymin').text
        xmax=bbox.find('xmax').text
        ymax=bbox.find('ymax').text
        dstf.write(' ')
        dstf.write(' '.join([xmin,ymin,xmax,ymax]))
    dstf.write('\n')
dstf.close()
print('Done!')
 