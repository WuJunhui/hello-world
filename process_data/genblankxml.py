import os
import cv2
import xml.etree.ElementTree as ET

def gen_person_xml(imgName):
    im=cv2.imread(imgName)
    h,w,ch=im.shape

    xmlroot = ET.Element('annotation')
    ET.SubElement(xmlroot,'folder').text='clothdata'    
    ET.SubElement(xmlroot,'filename').text=imgName
    size_tag=ET.SubElement(xmlroot,'size')
    ET.SubElement(size_tag, 'depth').text = str(ch)
    ET.SubElement(size_tag, 'height').text = str(h)
    ET.SubElement(size_tag, 'width').text = str(w)
    # format objects
    tree=ET.ElementTree(xmlroot)
    xmlfile=imgName.split('.')[0]+'.xml'
    tree.write(xmlfile)
 