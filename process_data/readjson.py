import os
import json
import numpy as np
import cv2
import xml.etree.ElementTree as ET

def isall999(alist):
    flag=True
    for i in alist:
        if not (int(i) == 999 or int(i) ==100):
            flag=False 
    return flag

def parse_json_line(line):
    '''return imgName as string, rects as list of [personID(as str),xmin(as int),ymin,width,height]
    '''
    anno = json.loads(line)
    imgName = anno['imgName']
    rectsjson = anno['data']

    rects=[]
    for rect in rectsjson:
        xmin=rect['x'] 
        ymin=rect['y']
        width=rect['width']
        height=rect['height']
        personID=rect['code']
        rects.append([personID,xmin,ymin,width,height])
    return imgName, rects

def find_not_inline(jsonname):
    '''
    '''
    imgName_before='_00.'
    with open(jsonname,'r') as f:
        lines = f.readlines()
        for line in lines:
            if line.strip() != '':
                imgName, rects = parse_json_line(line)
                #print imgName
                if not int(imgName_before.split('_')[-1].split('.')[0])<int(imgName.split('_')[-1].split('.')[0]) :
                    print 'before:', imgName_before
                    print 'now:', imgName
                imgName_before=imgName
    print 'Find not inline Done!'    

def see_annos(imroot,jsonname,only_inside_person=True):
    with open(jsonname,'r') as f:
        lines = f.readlines()
    cnt=0
    cnt_p=0
    for line in lines:
        if line.strip() != '':
            imgName, rects = parse_json_line(line)
            im=cv2.imread(imroot+imgName)
            im_toshow=im.copy()
            personids=[]
            for rect in rects:
                personids.append(rect[0])
                if only_inside_person:
                    if (int(rect[0])==999 or int(rect[0])==100):
                        continue
                cnt+=1
                print 'person_id, person_cnt:', rect[0],cnt
                cv2.rectangle(im_toshow,(rect[1],rect[2]),(rect[1]+rect[3],rect[2]+rect[4]),(0,0,255),2)
                cv2.putText(im_toshow,rect[0],(rect[1],rect[2]),cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
                
            if not isall999(personids):
                cnt_p+=1
                print 'img_num:', cnt_p
                cv2.imshow('imgName',im_toshow)
                cv2.waitKey(100)
    print 'See anno Done!'

def crop_person(imroot,jsonname,cropdir,name_prefix=''):
    with open(jsonname,'r') as f:
        lines = f.readlines()
    cnt=0
    for line in lines:
        if line.strip() != '':
            imgName, rects = parse_json_line(line)
            im=cv2.imread(imroot+imgName)
            personids=[]
            for rect in rects:
                personids.append(rect[0])
                if not (int(rect[0])==999 or int(rect[0])==100):
                    cnt+=1
                    print 'person_id, person_cnt:', rect[0],cnt
                    im_crop=im[rect[2]:rect[2]+rect[4],rect[1]:rect[1]+rect[3]]
                    dstdir=cropdir+'/'+name_prefix+'_'+rect[0]
                    if not os.path.exists(dstdir):
                        os.mkdir(dstdir)
                    im_crop_name=dstdir+'/'+name_prefix+'_'+rect[0]+'_'+imgName
                    #print 'im_crop_name:',im_crop_name
                    cv2.imwrite(im_crop_name,im_crop)
                    #cv2.imshow('im_crop',im_crop)
                    #cv2.waitKey(0)
    print 'Crop person Done!'

def gen_person_xml(imroot,jsonname,xmldir):
    lines=[]
    with open(jsonname,'r') as f:
        l=f.readline()
        while l:
            if l.strip() != '':
                lines.append(l)
            l=f.readline()
    print 'len:',len(lines)
    cnt_p=0
    for line in lines:
        imgName, rects = parse_json_line(line)
        im=cv2.imread(imroot+imgName)
        h,w,ch=im.shape
        #im_toshow=im.copy()
        personids=[]
        for rect in rects:
            personids.append(rect[0])
            #cv2.rectangle(im_toshow,(rect[1],rect[2]),(rect[1]+rect[3],rect[2]+rect[4]),(0,0,255),2)
            #cv2.putText(im_toshow,rect[0],(rect[1],rect[2]),cv2.FONT_HERSHEY_COMPLEX, 2, (0,0,255))
        if not isall999(personids):
            cnt_p+=1
            print 'img_num:', cnt_p
            #cv2.imshow('imgName',im_toshow)
            #cv2.waitKey(100)
            xmlroot = ET.Element('annotation')
            ET.SubElement(xmlroot,'folder').text='persondata'    
            ET.SubElement(xmlroot,'filename').text=imgName
            size_tag=ET.SubElement(xmlroot,'size')
            ET.SubElement(size_tag, 'depth').text = str(ch)
            ET.SubElement(size_tag, 'height').text = str(h)
            ET.SubElement(size_tag, 'width').text = str(w)
            # format objects
            for rect in rects:
                obj_tag = ET.SubElement(xmlroot,'object')
                ET.SubElement(obj_tag, 'name').text = 'person'
                ET.SubElement(obj_tag,'difficult').text = '0'
                bndbox_tag = ET.SubElement(obj_tag,'bndbox')
                xmin=rect[1]
                ymin=rect[2]
                xmax=rect[1]+rect[3]
                ymax=rect[2]+rect[4]
                ET.SubElement(bndbox_tag, "xmin").text = str(xmin)
                ET.SubElement(bndbox_tag, "ymin").text = str(ymin)
                ET.SubElement(bndbox_tag, "xmax").text = str(xmax)
                ET.SubElement(bndbox_tag, "ymax").text = str(ymax)
            tree=ET.ElementTree(xmlroot)
            xmlfile=xmldir+'/'+imgName.split('.')[0]+'.xml'
            #xmlfile='tmp.xml'
            tree.write(xmlfile)
    print 'Gen xml Done!'


if __name__=='__main__':
    imroot='frames_fourp1/'
    jsonname='anno_frames_fourp1.txt'
    #find_not_inline(jsonname)
    #see_annos(imroot,jsonname,False)
    cropdir='cropped_person_fourp1'
    if not os.path.exists(cropdir):
        os.mkdir(cropdir)
    name_prefix='8230003'
    crop_person(imroot,jsonname,cropdir,name_prefix)
    #dirsufs=['08240100','08240101','08240102','08240200','08240201','08240202','08240300','08240301']
    #xmldir='XML_38'
    #gen_person_xml(imroot,jsonname,xmldir)
    #for dirsuf in dirsufs:
    #    imroot='frames_'+dirsuf+'/'
    #    jsonname='anno_frames_'+dirsuf+'.txt'
    #    gen_person_xml(imroot,jsonname,xmldir)
