# -*- coding: utf-8 -*-
import requests
import os



def getAlbums(homeUrl):
    '''return a list of tuple: [(album_url1,album_name1),...]
    '''
    contents=requests.get(url=homeUrl).content.split('\n')
    albums=[]
    for item in contents:
        if '/album/' in item and 'https:' in item and not 'album_photo' in item and not 'tags' in item:
            _,album_url,album_name=item.split('"')
            album_name=album_name.strip('</a>')
            albums.append((album_url,album_name))
    return albums

def getPhotos(album_url):
    '''return a list of photo_url:[photo_url1,photo_url2...]
    '''
    contents=requests.get(url=album_url).content.split('\n')
    photo_urls=[]
    for item in contents:
        if 'photolst_photo' in item and 'https' in item:
            photo_url=item.split('"')[1]
            photo_urls.append(photo_url)
        if 'rel="next"' in item:
            next_page=item.split('"')[3]
            photo_urls+=getPhotos(next_page)
    return photo_urls

def downloadPhoto(photo_url,prefix):
    '''download photo in prefix/imname_uploaddata_imgtitle.jpg
    '''
    req=requests.get(url=photo_url)
    contents=req.content.split('\n')
    up_data=''
    for item in contents:
        if '上传于' in item:
            up_data=(item.split('上传于')[-1]).split('&')[0]
            #print 'up load data :', up_data
    large_url=photo_url + 'large'
    contents=requests.get(url=large_url).content.split('\n')
    for item in contents:
        if 'img src=' and '/large/' in item:
            download_url=item.split('"')[5]
            img_title=item.split('"')[7]
            #print 'download url:',download_url,img_title
    imname='_'.join([download_url.split('/')[-1].rstrip('.jpg'), up_data, img_title])+'.jpg'
    #print imname
    impath=os.path.join(prefix,imname)
    with open(impath,'wb') as fp:
        fp.write(requests.get(url=download_url).content)

if __name__=='__main__':
    rootdir='my_douban_photo'
    if not os.path.exists(rootdir):
        os.mkdir(rootdir)

    homeUrl='https://www.douban.com/people/bigbigsun/photos'
    albums=getAlbums(homeUrl)
    for album in [albums[3]]:
        album_url, album_name=album
        if not os.path.exists(os.path.join(rootdir,album_name)):
            os.mkdir(os.path.join(rootdir,album_name))

        print 'album_url',album_url,album_name
        photo_urls=getPhotos(album_url)
        print 'len photo_urls:', len(photo_urls)
        for photo_url in photo_urls:
            downloadPhoto(photo_url,os.path.join(rootdir,album_name))

