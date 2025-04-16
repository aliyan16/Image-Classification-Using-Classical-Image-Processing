import cv2 as cv
import numpy as np


def padding(img,pad):
    img[-pad:,:]=0
    img[:pad,:]=0
    img[:,-pad:]=0
    img[:,:pad]=0
    return img

def conv(img,filter):
    rows,cols=img.shape
    value=0
    for i in range(rows):
        for j in range(cols):
            value=value+(img[i,j]*filter[i,j])
    return value

def magnitude(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy
    value=0
    for i in range(rows):
        for j in range(cols):
            value=np.sqrt((img1[i,j]**2)+(img2[i,j]**2))
            resultant[i,j]=value
    resultant=norm(resultant)
    return resultant

def Phase(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy
    value=0
    for i in range(rows):
        for j in range(cols):
            value=np.arctan2(img1[i,j],img2[i,j])
            resultant[i,j]=value
    resultant=norm(resultant)
    return resultant

def norm(img):
    imgm = (img / np.max(img) * 255).astype(np.uint8)
    return imgm

def hogImplementation(cell,sobelx,sobely):
    Crows,Ccols=cell.shape

    for i in range(Crows):
        for j in range(Ccols):
            gx=conv(cell,sobelx)
            gy=conv(cell,sobely)
            SobelMag=magnitude(gx,gy)
            SobelPhase=          


def hog(img,sobelx,sobely):
    rows,cols=img.shape
    blockSize=(img[0]//4,img[1]//4)
    CellSize=(blockSize[0]//2,blockSize[1]//2)

    for i in range(rows):
        for j in range(cols):
            block=img[i:i+blockSize[0],j:j+blockSize[1]]
            for m in range(blockSize[0]):
                for n in range(blockSize[1]):
                    cell=block[m:m+CellSize[0],n:n+CellSize[1]]

            






if __name__=='__main__':
    img=cv.imread('C:\AllData\Semester6\DIP\Assignment\Assignment2\A2_wbc_data\wbc_data\Train\Basophil\Basophil_1.jpg',cv.IMREAD_GRAYSCALE)
    cv.imshow('img',img)
    cv.waitKey(2000)
    img=padding(img,2)
    sobelx=[[-1,2,-1],[0,0,0],[1,2,1]]
    sobely=[[-1,0,1],[-2,0,-2],[-1,0,1]]
    hogImg=hog(img,sobelx,sobely)
    