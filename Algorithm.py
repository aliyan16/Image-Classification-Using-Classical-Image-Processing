import cv2 as cv
import numpy as np
import os
from sklearn.metrics import mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


def padding(img,pad):
    img[-pad:,:]=0
    img[:pad,:]=0
    img[:,-pad:]=0
    img[:,:pad]=0
    return img

def conv(img,filter):
    img = img.astype(np.int16)
    rows,cols=img.shape
    value=0
    for i in range(rows):
        for j in range(cols):
            # value=value+((img[i,j]*filter[i,j]))
            value=(value)+((img[i,j])*(filter[i][j]))
    return value

# def conv(img, filter):
#     img = img.astype(np.float32)
#     filter = np.array(filter, dtype=np.float32)
#     f_rows, f_cols = filter.shape
#     i_rows, i_cols = img.shape
#     pad_x, pad_y = f_rows // 2, f_cols // 2
#     padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
#     output = np.zeros_like(img, dtype=np.float32)

#     for i in range(i_rows):
#         for j in range(i_cols):
#             region = padded_img[i:i+f_rows, j:j+f_cols]
#             output[i, j] = np.sum(region * filter)
#     return output


def magnitude(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy()
    value=0
    # for i in range(rows):
    #     for j in range(cols):
    #         value=np.sqrt((img1[i,j]**2)+(img2[i,j]**2))
    #         resultant[i,j]=value
    resultant=np.sqrt((img1**2)+(img2**2))

    resultant=norm(resultant)
    return resultant

def Phase(img1,img2):
    rows,cols=img1.shape
    resultant=img1.copy()
    value=0
    # for i in range(rows):
    #     for j in range(cols):
    #         value=np.arctan2(img1[i,j],img2[i,j])
    #         resultant[i,j]=value
    resultant = np.arctan2(img2, img1) * (180.0 / np.pi)
    resultant=norm(resultant)
    return resultant

def norm(img):
    imgm = (img / np.max(img) * 255).astype(np.uint8)
    return imgm



def sobel(img,sobelx,sobely):
    rows,cols=img.shape
    magImg=img.copy()
    phaseImg=img.copy()
    # frows,fcols=sobelx.shape

    for i in range(rows-2):
        for j in range(cols-2):
            window=img[i:i+3,j:j+3]
            value1=conv(window,sobelx)
            value2=conv(window,sobely)
            magImg[i,j]=value1
            phaseImg[i,j]=value2
    return magImg,phaseImg

def hogImplementation(magcell,phaseCell):
    # Crows,Ccols=cell.shape
    CellList = np.zeros(6, dtype=np.float64)
    # gx=conv(cell,sobelx)
    # gy=conv(cell,sobely)
    # SobelMag=magnitude(gx,gy)
    # SobelPhase= Phase(gx,gy)
    # SobelPhase[SobelPhase<0]+=180
    # SobelPhase[SobelPhase>180]-=180
    rows,cols=magcell.shape
    for i in range(rows):
        for j in range(cols):
            value=magcell[i,j]
            angle=phaseCell[i,j]
            index=angle//30
            if index>5:
                index=5
            CellList[index]+=value
            # if angle>=0 and angle <=30:
            #     CellList[0]+=value
            # elif angle>30 and angle<=60:
            #     CellList[30]+=value
            # elif angle>60 and angle<=90:
            #     CellList[60]+=value
            # elif angle>90 and angle<=120:
            #     CellList[90]+=value
            # elif angle>120 and angle<=150:
            #     CellList[120]+=value
            # elif angle>150 and angle<=180:
            #     CellList[150]+=value
    # for m in range(0,180,30):
    #     CellList[m//30]=np.sum(SobelMag[(SobelPhase>=m)&(SobelPhase<m+30)])
            
    return CellList

def hog(img,sobelx,sobely):
    rows,cols=img.shape
    # blockSize=(rows//4,cols//4)
    # CellSize=(blockSize[0]//4,blockSize[1]//4)
    blockSize=(16,16)
    CellSize=(4,4)
    BlockList=[]

    gx,gy=sobel(img,sobelx,sobely)
    SobelMag=magnitude(gx,gy)
    SobelPhase= Phase(gx,gy)
    SobelPhase[SobelPhase<0]+=180
    SobelPhase[SobelPhase>180]-=180

    for i in range(0,rows-blockSize[0]+1,blockSize[0]):
        for j in range(0,cols-blockSize[1],blockSize[1]):
            magblock=SobelMag[i:i+blockSize[0],j:j+blockSize[1]]
            phaseBlock=SobelPhase[i:i+blockSize[0],j:j+blockSize[1]]
            for m in range(0,blockSize[0],CellSize[0]):
                for n in range(0,blockSize[1],CellSize[1]):
                    magcell=magblock[m:m+CellSize[0],n:n+CellSize[1]]
                    phaseCell=phaseBlock[m:m+CellSize[0],n:n+CellSize[1]]
                    if magcell.shape[0]==CellSize[0] and magcell.shape[1]==CellSize[1]:
                        CellList=hogImplementation(magcell,phaseCell)
                        BlockList.append(CellList)
                        # print(BlockList)
    BlockList=np.array(BlockList)
    BlockList=BlockList.flatten()
    return BlockList
            

def color_hog(img, sobelx, sobely):
    # Split into R, G, B channels
    channels = cv.split(img)
    featureVector = []

    for ch in channels:
        ch_padded = padding(ch, 1)
        hog_features = hog(ch_padded, sobelx, sobely)
        featureVector.extend(hog_features)

    return np.array(featureVector)







if __name__=='__main__':
    # img=cv.imread(r'C:\AllData\Semester6\DIP\Assignment\Assignment2\A2_wbc_data\wbc_data\Train\Basophil\Basophil_1.jpg',cv.IMREAD_GRAYSCALE)
    # cv.imshow('img',img)
    # cv.waitKey(2000)
    # # print(img.shape)
    # # resized_img = cv.resize(img, (64, 64), interpolation=cv.INTER_AREA)
    # # print(resized_img.shape)
    # img=padding(img,2)
    # sobelx=[[-1,2,-1],[0,0,0],[1,2,1]]
    # sobely=[[-1,0,1],[-2,0,-2],[-1,0,1]]
    # imgList=hog(img,sobelx,sobely)
    # print(imgList)
    trainPath = r'C:\AllData\Semester6\DIP\Assignment\Assignment2\A2_wbc_data\wbc_data\Train_Cropped'
    classAverages = {}

    # Loop over all class folders (Basophil, Neutrophil, etc.)
    for className in os.listdir(trainPath):
        classPath = os.path.join(trainPath, className)
        if not os.path.isdir(classPath):
            continue

        print(f"Processing {className}")

        allFeatures = []

        for imgFile in os.listdir(classPath):
            imgPath = os.path.join(classPath, imgFile)
            img = cv.imread(imgPath)
            if img is None:
                print(f"Skipping {imgPath}, could not load.")
                continue

            img = cv.resize(img, (64, 64))
            # img = padding(img, 1)  # Padding size 1 for 3x3 filters

            sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

            featureVector = color_hog(img, sobelx, sobely)
            # featureVector = hog(img, sobelx, sobely)
            allFeatures.append(featureVector)

        if allFeatures:
            avgVector = np.mean(allFeatures, axis=0)
            classAverages[className] = avgVector
            print(f"{className} average feature vector length: {len(avgVector)}")

    print("\nDone! Average HOG features per class:")
    for cls, avgVec in classAverages.items():
        print(f"{cls}: Length = {len(avgVec)}\n {avgVec}\n\n")
    
    true_labels = []
    predicted_labels = []
    accuracy=0.0
    total=0
    testPath = r'C:\AllData\Semester6\DIP\Assignment\Assignment2\A2_wbc_data\wbc_data\Test_Cropped'
    ClassAccuracy=[]
    avClassName=[]
    print('Making Predictions')
    for className in os.listdir(testPath):
        ClassPath=os.path.join(testPath,className)
        if not os.path.isdir(ClassPath):
            continue
        print(f'Processing for {className}')

        for imgFile in os.listdir(ClassPath):
            imgPath=os.path.join(ClassPath,imgFile)
            img=cv.imread(imgPath)
            if img is None:
                print(f"Skipping {imgPath}, could not load.")
                continue
            img=cv.resize(img,(64,64))
            # paddedimg=padding(img,1)
            testHogimg=color_hog(img,sobelx,sobely)
            # testHogimg=hog(img,sobelx,sobely)
            mse_scores = {}
            for cls, avg_vector in classAverages.items():
                mse = mean_squared_error(testHogimg, avg_vector)
                mse_scores[cls] = mse

            # Get predicted class (lowest MSE)
            testHogimg = testHogimg / np.linalg.norm(testHogimg)
            avg_vector = avg_vector / np.linalg.norm(avg_vector)
            predicted_class = min(mse_scores, key=mse_scores.get)
            true_labels.append(className)
            predicted_labels.append(predicted_class)
            if className.lower()==predicted_class.lower():
                accuracy+=1
            total+=1
            if total%50==0:
                clsAv=(accuracy/total)*100
                ClassAccuracy.append(clsAv)
                avClassName.append(className)
                

            # print(f"Image: {imgFile} | True Class: {className} | Predicted: {predicted_class}")

    # Plot confusion matrix
    labels = sorted(classAverages.keys())  # Make sure order is consistent
    cm = confusion_matrix(true_labels, predicted_labels, labels=labels)
    print('Following is the Per Class Accuracy')
    for i in range(len(ClassAccuracy)):
        print('Accuracy for Class: ',avClassName[i],f' : {ClassAccuracy[i]}')
    print('Overall Accuracy is:',(accuracy/total)*100)


    outputFile='Output.txt'

    with open(outputFile, 'w') as f:
        f.write('My Output File \n\n')
        f.write('Feature Vectors \n\n')
        
        for cls, avVec in classAverages.items():
            f.write(f"{cls} feature vector length: {len(avVec)}\n")
        
        f.write('\n\nPer Class Accuracy\n\n')
        
        for i in range(len(ClassAccuracy)):
            f.write(f"{avClassName[i]} Accuracy is: {ClassAccuracy[i]:.2f}%\n")
        
        f.write('\n\nOverall Accuracy\n\n')
        f.write(f"Algorithm Accuracy is: {(accuracy/total)*100:.2f}%") 

    plt.figure(figsize=(8, 6))
    sns.heatmap(
    cm, 
    annot=True, 
    fmt='d', 
    cmap='viridis',  # Try: 'Reds', 'Greens', 'coolwarm', 'YlOrBr', 'magma'
    xticklabels=labels, 
    yticklabels=labels,
    linewidths=0.5,   # Add grid lines
    linecolor='gray'  # Grid line color
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


