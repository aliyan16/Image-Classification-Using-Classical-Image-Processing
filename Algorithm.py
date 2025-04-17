import cv2 as cv
import numpy as np
import os


def padding(img,pad):
    img[-pad:,:]=0
    img[:pad,:]=0
    img[:,-pad:]=0
    img[:,:pad]=0
    return img

# def conv(img,filter):
#     img = img.astype(np.int16)
#     rows,cols=img.shape
#     value=0
#     for i in range(rows):
#         for j in range(cols):
#             # value=value+((img[i,j]*filter[i,j]))
#             value=(value)+((img[i,j])*(filter[i][j]))
#     return value

def conv(img, filter):
    img = img.astype(np.float32)
    filter = np.array(filter, dtype=np.float32)
    f_rows, f_cols = filter.shape
    i_rows, i_cols = img.shape
    pad_x, pad_y = f_rows // 2, f_cols // 2
    padded_img = np.pad(img, ((pad_x, pad_x), (pad_y, pad_y)), mode='constant')
    output = np.zeros_like(img, dtype=np.float32)

    for i in range(i_rows):
        for j in range(i_cols):
            region = padded_img[i:i+f_rows, j:j+f_cols]
            output[i, j] = np.sum(region * filter)
    return output


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

def hogImplementation(cell,sobelx,sobely):
    Crows,Ccols=cell.shape
    CellList=[0]*6
    gx=conv(cell,sobelx)
    gy=conv(cell,sobely)
    SobelMag=magnitude(gx,gy)
    SobelPhase= Phase(gx,gy)
    SobelPhase[SobelPhase<180]+=180
    SobelPhase[SobelPhase>180]-=180
    for m in range(0,180,30):
        CellList[m//30]=np.sum(SobelMag[(SobelPhase>=m)&(SobelPhase<m+30)])
            
    return CellList


def hog(img,sobelx,sobely):
    rows,cols=img.shape
    blockSize=(rows//2,cols//2)
    CellSize=(blockSize[0]//2,blockSize[1]//2)
    BlockList=[]
    for i in range(0,rows-blockSize[0]+1,blockSize[0]):
        for j in range(0,cols-blockSize[1],blockSize[1]):
            block=img[i:i+blockSize[0],j:j+blockSize[1]]
            for m in range(0,blockSize[0],CellSize[0]):
                for n in range(0,blockSize[1],CellSize[1]):
                    cell=block[m:m+CellSize[0],n:n+CellSize[1]]
                    if cell.shape[0]==CellSize[0] and cell.shape[1]==CellSize[1]:
                        CellList=hogImplementation(cell,sobelx,sobely)
                        BlockList.append(CellList)
                        # print(BlockList)
    BlockList=np.array(BlockList)
    BlockList=BlockList.flatten()
    return BlockList
            
            






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
    base_path = r'C:\AllData\Semester6\DIP\Assignment\Assignment2\A2_wbc_data\wbc_data\Train'
    class_averages = {}

    # Loop over all class folders (Basophil, Neutrophil, etc.)
    for class_name in os.listdir(base_path):
        class_path = os.path.join(base_path, class_name)
        if not os.path.isdir(class_path):
            continue

        print(f"Processing {class_name}...")

        all_features = []

        for img_file in os.listdir(class_path):
            img_path = os.path.join(class_path, img_file)
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                print(f"Skipping {img_path}, could not load.")
                continue

            img = cv.resize(img, (64, 64))
            img = padding(img, 1)  # Padding size 1 for 3x3 filters

            sobelx = [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]
            sobely = [[-1, -2, -1], [0, 0, 0], [1, 2, 1]]

            feature_vector = hog(img, sobelx, sobely)
            all_features.append(feature_vector)

        if all_features:
            avg_vector = np.mean(all_features, axis=0)
            class_averages[class_name] = avg_vector
            print(f"{class_name} average feature vector length: {len(avg_vector)}")

    print("\nDone! Average HOG features per class:")
    for cls, avg_vec in class_averages.items():
        print(f"{cls}: Length = {len(avg_vec)}\n {avg_vec}\n\n")
    