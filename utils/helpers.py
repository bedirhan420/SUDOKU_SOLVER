import cv2
import numpy as np
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt


def initialize_prediction_model():
    try:
        model_path = r'C:\Users\bedir\Models\model_deneme.h5'
        model = load_model(model_path)
        print("Model başarıyla yüklendi.")
        return model
    except Exception as e:
        print("Model yükleme hatası:", e)

def preprocess_image(image):
    # Resmi numpy array'e dönüştür
    img = np.asarray(image)
    
    # Resmi normalize et
    img = img / 255.0
    
    # Resmi yeniden boyutlandır ve şekillendir
    img = cv2.resize(img, (28, 28))
    img = img.reshape(1, 28, 28, 1)
    
    return img

def get_prediction(boxes, model, threshold=0.7):
    result = []
    for image in boxes:
        try:
            # Görüntü ön işleme
            img = preprocess_image(image)
            # plt.imshow(img[0], cmap='gray')  # img[0] because img is likely shaped as (1, 28, 28, 1) or (1, 28, 28)
            # plt.title("Processed Image")
            # plt.show()
            # Tahmin yapma
            predictions = model.predict(img)
            classIndex = np.argmax(predictions, axis=-1)
            probabilityValue = np.max(predictions)
            
            # Tahmini sonuçları ekleme
            if probabilityValue > threshold:
                result.append(classIndex[0])
            else:
                result.append(0)

            print(probabilityValue)
        
        except Exception as e:
            print(f"Error processing image: {e}")
            result.append((0, 0))
    
    return result

def preprocess(img):
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img_blur = cv2.GaussianBlur(img_gray,(5,5),1)
    img_threshold = cv2.adaptiveThreshold(img_blur,255,1,1,11,2)
    return img_threshold

def reorder(my_points):
    my_points = my_points.reshape((4,2))
    my_points_new = np.zeros((4,1,2),dtype=np.int32)
    add = my_points.sum(1)
    my_points_new[0] = my_points[np.argmin(add)]
    my_points_new[3] = my_points[np.argmax(add)]
    diff = np.diff(my_points, axis=1)
    my_points_new[1] = my_points[np.argmin(diff)]
    my_points_new[2] = my_points[np.argmax(diff)]
    return my_points_new

def biggest_contour(contours):
    biggest = np.array([])
    max_area = 0
    for i in contours:
        area = cv2.contourArea(i)
        if area >50:
            peri = cv2.arcLength(i, True)
            approx = cv2.approxPolyDP(i, 0.02 * peri, True)
            if area > max_area and len(approx) == 4:
                biggest = approx
                max_area = area
    return biggest,max_area

def split_boxes(img):
    rows = np.vsplit(img,9)
    boxes=[]
    for r in rows:
        cols= np.hsplit(r,9)
        for box in cols:
            boxes.append(box)
    return boxes

def display_numbers(img,numbers,color = (0,255,0)):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for x in range (0,9):
        for y in range (0,9):
            if numbers[(y*9)+x] != 0 :
                 cv2.putText(img, str(numbers[(y*9)+x]),
                               (x*secW+int(secW/2)-10, int((y+0.8)*secH)), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                            2, color, 2, cv2.LINE_AA)
    return img

def draw_grid(img):
    secW = int(img.shape[1]/9)
    secH = int(img.shape[0]/9)
    for i in range (0,9):
        pt1 = (0,secH*i)
        pt2 = (img.shape[1],secH*i)
        pt3 = (secW * i, 0)
        pt4 = (secW*i,img.shape[0])
        cv2.line(img, pt1, pt2, (255, 255, 0),2)
        cv2.line(img, pt3, pt4, (255, 255, 0),2)
    return img

def stack_images(imgArray,scale):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range ( 0, rows):
            for y in range(0, cols):
                imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
            hor_con[x] = np.concatenate(imgArray[x])
        ver = np.vstack(hor)
        ver_con = np.concatenate(hor)
    else:
        for x in range(0, rows):
            imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor= np.hstack(imgArray)
        hor_con= np.concatenate(imgArray)
        ver = hor
    return ver