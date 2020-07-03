import os 
from skimage.exposure.exposure import rescale_intensity
import cv2
from skimage.draw import polygon
import numpy as np
from PIL import Image


def get_contour(image):
  img = rescale_intensity(image,out_range=(0,255))
  img = img.astype(np.uint8)
  kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  body = cv2.erode(img,kernel,iterations=1)
  kernel_1 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
  blur = cv2.GaussianBlur(body,(5,5),0)
  ret,body = cv2.threshold(blur, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

  body = cv2.morphologyEx(body, cv2.MORPH_CLOSE, kernel_1,iterations=3)

  contours, hierarchy = cv2.findContours(body,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  area =[[i,cv2.contourArea(contours[i])] for i in range(len(contours))]
  area.sort(key=lambda x : x[1],reverse=True)
 
  contour = contours[area[0][0]]
  r = contour[:,0,1]
  c = contour[:,0,0]
  rr,cc = polygon(r,c)
  body[rr,cc] = 1

  body = cv2.medianBlur(body,5)

  return body



def crop_by_contour_single(input_path,save_path):
  img = Image.open(input_path).convert('L')
  contour = get_contour(img)
  x,y,w,h = cv2.boundingRect(contour)
  new_img = img.crop((x-5,y-5,x+w+5,y+h+5))
  new_img.save(save_path)



def crop_by_contour(input_path,save_path):
  if os.path.isdir(input_path):
    entry_iterator = os.scandir(input_path)
    for item in entry_iterator:
      if item.is_dir():
        temp_path = os.path.join(save_path,item.name)
        if not os.path.exists(temp_path):
          os.makedirs(temp_path)
        crop_by_contour(item.path,temp_path)  

      elif item.is_file() and os.path.splitext(item.name)[1] == '.png':
        temp_path = os.path.join(save_path,item.name)
        crop_by_contour_single(item.path,temp_path)
        print("%s done!" % item.path)
  
  elif os.path.isfile(input_path) and os.path.splitext(os.path.basename(input_path))[1] == '.png':
    if not os.path.exists(save_path):
      os.makedirs(save_path)
    name = os.path.basename(input_path)
    temp_path = os.path.join(save_path,name)
    crop_by_contour_single(input_path,temp_path)
    print("%s done!" % input_path)


if __name__ == "__main__":
  
  input_path = '../dataset/pre_data/'
  save_path = '../dataset/pre_crop_data/'
  crop_by_contour(input_path,save_path)