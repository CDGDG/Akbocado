import os
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import random
import os

from music.OCR.OCR import find_chars, rgb

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/keys/akbocado-5c075ac55e91.json"
from google.cloud import vision

from io import BytesIO
import base64


def getArtist(img_ori):
    client = vision.ImageAnnotatorClient()

    # base_path = r"C:\DevRoot\dataset\avokado"
    # img = cv2.imread(os.path.join(base_path, 'ann.png'))
    # img

    cropped = img_ori[:img_ori.shape[1] // 4 , :]

    height, width, channel = cropped.shape
    # img.shape


    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    img_thresh = cv2.adaptiveThreshold(
        gray,  
        maxValue = 255.0,   
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
        thresholdType=cv2.THRESH_BINARY_INV,   
        blockSize=19,  
        C=9,
    )


    img_blurred = cv2.GaussianBlur(gray, 
                                ksize=(3, 3), 
                                sigmaX=0,  
                                )

    img_blur_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue = 255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,  
            blockSize=39,
            C=9,
    )

    img_thresh = cv2.adaptiveThreshold(
        gray,  
        maxValue = 255.0,   
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  
        thresholdType=cv2.THRESH_BINARY_INV,   
        blockSize=19,  
        C=9,
    )

    # blur and threshold
    img_blurred = cv2.GaussianBlur(gray, 
                                ksize=(3, 3), 
                                sigmaX=0, 
                                )

    img_blur_thresh = cv2.adaptiveThreshold(
            img_blurred,
            maxValue = 255.0,
            adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            thresholdType=cv2.THRESH_BINARY_INV,  
            blockSize=39,
            C=9,
    )

    contours, _ = cv2.findContours(
        img_blur_thresh,
        mode=cv2.RETR_LIST,
        method=cv2.CHAIN_APPROX_SIMPLE,
    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    cv2.drawContours(temp_result,
                    contours=contours,
                    contourIdx=-1,   
                    color=(255, 255, 255)
                    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []   

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
    
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),  
            'cy': y + (h / 2),  
        })
        
    MIN_AREA = 100  
    MIN_WIDTH, MIN_HEIGHT = 2, 8  
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  

    possible_contours = []  

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']   
        ratio = d['w'] / d['h']  
        
        
        if area > MIN_AREA\
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT:
            
            d['idx'] = cnt    
            cnt +=1           
            
            possible_contours.append(d)
            
    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    temp_original = img_ori.copy()

    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        cv2.rectangle(temp_original, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(0, 0, 255), thickness=2)

    MAX_DIAG_MULTIPLYER = 10
    MAX_ANGLE_DIFF = 1000.0
    MAX_AREA_DIFF = 1000.0
    MAX_WIDTH_DIFF = 1000.0
    MAX_HEIGHT_DIFF = 1000.0
    MIN_N_MATCHED = 5    
        
    result_idx = find_chars(possible_contours, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF,MIN_N_MATCHED, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, possible_contours)    
        
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.8


    MIN_PLATE_RATIO = 1
    MAX_PLATE_RATIO = 1.3


    plate_imgs = []
    plate_infos = []



    for i, matched_chars in enumerate(matched_result):
        
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        sorted_chars_y = sorted(matched_chars, key=lambda x: x['cy'])
        
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        

        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        

        plate_height = (sorted_chars_y[-1]['y'] + sorted_chars_y[-1]['h'] - sorted_chars_y[0]['y']) * PLATE_HEIGHT_PADDING

        img_cropped = cv2.getRectSubPix(
            img_blur_thresh,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy)),
        )
        
        if img_cropped.shape[1] / img_cropped.shape[0] < MIN_PLATE_RATIO \
        or img_cropped.shape[1] / img_cropped.shape[0] > MAX_PLATE_RATIO:
            continue
            
        plate_imgs.append(img_cropped)
        plate_infos.append({
            'x': int(plate_cx - plate_width / 2),
            'y': int(plate_cy - plate_height / 2),
            'w': int(plate_width),
            'h': int(plate_height),
        })    

    longest_idx, longest_text = -1, 0
    plate_chars = []   
    imglist = []

    cnt = 1
    SIZE = 8.0
    for i, plate_img in enumerate(plate_imgs):
        
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=SIZE, fy=SIZE)
        
        _, plate_img = cv2.threshold(
            plate_img,
            thresh=0.0,
            maxval=255.0,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        
        contours, _ = cv2.findContours(
            plate_img, 
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0
        
        temp_result = plate_img.copy()
        
        for contour in contours:  
            x, y, w, h = cv2.boundingRect(contour)  
        
            area = w * h  
            
            if x < plate_min_x: plate_min_x = x
            if y < plate_min_y: plate_min_y = y
            if x + w > plate_max_x: plate_max_x = x + w
            if y + h > plate_max_y: plate_max_y = y + h
                
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        
    img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
    _, img_result = cv2.threshold(
        img_result,
        thresh=0.0,
        maxval=255.0,
        type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    )
        

    border = 20 * int(SIZE)
    img_result = cv2.copyMakeBorder(
        img_result,
        top=border, bottom=border, left=border, right=border, 
        borderType=cv2.BORDER_CONSTANT, 
        value=(0,0,0),  
    )
    imglist.append(img_result)

    # chars = pytesseract.image_to_string(
    #     img_result,
    #     lang='kor',
    # )
    # result_chars = ''
    # for c in chars:
    #     result_chars += c

    # print(chars)
    # plate_chars.append(result_chars)
        
    charlist = []
    for img in imglist:
        pillow = Image.fromarray(img)
        data = BytesIO()
        pillow.save(data, "png")
        content = data.getvalue()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        texts = response.text_annotations
        charlist.append(texts[0].description)
        
    return charlist[0].split('\n'), imglist