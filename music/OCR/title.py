import cv2
import pytesseract
import numpy as np
import os

from music.OCR.OCR import find_chars

def getTitle(img_ori):

    akbo = cv2.resize(img_ori, dsize=(1600,2000))

    cropped = akbo[:akbo.shape[0]*2 // 10, akbo.shape[1] // 4:akbo.shape[1]*3 // 4]

    height, width, channel = cropped.shape

    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    img_blurred = cv2.GaussianBlur(gray, 
                                ksize=(3, 3), # kernel size . (0, 0) 을 지정하면 sigma 값에 의해 자동 결정
                                sigmaX=0,  # x 방향 sigma
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
                    contourIdx=-1,   # contourIdx=-1  <- 전체 contour 를 다 그리겠다는 뜻.
                    color=(255, 255, 255)
                    )

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)

    contours_dict = []   # <- contour 들의 정보를 다 저장해보겠습니다.

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        
        cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255, 255), thickness=2)
        
        # 데이타 만들기
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),  # center X
            'cy': y + (h / 2),  # center Y
        })
        
    MIN_AREA = 1000  # 최소 넓이
    # MAX_AREA = 1000
    MIN_WIDTH, MIN_HEIGHT = 2, 8  # 최소 너비, 높이
    MIN_RATIO, MAX_RATIO = 0.25, 1.0  # 너비-높이 비율의 최대/최소

    possible_contours = []  # 위 조건에 맞는 것들만 걸러내어 담아보겠습니다

    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h']   # 넓이 계산
        ratio = d['w'] / d['h']  # 너비-높이 비율 계산
        
        # 조건에 맞는 것들만 골라서 possible_contours 에 담기
        if area > MIN_AREA\
        and d['w'] > MIN_WIDTH and d['h'] > MIN_HEIGHT:
            
            d['idx'] = cnt    # 선별된 contour 에 idx 값 매겨놓기
            cnt +=1           # 나중에 조건에 맞는 contour 들의 idx 만 따로 빼낼거임.
            
            possible_contours.append(d)

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)
    temp_original = akbo.copy()
    for d in possible_contours:
        cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), color=(255, 255, 255), thickness=2)
        cv2.rectangle(temp_original, pt1=(d['x']+akbo.shape[1] // 4, d['y']), pt2=(d['x']+d['w']+akbo.shape[1] // 4, d['y']+d['h']), color=(0, 0, 255), thickness=2)

        
    MAX_DIAG_MULTIPLYER = 100
    MAX_ANGLE_DIFF = 1000.0
    MAX_AREA_DIFF = 1000.0
    MAX_WIDTH_DIFF = 1000.0
    MAX_HEIGHT_DIFF = 1000.0
    MIN_N_MATCHED = 2

    result_idx = find_chars(possible_contours)

    # 위 결과를 시각화
    import random
    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    temp_result = np.zeros((height, width, channel), dtype=np.uint8)


    for r in matched_result:
        color = (random.randint(100, 200),random.randint(100, 200), random.randint(100, 200))
        for d in r:
            cv2.rectangle(temp_result, pt1=(d['x'], d['y']), pt2=(d['x']+d['w'], d['y']+d['h']), 
                        color=color, thickness=2)


    PLATE_WIDTH_PADDING = 1.3
    PLATE_HEIGHT_PADDING = 1.8

    # 번호판 전체 가로세로비 MIN, MAX
    MIN_PLATE_RATIO = 1
    MAX_PLATE_RATIO = 10


    plate_imgs = []
    plate_infos = []

    # print(len(matched_result))

    for i, matched_chars in enumerate(matched_result):
        # 일단 center X 기준으로 정렬해준다 (직전까지는 순서가 뒤죽박죽이었을테니..)
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        sorted_chars_y = sorted(matched_chars, key=lambda x: x['cy'])
        
        # 번호판의 center 좌표를 계산해봅니다. (처음 contour ~ 마지막 countour 의 center 거리)
        plate_cx = (sorted_chars[0]['cx'] + sorted_chars[-1]['cx']) / 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2
        
        # 번호판의 width 계산
        plate_width = (sorted_chars[-1]['x'] + sorted_chars[-1]['w'] - sorted_chars[0]['x']) * PLATE_WIDTH_PADDING
        
        # 번호판의 height 계산
        plate_height = (sorted_chars_y[-1]['y'] + sorted_chars_y[-1]['h'] - sorted_chars_y[0]['y']) * PLATE_HEIGHT_PADDING
    #     sum_height = 0
    #     for d in sorted_chars:
    #         sum_height += d['h']
            
    #     plate_height = int(sum_height / len(sorted_chars) * PLATE_HEIGHT_PADDING)
        
        # 번호판이 비뚤어져 있을것이다
        # 회전 각도를 구해보자.
    #     triangle_height = sorted_chars[-1]['cy'] - sorted_chars[0]['cy']   # 높이
    #     triangle_hypotenus = np.linalg.norm( # 빗변
    #         np.array([sorted_chars[0]['cx'], sorted_chars[0]['cy']]) - 
    #         np.array([sorted_chars[-1]['cx'], sorted_chars[-1]['cy']])
    #     )
    #     # 높이와 빗변을 사용해서 arcsin() 함수로 각도 계산
    #     angle = np.degrees(np.arcsin(triangle_height / triangle_hypotenus))
        
    #     # 위 각도가 나오면 회전하는 transformation matrix구하기
    #     rotation_matrix = cv2.getRotationMatrix2D(center=(plate_cx, plate_cy), angle=angle, scale=1.0)
        
    #     # 위 matrix 를 이미지에 적용
    #     img_rotated = cv2.warpAffine(img_blur_thresh, M=rotation_matrix, dsize=(width, height))
        
        # 번호판 부분을 잘라낸다
        img_cropped = cv2.getRectSubPix(
            img_blur_thresh,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy)),
        )
        
        # 잘라낸 번호판의 가로세로 비율이 조건에 맞는지 확인
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
        

    # 번호판 문자열 인식후 이를 담기 위한 변수들
    longest_idx, longest_text = -1, 0
    plate_chars = []   # <- 여기에 번호판(들)의 문자를 담을거다

    cnt = 1
    SIZE = 15
    for i, plate_img in enumerate(plate_imgs):
        # x1.6배 확대
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=SIZE, fy=SIZE)
        
        # threshold 이진화
        _, plate_img = cv2.threshold(
            plate_img,
            thresh=0.0,
            maxval=255.0,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        
        # 또 한번 contour 찾기
        contours, _ = cv2.findContours(
            plate_img, 
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )
        
        # 여기서 번호판이 맞는지 확인해보고 글자도 추출할거다
        # ↓ (앞서 했던것과 거의 동일)
        
        # 잠시후, 이미지 안에서 추출할 '번호판 부분' 의 좌표값을 일단 초기화
        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0
        
        temp_result = plate_img.copy()
        
        for contour in contours:  # 각 contour  별로
            x, y, w, h = cv2.boundingRect(contour)  # boundingRect 구하고
        
            area = w * h  # 면적과
            ratio = w / h # 가로세로비율 구하고
            
            # 앞서 정해둔 설정기준에 맞는지 체크해서
    #         if area > MIN_AREA \
    #         and w > MIN_WIDTH and h > MIN_HEIGHT \
    #         and MIN_RATIO < ratio < MAX_RATIO:
            # '번호판 부분' 좌표의 최대, 최소값 구하기
            if x < plate_min_x: plate_min_x = x
            if y < plate_min_y: plate_min_y = y
            if x + w > plate_max_x: plate_max_x = x + w
            if y + h > plate_max_y: plate_max_y = y + h
                
            cv2.rectangle(temp_result, pt1=(x, y), pt2=(x+w, y+h), color=(255, 255,255), thickness=2)
                    
        # 위 에서 결정된 '번호판 부분' 좌표를 사용하여 '번호판 부분' 만 잘라내기 (crop)
        img_result = plate_img[plate_min_y:plate_max_y, plate_min_x:plate_max_x]
        
        cnt += 1
        
        # ↓ OCR 글자 인식하기 위해(검출률 향상을 위해) 약간의 처리를 추가합니다
        
        # 1. 노이즈 제거 (blur + threshold) 하겠습니다
        img_result = cv2.GaussianBlur(img_result, ksize=(21, 21), sigmaX=0)
        _, img_result = cv2.threshold(
            img_result,
            thresh=0.0, 
            maxval=255.0,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )
        
        # 2. 이미지에 패딩을 넣어준다
        # 검정색 패딩(여백) 을 주어 tesseract 가 좀더 인식을 수월하게 할수 있도록 한다
        border = 40 * int(SIZE)
        img_result = cv2.copyMakeBorder(
            img_result,
            top=border, bottom=border, left=border, right=border,  # 상하좌우 두께 10
            borderType=cv2.BORDER_CONSTANT, 
            value=(0,0,0),  # 검정색
        )
        
        # 드디어! OCR 문자 인식!
        chars = pytesseract.image_to_string(
            img_result,
            lang='kor',
            config='--psm 6 --oem 2'
        )
        
        # ■ 이렇게 tesseract 로 읽어낸 문자열 안에 이상한 글자들도 있다. 이들을 걸러내주어야 한다
        # 어짜피 '자동차 번호판' 은 글자 + 숫자로 이루어져 있다.
        # - 반드시 숫자는 있어야 한다.
        result_chars = ''
        for c in chars:
            result_chars += c
                
        print(chars)
        plate_chars.append(result_chars)
        print(result_chars)
        
        # 가장 문자열이 긴 번호판을 우리가 찾는 번호판이라 하자.
    #     print(i)
        if len(result_chars) > longest_text:
            longest_idx = i
            
        # 시각화
        cnt+=1

    return result_chars



