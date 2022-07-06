import cv2 
import pytesseract
import numpy as np
from .OCR import find_chars

def getLyrics(img_ori):
    WIDTH = 1500
    HEIGHT = 2000
    img_ori = cv2.resize(img_ori, dsize=(WIDTH,HEIGHT))
    # height, width, channel = img_ori.shape

    gray = cv2.cvtColor(img_ori, cv2.COLOR_BGR2GRAY)
    img_thresh = cv2.adaptiveThreshold(gray, maxValue = 255.0, adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C, thresholdType=cv2.THRESH_BINARY_INV, C=9,
        blockSize=57,
    )
    contours, _ = cv2.findContours(img_thresh, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE)

    contours_dict = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        # 데이터 만들기
        contours_dict.append({
            'contour': contour,
            'x': x,
            'y': y,
            'w': w,
            'h': h,
            'cx': x + (w / 2),
            'cy': y + (h / 2),
        })

    OLINE_MIN_AREA = 25000
    OLINE_MIN_RATIO, OLINE_MAX_RATIO = 7, 25
    oline_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h'] # 넓이 계산
        ratio = d['w'] / d['h'] # 너비 - 높이 비율 계산

        # 조건에 맞는 것들만 골라서 possible_contours 에 담기
        if area > OLINE_MIN_AREA \
        and OLINE_MIN_RATIO < ratio < OLINE_MAX_RATIO:

            d['idx'] = cnt # 선별된 contour 에 idx 값 매겨놓기
            cnt += 1       # 나중에 조건에 맞는 contour 들의 idx만 따로 빼놓을 거임

            oline_contours.append(d)

    LYRIC_MIN_AREA, LYRIC_MAX_AREA = 20, 400
    LYRIC_MIN_RATIO = 0.1

    possible_contours = []
    cnt = 0
    for d in contours_dict:
        area = d['w'] * d['h'] # 넓이 계산
        ratio = d['w'] / d['h'] # 너비 - 높이 비율 계산

        flag = False
        # 조건에 맞는 것들만 골라서 possible_contours 에 담기
        if LYRIC_MIN_AREA < area < LYRIC_MAX_AREA \
        and LYRIC_MIN_RATIO < ratio:
            if len(oline_contours) < 3:
                d['idx'] = cnt # 선별된 contour 에 idx 값 매겨놓기
                cnt += 1       # 나중에 조건에 맞는 contour 들의 idx만 따로 빼놓을 거임

                possible_contours.append(d)

            else:
                for oline in oline_contours:
                     if oline['y'] + (oline['h']/10 * 9) < d['y'] < oline['y'] + oline['h'] + oline['h']/2:
                            flag = True
                            break
                if flag:
                    d['idx'] = cnt # 선별된 contour 에 idx 값 매겨놓기
                    cnt += 1       # 나중에 조건에 맞는 contour 들의 idx만 따로 빼놓을 거임

                    possible_contours.append(d)

    LYRIC_MAX_DIAG_MULTIPLYER = 300
    LYRIC_MAX_ANGLE_DIFF = 5.0
    LYRIC_MAX_AREA_DIFF = 10.0
    LYRIC_MIN_N_MATCHED = 10

    result_idx = find_chars(
        possible_contours, 
        LYRIC_MAX_DIAG_MULTIPLYER,
        LYRIC_MAX_ANGLE_DIFF,
        LYRIC_MAX_AREA_DIFF,
        LYRIC_MIN_N_MATCHED,
        1000,
        1000,
        possible_contours,
    )

    matched_result = []
    for idx_list in result_idx:
        matched_result.append(np.take(possible_contours, idx_list))

    PLATE_WIDTH_PADDING = 1.2
    PLATE_HEIGHT_PADDING = 1.7;

    MIN_PLATE_RARIO = 0
    MAX_PLATE_RATIO = 100

    plate_imgs = []
    plate_infos = []

    for i, matched_chars in enumerate(reversed(matched_result)):
        # 일단 center X 기준으로 정렬해준다
        sorted_chars = sorted(matched_chars, key=lambda x: x['cx'])
        sorted_chars_y = sorted(matched_chars, key=lambda x: x['cy'])

        # 번호판의 center 좌표를 계산해봅니다. (처음 contour ~ 마지막 countour 의 center 거리)
        plate_cx = WIDTH // 2
        plate_cy = (sorted_chars[0]['cy'] + sorted_chars[-1]['cy']) / 2

        # 번호판의 width 계산
        plate_width = int((WIDTH / 20) * 19)

        # 번호판의 height 계산
        plate_height = (sorted_chars_y[-1]['y'] + sorted_chars_y[-1]['h'] - sorted_chars_y[0]['y']) * PLATE_HEIGHT_PADDING

        # 번호판 부분을 잘라낸다
        img_cropped = cv2.getRectSubPix(
            img_thresh,
            patchSize=(int(plate_width), int(plate_height)),
            center=(int(plate_cx), int(plate_cy)),
        )

        # 가사 영역 중복 확인
        areaflag = True
        for areacheck in plate_infos:
            if areacheck['y'] <= plate_cy <= areacheck['y']+areacheck['h']:
                areaflag = False
                break
            
        if areaflag:
            plate_imgs.append(img_cropped)
            plate_infos.append({
                'x': int(plate_cx - plate_width / 2),
                'y': int(plate_cy - plate_height / 2),
                'w': int(plate_width),
                'h': int(plate_height),
            })

    plate_chars = [] 
    lyrics_senc = []
    for i, plate_img in enumerate(plate_imgs):
        # x1.6배 확대
        plate_img = cv2.resize(plate_img, dsize=(0, 0), fx=3.0, fy=3.0)

        # threshhold 이진화
        _, plate_img = cv2.threshold(
            plate_img,
            thresh=120,
            maxval=255.0,
            type=cv2.THRESH_BINARY,
        )

        # 또 한번 contour 찾기
        contours, _ = cv2.findContours(
            plate_img,
            mode=cv2.RETR_LIST,
            method=cv2.CHAIN_APPROX_SIMPLE,
        )

        plate_min_x, plate_min_y = plate_img.shape[1], plate_img.shape[0]
        plate_max_x, plate_max_y = 0, 0

        for contour in contours: # 각 contour 별로
            x, y, w, h = cv2.boundingRect(contour) # boundingRect 구하고

            area = w * h # 면적과
            ratio = w / h # 가로세로 비율 구하고
            if 0.3 < ratio < 1.4 \
            and area < 25000:
                if x < plate_min_x: plate_min_x = x
                if y < plate_min_y: plate_min_y = y
                if x + w > plate_max_x: plate_max_x = x + w
                if y + h > plate_max_y: plate_max_y = y + h

        img_result = plate_img[plate_min_y: plate_max_y, plate_min_x-50: plate_max_x+50]
        if not img_result.any():
            plate_chars.append('')
            continue
        img_result = cv2.GaussianBlur(img_result, ksize=(3, 3), sigmaX=0)
        _, img_result = cv2.threshold(
            img_result,
            thresh=0.0,
            maxval=255.0,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        # 2. 이미지에 패딩을 넣어준다
        # 검정색 패딩(여백) 을 주어 tesseract 가 좀 더 인식을 수월하게 할 수 있도록 한다.
        img_result = cv2.copyMakeBorder(
            img_result,
            top=100, bottom=100, left=100, right=100,
            borderType=cv2.BORDER_CONSTANT, 
            value=(0,0,0), # 검정색
        )

        lyrics_senc.append(img_result)
        # OCR 문자 인식
        chars = pytesseract.image_to_string(
            img_result,
            lang='kor',
            config= '--oem 2'
        )
        result_chars = ''
        for c in chars:
            if c.isalpha() or c.isdigit() or (ord('가') <= ord(c) <= ord('힣')):
                result_chars += c

        plate_chars.append(result_chars)

    return plate_chars, lyrics_senc