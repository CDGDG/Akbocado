from io import BytesIO
from bs4 import BeautifulSoup
import cv2
from django.shortcuts import render
import numpy as np
import requests
from PIL import Image
from music.models import Input
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/keys/akbocado-5c075ac55e91.json"
from google.cloud import vision

client = vision.ImageAnnotatorClient()

def analyze(request):
    title = None; artist = None

    image = request.FILES.get('akbo')
    buffer = image.read()

    melon_image = cv2.imdecode(np.frombuffer(buffer , np.uint8), cv2.IMREAD_UNCHANGED)

    original = Input(
        img=image,
        img_original=image.name,
    )

    original.save()

    # 분석
    titleinfo, artistinfo = getMelon(melon_image)
    title = titleinfo[0] or '분석 실패'
    artist = artistinfo[0] or '분석 실패'

    print(f'멜론: {artist} - {title}')

    # 멜론 정보
    info = getMelonInfo(title, artist)

    context = {
        'image': original,
        'title': title,
        'artist': artist,
        'info': info,
    }
    return render(request, 'mresult.html', context)

def getMelonInfo(track, artist):
    track2 = track.replace('953964', '&amp;')
    query = f'{track2} {artist}'

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    q_url = 'https://www.melon.com/search/song/index.htm?q='+query
    try:
        trackid = BeautifulSoup(requests.get(q_url, headers=headers).text, 'html.parser').select_one('table tbody tr td div.wrap.pd_none.left input')['value']
        url = 'https://www.melon.com/song/detail.htm?songId='+trackid
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        data = {
            'album': soup.select_one('dl.list > dd').text.strip(),
            'release': soup.select('dl.list > dd')[1].text.strip(),
            'image': soup.select_one('div.wrap_info > div.thumb > a > img')['src'].strip(),
            'name': soup.select_one('div.info > div.song_name').text.replace('곡명', '').strip(),
            'artist': soup.select_one('div.info > div.artist > a.artist_name > span').text.strip(),
            'lyrics': BeautifulSoup(str(soup.select_one('#d_video_summary')).replace('<br/>', '\n'), 'html.parser').text.split('\n'),
            'id': trackid,
            }
    except Exception as e:
        return ('곡 정보 불러오기 실패', e)
    return data

def getMelon(img_ori):
    # 이미지 정제
    total_image = [] # title, artist 부분 slice한 이미지 담기
    player_chars = [] # 분석 결과 문자 담기
    player_images = [] # 분석 이미지 담기

    img_ori = cv2.resize(img_ori,(690,1450))

    total_image.append(img_ori[130:200,70:610]) # artist
    total_image.append(img_ori[60:200,70:610]) # title

    for i,slice_image in enumerate(total_image):
        gray = cv2.cvtColor(slice_image,cv2.COLOR_BGR2GRAY)

        if i == 1:
            # 제목만 걸러내기
            _,gray = cv2.threshold(
                    gray,
                    thresh=180,
                    maxval = 255.0,
                    type=cv2.THRESH_BINARY,
                )


        # blur and threshold
        img_blurred = cv2.GaussianBlur(gray,
                                    ksize=(5,5), # kernel size. (0,0)을 지정하면 sigma 값에 의해 자동 결정
                                    sigmaX = 0, # X 방향 sigma 
                                    )

        img_blur_thresh = cv2.adaptiveThreshold(
                img_blurred,
                maxValue = 255.0,
                adaptiveMethod = cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                thresholdType=cv2.THRESH_BINARY_INV,
                blockSize=19,
                C=2,
        )

        # 윤곽선 검출
        contours,_ = cv2.findContours(
            img_blur_thresh,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE,
        )

        # bounding rect
        contours_dict = []    # <- contour 들의 정보를 다 저장

        for contour in contours:
            x, y , w, h = cv2.boundingRect(contour)

            # 데이터 만들기
            contours_dict.append({
                'contour':contour,
                'x':x,
                'y':y,
                'w':w,
                'h':h,
                'cx':x + (w/2), #center X
                'cy':y + (h/2), # center Y
            })

        possible_contours = [] # 위 조건에 맞는 것들만 걸러냄

        cnt = 0

        for d in contours_dict:
            d['idx'] = cnt #선별된 contour에 idx 값
            cnt += 1       #나중에 조건에 맞는 contour 들의 idx만
            possible_contours.append(d)   

        # x1.6배 확대
        title_image = cv2.resize(gray,dsize=(0,0),fx=1.6,fy=1.6)

        # threshold 이진화
        _,title_image = cv2.threshold(
            title_image,
            thresh=0.0,
            maxval=255.0,
            type = cv2.THRESH_BINARY | cv2.THRESH_OTSU,

        )

        # 또 한번 contour 찾기
        contours, _ = cv2.findContours(
            title_image,
            mode = cv2.RETR_LIST,
            method = cv2.CHAIN_APPROX_SIMPLE,

        )

        # 이미지 안에서 추출할 '이름' 부분의 좌표값을 일단 초기화
        title_min_x, title_min_y = title_image.shape[1],title_image.shape[0]
        title_max_x, title_max_y = 0,0

        for contour in contours: # 각 contour 별로
            x, y ,w, h = cv2.boundingRect(contour) # boundingRect 구하고

            # '번호판 부분' 좌표의 최대, 최소값 구하기
            if x <title_min_x: title_min_x = x
            if y <title_min_y: title_min_y = y
            if x + w > title_max_x : title_max_x = x + w
            if y + h > title_max_y : title_max_y = y + h

        # 위에서 결정된 '이름' 부분만 잘라내기
        img_result = title_image[title_min_y:title_max_y,title_min_x:title_max_x]

        # 1. 노이즈 제거 (blur + threshold)
        img_result = cv2.GaussianBlur(img_result,ksize=(3,3),sigmaX = 0)
        _, img_result = cv2.threshold(
            img_result,
            thresh=0.0,
            maxval = 255.0,
            type=cv2.THRESH_BINARY | cv2.THRESH_OTSU,
        )

        # 2. 이미지에 패딩을 넣어준다
        # 검정색 패딩(여백)
        img_result = cv2.copyMakeBorder(
            img_result,
            top=50,bottom=50,left=50,right=50, # 상하좌우 두께 50
            borderType=cv2.BORDER_CONSTANT, 
            value=(0,0,0), # 검정색
        )

        # OCR 문자 인식
        # chars = pytesseract.image_to_string(
        #     img_result,
        #     lang='kor',
        #     config='--psm 7 --oem 2' 

        # )

        # google vision api
        pillow = Image.fromarray(img_result)
        data = BytesIO()
        pillow.save(data, "png")
        content = data.getvalue()
        image = vision.Image(content=content)
        response = client.text_detection(image=image)
        chars = response.text_annotations

        player_chars.append(chars[0].description)
        player_images.append(img_result)

    artist = (player_chars[0], player_images[0])
    title = (player_chars[1], player_images[0])

    return title, artist