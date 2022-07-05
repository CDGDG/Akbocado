from anyio import open_file
from bs4 import BeautifulSoup
from django.http import JsonResponse
from django.shortcuts import render
from matplotlib import artist
import requests
from music.models import Input

from tensorflow.keras.models import load_model
import PIL.ImageOps as ops
from PIL import Image
import numpy as np
import cv2 

model = load_model('model/akbo_model_200.h5')

def home(request):
    return render(request, 'home.html')

def index(request):
    return render(request, 'index.html')

def analyze(request):

    akbo = None; title = None; artist = None; lyrics = None

    akbo = request.FILES.get('akbo')

    original = Input(
        img=akbo,
        img_original=akbo.name,
    )
    original.save()

    # 분석 작업
    image = original

    title = title or '인스타그램'
    artist = artist or '딘'
    lyrics = lyrics or '알 수 없음'
    context = {
        'image': image,
        'original': original,
        'title': title,
        'artist': artist,
        'lyrics': lyrics,
    }

    return render(request, 'result.html', context)

def search(request):
    type = request.GET.get('type')
    item = request.GET.get('item')
    artist = request.GET.get('artist')

    print(f'입력 정보: {type}, {item}, {artist}')
    context = {}
    try:
        if type=='artist':
            melon_data = getMelonArtist(item)
            print(melon_data)
            context['image'] = melon_data['image']
            context['artist'] = melon_data['artist']
            context['tracks'] = melon_data['tracks']
        elif type=='track':
            melon_data = getMelonInfo(item, artist)
            context['album'] = melon_data['album']
            context['release'] = melon_data['release']
            context['image'] = melon_data['image']
            context['track'] = melon_data['track']
            context['artist'] = melon_data['artist']
            context['artists'] = melon_data['artist']
            context['lyrics'] = melon_data['lyrics']
    except IndexError as ie:
        context['error'] = 'IndexError'

    return JsonResponse(context)

def getMelonInfo(track, artist):
    query = f'{track} {artist}'

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
            'track': soup.select_one('div.info > div.song_name').text.replace('곡명', '').strip(),
            'artist': soup.select_one('div.info > div.artist > a.artist_name > span').text.strip(),
            'lyrics': BeautifulSoup(str(soup.select_one('#d_video_summary')).replace('<br/>', '\n'), 'html.parser').text.split('\n'),
            }
    except Exception as e:
        return ('곡 정보 불러오기 실패', e)
    return data

def getMelonArtist(artist):
    q_url = 'https://www.melon.com/search/artist/index.htm?q='+artist
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/100.0.4896.127 Safari/537.36'
    }
    data = {}
    try:
        artistid = BeautifulSoup(requests.get(q_url, headers=headers).text, 'html.parser').select_one('#pageList .d_artist_list ul li .wrap_atist12 button.btn_join_fan')['data-artist-no']
        url = f"https://www.melon.com/artist/song.htm?artistId={artistid}#params%5BorderBy%5D=POPULAR_SONG_LIST&params%5B"
        soup = BeautifulSoup(requests.get(url, headers=headers).text, 'html.parser')
        data['image'] = soup.select_one('#artistImgArea img').get('src')
        data['artist'] = soup.select_one('.title_atist').text.replace('아티스트명', '')
        data['tracks'] = [{
            'title': tr.select_one('.ellipsis a.fc_gray').text,
            'artist': tr.select_one('#artistName a.fc_mgray').text,
            'album': tr.select_one('.ellipsis:not(#artistName) a.fc_mgray').text,
        }
        for tr in soup.select('#frm div.tb_list.d_song_list.songTypeOne table tbody tr')]
    except Exception as e:
        return ('아티스트 정보 불러오기 실패', e)
    return data



def checkAkboImage(request):
    akbo = "악보"
    akboImage = request.FILES.get('file')
    print(type(akboImage))
    img = akboImage.read()
    image = cv2.imdecode(np.frombuffer(img , np.uint8), cv2.IMREAD_UNCHANGED)
    # print("==========",img,"=============")
    print(predict(image)) 
    if predict(image) == 1:
        print("=========악보 아님==============")
        akbo = "일반"
    return JsonResponse({'data': akbo})




def predict(file):
    class_names = [0,1]
    # img = cv2.imread(file)
    image = Image.fromarray(file)
    mono8img = image.convert('L')
    invImg = ops.invert(mono8img)
    resizeImg = invImg.resize((200, 200))
    list(resizeImg.getdata())[:10]
    dataImg = list(map(lambda n: int(n)/255, list(resizeImg.getdata())))
    data_arr = np.array(dataImg).reshape(1, 200, 200, 1)
    preds = model.predict(data_arr)

    return class_names[np.argmax(preds[0])]
    





def rgb(frame):
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def find_chars(contour_list, MAX_DIAG_MULTIPLYER, MAX_ANGLE_DIFF, MAX_AREA_DIFF,MIN_N_MATCHED, MAX_WIDTH_DIFF, MAX_HEIGHT_DIFF, possible_contours):
    
    matched_result_idx = [] # possible_contours 에서 조건에 맞는 것들의 idx 값들을 담아서 리턴할 거시다.
    
    # contour_list 에서 2개의 contour (d1, d2) 의 '모든 조합'을 비교
    for d1 in contour_list:
        
        matched_contours_idx = []
        
        for d2 in contour_list:
            if d1['idx'] == d2['idx']: continue # d1, d2 가 같은 contour 이면 비교할 필요 없다.
                
            # 두 contour 사이의 거리계산
            # d1 ~ d2 의 center 좌표 거리
            dx = abs(d1['cx'] - d2['cx'])
            dy = abs(d1['cy'] - d2['cy'])
            distance = np.linalg.norm(np.array([d1['cx'], d1['cy']]) - np.array([d2['cx'], d2['cy']]))
            
            # d1의 대각선 길이
            diagonal_length1 = np.sqrt(d1['w'] ** 2 + d1['h'] ** 2)
            
            # 각도 구하기 (아래 그림 참조)
            if dx == 0:
                angle_diff = 90
            else:
                angle_diff = np.degrees(np.arctan(dy / dx))
                
            # 면적 비율, 폭 비율, 높이 비율
            area_diff = abs(d1['w'] * d1['h'] - d2['w'] * d2['h']) / (d1['w'] * d1['h'])
            width_diff = abs(d1['w'] - d2['w']) / d1['w']
            height_diff = abs(d1['h'] - d2['h']) / d1['h']
            
            # 설정에 따른 기준에 맞는 contour 들만 골라서 
            # matched_contours_idx 에 contour 의 index 추가
            # 즉! d1 과 같은 번호판 후보군, d2(들)을 추가
            if distance < diagonal_length1 * MAX_DIAG_MULTIPLYER \
            and angle_diff < MAX_ANGLE_DIFF and area_diff < MAX_AREA_DIFF \
            and width_diff < MAX_WIDTH_DIFF and height_diff < MAX_HEIGHT_DIFF:
                matched_contours_idx.append(d2['idx'])
        
        # 마지막에 d1도 넣어준다
        matched_contours_idx.append(d1['idx'])
        
        # matched_contours_idx 에 MIN_N_MATCHED 보다 개수가 적으면 번호판으로 인정안함
        if len(matched_contours_idx) < MIN_N_MATCHED:
            continue
        
        # 위 조건들을 다 통과하면 최종 후보군 matched_result_idx 에 인덱스 추가
        matched_result_idx.append(matched_contours_idx)
        
        # 그런데, 여기서 끝내지 않고
        # 최종 후보군에 들어있지 않은 것들을 한번 더 비교해 봅니다
        
        # 일단 matched_contours_idx 에 들지 않은 것들을 아래의 unmatched_contours_idx 에 넣고
        unmatched_contours_idx = []
        for d4 in contour_list:
            if d4['idx'] not in matched_contours_idx:
                unmatched_contours_idx.append(d4['idx'])
                
        # np.take(a, idx)
        # a에서 idx 와 같은 인덱스의 값만 추출
        
        unmatched_contour =  np.take(possible_contours, unmatched_contours_idx)
        
        # 재귀호출
        recursive_contour_list = find_chars(unmatched_contour)
        
        for idx in recursive_contour_list:
            matched_result_idx.append(idx)
        break
            
    return matched_result_idx


def getLyrics(img_ori):
    WIDTH = 1500
    HEIGHT = 2000
    
    img_ori = cv2.resize(img_ori, dsize=(WIDTH,HEIGHT))
    
    height, width, channel = img_ori.shape
