from anyio import open_file
from bs4 import BeautifulSoup
from django.http import JsonResponse
from django.shortcuts import render
from matplotlib import artist
import requests
from music.models import Input
import functions as fs
import modules

from tensorflow.keras.models import load_model
import PIL.ImageOps as ops
from PIL import Image
import numpy as np
import cv2 
from .OCR.lyrics import getLyrics

from io import BytesIO
import base64

model = load_model('model/akbo_model_200.h5')

def home(request):
    return render(request, 'home.html')

def index(request):
    return render(request, 'index.html')

def analyze_title(request):
    # original = Input.objects.get(pk = request.POST.get('img_id'))
    # image = original.img
    # akbo_image = cv2.imdecode(np.frombuffer(image.read() , np.uint8), cv2.IMREAD_UNCHANGED)
    pass


def analyze_lyrics(request):
    lyrics = None
    print("======================",request.GET.get('img_id'))
    original = Input.objects.get(pk = request.GET.get('img_id'))
    image = original.img
    akbo_image = cv2.imdecode(np.frombuffer(image.read() , np.uint8), cv2.IMREAD_UNCHANGED)
    lyrics, lyrics_imgs = getLyrics(akbo_image)
    lyrics_uri = [to_data_uri(l) for l in lyrics_imgs]

    lyrics = lyrics or '알 수 없음'
    context = {
        'lyrics': lyrics,
        'lyrics_uri' : lyrics_uri,
    }

    return JsonResponse(context)




def analyze(request):
    akbo = None; title = None; artist = None; lyrics = None

    akbo = request.FILES.get('akbo')
    buffer = akbo.read()

    original = Input(
        img=akbo,
        img_original=akbo.name,
    )

    original.save()

    # 분석 작업(이미지 -> np)
    image = original

    # npimg = np.fromstring(buffer,np.uint8)
    # akbo_image = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    akbo_image = cv2.imdecode(np.frombuffer(buffer , np.uint8), cv2.IMREAD_UNCHANGED)

    # 음표 분석
    image_1 = modules.remove_noise(akbo_image) # 1. 보표 영역 추출 및 그 외 노이즈 제거
    image_2, staves = modules.remove_staves(image_1) # 2. 오선 제거
    image_3, staves = modules.normalization(image_2, staves, 10) # 3. 악보 이미지 정규화
    image_4, objects = modules.object_detection(image_3, staves) # 4. 객체 검출 과정
    image_5, objects = modules.object_analysis(image_4, objects) # 5. 객체 분석 과정
    image_6, key, beats, pitches = modules.recognition(image_5, staves, objects) # 6. 인식 과정

    # 가사 분석
    lyrics, lyrics_imgs = getLyrics(akbo_image)

    lyrics_uri = [to_data_uri(l) for l in lyrics_imgs]


    
    title = title or '인스타그램'
    artist = artist or '딘'
    lyrics = lyrics or '알 수 없음'
    context = {
        'image': image,
        'original': original,
        'title': title,
        'artist': artist,
        'lyrics': lyrics,
        'lyrics_uri' : lyrics_uri,
    }

    print("=================",original.pk)

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

def to_data_uri(img):
    image = Image.fromarray(img)
    data = BytesIO()
    image.save(data, "png") # pick your format
    data64 = base64.b64encode(data.getvalue())
    return u'data:img/png;base64,'+data64.decode('utf-8') 





