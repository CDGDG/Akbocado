from bs4 import BeautifulSoup
import cv2
from django.shortcuts import render
import numpy as np
import requests

from music.models import Input

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
    title = '인스타그램'
    artist = '딘'

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
