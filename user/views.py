from django.http import JsonResponse
from django.shortcuts import redirect, render
from django.contrib.auth.hashers import make_password, check_password
from user.forms import JoinForm

from user.models import Akbo, User
from music.models import Input

def login(request):
    if request.method == 'GET':
        return render(request, 'login.html')
    elif request.method == 'POST':
        userid = request.POST.get('userid', None)
        password = request.POST.get('password', None)

        res_data = {}

        if not(userid and password):  # 값이 다 입력되었는지 확인
            res_data['error'] = '모든 값을 입력해야 합니다'
        else:
            # 모델로부터 데이터를 가져와야 한다
            user = User.objects.get(userid=userid)
            # 비밀번호 비교
            if check_password(password, user.password):
                # 로그인 처리 (세션 사용)
                request.session['user'] = {'id': user.id, 'userid': user.userid}
                return redirect('/')   # 로그인 성공후 home 으로 redirect
            else:
                # 비밀번호 불일치.  로그인 실패 처리
                res_data['error'] = '비밀번호를 틀렸습니다'

        return render(request, 'login.html', res_data)

def logout(request):
    if request.session.get('user'):
        del(request.session['user'])
    return redirect('/')

def join(request):
    # 회원가입 처리
    if request.method=="POST":
        form = JoinForm(request.POST)
        if form.is_valid():
            user = User(
                userid = form.userid,
                password = make_password(form.password),
            )
            user.save()
        else: 
            print("join 실패")
        return redirect("/user/login/")
    else:
        form = JoinForm()
        return render(request,'join.html',{'form':form})

def checkid(request):
    userid = request.GET.get('userid')
    context={}
    try:
        User.objects.get(userid=userid)
    except:
        context['data'] = "not exist" # 아이디 중복 없음

    return JsonResponse(context)

def myakbo(request):
    user = User.objects.get(id=request.session.get('user').get('id'))
    akbos = Akbo.objects.filter(user=user)
    return render(request, 'myakbo.html', {'akbos': akbos})

def save(request):
    title = request.GET.get('title')
    artist = request.GET.get('artist')
    lyrics = request.GET.get('lyrics')
    image = request.GET.get('image')
    print('저장하기 ----')
    print('제목:', title)
    print('아티스트:', artist)
    print('가사:', lyrics)
    print('이미지:', image)
    akbo = Akbo(
        user=User.objects.get(id=request.session.get('user').get('id')),
        title=title,
        artist=artist,
        lyrics=lyrics,
        image=Input.objects.get(id=image).img,
    )
    akbo.save()
    return JsonResponse({'data': 'success'})

def akboinfo(request, pk):
    akbo = Akbo.objects.get(id=pk)
    print('내 악보정보-----')
    print('akbo')
    return render(request, 'akboinfo.html', {'akbo': akbo})