from django.db import models
from matplotlib import artist
from matplotlib.pyplot import title

import user

class User(models.Model):
    userid = models.CharField(max_length=20, verbose_name='아이디')
    password = models.CharField(max_length=500, verbose_name='비밀번호')

    class Meta:
        db_table = 'Akbo_user'
        verbose_name = '유저'
        verbose_name_plural = '유저(들)'

    def __str__(self):
        return f'{self.userid} 유저'

class Akbo(models.Model):
    user = models.ForeignKey('user.User', on_delete=models.CASCADE)
    title = models.CharField(max_length=30, verbose_name='제목')
    artist = models.CharField(max_length=20, verbose_name='아티스트', null=True)
    image = models.FileField(upload_to='akbo_image/', null=False)

    class Meta:
        db_table = 'Akbo_akbo'
        verbose_name = '악보'
        verbose_name_plural = '악보(들)'

        def __str__(self):
            return f'{self.id} - {self.title} 악보'