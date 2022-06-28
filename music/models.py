from django.db import models

class Input(models.Model):
    img = models.FileField(upload_to='input/', default="")
    img_original = models.CharField(max_length=200, null=False)

    class Meta:
        db_table = 'Akbo_input'
        verbose_name = '게시판그림'
        verbose_name_plural = '게시판그림(들)'