# Generated by Django 3.2.5 on 2022-07-15 00:02

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('user', '0002_akbo_lyrics'),
    ]

    operations = [
        migrations.AlterField(
            model_name='akbo',
            name='artist',
            field=models.CharField(max_length=50, null=True, verbose_name='아티스트'),
        ),
    ]
