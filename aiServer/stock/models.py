from django.db import models


class NewsData(models.Model):
    title = models.CharField(max_length=255)
    date = models.DateField()

    class Meta:
        unique_together = ('title', 'date')


class StockModelInfo(models.Model):
    stock_name = models.CharField(max_length=100, unique=True)
    model_file_path = models.CharField(max_length=255, default='')  # 기본값 추가
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.stock_name
