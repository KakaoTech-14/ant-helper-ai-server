
from django.urls import path

from stock import views

urlpatterns = [
    path('stocks/recommended-stock', views.recommended_stock_list),
    path('stocks/evaluation', views.stock_evaluate),
    path('stocks/crawling', views.crawl_news)
]
