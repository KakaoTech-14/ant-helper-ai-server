
from django.urls import path

from stock import views

urlpatterns = [
    path('/stocks/recommended-stock', views.recommended_stock_list),
    path('/stocks/evaluating', views.stock_evaluate),
]
