from django.shortcuts import render
from rest_framework.decorators import api_view


@api_view(['GET'])
def recommended_stock_list(request):
    return None

@api_view(['GET'])
def stock_evaluate(request):
    return None