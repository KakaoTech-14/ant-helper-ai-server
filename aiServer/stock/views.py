from django.http import JsonResponse
from rest_framework.decorators import api_view

from stock.ai_model import get_stock_order_ratio


@api_view(['GET'])
def recommended_stock_list(request):
    return None


@api_view(['POST'])
def stock_evaluate(request) -> JsonResponse:
    stocks = request.data['stocks']
    amount = request.data['amount']  # 사용 가능 금액

    response = get_stock_order_ratio(stocks)

    print(response)

    return JsonResponse({"stocks": response}, safe=True)
