from django.http import JsonResponse
from rest_framework.decorators import api_view

from stock.ai_model import Output, get_stock_order_ratio


@api_view(['GET'])
def recommended_stock_list(request):
    return None


@api_view(['POST'])
def stock_evaluate(request) -> JsonResponse:
    stocks = request.data['stocks']
    amount = request.data['amount']  # 사용 가능 금액
    print(request.data)

    response = get_stock_order_ratio(amount, stocks)

    # output1 = Output(
    #     product_number="005930",  # 회사코드
    #     name="삼성전자",  # 회사 이름
    #     quantity=1  # 몇 주 구매할지 나타남
    # )
    #
    # output2 = Output(
    #     product_number="066570",  # 회사코드
    #     name="LG전자",  # 회사 이름
    #     quantity=1  # 몇 주 구매할지 나타남
    # )
    #
    # response = []
    # response.append(output1.to_dict())
    # response.append(output2.to_dict())
    #
    # print(response)

    return JsonResponse({"stocks": response}, safe=True)
