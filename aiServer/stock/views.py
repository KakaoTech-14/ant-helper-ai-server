from django.http import JsonResponse
from rest_framework.decorators import api_view
from stock.ai_model import Output, get_stock_order_ratio, crawl_and_store_news_data


@api_view(['GET'])
def recommended_stock_list(request):
    return None


@api_view(['POST'])
def stock_evaluate(request) -> JsonResponse:
    stocks = request.data['stocks']
    amount = request.data['amount']  # 사용 가능 금액
    print(request.data)

    response = get_stock_order_ratio(amount, stocks)

    print(response)

    return JsonResponse({"stocks": response}, safe=True)


@api_view(['GET'])
def crawl_news(request) -> JsonResponse:
    """
    뉴스 크롤링을 수행하고 결과를 DB에 저장하는 API 엔드포인트
    """
    news_url_base = 'https://www.sedaily.com/NewsList/GD05'
    page_count = 200  # 크롤링할 페이지 수
    title_list, date_list = crawl_and_store_news_data(news_url_base, page_count)

    return JsonResponse({"message": "News data crawled and stored successfully.", "titles": title_list}, safe=True)
