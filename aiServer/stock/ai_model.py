import math
import pickle
from typing import List, Dict, Any
import datetime
import FinanceDataReader as fdr
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .models import NewsData, StockModelInfo  # NewsData, StockModelInfo 모델 임포트


class Output:
    def __init__(self, product_number, name, quantity):
        self.product_number = product_number
        self.name = name
        self.quantity = quantity

    def to_dict(self):
        return {
            'productNumber': self.product_number,
            'name': self.name,
            "quantity": self.quantity,
        }


def get_stock_order_ratio(amount: int, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for stock in stocks:
        stock_code = stock['productNumber']
        today_price = get_today_open_price(stock_code)
        stock['today_price'] = today_price

    open_dif_data_list = prepare_stock_data(stocks)
    print("open_dif_data_list" + str(open_dif_data_list))
    newslabel_match_openchange = predict_add_news_label(open_dif_data_list)
    print("newslabel_match_openchange=" + str(newslabel_match_openchange))

    predicted_results = predict_result(newslabel_match_openchange, open_dif_data_list, stocks)
    print("predicted_result=" + str(predicted_results))
    stock_orders = calculate_stock_amounts(predicted_results, amount, stocks)
    print("stock_orders=" + str(stock_orders))

    outputs = []
    for order in stock_orders:
        output = Output(order['productNumber'], order['name'], order['quantity'])
        outputs.append(output.to_dict())

    return outputs


def prepare_stock_data(stocks: List[Dict[str, Any]]) -> List[tuple]:
    start = datetime.datetime(2000, 1, 1)
    end = datetime.date.today()

    open_dif_data_list = []
    for stock in stocks:
        code = stock['productNumber']
        company_name = stock['name']
        stock_data = calculate_open_diff(fdr.DataReader(code, start=start, end=end))
        open_dif_data_list.append((company_name, stock_data))

    return open_dif_data_list


def calculate_open_diff(df: pd.DataFrame) -> pd.DataFrame:
    data = df['Open'][df['Volume'] != 0]
    data = data.to_frame()
    data['Change'] = data['Open'].diff().fillna(0)  # 더 간단하게 수정

    return data


def predict_add_news_label(open_dif_data_list: List[tuple]) -> Dict[str, pd.DataFrame]:
    newslabel_match_openchange = {}
    for company_name, data in open_dif_data_list:
        labeled_data = add_news_label(data, company_name)
        newslabel_match_openchange[company_name] = labeled_data

    return newslabel_match_openchange


def add_news_label(data: pd.DataFrame, name: str) -> pd.DataFrame:
    today_news_data = NewsData.objects.filter(title__contains=name)
    today_news_title_date = pd.DataFrame(list(today_news_data.values('title', 'date')))
    today_news_title_date.rename(columns={'date': 'Date'}, inplace=True)

    if today_news_title_date.empty:
        data['title_label'] = 0
        return data

    SA_lr_best = joblib.load('./static/SA_lr_best.pkl')
    tfidf = joblib.load('./static/tfidf.pkl')
    today_data_title_tfidf = tfidf.transform(today_news_title_date['title'])
    today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)
    today_news_title_date['title_label'] = today_data_title_predict

    newslabel_match_openchange = pd.merge(today_news_title_date[['Date', 'title_label']], data, on='Date', how='right')
    return newslabel_match_openchange


import os  # os 모듈 추가

def make_model(data: pd.DataFrame, name: str) -> LinearRegression:
    if data is None or len(data) <= 1:
        print(f"Insufficient data for training model for {name}")
        return joblib.load('./static/tomorrow_stock.pkl')

    X = data[['title_label', 'Change']].values
    y = data['Open'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    # 모델 저장 경로 설정
    model_directory = 'models'
    model_file_path = f'{model_directory}/{name}_model.pkl'

    # 디렉토리 존재 여부 확인 및 생성
    if not os.path.exists(model_directory):
        os.makedirs(model_directory)

    # 모델 저장
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    # StockModelInfo 데이터베이스 업데이트
    StockModelInfo.objects.update_or_create(stock_name=name, defaults={'model_file_path': model_file_path})

    return model



def predict_result(newslabel_match_openchange: Dict[str, pd.DataFrame], open_dif_data_list: List[tuple], stocks) -> Dict[str, Dict[str, float]]:
    predicted_stock_openprice = {}

    for company_name, data in newslabel_match_openchange.items():
        stock_code = next((stock['productNumber'] for stock in stocks if stock['name'] == company_name), None)
        if not stock_code:
            print(f"Stock code for {company_name} not found.")
            continue

        model = make_model(data, company_name)
        if model is not None:
            predicted_price = predicted_tomorrow_openprice(data, company_name, model)
            today_price = get_today_open_price(stock_code)
            if today_price is not None:
                predicted_stock_openprice[company_name] = {
                    "predicted_price": predicted_price,
                    "today_price": today_price
                }

    return predicted_stock_openprice


def predicted_tomorrow_openprice(data: pd.DataFrame, name: str, model: LinearRegression) -> int:
    if data['title_label'].isna().any():
        return 0

    X_last = data[['title_label', 'Change']].values[-1].reshape(1, -1)
    predicted_price = model.predict(X_last)

    return int(predicted_price[0])


def calculate_stock_amounts(predicted_results: Dict[str, Dict[str, float]], amount: float, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    stock_orders = []

    # 상승 비율 계산 및 필터링 (0 이하 제거)
    for company_name, prices in predicted_results.items():
        predicted_price = prices['predicted_price']
        today_price = prices['today_price']

        if today_price == 0:
            print(f"Skipping {company_name} due to today_price being 0.")
            continue

        # 상승률 계산
        increase_decrease_rate = (predicted_price - today_price) / today_price * 100

        # 상승률이 0 이하인 경우는 제외
        if increase_decrease_rate > 0:
            stock_order = {
                'name': company_name,
                'increase_decrease_rate': increase_decrease_rate
            }
            stock_orders.append(stock_order)

    # 상승률로 정렬 (내림차순)
    stock_orders.sort(key=lambda x: x['increase_decrease_rate'], reverse=True)
    total_increase_decrease_rate = sum(order['increase_decrease_rate'] for order in stock_orders)

    # 상승률이 양수인 주식이 없는 경우 처리
    if total_increase_decrease_rate == 0:
        print("No stocks with a positive increase rate. Check model predictions and data.")
        return []

    for order in stock_orders:
        company_name = order['name']
        rate = order['increase_decrease_rate'] / total_increase_decrease_rate
        allocated_amount = amount * rate
        today_price = predicted_results[company_name]['today_price']

        # 주식 수량 계산 (할당된 금액 기준)
        quantity = math.floor(allocated_amount / today_price)
        stock_code = next(stock['productNumber'] for stock in stocks if stock['name'] == company_name)

        order['quantity'] = int(quantity)
        order['productNumber'] = stock_code

        # 디버깅 정보 출력
        print(f"Order for {company_name}: Quantity: {order['quantity']}, Allocated Amount: {allocated_amount:.2f}, Rate: {rate:.2f}")

    return stock_orders



def get_today_open_price(stock_code: str) -> float:
    today = datetime.date.today().strftime('%Y-%m-%d')
    df = fdr.DataReader(stock_code, today, today)

    if not df.empty:
        return df['Open'].iloc[0]
    else:
        print(f"오늘의 시가 데이터를 찾을 수 없습니다: {stock_code}")
        return None


def calculate_weighted_average(sentiments: np.ndarray, weights: np.ndarray) -> float:
    if len(sentiments) != len(weights):
        raise ValueError("감정 점수 리스트와 가중치 리스트의 길이가 일치해야 합니다.")

    weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("총 가중치의 합이 0이 될 수 없습니다.")

    return weighted_sum / total_weight


def crawl_and_store_news_data(url_base: str, page_count: int) -> (List[str], List[str]):
    """
    주어진 URL 베이스와 페이지 수를 기반으로 뉴스를 크롤링하고 DB에 저장하는 함수
    """
    title_list = []
    date_list = []

    for i in range(1, page_count + 1):
        url = url_base if i == 1 else f'{url_base}/New/{i}'
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        titles = soup.select('.article_tit')
        rel_times = soup.select('.rel_time')

        for title, date in zip(titles, rel_times):
            title_text = title.get_text().strip()
            date_text = datetime.datetime.strptime(date.get_text().strip(), '%Y.%m.%d').date()

            # 중복 검사 후 DB에 저장
            if not NewsData.objects.filter(title=title_text, date=date_text).exists():
                NewsData.objects.create(title=title_text, date=date_text)
                title_list.append(title_text)
                date_list.append(date_text)

    return title_list, date_list
