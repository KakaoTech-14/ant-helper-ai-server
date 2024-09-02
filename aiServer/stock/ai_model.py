import datetime
from typing import Any

import FinanceDataReader as fdr
import joblib
import numpy as np
import pandas as pd
import requests
import datetime
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split




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

"""
# 요청된 주식 목록에 따라 각 주식의 구매 비율을 리턴하는 메서드
def get_stock_order_ratio(stocks) -> list[dict[str, Any]]:
    open_dif_data_list = start_predict(stocks)
    newslabel_match_openchange = predict_add_news_label(open_dif_data_list)
    print(newslabel_match_openchange)

    outputs = predict_result(newslabel_match_openchange, open_dif_data_list)
    # outputs = []
    # output = Output("005930", "삼성전자", 1)
    # outputs.append(output.to_dict())
    return outputs"""

def get_stock_order_ratio(stocks) -> list[dict[str, Any]]:
    open_dif_data_list = start_predict(stocks)
    newslabel_match_openchange = predict_add_news_label(open_dif_data_list)
    
    # 예측 결과 얻기 (내일의 시가와 오늘의 시가 포함)
    predicted_results = predict_result(newslabel_match_openchange, open_dif_data_list)
    
    outputs = []
    for company_name, prices in predicted_results.items():
        output = Output(
            product_number=stocks[company_name]['productNumber'],  # 회사코드
            name=company_name,  # 회사 이름
            quantity=0  # 몇 주 구매할지 나타남
        )
        outputs.append(output.to_dict())
    


title_list = []
date_list = []

# 최신~ 30페이지까지 로드
for i in range(1, 100):

    if i == 1:
        url = 'https://www.sedaily.com/NewsList/GD05'
    else:
        url = f'https://www.sedaily.com/NewsList/GD05/New/{i}'

    # 웹 페이지 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공했는지 확인

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.content, 'html.parser')

    titles = soup.select('.article_tit')
    rel_times = soup.select('.rel_time')
    dates = soup.select('.date')

    for title in titles:
        title_list.append(title.get_text())
    for rel_time in rel_times:
        date_list.append(rel_time.get_text())

    for date in dates:
        date_list.append(date.get_text())

# 다음날 주가 예측 위한 최신 뉴스 기사들 모음
news_list = pd.DataFrame({'title': title_list, 'date': date_list})


# 감정 점수 가중 평균 계산 함수
def weighted_sentiment_average(sentiments, weights):
    """
    감정 점수의 가중 평균을 계산하는 함수.

    Parameters:
    sentiments (list of float): 각 기사의 감정 점수 리스트 (예: [1, -1, 0])
    weights (list of float): 각 기사의 가중치 리스트 (예: [0.5, 1.0, 0.8])

    Returns:
    float: 가중 평균 감정 점수
    """
    if len(sentiments) != len(weights):
        raise ValueError("감정 점수 리스트와 가중치 리스트의 길이가 일치해야 합니다.")

    weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("총 가중치의 합이 0이 될 수 없습니다.")

    return weighted_sum / total_weight


# 과거 기사들 모음. 과거 변화량과의 비교로 각 주식들 모델들의 변화량에 따른 주식 예측 모델 생성 위함
makemodel_title_list = []
makemodel_date_list = []
# 최신~ 200페이지까지 로드
for i in range(1, 200):

    if i == 1:
        url = 'https://www.sedaily.com/NewsList/GD05'
    else:
        url = f'https://www.sedaily.com/NewsList/GD05/New/{i}'

    # 웹 페이지 요청
    response = requests.get(url)
    response.raise_for_status()  # 요청이 성공했는지 확인

    # BeautifulSoup 객체 생성
    soup = BeautifulSoup(response.content, 'html.parser')

    titles = soup.select('.article_tit')
    rel_times = soup.select('.rel_time')
    dates = soup.select('.date')

    for title in titles:
        makemodel_title_list.append(title.get_text())
    for rel_time in rel_times:
        makemodel_date_list.append(rel_time.get_text())

    for date in dates:
        makemodel_date_list.append(date.get_text())


# 1번. 각 종목별 변화량 차이 만들기
def start_predict(stocks):
    start = (2000, 1, 1)  # 2020년 01년 01월
    start = datetime.datetime(*start)
    end = datetime.date.today()  # 현재

    print(stocks)

    open_dif_data_list = {}
    for stock in stocks:
        print(stock)
        print(stock['productNumber'])
        print(stock['name'])
        code = stock['productNumber']
        company_name = stock['name']
        open_dif_data_list[company_name] = make_diff(fdr.DataReader(code, start=start, end=end))

    return open_dif_data_list


# 2번. 뉴스 라벨 붙이기
def predict_add_news_label(open_dif_data_list):
    newslabel_match_openchange = {}
    for i in range(len(open_dif_data_list)):
        company_name = list(open_dif_data_list.keys())[i]
        newslabel_match_openchange[company_name] = add_news_label(list(open_dif_data_list.values())[i], company_name)

    return newslabel_match_openchange


# 3번. 시가 예측하기
def predict_result(newslabel_match_openchange, open_dif_data_list):
    predicted_stock_openprice = {}
    for i in range(len(newslabel_match_openchange)):
        company_name = list(newslabel_match_openchange.keys())[i]
        data = list(newslabel_match_openchange.values())[i]
        raw_data = list(open_dif_data_list.values())[i]

        try:
            model = make_model(raw_data, company_name)
            predicted_price = predicted_tomorrow_openprice(data, company_name, model)
            today_price = get_today_open_price(company_name)  # 오늘의 시가 가져오기 추가함
            if predicted_price != 0:
                predicted_stock_openprice[company_name] = {
                    "predicted_price": predicted_price,
                    "today_price": today_price
                }
        except (IndexError, KeyError) as e:
            print(f"Warning: Skipping {company_name} due to error: {e}")

    predict_stock_list = {}
    predict_open_list = []
    for s in predicted_stock_openprice:
        if predicted_stock_openprice[s] != 0:
            predict_stock_list[s] = predicted_stock_openprice[s]

    return predict_stock_list


def make_diff(df):
    data = df['Open'][df['Volume'] != 0]
    data = data.to_frame()
    open_dif = []

    for i in range(len(data)):
        if i == 0:
            open_dif.append(0)
        else:
            open_dif.append(data['Open'].iloc[i] - data['Open'].iloc[i - 1])

    data['Change'] = open_dif

    return data


# 4번. 최신 뉴스 감정분석 라벨 붙이기
def add_news_label(data, name):
    today_news_title = []
    today_date_list = []
    for i in range(len(title_list)):
        if name in title_list[i]:
            today_news_title.append(title_list[i])
            today_date_list.append(date_list[i])

    today_news_title_date = pd.DataFrame({'title': today_news_title, 'Date': today_date_list})
    today_news_title_date['Date'] = pd.to_datetime(today_news_title_date['Date'])  # Convert 'Date' column to datetime

    if len(today_news_title) == 0:
        today_news_title_date['title_label'] = 0
        today_news_title_date['title'] = ''
        today_news_title_date['Date'] = date_list
        today_news_title_date['Date'] = pd.to_datetime(today_news_title_date['Date'])
        newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')
        return newslabel_match_openchange

    SA_lr_best = joblib.load('./static/SA_lr_best.pkl')
    tfidf = joblib.load('./static/tfidf.pkl')
    # # 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
    today_data_title_tfidf = tfidf.transform(today_news_title_date['title'])
    # # 2) 최적 파라미터 학습모델에 적용하여 감성 분석
    today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)
    # # 3) 감성 분석 결과값을 데이터 프레임에 저장
    today_news_title_date['title_label'] = today_data_title_predict
    # print(today_data_title_tfidf.shape)

    newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')

    return newslabel_match_openchange


# 5번. 모델 만들기
def make_model(data, name):
    today_news_title = []
    today_date_list = []
    for i in range(len(makemodel_title_list)):
        if name in makemodel_title_list[i]:
            today_news_title.append(makemodel_title_list[i])
            today_date_list.append(makemodel_date_list[i])

    today_news_title_date = pd.DataFrame({'title': today_news_title, 'Date': today_date_list})
    today_news_title_date['Date'] = pd.to_datetime(today_news_title_date['Date'])  # Convert 'Date' column to datetime

    if len(today_news_title) == 0:
        tomorrow_stock = joblib.load('./static/tomorrow_stock.pkl')
        return tomorrow_stock

    SA_lr_best = joblib.load('./static/SA_lr_best.pkl')
    tfidf = joblib.load('./static/tfidf.pkl')
    # # 1) 분석할 데이터의 피처 벡터화 ---<< title >> 분석
    today_data_title_tfidf = tfidf.transform(today_news_title_date['title'])
    # # 2) 최적 파라미터 학습모델에 적용하여 감성 분석
    today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)
    # # 3) 감성 분석 결과값을 데이터 프레임에 저장
    today_news_title_date['title_label'] = today_data_title_predict

    newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')

    ###########
    sentiments = newslabel_match_openchange['title_label']  # 예시 감정 점수 리스트
    weights = np.random.rand(len(newslabel_match_openchange))  # 예시 가중치 리스트

    # Ensure weights don't sum to zero
    if sum(weights) == 0:
        weights[0] = weights[0] + 0.5104  # Add a small constant to all weights
    # weights = weights / weights.sum()  # Normalize weights
    weighted_avg = weighted_sentiment_average(sentiments, weights)

    label_dif = list()

    for i in range(len(newslabel_match_openchange)):
        label_dif.append([newslabel_match_openchange['title_label'][i], newslabel_match_openchange['Change'][i]])

    ylist = list()

    for i in range(len(newslabel_match_openchange)):
        ylist.append(newslabel_match_openchange['Open'][i])

    X = np.array(
        label_dif
    )  # 예: [weighted_avg, stock_change]
    y = np.array(ylist)  # 예: next_day_stock_price

    if len(X) <= 1:
        tomorrow_stock = joblib.load('./static/tomorrow_stock.pkl')
        return tomorrow_stock
    # 데이터셋을 학습용과 테스트용으로 나누기
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 선형 회귀 모델 초기화 및 학습
    model = LinearRegression()
    model.fit(X_train, y_train)

    return model


# 6번. 내일 시가 예측
def predicted_tomorrow_openprice(data, name, model):
    model = model

    if np.isnan(data['title_label'].iloc[0]):
        return 0

    weight_seed = len(data['title_label'])
    sentiments = data['title_label']  # 예시 감정 점수 리스트
    weights = np.random.rand(weight_seed)

    # # 가중 평균 계산
    # Ensure weights don't sum to zero
    if sum(weights) == 0:
        weights[0] = weights[0] + 0.5104  # Add a small constant to all weights
        # weights = weights / weights.sum()  # Normalize weights
    weighted_avg = weighted_sentiment_average(sentiments, weights)

    # With:
    input_data = np.array([weighted_avg, weights if np.isscalar(weights) else weights[0]]).reshape(1, -1)
    # This change checks if 'weights' is already a scalar. If not, it takes the first element of 'weights'.
    predicted_price = model.predict(input_data)

    return int(predicted_price[0])

#7 오늘 시가 받아오기
def calculate_stock_amounts(predicted_results: dict, amount: float) -> dict:
    stock_orders = []
    
    for company_name, prices in predicted_results.items():
        predicted_price = prices['predicted_price']
        today_price = prices['today_price']
        
        in_de_rate = (predicted_price - today_price) / today_price * 100
        
        # 매수/매도 판단 (증가율이 양수일 때만 매수)
        if in_de_rate > 0:
            stock_order = {}
            stock_order['company_name'] = company_name
            stock_order['in_de_rate'] = in_de_rate
            stock_orders.append(stock_order)
    
    total_in_de_rate = sum(order['in_de_rate'] for order in stock_orders)
    
    for order in stock_orders:
        company_name = order['company_name']
        rate = order['in_de_rate'] / total_in_de_rate  # 종목별 구매 비율
        allocated_amount = amount * rate  # 종목별 할당된 예산
        stock_num = allocated_amount // predicted_results[company_name]['today_price']  # 총 몇 주를 구매할 수 있는지
        
        order['rate'] = rate * 100
        order['stock_amount'] = allocated_amount
        order['stock_num'] = int(stock_num)
    
    return stock_orders