from typing import List, Dict, Any
from datetime import datetime
import FinanceDataReader as fdr
import joblib
import numpy as np
import pandas as pd
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from .models import NewsData
import pickle
from .models import StockModelInfo

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


open_dif_data_list = []
newslabel_match_openchange = []
predicted_results = []
stock_orders = []


def get_stock_order_ratio(amount: int, stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for stock in stocks:
        stock_code = stock['productNumber']
        today_price = get_today_open_price(stock_code)
        stock['today_price'] = today_price
        

    open_dif_data_list = prepare_stock_data(stocks)
    print("open_dif_data_list" + str(open_dif_data_list))

    # 모델 만드는 코드
    for i in range(len(stocks)):
        make_model(next((item[1] for item in open_dif_data_list if item[0] == stocks[i]['name']), None), stocks[i]['name'])

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
    start = (2000, 1, 1)
    start = datetime.datetime(*start)
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
    open_dif = []

    for i in range(len(data)):
        if i == 0:
            open_dif.append(0)
        else:
            open_dif.append(data['Open'].iloc[i] - data['Open'].iloc[i - 1])
    data['Change'] = open_dif

    return data


def scrape_news_titles_and_dates():
    title_list = []
    date_list = []
    url_base = 'https://www.sedaily.com/NewsList/GD05'

    # Get existing data from the database
    existing_news = NewsData.objects.all()
    existing_titles_dates = set((news.title, news.date) for news in existing_news)

    for i in range(1, 5):
        url = url_base if i == 1 else f'{url_base}/New/{i}'
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        
        titles = soup.select('.article_tit')
        dates = soup.select('.date')

        for title in titles:
            if title not in existing_titles_dates:
                title_list.append(title.get_text())
        for date in dates:
            if date not in existing_titles_dates:
                date = datetime.strptime(date.get_text(), '%Y.%m.%d').date()
                date_list.append(date)

        for i in range(min(len(title_list), len(date_list))):
            # 기존 데이터 확인
            if not NewsData.objects.filter(title=title_list[i], date=date_list[i]).exists():
                # 데이터가 없을 때만 저장
                news_data = NewsData(title=title_list[i], date=date_list[i])
                news_data.save()
            else:
                print(f"Duplicate entry found for title: {title_list[i]}, date: {date_list[i]}. Skipping insertion.")

    return title_list, date_list

# news_url_base = 'https://www.sedaily.com/NewsList/GD05'

# title_list, date_list = scrape_news_titles_and_dates(news_url_base, 100)
# makemodel_title_list, makemodel_date_list = scrape_news_titles_and_dates(news_url_base, 200)


def predict_add_news_label(open_dif_data_list: List[tuple]) -> Dict[str, pd.DataFrame]:
    newslabel_match_openchange = {}
    for company_name, data in open_dif_data_list:
        labeled_data = add_news_label(data, company_name)
        newslabel_match_openchange[company_name] = labeled_data

    return newslabel_match_openchange


def add_news_label(data: pd.DataFrame, name: str) -> pd.DataFrame:
    today_news_title = []
    today_date_list = []

    scrape_news_titles_and_dates()

    # 데이터베이스의 데이터 중 절반만 쿼리
    half_data = NewsData.objects.all()[:NewsData.objects.count() // 2]

    # 절반 데이터를 출력하거나 처리
    for data in half_data:
        # print(data.title, data.date)
        if name in data.title:
            today_news_title.append(data.title)
            today_date_list.append(data.date)

    today_news_title_date = pd.DataFrame({'title': today_news_title, 'Date': today_date_list})
    today_news_title_date['Date'] = pd.to_datetime(today_news_title_date['Date'])

    if len(today_news_title) == 0:
        today_news_title_date['title_label'] = 0
        newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')
        return newslabel_match_openchange

    SA_lr_best = joblib.load('./static/SA_lr_best.pkl')
    tfidf = joblib.load('./static/tfidf.pkl')
    today_data_title_tfidf = tfidf.transform(today_news_title_date['title'])
    today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)
    today_news_title_date['title_label'] = today_data_title_predict

    newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')
    print(newslabel_match_openchange)
    return newslabel_match_openchange


def make_model(data: pd.DataFrame, name: str) -> LinearRegression:
    today_news_title = []
    today_date_list = []

    all_news_data = NewsData.objects.all()

    for data in all_news_data:
        if name in data.title:
            today_news_title.append(data.title)
            today_date_list.append(data.date)

    today_news_title_date = pd.DataFrame({'title': today_news_title, 'Date': today_date_list})
    today_news_title_date['Date'] = pd.to_datetime(today_news_title_date['Date'])

    if len(today_news_title) == 0:
        tomorrow_stock = joblib.load('./static/tomorrow_stock.pkl')
        return tomorrow_stock

    SA_lr_best = joblib.load('./static/SA_lr_best.pkl')
    tfidf = joblib.load('./static/tfidf.pkl')
    today_data_title_tfidf = tfidf.transform(today_news_title_date['title'])
    today_data_title_predict = SA_lr_best.predict(today_data_title_tfidf)
    today_news_title_date['title_label'] = today_data_title_predict

    newslabel_match_openchange = pd.merge(today_news_title_date, data, on='Date')
    sentiments = newslabel_match_openchange['title_label']
    weights = np.random.rand(len(newslabel_match_openchange))

    if sum(weights) == 0:
        weights[0] = weights[0] + 0.5104
    weighted_avg = calculate_weighted_average(sentiments, weights)

    label_dif = list()

    for i in range(len(newslabel_match_openchange)):
        label_dif.append([newslabel_match_openchange['title_label'][i], newslabel_match_openchange['Change'][i]])

    ylist = list()

    for i in range(len(newslabel_match_openchange)):
        ylist.append(newslabel_match_openchange['Open'][i])

    X = np.array(label_dif)
    y = np.array(ylist)

    if len(X) <= 1:
        tomorrow_stock = joblib.load('./static/tomorrow_stock.pkl')
        return tomorrow_stock

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)

    if StockModelInfo.objects.filter(stock_name=name).exists():
        print(f"Model for stock {name} already exists. Skipping save.")
        return

    # 데이터베이스에 모델 저장
    model_file_path = f'models/{name}_model.pkl'

    # Save the model to a .pkl file
    with open(model_file_path, 'wb') as f:
        pickle.dump(model, f)

    # Save the model information to the database
    StockModelInfo.objects.update_or_create(
        stock_name=name,
        defaults={'model_file_path': model_file_path}
    )


def load_stock_model(name: str):
    try:
        # Retrieve the model information from the database
        stock_model_info = StockModelInfo.objects.get(stock_name=name)
        model_file_path = stock_model_info.model_file_path

        # Load the model from the file
        with open(model_file_path, 'rb') as f:
            model = pickle.load(f)

        return model
    except StockModelInfo.DoesNotExist:
        print(f"No model found for stock: {name}")
        return None


def predict_result(newslabel_match_openchange: Dict[str, pd.DataFrame], open_dif_data_list: List[tuple], stocks) -> Dict[str, Dict[str, float]]:
    predicted_stock_openprice = {}

    stock_code = ''
    for company_name, data in newslabel_match_openchange.items():
        # open_dif_data_list에서 company_name에 맞는 데이터를 찾아서 사용
        for stock in stocks:
            if company_name == stock['name']:
                stock_code = stock['productNumber']

        raw_data = next((item[1] for item in open_dif_data_list if item[0] == company_name), None)

        if raw_data is None:
            print(f"Warning: Data for {company_name} not found in open_dif_data_list")
            continue

        try:
            model = load_stock_model(company_name)
            predicted_price = predicted_tomorrow_openprice(data, company_name, model)
            today_price = get_today_open_price(stock_code)  # 오늘의 시가 가져오기 추가함
            if predicted_price != 0:
                predicted_stock_openprice[company_name] = {
                    "predicted_price": predicted_price,
                    "today_price": today_price
                }
        except (IndexError, KeyError) as e:
            print(f"Warning: Skipping {company_name} due to error: {e}")


    return predicted_stock_openprice


def predicted_tomorrow_openprice(data: pd.DataFrame, name: str, model: LinearRegression) -> int:
    if np.isnan(data['title_label'].iloc[0]):
        return 0

    weight_seed = len(data['title_label'])
    sentiments = data['title_label']
    weights = np.random.rand(weight_seed)

    if sum(weights) == 0:
        weights[0] = weights[0] + 0.5104
    weighted_avg = calculate_weighted_average(sentiments, weights)
    input_data = np.array([weighted_avg, weights if np.isscalar(weights) else weights[0]]).reshape(1, -1)
    predicted_price = model.predict(input_data)

    return int(predicted_price[0])


def calculate_weighted_average(sentiments: np.ndarray, weights: np.ndarray) -> float:
    if len(sentiments) != len(weights):
        raise ValueError("감정 점수 리스트와 가중치 리스트의 길이가 일치해야 합니다.")

    weighted_sum = sum(s * w for s, w in zip(sentiments, weights))
    total_weight = sum(weights)

    if total_weight == 0:
        raise ValueError("총 가중치의 합이 0이 될 수 없습니다.")

    return weighted_sum / total_weight


def get_today_open_price(stock_code: str) -> float:
    today = datetime.datetime.today().strftime('%Y-%m-%d')
    df = fdr.DataReader(stock_code, today, today)

    if not df.empty:
        return df['Open'].iloc[0]
    else:
        print(f"오늘의 시가 데이터를 찾을 수 없습니다: {stock_code}")
        return None


def calculate_stock_amounts(predicted_results: Dict[str, Dict[str, float]], amount: float, stocks) -> List[Dict[str, Any]]:
    stock_orders = []
    stock_code = ''
    for company_name, prices in predicted_results.items():
        predicted_price = prices['predicted_price']
        today_price = prices['today_price']

        increase_decrease_rate = (predicted_price - today_price) / today_price * 100

        if increase_decrease_rate > 0:
            stock_order = {
                'name': company_name,
                'increase_decrease_rate': increase_decrease_rate
            }
            stock_orders.append(stock_order)

    stock_orders.sort(key=lambda x: x['increase_decrease_rate'], reverse=True)
    total_increase_decrease_rate = sum(order['increase_decrease_rate'] for order in stock_orders)

    for order in stock_orders:
        company_name = order['name']
        rate = order['increase_decrease_rate'] / total_increase_decrease_rate
        allocated_amount = amount * rate
        quantity = allocated_amount // predicted_results[company_name]['today_price']

        for stock in stocks:
            if stock['name'] == company_name:
                stock_code = stock['productNumber']

        # order['rate'] = rate * 100
        # order['stock_amount'] = allocated_amount
        order['quantity'] = int(quantity)
        order['productNumber'] = stock_code
        order['name'] = company_name

    return stock_orders


def get_stocklist_top50(predicted_results: Dict[str, Dict[str, float]], stocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    sorted_stock_orders = calculate_stock_amounts(predicted_results, 0, stocks)
    top_50_stocks = sorted_stock_orders[:50]

    return top_50_stocks