import requests as rq
from io import BytesIO
import pandas as pd
from datetime import datetime, timedelta
import FinanceDataReader as fdr
import requests
from bs4 import BeautifulSoup
import joblib

def get_krx_stock_data():
    date = (datetime.today() - timedelta(days=1)).strftime('%Y%m%d')

    gen_otp_url = 'http://data.krx.co.kr/comm/fileDn/GenerateOTP/generate.cmd'
    gen_otp_stk = {
        'mktId': 'STK',
        'trdDd': date,
        'money': '1',
        'csvxls_isNo': 'false',
        'name': 'fileDown',
        'url': 'dbms/MDC/STAT/standard/MDCSTAT03901'
    }
    headers = {
        'Referer': 'http://data.krx.co.kr/contents/MDC/MDI/mdiLoader/index.cmd?menuId=MDC0201010101',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/92.0.4515.107 Safari/537.36'
    }

    otp_stk = rq.post(gen_otp_url, gen_otp_stk, headers=headers).text

    if not otp_stk.strip():
        print("OTP 생성에 실패했습니다.")
        return None
    else:
        down_url = 'http://data.krx.co.kr/comm/fileDn/download_csv/download.cmd'
        down_sector_stk = rq.post(down_url, {'code': otp_stk}, headers=headers)

        if down_sector_stk.status_code == 200:
            try:
                sector_stk = pd.read_csv(BytesIO(down_sector_stk.content), encoding='EUC-KR')
                return sector_stk
            except pd.errors.EmptyDataError:
                print("다운로드한 데이터가 없습니다.")
                return None
        else:
            print("다운로드 요청이 실패했습니다.")
            return None


def categorize_industries(sector_stk):
    industry_mapping = {
        '기타금융': '기타 금융 서비스업',
        '증권': '전통 금융업',
        '보험': '전통 금융업',
        '은행': '전통 금융업',
        '운수장비': '운송',
        '운수창고업': '운송',
        '음식료품': '소비재',
        '섬유의복': '소비재',
        '종이목재': '소비재',
        '철강금속': '소재',
        '비금속광물': '소재',
        '기타제조업': '소재',
        '유통업': '유통업',
        '의약품': '의료',
        '의료정밀': '의료',
        '기계': '기계',
        '건설업': '건설업',
        '전기가스업': '에너지',
        '농업, 임업 및 어업': '기타',
        '기타': '기타',
        '통신업': '통신업',
        '서비스업': '서비스업',
        '전기전자': '전기전자'
    }

    sector_stk['대분류'] = sector_stk['업종명'].map(industry_mapping)
    sector_stk['대분류'] = sector_stk['대분류'].fillna('기타')

    return sector_stk[['종목코드', '종목명', '업종명', '대분류']]


def filter_stocks_by_sectors(sector_stk, selected_sectors):
    filtered_stocks = sector_stk[sector_stk['대분류'].isin(selected_sectors)]
    return filtered_stocks


def get_stock_open_price(stock_code, days_ago=1):
    # today = datetime.today().strftime('%Y-%m-%d'# )
    today = "2024-09-06"
    df = fdr.DataReader(stock_code, today, today)

    if not df.empty:
        return df['Open'].iloc[0]
    else:
        return None


def get_stock_open_diff(stock_code):
    #today = (datetime.today() - timedelta(days=1)).strftime('%Y-%m-%d')
    #yesterday = (datetime.today() - timedelta(days=2)).strftime('%Y-%m-%d')
    today = "2024-09-06"
    yesterday = "2024-09-05"
    df_today = fdr.DataReader(stock_code, today, today)
    df_yesterday = fdr.DataReader(stock_code, yesterday, yesterday)

    if not df_today.empty and not df_yesterday.empty:
        today_open = df_today['Open'].iloc[0]
        yesterday_open = df_yesterday['Open'].iloc[0]
        open_diff = today_open - yesterday_open
        return open_diff  # 시가 차이 반환
    else:
        print(f"시가 데이터를 찾을 수 없습니다: {stock_code}")
        return None


def get_today_open_prices(filtered_stocks):
    filtered_stocks = filtered_stocks.copy()

    filtered_stocks.loc[:, '오늘의 시가'] = None

    for index, row in filtered_stocks.iterrows():
        stock_code = row['종목코드']
        today_open_price = get_stock_open_price(stock_code)
        filtered_stocks.loc[index, '오늘의 시가'] = today_open_price

    return filtered_stocks


def calculate_open_diff(filtered_stocks):
    filtered_stocks.loc[:, '시가 차이'] = None

    for index, row in filtered_stocks.iterrows():
        stock_code = row['종목코드']
        open_diff = get_stock_open_diff(stock_code)

        if open_diff is not None:
            filtered_stocks.loc[index, '시가 차이'] = open_diff

    return filtered_stocks


def scrape_news_titles_and_dates(url_base: str, page_count: int):
    title_list = []
    date_list = []

    for i in range(1, page_count + 1):
        url = url_base if i == 1 else f'{url_base}/New/{i}'
        response = requests.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')

        titles = soup.select('.article_tit')
        dates = soup.select('.date')
        rel_times = soup.select('.rel_time')

        for title in titles:
            title_list.append(title.get_text())
        for rel_time in rel_times:
            date_list.append(rel_time.get_text())
        for date in dates:
            date_list.append(date.get_text())

    return title_list, date_list


news_url_base = 'https://www.sedaily.com/NewsList/GD05'

title_list, date_list = scrape_news_titles_and_dates(news_url_base, 100)
makemodel_title_list, makemodel_date_list = scrape_news_titles_and_dates(news_url_base, 200)

print(title_list)


def add_news_label(data: pd.DataFrame, name: str, title_list: list, date_list: list) -> pd.DataFrame:
    relevant_news_titles = []
    relevant_news_dates = []

    for i in range(len(title_list)):
        if name in title_list[i]:  # 주식 이름이 뉴스 제목에 포함된 경우
            relevant_news_titles.append(title_list[i])
            relevant_news_dates.append(date_list[i])

    news_data = pd.DataFrame({'title': relevant_news_titles, 'Date': relevant_news_dates})
    news_data['Date'] = pd.to_datetime(news_data['Date'])

    if len(relevant_news_titles) == 0:
        news_data['title_label'] = 0
        merged_data = pd.merge(news_data, data, on='Date', how='right')
        return merged_data

    try:
        SA_lr_best = joblib.load('static/SA_lr_best.pkl')
        tfidf = joblib.load('static/tfidf.pkl')
    except FileNotFoundError as e:
        print(f"모델 파일을 찾을 수 없습니다: {e}")
        return data  # 에러 발생 시 원본 데이터 반환

    news_title_tfidf = tfidf.transform(news_data['title'])
    news_title_predict = SA_lr_best.predict(news_title_tfidf)
    news_data['title_label'] = news_title_predict

    merged_data = pd.merge(news_data, data, on='Date', how='right')
    print(merged_data)
    return merged_data


if __name__ == "__main__":
    sector_data = get_krx_stock_data()

    if sector_data is not None:
        categorized_data = categorize_industries(sector_data)
        selected_sectors = ['전통 금융업', '전기전자', '건설업']  # 예시
        filtered_data = filter_stocks_by_sectors(categorized_data, selected_sectors)

        final_data = get_today_open_prices(filtered_data)
        print(final_data)

        final_data_with_diff = calculate_open_diff(final_data)
        print(final_data_with_diff)

        # 뉴스 라벨 추가 함수 호출
        stock_name = "삼성전자"  # 예시로 사용될 주식 이름
        labeled_data = add_news_label(final_data_with_diff, stock_name, title_list, date_list)
        print(labeled_data)


