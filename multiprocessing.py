import requests
from bs4 import BeautifulSoup
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

start = time.time()


def scrape_single_page(url):
    response = requests.get(url)
    response.raise_for_status()
    soup = BeautifulSoup(response.content, 'html.parser')

    titles = [title.get_text() for title in soup.select('.article_tit')]
    dates = [rel_time.get_text() for rel_time in soup.select('.rel_time')]

    return titles, dates


def scrape_news_titles_and_dates(url_base: str, page_count: int):
    title_list = []
    date_list = []
    urls = [url_base if i == 1 else f'{url_base}/New/{i}' for i in range(1, page_count + 1)]

    # ThreadPoolExecutor로 병렬 크롤링
    with ThreadPoolExecutor(max_workers=10) as executor:
        future_to_url = {executor.submit(scrape_single_page, url): url for url in urls}

        for future in as_completed(future_to_url):
            try:
                titles, dates = future.result()
                title_list.extend(titles)
                date_list.extend(dates)
            except Exception as exc:
                print(f'Error fetching {future_to_url[future]}: {exc}')

    return title_list, date_list


news_url_base = 'https://www.sedaily.com/NewsList/GD05'
title_list, date_list = scrape_news_titles_and_dates(news_url_base, 100)

print("title_list: ", title_list)
print("time :", time.time() - start)
