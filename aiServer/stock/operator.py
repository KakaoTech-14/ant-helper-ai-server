import logging

from apscheduler.schedulers.background import BackgroundScheduler

from stock.ai_model import crawl_and_store_news_data

logger = logging.getLogger('django')


def run_apis():
    logger.info("operator.run_apis()")
    news_url_base = 'https://www.sedaily.com/NewsList/GD05'
    page_count = 5  # 크롤링할 페이지 수
    crawl_and_store_news_data(news_url_base, page_count)


def main():
    sche = BackgroundScheduler(timezone='Asia/Seoul')

    print("gkdl")
    sche.add_job(run_apis, 'cron', day='*', hour='0', minute='0', id='run_apis')
    sche.start()


if __name__ == "__main__":
    main()
