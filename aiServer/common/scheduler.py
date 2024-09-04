import logging

from apscheduler.schedulers.background import BackgroundScheduler

logger = logging.getLogger('django')


def run_apis():
    logger.info("operator.run_apis()")
    # 이 부분에 돌아가게 할 메서드 넣으면 됩니다.

def main():
    sche = BackgroundScheduler(timezone='Asia/Seoul')

    sche.add_job(run_apis, 'cron', day_of_week='mon-fri', hour='0', minute='0', id='run_apis')
    sche.start()


if __name__ == "__main__":
    main()
