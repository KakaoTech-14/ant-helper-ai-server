import os

from django.apps import AppConfig


class StockConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'stock'

    def ready(self):
        if os.environ.get('RUN_MAIN', None) is not None:
            from .operator import main
            main()
