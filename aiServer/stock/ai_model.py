from array import array
from typing import List, Dict, Any


class Output:
    def __init__(self, product_number, name, percentage):
        self.product_number = product_number
        self.name = name
        self.percentage = percentage

    def to_dict(self):
        return {
            'productNumber': self.product_number,
            'name': self.name,
            "percentage": self.percentage,
        }


# 요청된 주식 목록에 따라 각 주식의 구매 비율을 리턴하는 메서드
def get_stock_order_ratio(stocks) -> list[dict[str, Any]]:

    outputs = []
    output = Output("삼성전자", "000000", 0.1)
    outputs.append(output.to_dict())

    return outputs
