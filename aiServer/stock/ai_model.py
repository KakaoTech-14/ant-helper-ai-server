from typing import Any


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


# 요청된 주식 목록에 따라 각 주식의 구매 비율을 리턴하는 메서드
def get_stock_order_ratio(stocks) -> list[dict[str, Any]]:
    print(stocks)

    outputs = []
    output = Output("005930", "삼성전자", 1)
    outputs.append(output.to_dict())

    return outputs
