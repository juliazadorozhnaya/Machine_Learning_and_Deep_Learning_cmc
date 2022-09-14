import pytest


class BankCard:
    def __init__(self, total_sum):
        self.total_sum = total_sum

    def __call__(self, sum_spent):
        if self.total_sum - sum_spent <= 0:
            print('Not enough money to spent sum_spent dollars.')
            raise ValueError
        self.total_sum -= sum_spent
        print('You spent sum_spent dollars. total_sum dollars are left.')

    def put(self, sum_put):
        self.total_sum += sum_put
        print('You put sum_put dollars. total_sum dollars are left.')

    @property
    def balance(self):
        if self.total_sum == 0:
            print('Not enough money to learn the balance.')
            raise ValueError
        self.total_sum -= 1
        return self.total_sum

    def __repr__(self):
        return 'To learn the balance you should put the money on the card, ' \
            'spent some money or get the bank data. The last procedure ' \
            'is not free and costs 1 dollar.'


def test_bank_card():
    a = BankCard(100)
    assert a.total_sum == 100
    assert a.__repr__() == "To learn the balance you should put the money on the card, spent some money or get the bank data. The last procedure is not free and costs 1 dollar."
    a(50)
    assert a.total_sum == 50
    assert a.balance == 49
    try:
        a(50)
    except ValueError:
        pass
    a.put(30)
    assert a.balance == 78


test_bank_card()