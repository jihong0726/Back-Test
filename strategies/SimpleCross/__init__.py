from jesse.strategies import Strategy
import jesse.indicators as ta

class SimpleCross(Strategy):
    @property
    def short_ema(self):
        return ta.ema(self.candles, 20)

    @property
    def long_ema(self):
        return ta.ema(self.candles, 50)

    def should_long(self) -> bool:
        return self.short_ema > self.long_ema

    def should_short(self) -> bool:
        return self.short_ema < self.long_ema

    def should_cancel_entry(self) -> bool:
        return False

    def go_long(self):
        qty = self.capital / self.price
        self.buy = qty, self.price

    def go_short(self):
        qty = self.capital / self.price
        self.sell = qty, self.price
