from dataclasses import dataclass
from enum import Enum
import numpy as np


class OptionStyle(Enum):
    AMERICAN = "American option can be exercised at any time before expiration"
    EUROPEAN = "European option can only be exercised at expiration"


class OptionRight(Enum):
    CALL = "Call option gives the holder the right to buy the underlying asset"
    PUT = "Put option gives the holder the right to sell the underlying asset"


@dataclass
class Option:
    """
    Do not supply upside and downside if you are supplying volatility. Upside and downside will be calculated from volatility.
    """

    style: OptionStyle
    right: OptionRight
    spot_price: float
    strike_price: float
    time_to_maturity: float
    risk_free_rate: float
    number_of_steps: int
    upside: float = None
    downside: float = None
    sigma: float = None

    def __post_init__(self):
        if self.upside != None and self.downside != None:
            assert self.upside > self.downside, "upside must be greater than downside"
        self.sigma != None and self.confirm_movements_with_volatility()
        self.price()

    def confirm_movements_with_volatility(self):
        if self.upside != np.exp(self.sigma * np.sqrt(self.time_step)):
            self.upside = np.exp(self.sigma * np.sqrt(self.time_step))
        if self.downside != np.exp(-self.sigma * np.sqrt(self.time_step)):
            self.downside = np.exp(-self.sigma * np.sqrt(self.time_step))

    @property
    def time_step(self):
        return self.time_to_maturity / self.number_of_steps

    @property
    def risk_neutral_probability(self):
        return (np.exp(self.risk_free_rate * self.time_step) - self.downside) / (
            self.upside - self.downside
        )

    def __get_call_price_at_step(self, row, column, underlying_price, option_price):
        if self.right.name == "CALL":
            return max(
                underlying_price[row, column] - self.strike_price,
                option_price[row, column],
            )
        if self.right.name == "PUT":
            return max(
                self.strike_price - underlying_price[row, column],
                option_price[row, column],
            )

    # @lru_cache(maxsize=5)
    def price(self):
        option_price = np.zeros([self.number_of_steps + 1, self.number_of_steps + 1])
        underlying_price = np.zeros(
            [self.number_of_steps + 1, self.number_of_steps + 1]
        )
        delta_evolution = np.zeros([self.number_of_steps, self.number_of_steps])

        for column in range(self.number_of_steps + 1):
            underlying_price[self.number_of_steps, column] = (
                self.spot_price
                * (self.upside**column)
                * (self.downside ** (self.number_of_steps - column))
            )
            if self.right.name == "CALL":
                option_price[self.number_of_steps, column] = max(
                    underlying_price[self.number_of_steps, column] - self.strike_price,
                    0,
                )
            if self.right.name == "PUT":
                option_price[self.number_of_steps, column] = max(
                    self.strike_price - underlying_price[self.number_of_steps, column],
                    0,
                )

        for row in range(self.number_of_steps - 1, -1, -1):
            for column in range(row + 1):
                option_price[row, column] = np.exp(
                    -self.risk_free_rate * self.time_step
                ) * (
                    self.risk_neutral_probability * option_price[row + 1, column + 1]
                    + (1 - self.risk_neutral_probability)
                    * option_price[row + 1, column]
                )
                underlying_price[row, column] = (
                    self.spot_price
                    * (self.upside**column)
                    * (self.downside ** (row - column))
                )
                option_price[row, column] = self.__get_call_price_at_step(
                    row, column, underlying_price, option_price
                )
                delta_evolution[row, column] = (
                    option_price[row + 1, column + 1] - option_price[row + 1, column]
                ) / (underlying_price[row, column] * (self.upside - self.downside))

        self.option_price = option_price
        self.underlying_price = underlying_price
        self.delta_evolution = delta_evolution

        return option_price[0, 0]

    def __repr__(self):
        return f"The price of this {self.style.name} {self.right.name} option is {self.price():.2f}"
