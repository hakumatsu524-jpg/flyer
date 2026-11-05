#!/usr/bin/env python3
"""flyer: a toy fly-brain memecoin trading simulation.

This is a self-contained paper-trading experiment. It never connects to an
exchange, sends orders, or handles real money.
"""

from __future__ import annotations

import argparse
import math
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import DefaultDict, Deque, Dict, List, Tuple

ACTIONS = ("HOLD", "BUY", "SELL")


@dataclass
class Coin:
    symbol: str
    price: float
    momentum: float = 0.0
    volatility: float = 0.04
    sentiment: float = 0.0

    def tick(self, rng: random.Random, market_mood: float) -> float:
        shock = rng.gauss(0, self.volatility)
        self.momentum = 0.82 * self.momentum + 0.18 * (market_mood + shock)
        self.sentiment = max(-1.0, min(1.0, 0.88 * self.sentiment + 0.12 * rng.gauss(0, 0.8)))
        change = 0.001 + 0.018 * self.momentum + 0.008 * self.sentiment + shock * 0.25
        self.price = max(0.000001, self.price * math.exp(change))
        return self.price


@dataclass
class Position:
    symbol: str
    quantity: float
    entry_price: float


class FlyBrain:
    """Small tabular reinforcement learner standing in for a fly's brain."""

    def __init__(self, rng: random.Random, learning_rate: float = 0.14, discount: float = 0.9):
        self.rng = rng
        self.learning_rate = learning_rate
        self.discount = discount
        self.exploration = 0.28
        self.q: DefaultDict[Tuple[str, str], Dict[str, float]] = defaultdict(
            lambda: {action: 0.0 for action in ACTIONS}
        )
        self.memory: Deque[Tuple[str, str, float, str]] = deque(maxlen=500)

    def state(self, coin: Coin, cash_fraction: float, held: bool) -> str:
        trend = "UP" if coin.momentum > 0.025 else "DOWN" if coin.momentum < -0.025 else "FLAT"
        mood = "GREED" if coin.sentiment > 0.25 else "FEAR" if coin.sentiment < -0.25 else "CALM"
        cash = "CASH" if cash_fraction > 0.35 else "INVESTED"
        return f"{trend}:{mood}:{cash}:{'HELD' if held else 'EMPTY'}"

    def choose(self, state: str) -> str:
        if self.rng.random() < self.exploration:
            return self.rng.choice(ACTIONS)
        values = self.q[state]
        best = max(values.values())
        return self.rng.choice([action for action, value in values.items() if value == best])

    def learn(self, state: str, action: str, reward: float, next_state: str) -> None:
        current = self.q[state][action]
        future = max(self.q[next_state].values())
        self.q[state][action] = current + self.learning_rate * (reward + self.discount * future - current)
        self.memory.append((state, action, reward, next_state))
        self.exploration = max(0.035, self.exploration * 0.9995)


class Flyer:
    def __init__(self, seed: int = 7, starting_cash: float = 1000.0):
        self.rng = random.Random(seed)
        self.brain = FlyBrain(self.rng)
        self.cash = starting_cash
        self.starting_cash = starting_cash
        self.positions: Dict[str, Position] = {}
        self.trades = 0
        self.wins = 0
        self.coins = [
            Coin("BZZ", 0.0042, volatility=0.055),
            Coin("WING", 0.012, volatility=0.045),
            Coin("NECTAR", 0.082, volatility=0.035),
            Coin("SWAT", 0.00091, volatility=0.075),
        ]

    def portfolio_value(self) -> float:
        return self.cash + sum(
            self.positions[c.symbol].quantity * c.price
            for c in self.coins
            if c.symbol in self.positions
        )

    def step(self, tick: int, verbose: bool = False) -> None:
        market_mood = 0.035 * math.sin(tick / 17) + self.rng.gauss(0, 0.025)
        for coin in self.coins:
            coin.tick(self.rng, market_mood)

        coin = self.rng.choice(self.coins)
        held = coin.symbol in self.positions
        before = self.portfolio_value()
        state = self.brain.state(coin, self.cash / max(before, 1), held)
        action = self.brain.choose(state)

        if action == "BUY" and not held and self.cash > 1:
            allocation = min(self.cash * 0.12, self.cash - 1)
            self.positions[coin.symbol] = Position(coin.symbol, allocation / coin.price, coin.price)
            self.cash -= allocation
            self.trades += 1
        elif action == "SELL" and held:
            position = self.positions.pop(coin.symbol)
            proceeds = position.quantity * coin.price
            self.cash += proceeds
            self.trades += 1
            if coin.price > position.entry_price:
                self.wins += 1

        after = self.portfolio_value()
        reward = (after - before) / max(before, 1) * 100
        next_state = self.brain.state(coin, self.cash / max(after, 1), coin.symbol in self.positions)
        self.brain.learn(state, action, reward, next_state)

        if verbose and action != "HOLD":
            print(f"tick {tick:04d} | {action:4s} {coin.symbol:6s} @ ${coin.price:.6f} | reward {reward:+.3f}")

    def run(self, ticks: int, report_every: int = 100, verbose: bool = False) -> None:
        for tick in range(1, ticks + 1):
            self.step(tick, verbose)
            if report_every and tick % report_every == 0:
                print(self.report(tick))
        print(self.report(ticks, final=True))

    def report(self, tick: int, final: bool = False) -> str:
        value = self.portfolio_value()
        change = (value / self.starting_cash - 1) * 100
        label = "FINAL" if final else "STATUS"
        return (
            f"[{label}] tick={tick} | portfolio=${value:,.2f} ({change:+.2f}%) | "
            f"cash=${self.cash:,.2f} | trades={self.trades} | wins={self.wins} | "
            f"epsilon={self.brain.exploration:.3f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the flyer fly-brain paper trading simulation.")
    parser.add_argument("--ticks", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cash", type=float, default=1000.0)
    parser.add_argument("--quiet", action="store_true", help="Only print periodic reports.")
    args = parser.parse_args()
    if args.ticks < 1 or args.cash <= 0:
        parser.error("--ticks must be positive and --cash must be greater than zero")

    print("flyer // simulated fly-brain memecoin market")
    print("paper trading only: no exchange connection, no real orders, no real money")
    Flyer(seed=args.seed, starting_cash=args.cash).run(args.ticks, verbose=not args.quiet)


if __name__ == "__main__":
    main()
