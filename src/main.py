from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import AppConfig, load_config
from src.mt5_client import MT5Client, credentials_from_env
from src.strategy import generate_signal


@dataclass
class TradeDecision:
    symbol: str
    action: str
    sl: float
    tp: float


def _pip_to_price(symbol_info, pips: float) -> float:
    point = symbol_info.point
    pip_size = point * 10 if symbol_info.digits in (3, 5) else point
    return pips * pip_size


def _build_decision(
    client: MT5Client,
    symbol: str,
    action: str,
    sl_pips: float,
    tp_pips: float,
) -> TradeDecision:
    info = client.symbol_info(symbol)
    if info is None:
        raise RuntimeError(f"Symbol info missing for {symbol}")
    tick = client.symbol_info_tick(symbol)
    if tick is None:
        raise RuntimeError(f"Tick info missing for {symbol}")

    sl_offset = _pip_to_price(info, sl_pips)
    tp_offset = _pip_to_price(info, tp_pips)

    if action == "buy":
        sl = tick.bid - sl_offset
        tp = tick.bid + tp_offset
    else:
        sl = tick.ask + sl_offset
        tp = tick.ask - tp_offset

    return TradeDecision(symbol=symbol, action=action, sl=sl, tp=tp)


def _should_trade(spread: float, max_spread: float, open_positions: int, max_positions: int) -> bool:
    if spread > max_spread:
        return False
    if open_positions >= max_positions:
        return False
    return True


def run_bot(config: AppConfig) -> None:
    creds = credentials_from_env(
        config.mt5.login_env,
        config.mt5.password_env,
        config.mt5.server_env,
        config.mt5.path_env,
    )
    client = MT5Client(creds)
    client.initialize()

    try:
        while True:
            for symbol in config.trade.symbols:
                rates = client.get_rates(symbol, config.trade.timeframe, config.strategy.min_bars + 10)
                state = generate_signal(
                    rates,
                    config.strategy.ema_fast,
                    config.strategy.ema_slow,
                    config.strategy.min_bars,
                    config.strategy.allow_short,
                )

                spread = client.get_spread_pips(symbol)
                positions = client.get_open_positions(symbol=symbol, magic=config.trade.magic)

                if not _should_trade(spread, config.trade.max_spread_pips, len(positions), config.trade.max_positions):
                    print(f"[{symbol}] Risk block: spread={spread:.2f} positions={len(positions)}")
                    continue

                if state.signal == "hold":
                    print(f"[{symbol}] No signal. EMA fast={state.ema_fast:.5f}, slow={state.ema_slow:.5f}")
                    continue

                decision = _build_decision(client, symbol, state.signal, config.trade.sl_pips, config.trade.tp_pips)

                if config.trade.dry_run:
                    print(
                        f"[{symbol}] DRY RUN {decision.action.upper()} lot={config.trade.lot} "
                        f"sl={decision.sl:.5f} tp={decision.tp:.5f} spread={spread:.2f}"
                    )
                    continue

                client.place_market_order(
                    symbol=decision.symbol,
                    action=decision.action,
                    lot=config.trade.lot,
                    sl_price=decision.sl,
                    tp_price=decision.tp,
                    magic=config.trade.magic,
                )
                print(f"[{symbol}] Order sent: {decision.action}")

            time.sleep(config.trade.polling_seconds)
    finally:
        client.shutdown()


def main() -> None:
    config = load_config()
    run_bot(config)


if __name__ == "__main__":
    main()
