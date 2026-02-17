from __future__ import annotations

import time
from dataclasses import dataclass

import pandas as pd

from src.config import AppConfig, TradeSymbolConfig, load_config
from src.ai_filter import AITradeFilter, build_ai_config
from src.mt5_client import MT5Client, credentials_from_env
from src.strategy import generate_signal, compute_ema, compute_rsi, compute_atr
from src.logger import log_section, setup_logger
from src.trade_tracker import TradeTracker


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


def _resolve_symbol_trade(config: AppConfig, symbol: str) -> TradeSymbolConfig:
    override = config.trade.per_symbol.get(symbol)
    if override is None:
        raise KeyError(f"Missing trade config for symbol {symbol}")
    return override


def _evaluate_exit_risk(
    rates,
    action: str,
    sl_price: float,
    tp_price: float,
    ema_fast: int,
    ema_slow: int,
    rsi_period: int,
    atr_period: int,
) -> tuple[float, str, float, float, float, float]:
    close = rates["close"]
    ema_fast_series = compute_ema(close, ema_fast)
    ema_slow_series = compute_ema(close, ema_slow)
    curr_fast = float(ema_fast_series.iloc[-1])
    curr_slow = float(ema_slow_series.iloc[-1])
    rsi_series = compute_rsi(close, rsi_period)
    curr_rsi = float(rsi_series.iloc[-1]) if not pd.isna(rsi_series.iloc[-1]) else 50.0
    atr_series = compute_atr(rates, atr_period)
    curr_atr = float(atr_series.iloc[-1]) if not pd.isna(atr_series.iloc[-1]) else 0.0

    last_open = float(rates.iloc[-1]["open"])
    last_close = float(rates.iloc[-1]["close"])
    price = last_close

    risk = 0.0
    reasons = []

    if action == "buy" and curr_fast < curr_slow:
        risk += 0.35
        reasons.append("ema_cross_down")
    if action == "sell" and curr_fast > curr_slow:
        risk += 0.35
        reasons.append("ema_cross_up")

    if action == "buy" and curr_rsi < 50:
        risk += 0.2
        reasons.append("rsi_weak")
    if action == "sell" and curr_rsi > 50:
        risk += 0.2
        reasons.append("rsi_weak")

    if action == "buy" and last_close < last_open:
        risk += 0.15
        reasons.append("red_candle")
    if action == "sell" and last_close > last_open:
        risk += 0.15
        reasons.append("green_candle")

    if action == "buy":
        dist_sl = price - sl_price
        dist_tp = tp_price - price
    else:
        dist_sl = sl_price - price
        dist_tp = price - tp_price
    if dist_sl > 0 and dist_tp > 0 and dist_sl < dist_tp:
        risk += 0.2
        reasons.append("closer_to_sl")

    return min(risk, 1.0), ",".join(reasons) if reasons else "none", curr_fast, curr_slow, curr_rsi, curr_atr


def _manage_open_positions(client: MT5Client, tracker: TradeTracker, logger, config: AppConfig) -> None:
    """Check for closed positions and apply profit-take filter."""
    open_tracked_trades = tracker.get_open_trades()
    tracked_by_ticket = {t.ticket: t for t in open_tracked_trades if t.ticket is not None}

    mt5_positions = []
    try:
        mt5_positions = client.get_open_positions(symbol=None, magic=config.trade.magic)
    except Exception:
        mt5_positions = []

    for position in mt5_positions:
        ticket = getattr(position, "ticket", None)
        if ticket is None or ticket in tracked_by_ticket:
            continue
        action = "buy" if getattr(position, "type", None) == 0 else "sell"
        entry_price = float(getattr(position, "price_open", 0.0) or 0.0)
        sl_price = float(getattr(position, "sl", 0.0) or 0.0)
        tp_price = float(getattr(position, "tp", 0.0) or 0.0)
        lot = float(getattr(position, "volume", 0.0) or 0.0)
        tracker.record_trade(
            symbol=position.symbol,
            action=action,
            lot=lot,
            entry_price=entry_price,
            sl_price=sl_price,
            tp_price=tp_price,
            magic=config.trade.magic,
            spread_pips=None,
            ai_probability=None,
            ticket=ticket,
        )
        tracked_by_ticket[ticket] = tracker.get_open_trades()[-1]

    logger.info(
        "profit_filter scan tracked=%d mt5=%d",
        len(tracked_by_ticket),
        len(mt5_positions),
    )

    for trade in list(tracked_by_ticket.values()):
        if trade.ticket is None:
            continue

        position = client.get_position_by_ticket(trade.ticket)

        if position is None:
            logger.info(f"[{trade.symbol}] Position {trade.ticket} closed, updating records...")
            deals = client.get_deals_history()
            closing_deal = None

            for deal in deals:
                if hasattr(deal, "position_id") and deal.position_id == trade.ticket:
                    if hasattr(deal, "entry") and deal.entry == 1:  # Entry type 1 = OUT (exit)
                        closing_deal = deal
                        break

            if closing_deal:
                exit_price = closing_deal.price
                profit = closing_deal.profit

                tracker.update_trade_exit(
                    ticket=trade.ticket,
                    exit_price=exit_price,
                    profit=profit,
                    status="closed",
                )

                logger.info(
                    f"[{trade.symbol}] Trade closed: "
                    f"Entry={trade.entry_price:.5f} Exit={exit_price:.5f} "
                    f"Profit={profit:.2f}"
                )
            else:
                logger.warning(f"[{trade.symbol}] Position closed but no deal history found")
                tracker.update_trade_exit(
                    ticket=trade.ticket,
                    exit_price=trade.entry_price,
                    profit=0.0,
                    status="closed",
                )
            continue

        if not config.profit_take.enabled:
            continue

        profit = float(getattr(position, "profit", 0.0) or 0.0)
        take_profit_usd = config.profit_take.take_profit_usd_per_symbol.get(trade.symbol)
        if take_profit_usd is not None:
            logger.info(
                "[%s] profit_cap check profit=%.2f target=%.2f",
                trade.symbol,
                profit,
                take_profit_usd,
            )
        if take_profit_usd is not None and profit >= take_profit_usd:
            logger.info(
                "[%s] profit_cap action=close profit=%.2f target=%.2f",
                trade.symbol,
                profit,
                take_profit_usd,
            )
            client.close_position(trade.ticket)
            continue
        if profit < config.profit_take.trigger_usd:
            continue

        if trade.sl_price <= 0 or trade.tp_price <= 0:
            logger.info(
                "[%s] profit_filter skip=missing_sl_tp profit=%.2f",
                trade.symbol,
                profit,
            )
            continue

        rates = client.get_rates(trade.symbol, config.trade.timeframe, config.strategy.min_bars + 10)
        risk_score, reasons, ema_fast, ema_slow, rsi, atr = _evaluate_exit_risk(
            rates,
            trade.action,
            trade.sl_price,
            trade.tp_price,
            config.strategy.ema_fast,
            config.strategy.ema_slow,
            config.strategy.rsi_period,
            config.strategy.atr_period,
        )
        logger.info(
            "[%s] profit_filter profit=%.2f risk=%.2f reasons=%s ema_fast=%.5f ema_slow=%.5f rsi=%.1f atr=%.5f",
            trade.symbol,
            profit,
            risk_score,
            reasons,
            ema_fast,
            ema_slow,
            rsi,
            atr,
        )

        if risk_score >= config.profit_take.risk_threshold:
            logger.info(
                "[%s] profit_filter action=close risk=%.2f threshold=%.2f",
                trade.symbol,
                risk_score,
                config.profit_take.risk_threshold,
            )
            client.close_position(trade.ticket)


def run_bot(config: AppConfig) -> None:
    # Setup logging and tracking
    logger = setup_logger("forex_bot")
    tracker = TradeTracker()
    
    log_section(logger, "BOT START")
    logger.info("symbols=%s timeframe=%s dry_run=%s ai=%s", config.trade.symbols, config.trade.timeframe, config.trade.dry_run, config.ai.enabled)
    if config.profit_take.enabled and config.profit_take.take_profit_usd_per_symbol:
        logger.info("profit_cap targets=%s", config.profit_take.take_profit_usd_per_symbol)
    
    creds = credentials_from_env(
        config.mt5.login_env,
        config.mt5.password_env,
        config.mt5.server_env,
        config.mt5.path_env,
    )
    client = MT5Client(creds)
    client.initialize()
    log_section(logger, "MT5 CONNECTION")
    logger.info("connected")

    ai_filter = None
    if config.ai.enabled:
        log_section(logger, "AI FILTER")
        logger.info("init")
        ai_config = build_ai_config(
            enabled=config.ai.enabled,
            model_path=config.ai.model_path,
            train_data_path=config.ai.train_data_path,
            probability_threshold=config.ai.probability_threshold,
        )
        ai_filter = AITradeFilter(ai_config)
        logger.info("ready threshold=%.2f", config.ai.probability_threshold)

    try:
        log_section(logger, "TRADING LOOP")
        logger.info("started")
        last_trade_cycle = 0.0
        last_profit_check = 0.0
        loop_sleep = min(
            config.trade.polling_seconds,
            config.profit_take.check_interval_seconds if config.profit_take.enabled else config.trade.polling_seconds,
        )
        while True:
            now = time.time()
            if config.profit_take.enabled and now - last_profit_check >= config.profit_take.check_interval_seconds:
                _manage_open_positions(client, tracker, logger, config)
                last_profit_check = now

            if now - last_trade_cycle >= config.trade.polling_seconds:
                last_trade_cycle = now
                for symbol in config.trade.symbols:
                    try:
                        symbol_trade = _resolve_symbol_trade(config, symbol)
                        rates = client.get_rates(symbol, config.trade.timeframe, config.strategy.min_bars + 10)
                        state = generate_signal(
                            rates,
                            config.strategy.ema_fast,
                            config.strategy.ema_slow,
                            config.strategy.min_bars,
                            config.strategy.allow_short,
                            config.strategy.entry_delay_bars,
                            config.strategy.use_rsi,
                            config.strategy.rsi_period,
                            config.strategy.rsi_overbought,
                            config.strategy.rsi_oversold,
                            config.strategy.use_atr,
                            config.strategy.atr_period,
                            config.strategy.atr_min_thresholds.get(
                                symbol,
                                config.strategy.atr_min_threshold,
                            ),
                        )

                        spread = client.get_spread_pips(symbol)
                        positions = client.get_open_positions(symbol=symbol, magic=config.trade.magic)

                        if not _should_trade(spread, symbol_trade.max_spread_pips, len(positions), config.trade.max_positions):
                            logger.info(
                                "[%s] block=risk spread=%.2f positions=%d",
                                symbol,
                                spread,
                                len(positions),
                            )
                            continue

                        if state.signal == "hold":
                            logger.info(
                                "[%s] signal=hold ema_fast=%.5f ema_slow=%.5f rsi=%.1f atr=%.5f reason=%s",
                                symbol,
                                state.ema_fast,
                                state.ema_slow,
                                state.rsi,
                                state.atr,
                                state.reason,
                            )
                            continue

                        ai_probability = None
                        if ai_filter:
                            allowed, probability = ai_filter.evaluate(rates)
                            ai_probability = probability
                            logger.info(
                                "[%s] signal=%s rsi=%.1f atr=%.5f ai_prob=%.2f ai_allowed=%s",
                                symbol,
                                state.signal.upper(),
                                state.rsi,
                                state.atr,
                                probability,
                                allowed,
                            )
                            if not allowed:
                                continue
                        else:
                            logger.info(
                                "[%s] signal=%s rsi=%.1f atr=%.5f reason=%s",
                                symbol,
                                state.signal.upper(),
                                state.rsi,
                                state.atr,
                                state.reason,
                            )

                        info = client.symbol_info(symbol)
                        sl_pips = symbol_trade.sl_pips
                        tp_pips = symbol_trade.tp_pips
                        if info is not None and symbol_trade.sl_mode == "atr" and state.atr > 0:
                            pip_size = client._pip_size(info)
                            if pip_size > 0:
                                sl_pips = (state.atr * symbol_trade.atr_sl_multiplier) / pip_size
                                tp_pips = sl_pips * symbol_trade.rr_ratio

                        decision = _build_decision(client, symbol, state.signal, sl_pips, tp_pips)
                        info = client.symbol_info(symbol)
                        if info is not None:
                            pip_size = client._pip_size(info)
                            contract_size = float(getattr(info, "trade_contract_size", 0.0) or 0.0)
                            pip_value = contract_size * pip_size * symbol_trade.lot if contract_size > 0 else 0.0
                            logger.info(
                                "[%s] risk_params lot=%.4f digits=%s point=%.10f pip_size=%.10f pip_value=%.4f sl_mode=%s atr=%.5f sl_pips=%.2f tp_pips=%.2f",
                                symbol,
                                symbol_trade.lot,
                                getattr(info, "digits", None),
                                float(getattr(info, "point", 0.0) or 0.0),
                                pip_size,
                                pip_value,
                                symbol_trade.sl_mode,
                                state.atr,
                                sl_pips,
                                tp_pips,
                            )
                        tick = client.symbol_info_tick(symbol)
                        entry_price = tick.ask if decision.action == "buy" else tick.bid

                        if config.trade.dry_run:
                            logger.info(
                                "[%s] dry_run action=%s lot=%.2f entry=%.5f sl=%.5f tp=%.5f spread=%.2f",
                                symbol,
                                decision.action.upper(),
                                symbol_trade.lot,
                                entry_price,
                                decision.sl,
                                decision.tp,
                                spread,
                            )
                            # In dry run, use timestamp as fake ticket
                            fake_ticket = int(time.time() * 1000) % 1000000
                            tracker.record_trade(
                                symbol=symbol,
                                action=decision.action,
                                lot=symbol_trade.lot,
                                entry_price=entry_price,
                                sl_price=decision.sl,
                                tp_price=decision.tp,
                                magic=config.trade.magic,
                                spread_pips=spread,
                                ai_probability=ai_probability,
                                ticket=fake_ticket,
                            )
                            continue

                        logger.info("[%s] order_submit action=%s", symbol, decision.action.upper())
                        ticket = client.place_market_order(
                            symbol=decision.symbol,
                            action=decision.action,
                            lot=symbol_trade.lot,
                            sl_price=decision.sl,
                            tp_price=decision.tp,
                            magic=config.trade.magic,
                        )
                        logger.info(
                            "[%s] order_filled action=%s entry=%.5f ticket=%s",
                            symbol,
                            decision.action.upper(),
                            entry_price,
                            ticket,
                        )
                        tracker.record_trade(
                            symbol=symbol,
                            action=decision.action,
                            lot=symbol_trade.lot,
                            entry_price=entry_price,
                            sl_price=decision.sl,
                            tp_price=decision.tp,
                            magic=config.trade.magic,
                            spread_pips=spread,
                            ai_probability=ai_probability,
                            ticket=ticket,
                        )

                    except Exception as e:
                        logger.error(f"[{symbol}] Error processing symbol: {e}", exc_info=True)

            time.sleep(loop_sleep)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down MT5 connection")
        # Daily MT5 summary using magic number
        try:
            from datetime import datetime, time as dt_time

            now = datetime.now()
            day_start = datetime.combine(now.date(), dt_time.min)
            day_end = now
            deals = client.get_deals_history(from_date=day_start, to_date=day_end)
            bot_deals = [
                d
                for d in deals
                if getattr(d, "magic", None) == config.trade.magic
                and getattr(d, "entry", None) == 1
            ]

            total_trades = len(bot_deals)
            total_profit = sum(float(getattr(d, "profit", 0.0) or 0.0) for d in bot_deals if float(getattr(d, "profit", 0.0) or 0.0) > 0)
            total_loss = sum(float(getattr(d, "profit", 0.0) or 0.0) for d in bot_deals if float(getattr(d, "profit", 0.0) or 0.0) < 0)
            wins = [d for d in bot_deals if float(getattr(d, "profit", 0.0) or 0.0) > 0]
            win_rate = (len(wins) / total_trades) if total_trades > 0 else 0.0

            logger.info("=" * 50)
            logger.info("DAILY MT5 SUMMARY")
            logger.info("Date: %s", now.date().isoformat())
            logger.info("Total trades: %d", total_trades)
            logger.info("Winning trades: %d", len(wins))
            logger.info("Losing trades: %d", total_trades - len(wins))
            logger.info("Win rate: %.1f%%", win_rate * 100)
            logger.info("Total profit: %.2f", total_profit)
            logger.info("Total loss: %.2f", total_loss)
            logger.info("Net profit: %.2f", total_profit + total_loss)
            logger.info("=" * 50)
        except Exception as e:
            logger.error("Daily summary failed: %s", e, exc_info=True)
        finally:
            client.shutdown()


def main() -> None:
    config = load_config()
    run_bot(config)


if __name__ == "__main__":
    main()
