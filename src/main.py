from __future__ import annotations

import time
from dataclasses import dataclass

from src.config import AppConfig, load_config
from src.ai_filter import AITradeFilter, build_ai_config
from src.mt5_client import MT5Client, credentials_from_env
from src.strategy import generate_signal
from src.logger import log_section, setup_logger, get_logger
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


def _monitor_positions(client: MT5Client, tracker: TradeTracker, logger) -> None:
    """Check for closed positions and update trade records."""
    open_tracked_trades = tracker.get_open_trades()
    
    for trade in open_tracked_trades:
        if trade.ticket is None:
            continue
        
        # Check if position still exists in MT5
        position = client.get_position_by_ticket(trade.ticket)
        
        if position is None:
            # Position is closed - try to get the closing details from history
            logger.info(f"[{trade.symbol}] Position {trade.ticket} closed, updating records...")
            
            # Get recent deals to find closing price
            deals = client.get_deals_history()
            closing_deal = None
            
            for deal in deals:
                if hasattr(deal, 'position_id') and deal.position_id == trade.ticket:
                    if hasattr(deal, 'entry') and deal.entry == 1:  # Entry type 1 = OUT (exit)
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
                # Fallback: mark as closed but without exit data
                logger.warning(f"[{trade.symbol}] Position closed but no deal history found")
                tracker.update_trade_exit(
                    ticket=trade.ticket,
                    exit_price=trade.entry_price,
                    profit=0.0,
                    status="closed",
                )


def run_bot(config: AppConfig) -> None:
    # Setup logging and tracking
    logger = setup_logger("forex_bot")
    tracker = TradeTracker()
    
    log_section(logger, "BOT START")
    logger.info("symbols=%s timeframe=%s dry_run=%s ai=%s", config.trade.symbols, config.trade.timeframe, config.trade.dry_run, config.ai.enabled)
    
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
        while True:
            # Monitor existing positions first
            _monitor_positions(client, tracker, logger)
            
            for symbol in config.trade.symbols:
                try:
                    rates = client.get_rates(symbol, config.trade.timeframe, config.strategy.min_bars + 10)
                    state = generate_signal(
                        rates,
                        config.strategy.ema_fast,
                        config.strategy.ema_slow,
                        config.strategy.min_bars,
                        config.strategy.allow_short,
                        config.strategy.use_rsi,
                        config.strategy.rsi_period,
                        config.strategy.rsi_overbought,
                        config.strategy.rsi_oversold,
                        config.strategy.use_atr,
                        config.strategy.atr_period,
                        config.strategy.atr_min_threshold,
                    )

                    spread = client.get_spread_pips(symbol)
                    positions = client.get_open_positions(symbol=symbol, magic=config.trade.magic)

                    if not _should_trade(spread, config.trade.max_spread_pips, len(positions), config.trade.max_positions):
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

                    decision = _build_decision(client, symbol, state.signal, config.trade.sl_pips, config.trade.tp_pips)
                    tick = client.symbol_info_tick(symbol)
                    entry_price = tick.ask if decision.action == "buy" else tick.bid

                    if config.trade.dry_run:
                        logger.info(
                            "[%s] dry_run action=%s lot=%.2f entry=%.5f sl=%.5f tp=%.5f spread=%.2f",
                            symbol,
                            decision.action.upper(),
                            config.trade.lot,
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
                            lot=config.trade.lot,
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
                        lot=config.trade.lot,
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
                        lot=config.trade.lot,
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

            time.sleep(config.trade.polling_seconds)
    except KeyboardInterrupt:
        logger.info("Bot stopped by user")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        raise
    finally:
        logger.info("Shutting down MT5 connection")
        client.shutdown()
        
        # Show performance summary
        metrics = tracker.calculate_metrics()
        if metrics:
            logger.info("=" * 50)
            logger.info("PERFORMANCE SUMMARY")
            logger.info(f"Total trades: {metrics.total_trades}")
            logger.info(f"Win rate: {metrics.win_rate:.1%}")
            logger.info(f"Net profit: {metrics.net_profit:.2f}")
            logger.info(f"Profit factor: {metrics.profit_factor:.2f}")
            logger.info("=" * 50)


def main() -> None:
    config = load_config()
    run_bot(config)


if __name__ == "__main__":
    main()
