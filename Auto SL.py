# -*- coding: utf-8 -*-
"""
Created on Fri Oct 10 17:17:57 2025

@author: taten
"""

import MetaTrader5 as mt5

MT5_LOGIN = 34276110
MT5_PASSWORD = "[6yA[9hL"
MT5_SERVER = "Weltrade-Real"

# multiplier to decide how far stops are (in stop levels)
SL_MULTIPLIER = 3
TP_MULTIPLIER = 600

if not mt5.initialize(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER):
    print("❌ Initialization failed:", mt5.last_error())
    quit()
else:
    print("✅ Connected to MT5")

positions = mt5.positions_get()
if not positions:
    print("⚠️ No open positions found.")
else:
    for pos in positions:
        symbol = pos.symbol
        ticket = pos.ticket
        mt5.symbol_select(symbol, True)

        info = mt5.symbol_info(symbol)
        if info is None:
            print(f"⚠️ {symbol}: symbol_info() failed.")
            continue

        # ensure data is up to date
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            print(f"⚠️ {symbol}: could not get tick info.")
            continue

        point = info.point
        stop_level = info.trade_stops_level
        if stop_level == 0:
            stop_level = 10  # default fallback if broker doesn't provide one

        min_distance = stop_level * point

        if pos.type == 0:  # BUY
            sl = tick.bid - SL_MULTIPLIER * min_distance
            tp = tick.ask + TP_MULTIPLIER * min_distance
        else:              # SELL
            sl = tick.ask + SL_MULTIPLIER * min_distance
            tp = tick.bid - TP_MULTIPLIER * min_distance

        # round to symbol digits
        sl = round(sl, info.digits)
        tp = round(tp, info.digits)

        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "position": ticket,
            "symbol": symbol,
            "sl": sl,
            "tp": tp,
            "comment": "Dynamic SL/TP Auto",
        }

        result = mt5.order_send(request)
        if result is None:
            print(f"❌ {symbol} Ticket {ticket}: order_send() returned None — {mt5.last_error()}")
            continue

        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"✅ {symbol} Ticket {ticket}: SL={sl}, TP={tp} set successfully.")
        else:
            print(f"❌ {symbol} Ticket {ticket}: failed ({result.retcode}) — {result.comment}")

mt5.shutdown()
print("🔒 MT5 closed.")
