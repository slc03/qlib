"""
高频交易（日内做T）策略示例
"""

import os
import copy
import warnings
import numpy as np
import pandas as pd

from typing import Dict, List, Text, Tuple, Union
from abc import ABC

from .signal_strategy import BaseSignalStrategy

from qlib.backtest.position import Position
from qlib.backtest.decision import Order, OrderDir, TradeDecisionWO


class IntradayTStrategy(BaseSignalStrategy):
    # TODO:
    # 1. 更灵活的打印中间过程，便于调试
    # 2. 更灵活的定制买入和卖出策略
    # 3. 更灵活的持仓权重
    # 4. 重新设计init参数
    def __init__(
        self,
        *,
        forbid_all_trade_at_limit=True,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
    
    def generate_trade_decision(self, execute_result=None):
        """仅在做出交易决策时，才打印内容，以减少信息量"""
        # 1️⃣ 获取当前交易步及时间窗口
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)

        # 2️⃣ 获取预测分数
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or np.isfinite(pred_score).sum() == 0:
            return TradeDecisionWO([], self)
        
        # 3️⃣ 因为只有一个股票，因此我们选择第一个值即可
        code = pred_score.index[0]
        pred_score = pred_score.iloc[0]
        if pred_score == 0.0:
            return TradeDecisionWO([], self)

        # 4️⃣ 拷贝当前持仓
        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list, buy_order_list = [], []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()

        # 5️⃣ 根据信号确定买入/卖出还是持有，以及开仓数量
        if pred_score < 0 and len(current_stock_list) > 0:
            print(f"  [Datetime] {trade_end_time.strftime('%Y-%m-%d %H:%M:%S')}")
            stock_tradable, reason = self.trade_exchange.is_stock_tradable(
                stock_id = code,
                start_time = trade_start_time,
                end_time = trade_end_time,
                direction = None if self.forbid_all_trade_at_limit else OrderDir.SELL,
                specific_mode = True,
            )
            if not stock_tradable:
                print(f"[Skip] {code} 不可卖出。（{reason}）")
                return TradeDecisionWO([], self)
            
            hold_amount = current_temp.get_stock_amount(code=code)
            sell_amount = hold_amount * (-pred_score)
            sell_order = Order(
                stock_id = code,
                amount = sell_amount,
                start_time = trade_start_time,
                end_time = trade_end_time,
                direction = Order.SELL,
            )
            
            if self.trade_exchange.check_order(sell_order):
                sell_order_list.append(sell_order)
                trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                    sell_order, position=current_temp
                )
                cash += trade_val - trade_cost
                print(f"  [Sell] {code} 减持={sell_amount:.2f} 余量={hold_amount-sell_amount:.2f} 价格={trade_price:.2f} 收入={trade_val:.2f} 手续费={trade_cost:.2f} 现金={cash:.2f}")
            else:
                print(f"  [Invalid Order] {code} 卖单校验未通过。")
                
        elif pred_score > 0 and cash > 0:
            print(f"  [Datetime] {trade_end_time.strftime('%Y-%m-%d %H:%M:%S')}     [Code] {code}")
            stock_tradable, reason = self.trade_exchange.is_stock_tradable(
                stock_id = code,
                start_time = trade_start_time,
                end_time = trade_end_time,
                direction = None if self.forbid_all_trade_at_limit else OrderDir.BUY,
                specific_mode = True,
            )
            if not stock_tradable:
                print(f"  [Skip] {code} 不可买入。（{reason}）")
                return TradeDecisionWO([], self)
            
            buy_price = self.trade_exchange.get_deal_price(
                stock_id = code, start_time = trade_start_time, end_time = trade_end_time, direction = OrderDir.BUY
            )
            buy_amount = cash * pred_score / buy_price
            factor = self.trade_exchange.get_factor(stock_id = code, start_time = trade_start_time, end_time = trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id = code,
                amount = buy_amount,
                start_time = trade_start_time,
                end_time = trade_end_time,
                direction = Order.BUY,
            )
            buy_order_list.append(buy_order)
            print(f"  [Buy] {code} 计划增持={buy_amount:.2f} 价格={buy_price:.2f} 花费={buy_amount*buy_price:.2f}")

        return TradeDecisionWO(sell_order_list + buy_order_list, self)
