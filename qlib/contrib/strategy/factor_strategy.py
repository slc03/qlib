"""
自定义的根据因子调仓的选股策略
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


class MyTopkDropoutStrategy(BaseSignalStrategy):
    # TODO:
    # 1. 更灵活的打印中间过程，便于调试
    # 2. 更灵活的定制买入和卖出策略
    # 3. 更灵活的持仓权重
    def __init__(
        self,
        *,
        topk,
        n_drop,
        method_sell="bottom",
        method_buy="top",
        hold_thresh=1,
        only_tradable=False,
        forbid_all_trade_at_limit=True,
        stop_profit=None,
        stop_loss=None,
        **kwargs,
    ):
        """
        Parameters
        -----------
        topk : int
            the number of stocks in the portfolio.
        n_drop : int
            number of stocks to be replaced in each trading date.
        method_sell : str
            dropout method_sell, random/bottom.
        method_buy : str
            dropout method_buy, random/top.
        hold_thresh : int
            minimum holding days
            before sell stock , will check current.get_stock_count(order.stock_id) >= self.hold_thresh.
        only_tradable : bool
            will the strategy only consider the tradable stock when buying and selling.

            if only_tradable:

                strategy will make decision with the tradable state of the stock info and avoid buy and sell them.

            else:

                strategy will make buy sell decision without checking the tradable state of the stock.
        forbid_all_trade_at_limit : bool
            if forbid all trades when limit_up or limit_down reached.

            if forbid_all_trade_at_limit:

                strategy will not do any trade when price reaches limit up/down, even not sell at limit up nor buy at
                limit down, though allowed in reality.

            else:

                strategy will sell at limit up and buy ad limit down.
        stop_profit : float
            take profit threshold, e.g. 0.1 means taking profit when the price increase 10%
        stop_loss : float
            stop loss threshold, e.g. -0.1 means stopping loss when the price decrease 10%
        """
        super().__init__(**kwargs)
        self.topk = topk
        self.n_drop = n_drop
        self.method_sell = method_sell
        self.method_buy = method_buy
        self.hold_thresh = hold_thresh
        self.only_tradable = only_tradable
        self.forbid_all_trade_at_limit = forbid_all_trade_at_limit
        self.stop_profit = stop_profit
        self.stop_loss = stop_loss
    
    def generate_trade_decision(self, execute_result=None):
        """添加了中文注解"""
        print("\n========== GENERATE_TRADE_DECISION START ==========")

        # 1️⃣ 获取当前交易步及时间窗口
        trade_step = self.trade_calendar.get_trade_step()
        trade_start_time, trade_end_time = self.trade_calendar.get_step_time(trade_step)
        pred_start_time, pred_end_time = self.trade_calendar.get_step_time(trade_step, shift=1)
        print(f"[Step] 当前交易步: {trade_step}")
        print(f"[Time] trade: {trade_start_time} ~ {trade_end_time}")
        print(f"[Time] pred:  {pred_start_time} ~ {pred_end_time}")

        # 2️⃣ 获取预测分数
        pred_score = self.signal.get_signal(start_time=pred_start_time, end_time=pred_end_time)
        if isinstance(pred_score, pd.DataFrame):
            print("[Info] pred_score 为 DataFrame，取第一列")
            pred_score = pred_score.iloc[:, 0]
        if pred_score is None or np.isfinite(pred_score).sum() == 0:
            print("[Warning] 无有效预测信号，返回空交易决策。")
            return TradeDecisionWO([], self)

        # 🧩 统计基本信息
        total_count = len(pred_score)
        non_nan_count = pred_score.notna().sum()
        finite_count = np.isfinite(pred_score).sum()

        # 🧹 去除 ±inf，但保留 NaN（前者视为错误的计算，后者视为无此值）
        pred_score = pred_score[~np.isinf(pred_score)]
        print(f"[Signal] 获取到 {total_count} 条预测信号，"
              f"NaN信号数={total_count-non_nan_count}，"
              f"inf信号数={non_nan_count-finite_count}")

        # 3️⃣ 定义辅助函数（根据 only_tradable）
        if self.only_tradable:
            print("[Mode] 仅考虑可交易股票 (only_tradable=True)")

            def get_first_n(li, n, reverse=False):
                cur_n, res = 0, []
                for si in reversed(li) if reverse else li:
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    ):
                        res.append(si)
                        cur_n += 1
                        if cur_n >= n:
                            break
                return res[::-1] if reverse else res

            def get_last_n(li, n):
                return get_first_n(li, n, reverse=True)

            def filter_stock(li):
                return [
                    si
                    for si in li
                    if self.trade_exchange.is_stock_tradable(
                        stock_id=si, start_time=trade_start_time, end_time=trade_end_time
                    )
                ]
        else:
            print("[Mode] 不考虑可交易过滤 (only_tradable=False)")

            def get_first_n(li, n):
                return list(li)[:n]

            def get_last_n(li, n):
                return list(li)[-n:]

            def filter_stock(li):
                return li

        # 4️⃣ 拷贝当前持仓
        current_temp: Position = copy.deepcopy(self.trade_position)
        sell_order_list, buy_order_list = [], []
        cash = current_temp.get_cash()
        current_stock_list = current_temp.get_stock_list()
        print(f"[Position] 当前持仓股票数: {len(current_stock_list)}，现金: {cash:.2f}")

        # 5️⃣ 当前持仓按预测分数排序
        sorted_series = pred_score.reindex(current_stock_list).sort_values(ascending=False)
        last = sorted_series.index
        
        # 🟩 打印所有当前持仓及其分数及排名占比
        print("\n[Holding Rank] 当前持仓全部股票及对应预测分数及排名占比:")
        rank_all = pred_score.rank(ascending=False, pct=True)
        for code, score in sorted_series.items():
            if code in rank_all:
                pct_rank = rank_all[code] * 100
                print(f"    {code}: {score:.4f}  (前{pct_rank:.2f}%)")
            else:
                print(f"    {code}: {score:.4f}  (无排名)")

        # 🟦 打印当日股票池 top10
        print("\n[Signal Rank] 当日全股票池 Top10 股票及信号:")
        top10 = pred_score.sort_values(ascending=False).head(10)
        for code, score in top10.items():
            mark = " [持仓中]" if code in current_stock_list else ""
            print(f"    {code}: {score:.4f}{mark}")

        # 🟥 打印当日股票池 Last10
        print("\n[Signal Rank] 当日全股票池 Last10 股票及信号:")
        last10 = pred_score.sort_values(ascending=True).head(10)
        for code, score in last10.items():
            mark = " [持仓中]" if code in current_stock_list else ""
            print(f"    {code}: {score:.4f}{mark}")

        # 6️⃣ 计算买入候选 today
        if self.method_buy == "top":
            print("[Buy Method] top")
            today = get_first_n(
                pred_score[~pred_score.index.isin(last)].sort_values(ascending=False).index,
                self.n_drop + self.topk - len(last),
            )
        elif self.method_buy == "random":
            print("[Buy Method] random")
            topk_candi = get_first_n(pred_score.sort_values(ascending=False).index, self.topk)
            candi = list(filter(lambda x: x not in last, topk_candi))
            n = self.n_drop + self.topk - len(last)
            try:
                today = np.random.choice(candi, n, replace=False)
            except ValueError:
                today = candi
        else:
            raise NotImplementedError(f"[Error] method_buy={self.method_buy} 不支持")
        print(f"[Buy Candidate] today={list(today)}")

        # 7️⃣ 合并持仓+候选，排序
        comb = pred_score.reindex(last.union(pd.Index(today))).sort_values(ascending=False).index

        # 8️⃣ 决定卖出列表
        if self.method_sell == "bottom":
            print("[Sell Method] bottom")
            sell = last[last.isin(get_last_n(comb, self.n_drop))]
        elif self.method_sell == "random":
            print("[Sell Method] random")
            candi = filter_stock(last)
            try:
                sell = pd.Index(np.random.choice(candi, self.n_drop, replace=False) if len(last) else [])
            except ValueError:
                sell = candi
        else:
            raise NotImplementedError(f"[Error] method_sell={self.method_sell} 不支持")

        print(f"[Sell Candidate] sell={list(sell)}")
        
        # 🟪 止盈止损检查
        stop_sell = []
        if self.stop_profit is not None or self.stop_loss is not None:
            # print("\n[Risk Control] 检查止盈止损触发情况...")
            for code in current_stock_list:
                init_price = current_temp.get_stock_init_price(code=code)
                cur_price = current_temp.get_stock_price(code=code)
                pct_change = (cur_price / init_price - 1)
                if self.stop_profit is not None and pct_change >= self.stop_profit:
                    print(f"  [StopProfit] {code} 当前收益 {pct_change*100:.2f}% >= {self.stop_profit*100:.2f}% 触发止盈")
                    stop_sell.append(code)
                elif self.stop_loss is not None and pct_change <= self.stop_loss:
                    print(f"  [StopLoss] {code} 当前收益 {pct_change*100:.2f}% <= {self.stop_loss*100:.2f}% 触发止损")
                    stop_sell.append(code)

            # 合并止盈止损卖出到sell列表（不影响n_drop逻辑）
            stop_sell = pd.Index(stop_sell)
            sell = sell.union(stop_sell)    
            print(f"[Sell Candidate (含止盈止损)] sell={list(sell)}")

        # 9️⃣ 决定买入列表
        buy = today[: len(sell) + self.topk - len(last)]
        print(f"[Final Buy List] buy={list(buy)}")

        # 🔟 模拟卖出操作
        print("\n[Action] 开始卖出操作...")
        for code in current_stock_list:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.SELL,
            ):
                print(f"  [Skip] {code} 不可卖出。")
                continue
            if code in sell:
                time_per_step = self.trade_calendar.get_freq()
                hold_days = current_temp.get_stock_count(code=code, bar=time_per_step)
                if hold_days < self.hold_thresh:
                    print(f"  [Hold] {code} 持有时间不足阈值({self.hold_thresh})，跳过卖出。")
                    continue
                
                # 🟦 获取持仓信息
                hold_amount = current_temp.get_stock_amount(code=code)
                init_price = current_temp.get_stock_init_price(code=code)
                
                # 🧾 执行卖单
                sell_amount = hold_amount
                sell_order = Order(
                    stock_id=code,
                    amount=sell_amount,
                    start_time=trade_start_time,
                    end_time=trade_end_time,
                    direction=Order.SELL,
                )
                if self.trade_exchange.check_order(sell_order):
                    sell_order_list.append(sell_order)
                    trade_val, trade_cost, trade_price = self.trade_exchange.deal_order(
                        sell_order, position=current_temp
                    )
                    cash += trade_val - trade_cost
                    print(f"  [Sell] {code} 数量={sell_amount:.2f} 价格={trade_price:.2f} 收入={trade_val:.2f} 手续费={trade_cost:.2f} 当前现金={cash:.2f}")
                    print(f"  [Profit] {code} 持有交易日数={hold_days} 卖出日期: {trade_end_time.strftime('%Y-%m-%d')} 股价变化: {init_price:.2f}->{trade_price:.2f}={(trade_price/init_price-1)*100:.2f}%")
                else:
                    print(f"  [Invalid Order] {code} 卖单校验未通过。")

        # ⓫ 计算买入资金分配
        value = cash * self.risk_degree / len(buy) if len(buy) > 0 else 0
        print(f"[Cash] 卖出后现金={cash:.2f}，风险系数={self.risk_degree}，单只买入金额={value:.2f}")

        # ⓬ 生成买单（不执行）
        print("\n[Action] 开始买入操作...")
        for code in buy:
            if not self.trade_exchange.is_stock_tradable(
                stock_id=code,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=None if self.forbid_all_trade_at_limit else OrderDir.BUY,
            ):
                print(f"  [Skip] {code} 不可买入。")
                continue
            buy_price = self.trade_exchange.get_deal_price(
                stock_id=code, start_time=trade_start_time, end_time=trade_end_time, direction=OrderDir.BUY
            )
            buy_amount = value / buy_price
            factor = self.trade_exchange.get_factor(stock_id=code, start_time=trade_start_time, end_time=trade_end_time)
            buy_amount = self.trade_exchange.round_amount_by_trade_unit(buy_amount, factor)
            buy_order = Order(
                stock_id=code,
                amount=buy_amount,
                start_time=trade_start_time,
                end_time=trade_end_time,
                direction=Order.BUY,
            )
            buy_order_list.append(buy_order)
            print(f"  [Buy] {code} 数量={buy_amount:.2f} 价格={buy_price:.2f} 金额={buy_amount*buy_price:.2f}")

        print("\n========== GENERATE_TRADE_DECISION END ==========")
        print(f"[Summary] 卖单数={len(sell_order_list)} 买单数={len(buy_order_list)}\n")

        return TradeDecisionWO(sell_order_list + buy_order_list, self)
