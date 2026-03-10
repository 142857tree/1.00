
# ============ 第一部分：导入库 ============
import pandas as pd
import numpy as np
import warnings
#import time
#import timeit
warnings.filterwarnings('ignore')  # 忽略警告信息，让输出更干净


# ============ 第八部分：预测新数据函数 ============
def predict_new_data(self, E_row, sector_rows):
    [A_row,B_row,C_row,D_row] = sector_rows
    test_merged_data = merge_all_stocks_one_line(A_row,B_row,C_row,D_row,E_row)#只有一行五个股票的数据
    self.whole_dataframe=pd.concat((self.whole_dataframe,test_merged_data)).reset_index(drop=True)

    #self.whole_dataframe.to_csv("whole_data.csv", index=False)
    #print(self.whole_dataframe['Time'].iloc[-1])
    #print(".......;;;;;;")
    #只保留最近15min的数据（即1800行,适当多一些）
    #start=time.perf_counter()
    if len(self.whole_dataframe) > 1850:
        self.whole_dataframe=self.whole_dataframe.iloc[1:]
    #end = time.perf_counter()
    #print(f"切片操作：{end-start}s")
    test_features = create_all_features_enhanced(self.whole_dataframe, self.lst_tick_feature)
    self.lst_tick_feature = test_features.iloc[0]

    #test_features.to_csv("test_features.csv")
    #start=time.perf_counter()
    if len(self.whole_dataframe) == 1:
        self.whole_dataframe = pd.concat([self.whole_dataframe,test_features],axis=1)
    else:
        need_index=[]
        for stock_prefix in ['A','B','C','D','E']:
            for name in ['30s','1min','5min','10min']:
                need_index.append(f'{stock_prefix}_ma_{name}')
            need_index.append(f'{stock_prefix}_price_momentum_5')
            need_index.append(f'{stock_prefix}_cross_up_30s_1min')
            need_index.append(f'{stock_prefix}_cross_down_30s_1min')
            need_index.append(f'{stock_prefix}_order_imbalance')
        for index in need_index:
            self.whole_dataframe[index].iloc[-1] = test_features[index].iloc[0]
    #end=time.perf_counter()
    #print(f"合并用时：{end-start}s")
    missing_features = set(self.selected_features) - set(test_features.columns)
    extra_features = set(test_features.columns) - set(self.selected_features)



    if missing_features:
        for feature in missing_features:
            test_features[feature] = 0

    if extra_features:
        #print(f"移除{len(extra_features)}个多余特征")
        test_features = test_features.drop(columns=list(extra_features))

    # 按正确的顺序选择特征
    x_test = test_features[self.selected_features]
    #print(f"标准化前用时{time.perf_counter()-start}s")
    # 标准化
    x_test_scaled = self.x_scaler.transform(x_test)

    # 预测
    predictions = self.model.predict(x_test_scaled)
    return predictions[0]

def load_stock_data_one_line(row,stock_name):

    # 读取CSV文件
    df = row.to_frame().T
    #print(df)

    # 检查列名
    #print(f"  原始列名: {df.columns.tolist()[:5]}...")

    df.drop(columns=['Return5min'], inplace=True, errors='ignore')
    # 为所有列添加前缀（除了Time）
    rename_dict = {}
    for col in df.columns:
        if col == 'Time':
            continue  # 时间列保持不变
        elif col == 'Return5min':
            if stock_name == 'E':
                rename_dict[col] = 'target'  # E股票的目标变量
            else:
                rename_dict[col] = f'{stock_name}_Return5min'  # 其他股票的5分钟收益率
        else:
            rename_dict[col] = f'{stock_name}_{col}'

    df = df.rename(columns=rename_dict)

    # 显示处理后的列名
    #print(f"  处理后的列名示例: {[col for col in df.columns if 'Return5min' in col or col == 'target']}")

    return df


def merge_all_stocks_one_line(A_row,B_row,C_row,D_row,E_row):
    """
    合并五只股票的数据 - 修正版
    """
    #print("开始合并五只股票数据...")

    all_dfs = []


    df = load_stock_data_one_line(A_row,'A')
    all_dfs.append(df)
    df = load_stock_data_one_line(B_row, 'B')
    all_dfs.append(df)
    df = load_stock_data_one_line(C_row,'C')
    all_dfs.append(df)
    df = load_stock_data_one_line(D_row,'D')
    all_dfs.append(df)
    df = load_stock_data_one_line(E_row,'E')
    all_dfs.append(df)
    #print(all_dfs)
    # 使用reduce逐步合并 concat?
    from functools import reduce

    def merge_func(df1, df2):
        return pd.merge(df1, df2, on='Time', how='inner')

    merged_df = reduce(merge_func, all_dfs)

    #print(f"合并完成，总数据形状：{merged_df.shape}")
    #print(f"列名数量：{len(merged_df.columns)}")

    # 显示包含Return5min的列，确认没有重复
    return_cols = [col for col in merged_df.columns if 'Return5min' in col or 'target' in col]
    #print(f"收益率相关列：{return_cols}")

    return merged_df


# ============ 第三部分：特征工程函数 ============

def enhanced_stock_features(df, stock_prefix, lst_tick_feature):#参数增加上次计算的所有feature
    """增强版股票特征"""
    #start=time.perf_counter()
    features = {}
    df_now = df.iloc[-1]
    if len(df) > 120:
        df_before_1min = df.iloc[-121]#由于当前量和1min前的量计算较多，额外维护这两个量
    else:
        df_before_1min = pd.Series()
    # 基础价格列
    last_price_col = f'{stock_prefix}LastPrice'
    stock_lst_price = df[last_price_col].values
    if last_price_col in df.columns:
        # === 价格趋势特征 ===
        # 1. 价格动量（短期、中期）
        for period in [5, 10, 20, 30, 60, 120]:  # 增加更多时间尺度
            _period = min(period, len(df)-1)
            features[f'{stock_prefix}price_momentum_{period}'] = (stock_lst_price[-1] - stock_lst_price[-1-_period]) / stock_lst_price[-1-_period]
            #那这里最前面的几个值就NAN了是嘛（

        # 2. 价格波动范围
        for window in [60, 120, 300, 600]:  # 30秒到5分钟
            roll_max = stock_lst_price[-window:].max()
            roll_min = stock_lst_price[-window:].min()
            features[f'{stock_prefix}price_range_{window}'] = (roll_max - roll_min) / (roll_min + 1e-8)#何意味避免0嘛
        #print(f"1、2用时{time.perf_counter()-start}s")
        # 3. 移动平均线特征
        ma_windows = {
            '30s': 60,  # 30秒 (比5分钟预测窗口短)
            '1min': 120,  # 1分钟 (预测窗口的1/5)
            '2min': 240,  # 2分钟 (预测窗口的2/5)
            '3min': 360,  # 3分钟 (预测窗口的3/5)
            '5min': 600,  # 5分钟 (与预测窗口相同)
            '10min': 1200,  # 10分钟 (预测窗口的2倍)
            '15min': 1800,  # 15分钟 (预测窗口的3倍)
        }

        # 1. 计算所有移动平均线
        ma_dict = {}
        for name, window in ma_windows.items():
            ma_key = f'{stock_prefix}ma_{name}'
            # 使用适当的min_periods避免开头太多NaN(window中大于等于min_periods个数不为NAN，rolling就不为NAN)
            if len(df) == 1:
                ma_dict[name] = stock_lst_price[-1]
            elif len(df) <= window:
                ma_dict[name] = (lst_tick_feature[ma_key] * (len(df)-1) + stock_lst_price[-1]) / len(df)
            else:
                ma_dict[name] = (lst_tick_feature[ma_key] * window + stock_lst_price[-1] - stock_lst_price[-window-1]) / window
            features[ma_key] = ma_dict[name]
        # 2. 价格相对于移动平均线的位置（核心特征）
        # 偏离度 = (价格 - MA) / MA
        for name in ['30s', '1min', '5min', '10min']:
            if name in ma_dict:#意义何在
                ma_value = ma_dict[name]
                price = df[last_price_col].iloc[-1]
                features[f'{stock_prefix}price_vs_ma_{name}_pct'] = (price - ma_value) / (ma_value + 1e-8)

                # 价格是否在MA之上（布尔特征）
                features[f'{stock_prefix}above_ma_{name}'] = (price > ma_value).astype(int)

        # 3. 移动平均线的趋势特征（MA的斜率）
        # 计算各MA在一段时间内的变化率
        for name in ['1min', '5min', '10min']:
            ma_value = ma_dict[name]
            # 短期变化：最近1分钟的变化
            if len(df) > 120:
                ma_value_before_1min = df_before_1min[f'{stock_prefix}ma_{name}']
            else:
                ma_value_before_1min = np.nan
            change_1min = ma_value - ma_value_before_1min
            features[f'{stock_prefix}ma_{name}_change_1min'] = change_1min / (ma_value_before_1min + 1e-8)

            # 只保留需要的历史数据（span=300最多需要300个数据点）
            alpha_short = 2/61
            alpha_long = 2/301

            # 计算EMA后只取最后一个值（比全量计算快很多）
            if len(df) > 1:
                ema_short = lst_tick_feature[f'{stock_prefix}ma_{name}_ema_short'] * (1-alpha_short) + df_now[f'{stock_prefix}ma_{name}'] * alpha_short
                ema_long = lst_tick_feature[f'{stock_prefix}ma_{name}_ema_long'] * (1-alpha_long) + df_now[f'{stock_prefix}ma_{name}'] * alpha_long
            else:
                ema_short = stock_lst_price[-1]
                ema_long = ema_short

            # 趋势强度计算
            features[f'{stock_prefix}ma_{name}_ema_short'] = ema_short
            features[f'{stock_prefix}ma_{name}_ema_long'] = ema_long
            trend_strength = (ema_short - ema_long) / (ema_long + 1e-8)
            features[f'{stock_prefix}ma_{name}_trend_strength'] = trend_strength

        # 4. 金叉死叉特征（技术分析核心）
        if all(name in ma_dict for name in ['30s', '1min', '5min']):
            ma_30s = ma_dict['30s']
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']
            if len(df) > 1:
                ma_30s_last=df_now[f'{stock_prefix}ma_{"30s"}']#overall_dataframe里面的上一个均线
                ma_5min_last=df_now[f'{stock_prefix}ma_{"5min"}']
                ma_1min_last=df_now[f'{stock_prefix}ma_{"1min"}']
            else:
                ma_30s_last = np.nan
                ma_5min_last = np.nan
                ma_1min_last = np.nan


            #print(f"目前用时（金叉前）{time.perf_counter()-start}s")
            # 4.1 基本金叉死叉信号
            # 30秒线上穿1分钟线
            cross_up_30s_1min = (ma_30s > ma_1min) & (ma_30s_last <= ma_1min_last)
            cross_down_30s_1min = (ma_30s < ma_1min) & (ma_30s_last >= ma_1min_last)

            # 1分钟线上穿5分钟线（传统金叉）
            cross_up_1min_5min = (ma_1min > ma_5min) & (ma_1min_last <= ma_5min_last)
            cross_down_1min_5min = (ma_1min < ma_5min) & (ma_1min_last >= ma_5min_last)
            features[f'{stock_prefix}cross_up_30s_1min'] = cross_up_30s_1min.astype(int)
            features[f'{stock_prefix}cross_down_30s_1min'] = cross_down_30s_1min.astype(int)
            features[f'{stock_prefix}cross_up_1min_5min'] = cross_up_1min_5min.astype(int)
            features[f'{stock_prefix}cross_down_1min_5min'] = cross_down_1min_5min.astype(int)

            # 4.2 金叉/死叉后的时间（事件驱动特征）
            # 距离上次金叉的时间（tick数）
            for cross_name, cross_signal, name_in_dataframe in [('up_1min_5min', cross_up_1min_5min, f'{stock_prefix}cross_up_1min_5min'),
                                             ('down_1min_5min', cross_down_1min_5min, f'{stock_prefix}cross_down_1min_5min')]:
                # 标记金叉发生离现在的时间
                if (len(df) > 1) & (features[name_in_dataframe]):
                    features[f'{stock_prefix}ticks_since_{cross_name}'] = 0
                elif len(df) ==1:
                    features[f'{stock_prefix}ticks_since_{cross_name}'] = 9999
                else:
                    features[f'{stock_prefix}ticks_since_{cross_name}'] = max(9999, lst_tick_feature[name_in_dataframe] + 1)

            # 4.3 均线排列状态（多头/空头排列）
            # 多头排列：短期 > 中期 > 长期
            bull_alignment = (ma_30s > ma_1min) & (ma_1min > ma_5min)
            # 空头排列：短期 < 中期 < 长期
            bear_alignment = (ma_30s < ma_1min) & (ma_1min < ma_5min)

            features[f'{stock_prefix}bull_alignment'] = bull_alignment.astype(int)
            features[f'{stock_prefix}bear_alignment'] = bear_alignment.astype(int)

            # 排列强度：使用Z-score标准化
            ma_diff_30s_1min = (ma_30s - ma_1min) / (ma_1min + 1e-8)
            ma_diff_1min_5min = (ma_1min - ma_5min) / (ma_5min + 1e-8)
            alignment_strength = ma_diff_30s_1min + ma_diff_1min_5min
            features[f'{stock_prefix}alignment_strength'] = alignment_strength

        #print(f"目前用时：{time.perf_counter()-start}s")
        # 5. 移动平均线带宽特征（MA间的距离）
        if all(name in ma_dict for name in ['30s', '5min']):
            ma_30s = ma_dict['30s']
            ma_5min = ma_dict['5min']

            # 带宽 = (短期MA - 长期MA) / 长期MA
            ma_bandwidth = (ma_30s - ma_5min) / (ma_5min + 1e-8)
            features[f'{stock_prefix}ma_bandwidth'] = ma_bandwidth

            # 带宽的变化率
            if len(df) > 120:
                bandwidth_change = ma_bandwidth - df_before_1min[f'{stock_prefix}ma_bandwidth']  # 1分钟变化
            else:
                bandwidth_change = np.nan
            features[f'{stock_prefix}ma_bandwidth_change'] = bandwidth_change
        #print(f"5目前用时：{time.perf_counter() - start}s")

        # 6. 价格与MA的背离特征
        if all(name in ma_dict for name in ['1min', '5min']):
            ma_1min = ma_dict['1min']
            ma_5min = ma_dict['5min']

            # 计算价格和MA的动量
            if len(df) > 60:
                price_momentum = stock_lst_price[-1] - stock_lst_price[-61]  # 30秒动量
                ma_1min_momentum = ma_1min - df[f'{stock_prefix}ma_1min'].iloc[-60]
                ma_5min_momentum = ma_5min - df[f'{stock_prefix}ma_5min'].iloc[-60]
            else:
                price_momentum = np.nan
                ma_1min_momentum = np.nan
                ma_5min_momentum = np.nan

            # 背离：价格上涨但MA下跌，或反之
            divergence_1min = ((price_momentum > 0) & (ma_1min_momentum < 0)) | \
                              ((price_momentum < 0) & (ma_1min_momentum > 0))
            divergence_5min = ((price_momentum > 0) & (ma_5min_momentum < 0)) | \
                              ((price_momentum < 0) & (ma_5min_momentum > 0))

            features[f'{stock_prefix}divergence_1min'] = int(divergence_1min)
            features[f'{stock_prefix}divergence_5min'] = int(divergence_5min)
        #print(f"5目前用时：{time.perf_counter() - start}s")
        # 4. 收益率波动率（使用1期收益率）
        ret_1 = []
        for i in range(0,30):
            if len(df) > 30-i:
                ret_1.append((stock_lst_price[i-30] - stock_lst_price[i-31]) / stock_lst_price[i-31])
            else:
                ret_1.append(np.nan)
        ret_1=pd.Series(ret_1)
        features[f'{stock_prefix}ret_volatility_10'] = ret_1[-10:].std()
        features[f'{stock_prefix}ret_volatility_20'] = ret_1[-20:].std()
        features[f'{stock_prefix}ret_volatility_30'] = ret_1[-30:].std()
        #print(f"6目前用时：{time.perf_counter() - start}s")
    # === 成交量特征 ===
    # 成交量加权价格

    now_all_price = df_now[f'{stock_prefix}TradeBuyAmount'] + df_now[f'{stock_prefix}TradeSellAmount']
    now_volume = df_now[f'{stock_prefix}TradeBuyVolume'] + df_now[f'{stock_prefix}TradeSellVolume']
    price = now_all_price / now_volume

    for window in [120, 600, 1200]:  # 1分钟、5分钟、10分钟VWAP

        if len(df) == 1:
            all_price_sum = now_all_price
            volume_sum = now_volume
        elif len(df) <= window:
            all_price_sum = now_all_price + lst_tick_feature[f'{stock_prefix}all_price_{window}']
            volume_sum = now_volume + lst_tick_feature[f'{stock_prefix}volume_{window}']
        else:
            all_price_sum = now_all_price + lst_tick_feature[f'{stock_prefix}all_price_{window}'] - df[f'{stock_prefix}TradeBuyAmount'].iloc[-1-window] - df[f'{stock_prefix}TradeSellAmount'].iloc[-1-window]
            volume_sum = now_volume + lst_tick_feature[f'{stock_prefix}volume_{window}'] - df[f'{stock_prefix}TradeBuyVolume'].iloc[-1-window] - df[f'{stock_prefix}TradeSellVolume'].iloc[-1-window]
        features[f'{stock_prefix}all_price_{window}'] = all_price_sum
        features[f'{stock_prefix}volume_{window}'] = volume_sum
        vwap = all_price_sum / volume_sum
        features[f'{stock_prefix}vwap_{window}'] = vwap
        features[f'{stock_prefix}price_vwap_diff_{window}'] = price - vwap
        features[f'{stock_prefix}price_vwap_ratio_{window}'] = price / (vwap + 1e-8) - 1
    #print(f"7目前用时：{time.perf_counter() - start}s")
    # === 委托深度特征 ===
    bid_pos = [df.columns.get_loc(f'{stock_prefix}BidVolume{i}') for i in range(1, 6)]
    ask_pos = [df.columns.get_loc(f'{stock_prefix}AskVolume{i}') for i in range(1, 6)]

    # 一次性选取行和列，避免链式索引
    total_bid_depth = df_now.iloc[bid_pos].sum()
    total_ask_depth = df_now.iloc[ask_pos].sum()

    if len(df) > 120:
        total_bid_depth_before_1min = df_before_1min.iloc[bid_pos].sum()
        total_ask_depth_before_1min = df_before_1min.iloc[ask_pos].sum()
    else:
        total_bid_depth_before_1min = np.nan
        total_ask_depth_before_1min = np.nan
    total_bid_depth_change_1min = (total_bid_depth-total_bid_depth_before_1min)/total_bid_depth_before_1min
    total_ask_depth_change_1min = (total_ask_depth-total_ask_depth_before_1min)/total_ask_depth_before_1min

    features[f'{stock_prefix}depth_imbalance'] = (total_bid_depth - total_ask_depth) / (total_bid_depth + total_ask_depth + 1e-6)

    # 深度变化率
    features[f'{stock_prefix}depth_change'] = total_bid_depth_change_1min-total_ask_depth_change_1min


    #print(f"8目前用时：{time.perf_counter() - start}s")
    # === 订单流不平衡（高级版）===
    if (f'{stock_prefix}OrderBuyVolume' in df.columns and
            f'{stock_prefix}OrderSellVolume' in df.columns):
        # 订单流不平衡
        order_imbalance = (
                                  df_now[f'{stock_prefix}OrderBuyVolume'] -
                                  df_now[f'{stock_prefix}OrderSellVolume']
                          ) / (df_now[f'{stock_prefix}OrderBuyVolume'] + df_now[f'{stock_prefix}OrderSellVolume'] + 1e-6)

        features[f'{stock_prefix}order_imbalance'] = order_imbalance
        if len(df) > 120:
            features[f'{stock_prefix}order_imbalance_ma'] = (lst_tick_feature[f'{stock_prefix}order_imbalance_ma'] * 120 + features[f'{stock_prefix}order_imbalance'] - df[f'{stock_prefix}order_imbalance'].iloc[-120])/120
        elif len(df) > 1:
            features[f'{stock_prefix}order_imbalance_ma'] = (lst_tick_feature[f'{stock_prefix}order_imbalance_ma'] * (len(df)-1) + features[f'{stock_prefix}order_imbalance'])/len(df)
        else:
            features[f'{stock_prefix}order_imbalance_ma'] = features[f'{stock_prefix}order_imbalance']
    #end=time.perf_counter()
    #print(f"enhanced_stock_features用时：{end-start}s")
    return features




# 添加MA特征后处理的辅助函数
def post_process_ma_features(features_df, stock_prefix='E'):
    """对移动平均线特征进行后处理"""
    #print(type(features_df))
    #print(len(features_df))
    #print("*")
    # 3. 创建MA特征组合（交互特征）
    #start=time.perf_counter()
    if f'{stock_prefix}price_vs_ma_30s_pct' in features_df.columns and \
            f'{stock_prefix}price_vs_ma_5min_pct' in features_df.columns:
        # 短期偏离与长期偏离的差异
        features_df[f'{stock_prefix}ma_deviation_diff'] = (
                features_df[f'{stock_prefix}price_vs_ma_30s_pct'] -
                features_df[f'{stock_prefix}price_vs_ma_5min_pct']
        )

        # 偏离的一致性（符号是否相同）
        features_df[f'{stock_prefix}ma_deviation_consistent'] = (
                np.sign(features_df[f'{stock_prefix}price_vs_ma_30s_pct']) ==
                np.sign(features_df[f'{stock_prefix}price_vs_ma_5min_pct'])
        ).astype(int)

    #end=time.perf_counter()
    #print(f"post_process_ma_features用时：{end-start}s")
    return features_df.squeeze().to_dict()


def enhanced_sector_features(stock_features_dict):
    #start=time.perf_counter()
    sector_features = {}
    # 收集所有股票的ma_10min_pct特征
    ma_features = []
    for stock in ['A', 'B', 'C', 'D', 'E']:
        ma_feat = f'{stock}_price_vs_ma_10min_pct'
        if ma_feat in stock_features_dict:
            ma_features.append(stock_features_dict[ma_feat])

    if ma_features:
        ma_df = pd.Series(ma_features)
        # 板块平均偏离
        sector_features['sector_ma_deviation_avg'] = ma_df.mean()
        # 板块偏离离散度
        sector_features['sector_ma_deviation_std'] = ma_df.std()
        # E股在板块中的排名
        e_ma = ma_df.iloc[-1]
        sector_features['E_ma_deviation_rank'] = (ma_df.T > e_ma).sum() / len(ma_features)

    #end=time.perf_counter()
    #print(f"enhanced_sector_features用时：{end-start}s")
    return sector_features


def enhanced_time_features(now_time):
    """增强版时间特征"""
    #print(type(now_time))
    time_features = {}
    time_series = pd.Series([now_time])
    # 将整数时间转换为datetime
    time_str = time_series.astype(str).str.zfill(9)
    hours = time_str.str[:2].astype(int)
    minutes = time_str.str[2:4].astype(int)
    #print([hours[0], minutes[0]])
    # 只保留开盘/收盘标记
    time_features['is_opening_15min'] = ((hours == 9) & (minutes >= 30) & (minutes < 45))
    time_features['is_closing_15min'] = ((hours == 14) & (minutes >= 45))

    return time_features


def add_e_specific_features(df,stock_features_dict):
    """为E股添加特定特征（预测目标）"""
    #start=time.perf_counter()
    e_features = {}

    # E股的各种收益率
    if 'E_price_momentum_5' in stock_features_dict:
        # 1. E股动量强度
        if len(df) > 1:
            e_momentum_5 = df['E_price_momentum_5']
        else:
            e_momentum_5 = pd.Series()
        e_momentum_5[df.index[-1]] = stock_features_dict['E_price_momentum_5']
        e_std_20 = e_momentum_5.tail(20).std()
        e_features['E_momentum_strength'] = e_momentum_5.iloc[-1] / (e_std_20 + 1e-6)

        # 2. E股动量变化率
        if len(df) > 1:
            e_features['E_momentum_change'] = (e_momentum_5.iloc[-1]-e_momentum_5.iloc[-2])/(e_momentum_5 + 1e-6)
        else:
            e_features['E_momentum_change'] = np.nan
        # 3. E股动量符号
        e_features['E_momentum_sign'] = np.sign(e_momentum_5.iloc[-1])


        # 4. E股连续上涨/下跌次数
        sign_series = np.sign(e_momentum_5)

        lst_pos = e_momentum_5[sign_series * e_momentum_5 < 0].last_valid_index()
        if lst_pos is None:
            consecutive = 0
        else:
            consecutive = len(e_momentum_5)- lst_pos
        e_features['E_consecutive_direction'] = consecutive * sign_series.iloc[-1]

    # E股与其他股票的互动
    for stock in ['A', 'B', 'C', 'D']:
        if f'{stock}_price_momentum_5' in stock_features_dict:
            stock_momentum = stock_features_dict[f'{stock}_price_momentum_5']
            e_momentum = stock_features_dict['E_price_momentum_5']

            # 5. E股相对于其他股票的动量差
            e_features[f'E_vs_{stock}_momentum_diff'] = e_momentum - stock_momentum

            # 6. E股与其他股票动量的比率
            e_features[f'E_{stock}_momentum_ratio'] = e_momentum / (abs(stock_momentum) + 1e-6)

    #end=time.perf_counter()
    #print(f"add_e用时:{end-start}s")
    return e_features


def create_all_features_enhanced(df, lst_tick_feature):
    #start = time.perf_counter()
    """增强版特征创建features+target"""
    #print("开始创建增强版特征...")

    stock_features_dict = {}

    # 为每只股票创建增强特征
    for stock in ['A', 'B', 'C', 'D', 'E']:
        #print(f"  创建{stock}股增强特征...")
        # 增强特征
        features = enhanced_stock_features(df, f'{stock}_', lst_tick_feature)
        stock_features_dict.update(features)
        #print(type(features))
        # 对MA特征进行后处理（主要对E股票）
        if stock == 'E':
            ma_features = {k: v for k, v in features.items() if 'ma_' in k}
            if ma_features:
                ma_df = pd.Series(ma_features).to_frame().T
                processed_ma = post_process_ma_features(ma_df, f'{stock}_')
                stock_features_dict.update(processed_ma)
                #print(type(processed_ma))

    e_specific_features = add_e_specific_features(df,stock_features_dict)
    stock_features_dict.update(e_specific_features)
    #print(type(stock_features_dict))
    sector_features = enhanced_sector_features(stock_features_dict)
    #print(df['Time'].iloc[-1])
    #print("????????")
    time_features = enhanced_time_features(int(df['Time'].iloc[-1]))
    all_features_dict = {}
    all_features_dict.update(stock_features_dict)
    all_features_dict.update(sector_features)
    all_features_dict.update(time_features)


    all_features = pd.DataFrame([all_features_dict])
    #all_features.rename(index={0: df['Time'].iloc[-1]}, inplace=True)



    all_features = all_features.fillna(0)
    #end=time.perf_counter()
    #print(f"create_all_features总用时:{end-start}s")
    #input()
    return all_features.iloc[[-1]]


def feature_post_processing(features_df):
    print("\n特征后处理...")




    # 4.1 识别板块特征
    sector_cols = [col for col in features_df.columns if any(x in col for x in [
        '_lead_', '_corr_', 'sector_', 'E_sector', 'A_', 'B_', 'C_', 'D_'
    ]) and col not in ['target']]

    #print(f"    找到{len(sector_cols)}个板块相关特征")

    if sector_cols:

        # 4.3 创建板块特征组合
        return_cols = [col for col in sector_cols if '_return' in col]
        if len(return_cols) >= 2:
            features_df['sector_return_composite'] = features_df[return_cols].mean(axis=1)
            #print(f"    创建了板块收益率综合特征，基于{len(return_cols)}个特征")

    # 6. 最终检查和处理
    #print("  6. 最终检查...")

    # 移除与target相关性为NaN的特征
    if 'target' in features_df.columns:
        target_corr = features_df.corrwith(features_df['target']).abs()
        nan_corr_features = target_corr[target_corr.isna()].index.tolist()
        nan_corr_features = [f for f in nan_corr_features if f != 'target']

        if nan_corr_features:
            #print(f"    移除{len(nan_corr_features)}个与目标相关性为NaN的特征")
            features_df = features_df.drop(columns=nan_corr_features)

    # 确保没有inf或NaN值
    features_df = features_df.replace([np.inf, -np.inf], np.nan)
    features_df = features_df.fillna(0)

    #print(f"  后处理完成，最终特征数: {len(features_df.columns)}")
    return features_df

