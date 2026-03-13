import pandas as pd
import numpy as np
from utils import *
from MyModel import MyModel
#import time
#import timeit

def run_test():
    #########Load Model
    model = MyModel()
    print("开始测试，一天数据预计用时60分钟")
    #########Load Day Data
    days = get_day_folders("./data")

    #########Online Predict
    for day in days:
        day_data = load_day_data("./data", day)
        n_ticks = len(day_data['E'])

        ticktimes = day_data['E'].values.T[0, :]
        my_preds = np.zeros((n_ticks))



        for tick_index in range(n_ticks):
            ###########Get Tick Data(E and Sector)
            #tick_time=time.perf_counter()
            E_row_data = day_data['E'].iloc[tick_index]
            sector_row_datas = [
                day_data['A'].iloc[tick_index],
                day_data['B'].iloc[tick_index],
                day_data['C'].iloc[tick_index],
                day_data['D'].iloc[tick_index]
            ]


            ###########Predict
            my_preds[tick_index] = model.online_predict(E_row_data, sector_row_datas)
            #print(f"一次总用时：{time.perf_counter() - tick_time}s")

        ###########Save Data
        if os.path.exists("./output/"+day) is not True:
            os.makedirs("./output/"+day)
        out_frame = pd.DataFrame(np.vstack(([ticktimes, my_preds])).T)
        columns = ['Time', 'Predict']
        out_frame.columns = columns
        out_frame.to_csv("./output/"+day+"/E.csv", index=False)
        print ("Submit Day", day)
        print("开始重置数据")
        model.reset()
        #print(f"第{day}天结束，用时{time.perf_counter()-start_time}s")
        #input()

    
if __name__ == '__main__':
    run_test()
