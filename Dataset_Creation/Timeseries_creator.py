import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def TimeSeries_window(dataset:pd.DataFrame, catalog:pd.DataFrame, itv_front:int = 20, itv_back:int = 25):
    
    catalog = catalog.loc[catalog['cme_vel'].notnull(), :]
    # dataset.loc[dataset['B_filtered'].isnull(), 'B_filtered'] = 0 # Later we add B_AVG and B_filtered

    catalog.loc[:, 'start_time'] = pd.to_datetime(catalog['start_time'])
    catalog.loc[:, 'end_time'] = pd.to_datetime(catalog['end_time'])
    dataset.loc[:, 'Timestamp'] =  pd.to_datetime(dataset['Timestamp'])

    features = []
    count = 0
    for i in catalog.index:
        count += 1
        print(f'Complete {count*100 / len(catalog) : 1f}%', end = '\r')
        # define time-series window
        window_st = catalog.loc[i, 'start_time'] - pd.Timedelta(minutes = itv_front)
        window_end = catalog.loc[i, 'start_time'] + pd.Timedelta(minutes = itv_back)
        timeseries = dataset.loc[dataset['Timestamp'].between(window_st, window_end), ['Timestamp', 'B_AVG', 'B_filtered']]
        
        # remove flare x-ray flux other than flare of interest
        st = catalog.loc[i, 'start_time']
        end = catalog.loc[i, 'end_time']

        timeseries.loc[~timeseries['Timestamp'].between(st, end), 'B_AVG'] = None # only X-ray flux during interval of flare of interest
        timeseries = timeseries.assign( data = timeseries[['B_AVG', 'B_filtered']].sum(axis=1, min_count=1) )
        
        if timeseries['data'].isnull().any():
            # print(timeseries)
            # plt.plot(timeseries['data'])
            # plt.show()
            timeseries['data'] = timeseries['data'].interpolate(method = 'linear')
            # print(timeseries['data'])
            # break

        features.append([timeseries['data'], catalog.loc[i, 'goes_class'], catalog.loc[i, 'relative_X-ray_flux_increase'], catalog.loc[i, 'cme_vel']])

    df = pd.DataFrame(features)
    df.to_csv('FLTimeseries_CMEPred.csv', index = False)


df_xflux = pd.read_csv("/media/jh/maxone/Research/GSU/Research1_xray_flux/fixed_all_xrs_rxfi_2023.csv")
df_catal = pd.read_csv("/media/jh/maxone/Research/GSU/Research1_xray_flux/MultiwayIntegration_2010_to_2018_conf_rxfi.csv")

if __name__ == '__main__':
    TimeSeries_window(dataset = df_xflux, catalog = df_catal)

