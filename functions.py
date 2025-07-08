import pandas as pd
import numpy as np

# 시드 설정 (재현 가능한 결과)
np.random.seed(42)

from matplotlib import rc
rc('font', family = 'Malgun Gothic')

character_dict = {'cooler':'cooler condition(%)','valve':'valve condition(%)','pump':'internal pump leakage','accumulator':'hydraulic accumulator(bar)'}
sensor_list = ['CE','CP','EPS1','FS1','FS2','PS1','PS2','PS3','PS4','PS5','PS6','SE','TS1','TS2','TS3','TS4','VS1']
feature_list = ['mean', 'std', 'min', 'max', 'rms', 'p2p','slope', 'skew', 'kurtosis']


def slope_numpy_gradient(df):
    """numpy gradient 사용 (가장 빠름)"""
    if df.shape[1] < 2:
         return [0.0] * df.shape[0]
    gradients = np.gradient(df.values, axis=1)
    slopes = gradients.mean(axis=1)
    return slopes.tolist()

def feature_extract(sensor_name, num_windows, duplicate_ratio, window_tag, num_units, data_dict):
  sensor_length_dict = {'CE':60,'CP':60,'EPS1':6000,'FS1':600,'FS2':600,'PS1':6000,'PS2':6000,'PS3':6000,'PS4':6000,'PS5':6000,'PS6':6000,'SE':60,'TS1':60,'TS2':60,'TS3':60,'TS4':60,'VS1':60}
  df = data_dict[f'{sensor_name}_live_df']
  domain_length = sensor_length_dict[sensor_name]
  window_length = domain_length//num_windows
  unit_length = window_length//num_units

  result_dict = {feature : [] for feature in feature_list}
  window_range = [round(window_length*(window_tag-duplicate_ratio*0.01)) if window_tag != 0 else 0, round(window_length*(window_tag+1+0.01*duplicate_ratio)+1) if window_tag != num_windows -1 else domain_length]
  target_window_df = df.iloc[window_range[0]:window_range[1]]
  for unit_number in range(num_units):
    unit_range = [round(unit_length*(unit_number-duplicate_ratio*0.01)) if unit_number != 0 else 0, round(unit_length*(unit_number+1+0.01*duplicate_ratio)+1) if unit_number != num_units -1 else target_window_df.shape[1]]
    target_df = target_window_df.iloc[:,unit_range[0]:unit_range[1]]
    result_dict['mean'].append(target_df.mean().values)
    result_dict['std'].append(target_df.std().values)
    result_dict['min'].append(target_df.min().values)
    result_dict['max'].append(target_df.max().values)
    result_dict['rms'].append(np.sqrt((target_df**2).mean()).values)
    result_dict['p2p'].append((target_df.max() - target_df.min()).values)
    result_dict['slope'].append(slope_numpy_gradient(target_df.T))
    result_dict['skew'].append(target_df.skew().values)
    result_dict['kurtosis'].append(target_df.kurtosis().values)

  for feature in result_dict.keys():
    result_dict[feature] = pd.DataFrame(result_dict[feature]).T
    result_dict[feature].columns = list(map(lambda x : sensor_name + '_' + x + '_' + feature , [f'w{window_tag}_u{unit_number}' for unit_number in range(num_units)]))
  return result_dict

def feature_extracted_df(num_windows, duplicate_ratio, window_tag, num_units, data_dict) :
  df_list = []
  for sensor_name in sensor_list:
    df_list.append(pd.concat(feature_extract(sensor_name, num_windows, duplicate_ratio, window_tag, num_units, data_dict).values(),axis=1))
  return pd.concat(df_list,axis=1)

def prob_standard_select(sensor_name, num_windows, duplicate_ratio, window_tag):
  sensor_length_dict = {'CE':60,'CP':60,'EPS1':6000,'FS1':600,'FS2':600,'PS1':6000,'PS2':6000,'PS3':6000,'PS4':6000,'PS5':6000,'PS6':6000,'SE':60,'TS1':60,'TS2':60,'TS3':60,'TS4':60,'VS1':60}
  domain_length = sensor_length_dict[sensor_name]
  window_length = domain_length//num_windows
  window_range = [round(window_length*(window_tag-duplicate_ratio*0.01)) if window_tag != 0 else 0, round(window_length*(window_tag+1+0.01*duplicate_ratio)+1) if window_tag != num_windows -1 else domain_length]
  return window_range
