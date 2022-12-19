---
layout: post
title: Bike Sharing Demand
subtitle: 공유 자전거 사용량 예측
categories: Data Analysis
tags: [Data Analysis, EDA]
---

공유 자전거 사용량 예측 [[Kaggle]](https://www.kaggle.com/competitions/bike-sharing-demand/overview)

## 목표

- 워싱턴 D.C.의 Capital Bikeshare 수요를 시간대 별로 예측
- 자전거 대여 수요를 예측하기 위해 과거 사용 패턴과 날씨 데이터를 분석

## 데이터 필드

- 과거 사용 패턴: 시간(hour), 휴일 여부
- 날씨 데이터: 계절, 날씨, 온도, 체감 온도, 습도, 풍속

| 필드 | 내용 |
|-|-|
| `datetime` | 날짜 + 시간 타임스탬프 |
| `season` | 1: 봄, 2: 여름, 3: 가을, 4: 겨울 |
| `holiday` | 휴일 여부 |
| `weateher` | 1: Clear, Few clouds, Partly cloudy, Partly cloudy<br>2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist<br>3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds<br>4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
| `temp` | 섭씨 온도 |
| `atemp` | 섭씨 체감 온도 |
| `humidity` | 습도 |
| `windspeed` | 풍속 |
| **`casual`** | 비회원 대여량 |
| **`registered`** | 회원 대여량 |
| **`count`** | 총 대여량 |

`casual`, `registered`, `count` 필드를 예측해야 한다.

## EDA & 예측

### 라이브러리 로드


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
```


```python
%matplotlib inline
plt.rc("font", family="Malgun Gothic")
```

### 데이터셋 로드


```python
## This Python 3 environment comes with many helpful analytics libraries installed
## It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
## For example, here's several helpful packages to load

import numpy as np ## linear algebra
import pandas as pd ## data processing, CSV file I/O (e.g. pd.read_csv)

## Input data files are available in the read-only "../input/" directory
## For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

## You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
## You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
```

    /kaggle/input/bike-sharing-demand/sampleSubmission.csv
    /kaggle/input/bike-sharing-demand/train.csv
    /kaggle/input/bike-sharing-demand/test.csv



```python
train = pd.read_csv("/kaggle/input/bike-sharing-demand/train.csv", parse_dates=['datetime'])
test = pd.read_csv("/kaggle/input/bike-sharing-demand/test.csv", parse_dates=['datetime'])
sample = pd.read_csv("/kaggle/input/bike-sharing-demand/sampleSubmission.csv", parse_dates=['datetime'])
```

### 데이터셋 요약

#### 데이터 Shape


```python
print(train.shape, test.shape, sample.shape)
```

    (10886, 12) (6493, 9) (6493, 2)


#### 데이터 필드


```python
train.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 10886 entries, 0 to 10885
    Data columns (total 12 columns):
     ##   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    10886 non-null  datetime64[ns]
     1   season      10886 non-null  int64         
     2   holiday     10886 non-null  int64         
     3   workingday  10886 non-null  int64         
     4   weather     10886 non-null  int64         
     5   temp        10886 non-null  float64       
     6   atemp       10886 non-null  float64       
     7   humidity    10886 non-null  int64         
     8   windspeed   10886 non-null  float64       
     9   casual      10886 non-null  int64         
     10  registered  10886 non-null  int64         
     11  count       10886 non-null  int64         
    dtypes: datetime64[ns](1), float64(3), int64(8)
    memory usage: 1020.7 KB



```python
test.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 6493 entries, 0 to 6492
    Data columns (total 9 columns):
     ##   Column      Non-Null Count  Dtype         
    ---  ------      --------------  -----         
     0   datetime    6493 non-null   datetime64[ns]
     1   season      6493 non-null   int64         
     2   holiday     6493 non-null   int64         
     3   workingday  6493 non-null   int64         
     4   weather     6493 non-null   int64         
     5   temp        6493 non-null   float64       
     6   atemp       6493 non-null   float64       
     7   humidity    6493 non-null   int64         
     8   windspeed   6493 non-null   float64       
    dtypes: datetime64[ns](1), float64(3), int64(5)
    memory usage: 456.7 KB


학습 데이터에 존재하는 `casual`, `registered`, `count` 필드가 테스트 데이터에는 없다.

`sampleSubmission.csv`에 따르면 날짜 및 시간대 별로 `count`를 예측해야 한다.

#### 데이터 샘플


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



#### 결측치 확인


```python
train.isnull().sum()
```




    datetime      0
    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    casual        0
    registered    0
    count         0
    dtype: int64




```python
test.isnull().sum()
```




    datetime      0
    season        0
    holiday       0
    workingday    0
    weather       0
    temp          0
    atemp         0
    humidity      0
    windspeed     0
    dtype: int64




```python
fig, (ax1, ax2, ax3, ax4) = plt.subplots(ncols=4)
fig.set_size_inches(20, 10)
sns.regplot(data=train, x='temp', y='count', ax=ax1)
sns.regplot(data=train, x='atemp', y='count', ax=ax2)
sns.regplot(data=train, x='windspeed', y='count', ax=ax3)
sns.regplot(data=train, x='humidity', y='count', ax=ax4)
```




    <AxesSubplot:xlabel='humidity', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_19_1.png)
    



```python
print(len(train[train['windspeed'] == 0]), str(len(train[train['windspeed'] == 0])/len(train)*100)+"%")
```

    1313 12.061363218813154%


`windspeed`의 약 12%가 0에 분포하며, 다음 구간에 분포가 비어 있다.

결측치가 0으로 기입되어 있다고 가정할 수 있다.

### EDA

#### 시기별 대여량


```python
def build_datetime_features(df):
    ## 날짜 및 시간 피처 생성
    df['season_str'] = df['season'].map({1: "Spring", 2: "Summer", 3: "Fall", 4: "Winter"})
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['weekday'] = df['datetime'].dt.dayofweek
    df['hour'] = df['datetime'].dt.hour
    return df
```


```python
train = build_datetime_features(train)
```


```python
train.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>season</th>
      <th>holiday</th>
      <th>workingday</th>
      <th>weather</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>casual</th>
      <th>registered</th>
      <th>count</th>
      <th>season_str</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-01 00:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>81</td>
      <td>0.0</td>
      <td>3</td>
      <td>13</td>
      <td>16</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-01 01:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>8</td>
      <td>32</td>
      <td>40</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-01 02:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.02</td>
      <td>13.635</td>
      <td>80</td>
      <td>0.0</td>
      <td>5</td>
      <td>27</td>
      <td>32</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-01 03:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>3</td>
      <td>10</td>
      <td>13</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-01 04:00:00</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>9.84</td>
      <td>14.395</td>
      <td>75</td>
      <td>0.0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>




```python
## Barplot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
fig.set_size_inches(20, 15)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

ax1.set(title="Rental by year")
sns.barplot(data=train, x='year', y='count', orient='v', ax=ax1)

ax2.set(title="Rental by season")
sns.barplot(data=train, x='season_str', y='count', ax=ax2)

ax3.set(title="Monthly rental")
sns.barplot(data=train, x='month', y='count', ax=ax3)

ax4.set(title="Daily rental")
sns.barplot(data=train, x='day', y='count', ax=ax4)

ax5.set(title="Hourly rental")
sns.barplot(data=train, x='hour', y='count', ax=ax5)

ax6.set(title="Rental by day of week")
sns.barplot(data=train, x='weekday', y='count', ax=ax6,
            palette=['gray', 'gray', 'gray', 'gray', 'gray', 'blue', 'red'])

ax7.set(title="Weekday/holiday rental")
sns.barplot(data=train, x='holiday', y='count', ax=ax7)

ax8.set(title="Rental by weather")
sns.barplot(data=train, x='weather', y='count', ax=ax8)
```




    <AxesSubplot:title={'center':'Rental by weather'}, xlabel='weather', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_27_1.png)
    



```python
## Boxplot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
fig.set_size_inches(20, 20)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

ax1.set(title="Rental by year")
sns.boxplot(data=train, x='year', y='count', ax=ax1)

ax2.set(title="Rental by season")
sns.boxplot(data=train, x='season_str', y='count', ax=ax2)

ax3.set(title="Monthly rental")
sns.boxplot(data=train, x='month', y='count', ax=ax3)

ax4.set(title="Daily rental")
sns.boxplot(data=train, x='day', y='count', ax=ax4)

ax5.set(title="Hourly rental")
sns.boxplot(data=train, x='hour', y='count', ax=ax5)

ax6.set(title="Rental by day of week")
sns.boxplot(data=train, x='weekday', y='count', ax=ax6,
            palette=['gray', 'gray', 'gray', 'gray', 'gray', 'blue', 'red'])

ax7.set(title="Weekday/holiday rental")
sns.boxplot(data=train, x='holiday', y='count', ax=ax7)

ax8.set(title="Rental by weather")
sns.boxplot(data=train, x='weather', y='count', ax=ax8)
```




    <AxesSubplot:title={'center':'Rental by weather'}, xlabel='weather', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_28_1.png)
    



```python
## 시간대별 대여량
fig, (ax1, ax2, ax3, ax4, ax5, ax6) = plt.subplots(nrows=6)
fig.set_size_inches(20, 20)

sns.pointplot(data=train, x='hour', y='count', ax=ax1)
sns.pointplot(data=train, x='hour', y='count', hue='workingday', ax=ax2)
sns.pointplot(data=train, x='hour', y='count', hue='holiday', ax=ax3)
sns.pointplot(data=train, x='hour', y='count', hue='weekday', ax=ax4)
sns.pointplot(data=train, x='hour', y='count', hue='weather', ax=ax5)
sns.pointplot(data=train, x='hour', y='count', hue='season', ax=ax6)
```




    <AxesSubplot:xlabel='hour', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_29_1.png)
    


- 연도별 대여량: 2011 < 2012
- 계절별 대여량: 가을 > 여름 > 겨울 > 봄
- 월별 대여량: 6월 > 7~9월 > 10월 > 5월 > 11월 > 4월 > 12월 > 3월 > 2월 > 1월
- 시간별 대여량: 출퇴근 시간대 대여량과 편차가 큼
- 시기별 대여량: `workingday` 0과 토요일, 일요일은 비슷한 추세를 보이며, 출퇴근 시간대의 영향을 받지 않음


```python
print(train[train['season'] == 1]['month'].unique())
print(train[train['season'] == 2]['month'].unique())
print(train[train['season'] == 3]['month'].unique())
print(train[train['season'] == 4]['month'].unique())
```

    [1 2 3]
    [4 5 6]
    [7 8 9]
    [10 11 12]


`season`은 사전적 의미의 계절이 아니라 분기를 의미한다.

#### 이상치 제거


```python
train_normalized = train[np.abs(train['count'] - train['count'].mean()) <= (3*train['count'].std())]
```


```python
print("Shape of before normalization: ", train.shape)
print("Shape of after normalization: ", train_normalized.shape)
```

    Shape of before normalization:  (10886, 18)
    Shape of after normalization:  (10739, 18)



```python
## Boxplot
fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6), (ax7, ax8)) = plt.subplots(nrows=4, ncols=2)
fig.set_size_inches(20, 20)
plt.subplots_adjust(wspace=0.2, hspace=0.3)

ax1.set(title="Rental by year")
sns.boxplot(data=train_normalized, x='year', y='count', ax=ax1)

ax2.set(title="Rental by season")
sns.boxplot(data=train_normalized, x='season_str', y='count', ax=ax2)

ax3.set(title="Monthly rental")
sns.boxplot(data=train_normalized, x='month', y='count', ax=ax3)

ax4.set(title="Daily rental")
sns.boxplot(data=train_normalized, x='day', y='count', ax=ax4)

ax5.set(title="Hourly rental")
sns.boxplot(data=train_normalized, x='hour', y='count', ax=ax5)

ax6.set(title="Rental by day of week")
sns.boxplot(data=train_normalized, x='weekday', y='count', ax=ax6,
            palette=['gray', 'gray', 'gray', 'gray', 'gray', 'blue', 'red'])

ax7.set(title="Weekday/holiday rental")
sns.boxplot(data=train_normalized, x='holiday', y='count', ax=ax7)

ax8.set(title="Rental by weather")
sns.boxplot(data=train_normalized, x='weather', y='count', ax=ax8)
```




    <AxesSubplot:title={'center':'Rental by weather'}, xlabel='weather', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_36_1.png)
    


#### 결측치 보정

`windspeed`의 결측치는 0으로 되어 있다. 0인 것들의 `windspeed`를 예측하여 보정한다.


```python
from sklearn.ensemble import RandomForestClassifier
```


```python
wind0 = train.loc[train['windspeed'] == 0]
wind_not0 = train.loc[train['windspeed'] != 0]
```


```python
print("Number of rows with 0 windspeed before prediction: ", len(wind0))
```

    Number of rows with 0 windspeed before prediction:  1313



```python
## windspeed와의 상관계수 절대값 내림차순
corr = wind_not0.corr()[['windspeed']]
corr.rename(columns={'windspeed': 'corr'}, inplace=True)
corr['corr_abs'] = corr['corr'].abs()
corr.sort_values(by='corr_abs', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr</th>
      <th>corr_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>windspeed</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>-0.328272</td>
      <td>0.328272</td>
    </tr>
    <tr>
      <th>month</th>
      <td>-0.142505</td>
      <td>0.142505</td>
    </tr>
    <tr>
      <th>season</th>
      <td>-0.138272</td>
      <td>0.138272</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.126289</td>
      <td>0.126289</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>0.085342</td>
      <td>0.085342</td>
    </tr>
    <tr>
      <th>count</th>
      <td>0.085014</td>
      <td>0.085014</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>0.073669</td>
      <td>0.073669</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>-0.068576</td>
      <td>0.068576</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>-0.038902</td>
      <td>0.038902</td>
    </tr>
    <tr>
      <th>year</th>
      <td>-0.035825</td>
      <td>0.035825</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>-0.030849</td>
      <td>0.030849</td>
    </tr>
    <tr>
      <th>workingday</th>
      <td>0.021188</td>
      <td>0.021188</td>
    </tr>
    <tr>
      <th>holiday</th>
      <td>0.015603</td>
      <td>0.015603</td>
    </tr>
    <tr>
      <th>weather</th>
      <td>-0.011837</td>
      <td>0.011837</td>
    </tr>
    <tr>
      <th>day</th>
      <td>0.009141</td>
      <td>0.009141</td>
    </tr>
  </tbody>
</table>
</div>




```python
def predict_windspeed(df):
    df_wind0 = df.loc[df['windspeed'] == 0]
    df_wind_not0 = df.loc[df['windspeed'] != 0]
    
    columns = ['humidity', 'month', 'hour', 'season', 'weather', 'atemp', 'temp']
    
    rf_model = RandomForestClassifier()
    rf_model.fit(df_wind_not0[columns], df_wind_not0['windspeed'].astype('str'))
    rf_prediction = rf_model.predict(df_wind0[columns])
    df_wind0['windspeed'] = rf_prediction
    
    result = df_wind_not0.append(df_wind0)
    result.reset_index(inplace=True)
    result.drop('index', inplace=True, axis=1)
    
    return result
```


```python
train_before_wind = train.copy()
train = predict_windspeed(train)
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ## Remove the CWD from sys.path while we load stuff.



```python
print("Number of rows with 0 windspeed after prediction: ", len(train[train['windspeed'] == 0]))
```

    Number of rows with 0 windspeed after prediction:  0



```python
fig, (ax1, ax2) = plt.subplots(nrows=2)
fig.set_size_inches(20, 15)

sns.countplot(data=train_before_wind, x='windspeed', ax=ax1)
sns.countplot(data=train, x='windspeed', ax=ax2)
```




    <AxesSubplot:xlabel='windspeed', ylabel='count'>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_45_1.png)
    


#### Skewness & Kurtosis
Skewness(왜도)와 Kurtosis(첨도)를 통해 데이터 분포의 치우침을 확인하고 보정한다.


```python
print("Skewness: ", train['count'].skew())
print("Kurtosis: ", train['count'].kurt())
```

    Skewness:  1.242066211718077
    Kurtosis:  1.3000929518398299



```python
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(20, 10)

sns.distplot(train['count'], ax=ax1)
stats.probplot(train['count'], dist="norm", fit=True, plot=ax2)
```

    /opt/conda/lib/python3.7/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)





    ((array([-3.83154229, -3.60754977, -3.48462983, ...,  3.48462983,
              3.60754977,  3.83154229]),
      array([  1,   1,   1, ..., 968, 970, 977])),
     (169.82942673231386, 191.5741319125482, 0.9372682766213176))




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_48_2.png)
    


#### 로그 스케일 정규화

Skewness의 쏠림이 있으므로 로그 스케일 정규화를 할 것이다.


```python
train['count_log'] = np.log(train['count'])
```


```python
print("Skewness: ", train['count_log'].skew())
print("Kurtosis: ", train['count_log'].kurt())
```

    Skewness:  -0.9712277227866108
    Kurtosis:  0.24662183416964067



```python
fig, (ax1, ax2) = plt.subplots(ncols=2)
fig.set_size_inches(20, 10)

sns.distplot(train['count_log'], ax=ax1)
stats.probplot(np.log1p(train['count']), dist="norm", fit=True, plot=ax2)
```

    /opt/conda/lib/python3.7/site-packages/seaborn/distributions.py:2619: FutureWarning: `distplot` is a deprecated function and will be removed in a future version. Please adapt your code to use either `displot` (a figure-level function with similar flexibility) or `histplot` (an axes-level function for histograms).
      warnings.warn(msg, FutureWarning)





    ((array([-3.83154229, -3.60754977, -3.48462983, ...,  3.48462983,
              3.60754977,  3.83154229]),
      array([0.69314718, 0.69314718, 0.69314718, ..., 6.87626461, 6.87832647,
             6.88550967])),
     (1.3647396459244172, 4.591363690454027, 0.9611793780126952))




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_52_2.png)
    


#### One-hot encoding


```python
print(train['weather'].unique())
print(train['season'].unique())
print(train['workingday'].unique())
print(train['holiday'].unique())
```

    [2 1 3 4]
    [1 2 3 4]
    [0 1]
    [0 1]



```python
def one_hot_encoding(df):
    df = pd.get_dummies(df, columns=['weather'], prefix='weather')
    df = pd.get_dummies(df, columns=['season'], prefix='season')
    df = pd.get_dummies(df, columns=['workingday'], prefix='workingday')
    df = pd.get_dummies(df, columns=['holiday'], prefix='holiday')
    return df
```


```python
train_before_encoding = train.copy()
train = one_hot_encoding(train)
```


```python
train.columns
```




    Index(['datetime', 'temp', 'atemp', 'humidity', 'windspeed', 'casual',
           'registered', 'count', 'season_str', 'year', 'month', 'day', 'weekday',
           'hour', 'count_log', 'weather_1', 'weather_2', 'weather_3', 'weather_4',
           'season_1', 'season_2', 'season_3', 'season_4', 'workingday_0',
           'workingday_1', 'holiday_0', 'holiday_1'],
          dtype='object')



#### 상관관계 분석


```python
## Pearson 상관계수 히트맵 시각화
fix, (ax1, ax2) = plt.subplots(figsize=(20, 30), nrows=2)
sns.heatmap(train_before_encoding.corr(), annot=True, fmt=".2f", cmap="BuPu", ax=ax1)
sns.heatmap(train.corr(), annot=True, fmt=".2f", cmap="Blues", ax=ax2)
```




    <AxesSubplot:>




    
![png](/assets/visualizations/bike-sharing-demand_files/bike-sharing-demand_59_1.png)
    



```python
## count와의 상관계수 절대값 내림차순
corr = train.corr()[['count']]
corr.rename(columns={'count': 'corr'}, inplace=True)
corr['corr_abs'] = corr['corr'].abs()
corr.sort_values(by='corr_abs', ascending=False)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>corr</th>
      <th>corr_abs</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>1.000000</td>
      <td>1.000000</td>
    </tr>
    <tr>
      <th>registered</th>
      <td>0.970948</td>
      <td>0.970948</td>
    </tr>
    <tr>
      <th>count_log</th>
      <td>0.805773</td>
      <td>0.805773</td>
    </tr>
    <tr>
      <th>casual</th>
      <td>0.690414</td>
      <td>0.690414</td>
    </tr>
    <tr>
      <th>hour</th>
      <td>0.400601</td>
      <td>0.400601</td>
    </tr>
    <tr>
      <th>temp</th>
      <td>0.394454</td>
      <td>0.394454</td>
    </tr>
    <tr>
      <th>atemp</th>
      <td>0.389784</td>
      <td>0.389784</td>
    </tr>
    <tr>
      <th>humidity</th>
      <td>-0.317371</td>
      <td>0.317371</td>
    </tr>
    <tr>
      <th>year</th>
      <td>0.260403</td>
      <td>0.260403</td>
    </tr>
    <tr>
      <th>season_1</th>
      <td>-0.237704</td>
      <td>0.237704</td>
    </tr>
    <tr>
      <th>month</th>
      <td>0.166862</td>
      <td>0.166862</td>
    </tr>
    <tr>
      <th>season_3</th>
      <td>0.136942</td>
      <td>0.136942</td>
    </tr>
    <tr>
      <th>weather_3</th>
      <td>-0.117519</td>
      <td>0.117519</td>
    </tr>
    <tr>
      <th>weather_1</th>
      <td>0.105246</td>
      <td>0.105246</td>
    </tr>
    <tr>
      <th>season_2</th>
      <td>0.075681</td>
      <td>0.075681</td>
    </tr>
    <tr>
      <th>weather_2</th>
      <td>-0.041329</td>
      <td>0.041329</td>
    </tr>
    <tr>
      <th>season_4</th>
      <td>0.023704</td>
      <td>0.023704</td>
    </tr>
    <tr>
      <th>day</th>
      <td>0.019826</td>
      <td>0.019826</td>
    </tr>
    <tr>
      <th>workingday_1</th>
      <td>0.011594</td>
      <td>0.011594</td>
    </tr>
    <tr>
      <th>workingday_0</th>
      <td>-0.011594</td>
      <td>0.011594</td>
    </tr>
    <tr>
      <th>holiday_1</th>
      <td>-0.005393</td>
      <td>0.005393</td>
    </tr>
    <tr>
      <th>holiday_0</th>
      <td>0.005393</td>
      <td>0.005393</td>
    </tr>
    <tr>
      <th>weekday</th>
      <td>-0.002283</td>
      <td>0.002283</td>
    </tr>
    <tr>
      <th>weather_4</th>
      <td>-0.001459</td>
      <td>0.001459</td>
    </tr>
  </tbody>
</table>
</div>



### 모델

#### 피처 엔지니어링

EDA 과정에서 Train 데이터에 행한 과정을 Test 데이터에도 적용한다.


```python
test = build_datetime_features(test)
test = predict_windspeed(test)
test = one_hot_encoding(test)
```

    /opt/conda/lib/python3.7/site-packages/ipykernel_launcher.py:10: SettingWithCopyWarning: 
    A value is trying to be set on a copy of a slice from a DataFrame.
    Try using .loc[row_indexer,col_indexer] = value instead
    
    See the caveats in the documentation: https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
      ## Remove the CWD from sys.path while we load stuff.



```python
test.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>temp</th>
      <th>atemp</th>
      <th>humidity</th>
      <th>windspeed</th>
      <th>season_str</th>
      <th>year</th>
      <th>month</th>
      <th>day</th>
      <th>weekday</th>
      <th>...</th>
      <th>weather_3</th>
      <th>weather_4</th>
      <th>season_1</th>
      <th>season_2</th>
      <th>season_3</th>
      <th>season_4</th>
      <th>workingday_0</th>
      <th>workingday_1</th>
      <th>holiday_0</th>
      <th>holiday_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>10.66</td>
      <td>11.365</td>
      <td>56</td>
      <td>26.0027</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 03:00:00</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 04:00:00</td>
      <td>10.66</td>
      <td>12.880</td>
      <td>56</td>
      <td>11.0014</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 05:00:00</td>
      <td>9.84</td>
      <td>11.365</td>
      <td>60</td>
      <td>15.0013</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 06:00:00</td>
      <td>9.02</td>
      <td>10.605</td>
      <td>60</td>
      <td>15.0013</td>
      <td>Spring</td>
      <td>2011</td>
      <td>1</td>
      <td>20</td>
      <td>3</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



#### 필드 선택

1. `count`와 상관계수가 높은 필드
2. 의미가 중복되는 컬럼은 덜 분산된 필드 선택
  - ex. `workingday`와 `holiday`는 부의 상관관계가 있으나, `workingday`의 분산이 작으므로 `workingday` 선택


```python
test_datetime = test['datetime']
train.drop(['datetime', 'season_str', 'holiday_0', 'holiday_1', 'atemp', 'registered', 'casual'], axis=1, inplace=True)
test.drop(['datetime', 'season_str', 'holiday_0', 'holiday_1', 'atemp'], axis=1, inplace=True)
```


```python
print(train.columns)
print(test.columns)
```

    Index(['temp', 'humidity', 'windspeed', 'count', 'year', 'month', 'day',
           'weekday', 'hour', 'count_log', 'weather_1', 'weather_2', 'weather_3',
           'weather_4', 'season_1', 'season_2', 'season_3', 'season_4',
           'workingday_0', 'workingday_1'],
          dtype='object')
    Index(['temp', 'humidity', 'windspeed', 'year', 'month', 'day', 'weekday',
           'hour', 'weather_1', 'weather_2', 'weather_3', 'weather_4', 'season_1',
           'season_2', 'season_3', 'season_4', 'workingday_0', 'workingday_1'],
          dtype='object')


#### Gradient boosting


```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn import metrics
```


```python
x_train = train.drop(['count_log', 'count'], axis=1).values
target_label = train['count_log'].values
x_test = test.values
```


```python
x_train, x_val, y_train, y_val = train_test_split(x_train, target_label, test_size=0.2, random_state=2000)
```


```python
x_train
```




    array([[14.76, 50, 16.9979, ..., 1, 0, 1],
           [33.62, 43, 19.9995, ..., 0, 1, 0],
           [31.16, 58, 19.0012, ..., 0, 0, 1],
           ...,
           [22.96, 37, 19.0012, ..., 0, 0, 1],
           [18.86, 63, 8.9981, ..., 1, 0, 1],
           [17.22, 38, 19.9995, ..., 0, 0, 1]], dtype=object)




```python
gbr_model = GradientBoostingRegressor(
    n_estimators=2000,
    learning_rate=0.05,
    max_depth=5,
    min_samples_leaf=15,
    min_samples_split=10,
    random_state=42
)
gbr_model.fit(x_train, y_train)
```




    GradientBoostingRegressor(learning_rate=0.05, max_depth=5, min_samples_leaf=15,
                              min_samples_split=10, n_estimators=2000,
                              random_state=42)



##### Validation


```python
train_score = gbr_model.score(x_train, y_train)
validation_score = gbr_model.score(x_val, y_val)
print(train_score, validation_score)
```

    0.9866447588995861 0.957037659130984


#### 자전거 수요 예측


```python
gbr_prediction = gbr_model.predict(x_test)
predicted_count = np.exp(gbr_prediction)
```


```python
sample.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 01:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 02:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 03:00:00</td>
      <td>0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 04:00:00</td>
      <td>0</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission = pd.DataFrame()
submission['datetime'] = test_datetime
submission['count'] = predicted_count
```


```python
submission.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>datetime</th>
      <th>count</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011-01-20 00:00:00</td>
      <td>13.892524</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2011-01-20 03:00:00</td>
      <td>2.242351</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011-01-20 04:00:00</td>
      <td>2.509201</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2011-01-20 05:00:00</td>
      <td>5.775774</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2011-01-20 06:00:00</td>
      <td>31.641682</td>
    </tr>
  </tbody>
</table>
</div>




```python
submission.to_csv("bike.csv", index=False)
```
