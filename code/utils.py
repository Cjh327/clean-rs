import os
from datetime import datetime
from tqdm import tqdm
import pandas as pd
import numpy as np


def read_news_dataset(datadir):
    doc_info_path = os.path.join(datadir, 'doc_info.txt')
    user_info_path = os.path.join(datadir, 'user_info.txt')
    train_data_path = os.path.join(datadir, 'train_data.txt')
    test_data_path = os.path.join(datadir, 'test_data.txt')
    
    # # Read meta data
    doc_info = pd.read_csv(doc_info_path, delimiter='\t', names=['article_id', 'title', 'ctime', 'img_num', 'category_1', 'category_2', 'key_words'], low_memory=False)
    user_info = pd.read_csv(user_info_path, delimiter='\t', names=['user_id', 'device', 'os', 'province', 'city', 'age', 'gender'])
    
    print(f'doc_info: {doc_info.shape}')
    print(f'user_info: {user_info.shape}')
    
    doc_info = process_doc_info(doc_info)
    user_info = process_user_info(user_info)

    # Read log data
    train_data_reader = pd.read_csv(train_data_path, delimiter='\t', chunksize=100000, iterator=True, 
                             names=['user_id', 'article_id', 'expo_time', 'net_status', 'flush_nums', 'exop_position', 'click', 'duration'])
    
    
    article_ids = set(doc_info['article_id'])

    for idx, chunk in tqdm(enumerate(train_data_reader)):
        chunk = process_train_data_chunk(chunk, article_ids, doc_info)
        if idx == 0:
            print(chunk)
            chunk.to_csv(os.path.join(datadir, 'train_data_proceessed.csv'),index=False)
        else:
            chunk.to_csv(os.path.join(datadir, 'train_data_proceessed.csv'),index=False, mode = 'a',header = False)

    test_data = pd.read_csv(test_data_path, delimiter='\t', names=['index', 'user_id', 'article_id', 'expo_time', 'net_status', 'flush_nums'])
    print(f'test_data: {test_data.shape}')
    test_data = process_test_data(test_data)

def process_train_data_chunk(chunk, article_ids, doc_info, start_time='2021-06-30 00:00:00'):
    # 根据时间筛选， 只用后七天的数据
    chunk['expo_time'] = chunk['expo_time'].apply(lambda x: datetime.fromtimestamp(x/1000) \
                                                        .strftime('%Y-%m-%d %H:%M:%S'))
    chunk['expo_time'] = pd.to_datetime(chunk['expo_time'])
    chunk = chunk[chunk['expo_time'] >= start_time]
    
    # 去掉点击的文章不在总doc里面的记录
    chunk = chunk[chunk['article_id'].isin(article_ids)]
    
    # 拼接上doc的ctime， 然后去掉曝光时间小于上传时间的
    chunk = chunk.merge(doc_info[['article_id', 'ctime']], on='article_id', how='left')
    chunk = chunk[chunk['expo_time'] > chunk['ctime']]
    del chunk['ctime']
    
    # 标签
    chunk = chunk[chunk['click'].isin([0, 1])]
    
    # duration
    chunk = chunk[~((chunk['click']==1) & (chunk['duration']<3))]
    
    return chunk

def process_doc_info(df):
    print('\nProcessing doc_info...\nBefore:')
    print(df.head())
    print(df.isnull().sum())
    
    # Process ctime
    df['ctime'] = df['ctime'].str.replace('Android', '1625400960000')
    df['ctime'].fillna('1625400960000', inplace=True)
    df['ctime'] = df['ctime'].apply(lambda x: datetime.fromtimestamp(int(x)/1000) \
                                                            .strftime('%Y-%m-%d %H:%M:%S'))
    df['ctime'] = pd.to_datetime(df['ctime'])
    
    # Process missing values
    df['img_num'].fillna(0.0, inplace=True)
    df['category_1'].fillna(df['category_1'].mode()[0], inplace=True) # Fill with the most common value
    df['category_2'].fillna(df['category_2'].mode()[0], inplace=True) # Fill with the most common value
    
    print('After:')
    print(df.head())
    print(df.isnull().sum())
    
    return df

def process_user_info(df):
    print('\nProcessing user_info...\nBefore:')
    print(df.head())
    print(df.isnull().sum())
    
    def get_age_gender(x):
        if pd.isna(x):
            return x
        x_list = x.split(',')
        age_stage_val = list(map(lambda x: x.split(':'), x_list))
        age_stage = list(map(lambda x: x[0], age_stage_val))
        age_val = list(map(lambda x: x[1], age_stage_val))
        return age_stage[np.argmax(age_val)]

    df['age'] = df['age'].apply(lambda x: get_age_gender(x))
    df['gender'] = df['gender'].apply(lambda x: get_age_gender(x))
    
    df.fillna('nan', inplace=True)

    print('After:')
    print(df.head())
    print(df.isnull().sum())

    return df


def process_test_data(df):
    print('\nProcessing test_data...\nBefore:')
    print(df.head())
    print(df.isnull().sum())
    
    df['expo_time'] = df['expo_time'].apply(lambda x: datetime.fromtimestamp(int(x)/1000) \
                                                            .strftime('%Y-%m-%d %H:%M:%S'))
    
    print('After:')
    print(df.head())
    print(df.isnull().sum())
    
    return df


if __name__ == '__main__':
    read_news_dataset('dataset/raw_data')