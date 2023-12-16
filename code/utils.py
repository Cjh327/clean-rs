import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, data, doc_info, user_info):
        click_counts = data['click'].value_counts()
        print(click_counts)
        print(f'Click pos ratio: {click_counts[1] / (click_counts[0] + click_counts[1])}')
        
        data, self.user2index, self.doc2index = self.transform_ids(data, doc_info, user_info)
        self.feat_dict = self.generate_features(data, doc_info, user_info)
        self.label_dict = self.generate_labels(data)
        
    def transform_ids(self, data, doc_info, user_info):
        df_reset = user_info['user_id'].drop_duplicates().reset_index()
        user2index = df_reset.set_index('user_id')['index'].to_dict()
        
        df_reset = doc_info['article_id'].drop_duplicates().reset_index()
        doc2index = df_reset.set_index('article_id')['index'].to_dict()
        
        data['user_id'] = data['user_id'].map(user2index)
        data['article_id'] = data['article_id'].map(doc2index)
        
        return data, user2index, doc2index
        
    def generate_features(self, data, doc_info, user_info):
        uids = data['user_id'].values.astype(int)
        gids = data['article_id'].values.astype(int)
        metadata = data[['net_status', 'flush_nums', 'expo_position']].values.astype(np.float32)
        return {'uid': uids, 'gid': gids, 'metadata': metadata}
    
    def generate_labels(self, data):
        click = data['click'].values.astype(np.float32)
        duration = data['duration'].values.astype(np.float32)
        return {'click': click, 'duration': duration}
    
    def __getitem__(self, idx):
        return self.feat_dict['uid'][idx], self.feat_dict['gid'][idx], self.feat_dict['metadata'][idx], self.label_dict['click'][idx]
    
    def __len__(self):
        return len(self.feat_dict['uid'])


def read_processed_data(datadir, small=False):
    doc_info = pd.read_csv(os.path.join(datadir, 'doc_info_proceessed.csv'), low_memory=False)
    user_info = pd.read_csv(os.path.join(datadir, 'user_info_proceessed.csv'))
    if small:
        print('Small data set is used.')
        train_data = pd.read_csv(os.path.join(datadir, 'train_data_proceessed_small.csv'))
    else:
        train_data = pd.read_csv(os.path.join(datadir, 'train_data_proceessed.csv'))
        train_data.sample(100000).to_csv(os.path.join(datadir, 'train_data_proceessed_small.csv'), index=False)
    
    print(doc_info.shape, user_info.shape, train_data.shape)
    
    return doc_info, user_info, train_data



if __name__ == '__main__':
    read_processed_data('dataset/processed_data')