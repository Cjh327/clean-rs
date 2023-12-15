import os
import pandas as pd
from torch.utils.data import Dataset


class NewsDataset(Dataset):
    def __init__(self, data, doc_info, user_info):
        self.features = self.generate_features(data, doc_info, user_info).values
        self.click_label = data['click'].values
        self.duration_label = data['duration'].values
        
    def generate_features(self, data, doc_info, user_info):
        features = data[['user_id', 'article_id', 'expo_time', 'net_status', 'flush_nums', 'exop_position']]
        return features

    def __getitem__(self, idx):
        return self.features[idx], self.click_label[idx], self.duration_label[idx]

    def __len__(self):
        return len(self.features)


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