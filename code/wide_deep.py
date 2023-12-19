import argparse

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 
import torchmetrics

from torch.utils.data import DataLoader, random_split

from utils import read_processed_data, NewsDataset

class WideAndDeepModel(L.LightningModule):
    def __init__(self, u_size, g_size, metadata_dim, embed_dim, output_dim, lr=1e-3):
        super().__init__()
        self.u_embed = nn.Embedding(u_size, embed_dim)
        self.g_embed = nn.Embedding(g_size, embed_dim)
        self.wide = nn.Linear(embed_dim * 2 + metadata_dim, output_dim)
        self.deep = nn.Sequential(
                        nn.Linear(embed_dim * 2 + metadata_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, output_dim))
        self.sogmoid = nn.Sigmoid()
        self.lr = lr
        
        self.accuracy = torchmetrics.classification.BinaryAccuracy(threshold=0.5)
        self.auc = torchmetrics.classification.BinaryAUROC()

    def forward(self, uid, gid, metadata):
        u_embedding = self.u_embed(uid)
        g_embedding = self.g_embed(gid)
        
        X_wide = torch.cat([u_embedding, g_embedding, metadata], -1)
        X_deep = torch.cat([u_embedding, g_embedding, metadata], -1)
        
        wide_output = self.wide(X_wide)
        deep_output = self.deep(X_deep)
        y_out = self.sogmoid(wide_output + deep_output)
        return y_out
    
    def training_step(self, batch, batch_idx):
        uid, gid, metadata, y_click = batch # Lightning has turned the batch to tensor when reading dataloader
        y_hat = self(uid, gid, metadata)[:, 0]
        loss = F.binary_cross_entropy(y_hat, y_click)
        self.log_metrics('valid', y_hat, y_click)
        self.log('train/click_bce_loss', loss, prog_bar=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        uid, gid, metadata, y_click = batch
        y_hat = self(uid, gid, metadata)[:, 0]
        loss = F.binary_cross_entropy(y_hat, y_click)
        self.log_metrics('valid', y_hat, y_click)
        self.log('valid/click_bce_loss', loss, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    
    def log_metrics(self, prefix, y_hat, y_click):
        self.log(f'{prefix}/click_acc', self.accuracy(y_hat, y_click), on_epoch=True)
        self.log(f'{prefix}/click_auc', self.auc(y_hat, y_click), on_epoch=True)
    
def main(args):
    doc_info, user_info, data = read_processed_data('dataset/processed_data', args.dataset)
    dataset = NewsDataset(data, doc_info, user_info)
    
    # Split train validation set
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size])
    print(f'train size: {len(train_set)}, test size: {len(valid_set)}')

    # Create dataloader
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, persistent_workers=True)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, num_workers=8, persistent_workers=True)
    
    # Build model
    print(f'u_size: {len(dataset.user2index), len(user_info)}, g_size: {len(dataset.doc2index), len(doc_info)}')    
    
    model = WideAndDeepModel(u_size=len(user_info), 
                             g_size=len(doc_info),
                             metadata_dim=dataset.feat_dict['metadata'].shape[1],
                             embed_dim=16,
                             output_dim=1)
    trainer = L.Trainer(max_epochs=args.epochs, 
                        accelerator="cpu",
                        default_root_dir=f"outputs/wide_and_deep/{args.dataset}")
    trainer.fit(model, train_loader, valid_loader)
    result = trainer.validate(model, valid_loader)
    print(result)
    
    result_df = {'algorithm': ['Wide&Deep'],
                 'dataset': [args.dataset],
                 'epochs': [args.epochs],
                 'batch_size': [args.batch_size]}
    for k in result[0]:
        result_df[k] = [result[0][k]]
    result_df = pd.DataFrame(result_df)
    result_df.to_csv('results.csv', mode='a', index=False, header=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='small', help='Dataset for training. [small, medium, full]')
    parser.add_argument('--batch-size', type=int, default=256, help='Batch size for the model')
    parser.add_argument('--epochs', type=int, default=2, help='Batch size for the model')
    parser.add_argument('--write-result', action='store_true', help='Write result to file')
    args = parser.parse_args()
    
    main(args)