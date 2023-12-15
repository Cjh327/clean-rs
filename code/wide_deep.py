import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L 

from torch.utils.data import DataLoader, random_split

from utils import read_processed_data, NewsDataset

class WideAndDeepModel(L.LightningModule):
    def __init__(self, wide_input_dim, deep_input_dim, output_dim, lr=1e-3):
        super().__init__()
        self.wide = nn.Linear(wide_input_dim, output_dim)
        self.deep = nn.Sequential(
                        nn.Linear(deep_input_dim, 64),
                        nn.ReLU(),
                        nn.Linear(64, output_dim))
        self.lr = lr
    
    def forward(self, X_wide, X_deep):
        wide_output = self.wide(X_wide)
        deep_output = self.deep(X_deep)
        return wide_output + deep_output
    
    def training_step(self, batch, batch_idx):
        X_wide, X_deep, y = batch
        y_hat = self(X_wide, X_deep)
        loss = F.mse_loss(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
    

def main():
    doc_info, user_info, data = read_processed_data('dataset/processed_data', small=True)
    dataset = NewsDataset(data, doc_info, user_info)
    
    train_set_size = int(len(dataset) * 0.8)
    valid_set_size = len(dataset) - train_set_size
    seed = torch.Generator().manual_seed(42)
    train_set, valid_set = random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    
    print(len(train_set), len(valid_set))

    # train_dataloader = DataLoader(TensorDataset(wide_data, deep_data, target), batch_size=64)

    # model = WideAndDeepModel(wide_data.size(1), deep_data.size(1), 1)
    # trainer = L.Trainer(max_epochs=10)
    # trainer.fit(model, train_dataloader)
    


if __name__ == '__main__':
    main()