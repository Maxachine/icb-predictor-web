import torch
from torch.optim import Adam
from torch.utils.data import DataLoader,Dataset
from tqdm import trange
from model import ScoreNet
from utils import marginal_prob_std_fn,device,loss_fn
import numpy as np
import pdb
 

class MedicalDataset(Dataset):
    def __init__(self, file_path):
        """
        Args:
            file_path (str): Path to the .txt file containing the dataset.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        # Read the data
        with open(file_path, 'r') as file:
            data = [list(map(float, line.strip().split(','))) for line in file]

        # Convert data to a NumPy array for easier processing
        data_array = np.array(data)

        # Separate the features (all but the last column) and the last column
        features = data_array # All columns except the last
        # last_column = data_array[:, -1]  # The last column

        # Standardize the features (subtract mean, divide by standard deviation)
        mean_vals = features.mean(axis=0)
        std_vals = features.std(axis=0)
        normalized_features = (features - mean_vals) / std_vals


        # Combine the normalized features with the last column
        # normalized_data = np.hstack((normalized_features, last_column[:, np.newaxis]))

        # Convert to a PyTorch tensor
        # self.data = torch.tensor(normalized_features, dtype=torch.float32)  # shape [N, D]
        self.raw_data = data_array
        self.data = torch.tensor(normalized_features[:,:], dtype=torch.float32)  # shape [N, D]
        self.mean_vals = mean_vals
        self.std_vals = std_vals

    def normalize(self,):
        tmp_raw_data = self.raw_data.copy()
        tmp_raw_data[:,-1] = 0
        self.data = (tmp_raw_data - self.mean_vals) / self.std_vals
    
    def __len__(self):
        # Return the number of samples in the dataset
        return len(self.data)
    
    def __getitem__(self, idx):
        # Return a single sample at the given index
        sample = self.data[idx]
        raw_sample = self.raw_data[idx]
        y = raw_sample[-1]
        return sample, y

if __name__ == '__main__':
    score_model = torch.nn.DataParallel(ScoreNet(marginal_prob_std=marginal_prob_std_fn, input_dim=12))
    score_model = score_model.to(device)

    n_epochs = 1200
    ## size of a mini-batch
    batch_size = 32 
    ## learning rate
    lr=1e-4 

    file_path = './train_files_11/11111111111.txt'
    # file_path = 'train_type123.txt'
    dataset = MedicalDataset(file_path)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # vae = VAE(input_dim=6, hidden_dim=256, latent_dim=5)
    # vae.load_state_dict(torch.load('vae_model.pth'))
    # vae.eval()

    optimizer = Adam(score_model.parameters(), lr=lr)
    tqdm_epoch = trange(n_epochs)
    for epoch in tqdm_epoch:
        avg_loss = 0.
        num_items = 0
        for x,_ in data_loader:
            x = x.to(device)
           
            # encoded,_,_ = vae.encode(x[:,:-1])
            # # response = x[:,-1].unsqueeze(1)
            # # x = torch.cat((encoded,response), dim=1)
            # x = encoded
            # # pdb.set_trace()

            loss = loss_fn(score_model, x, marginal_prob_std_fn)
            optimizer.zero_grad()
            loss.backward()    
            optimizer.step()
            avg_loss += loss.item() * x.shape[0]
            num_items += x.shape[0]
        # Print the averaged training loss so far.
        tqdm_epoch.set_description('Average Loss: {:5f}'.format(avg_loss / num_items))
        # Update the checkpoint after each epoch of training.
        torch.save(score_model.state_dict(), './checkpoints/ckpt_11111111111.pth')