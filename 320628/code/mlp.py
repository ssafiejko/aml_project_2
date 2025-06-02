import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np

class MLP(nn.Module):
    def __init__(self, input_size):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 8),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(8, 1),
        )

    def forward(self, x):
        return self.net(x)

class MLPClassifier:
    def __init__(self,
                epochs=10, lr=1e-3, weight_decay=1e-4, batch_size=64, device="mps"):
        self.device = device
        self.model = MLP(input_size=500).to(device)

        self.epochs = epochs
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size =  batch_size
    
    def fit(self, X_train, y_train):
        input_size = X_train.shape[1]
        self.model = MLP(input_size=input_size).to(self.device)
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
        )
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        criterion = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        for _ in range(self.epochs):
            self.model.train()
            for x_batch, y_batch in dataloader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = criterion(logits, y_batch)
                loss.backward()
                optimizer.step()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        self.model.eval()
        with torch.no_grad():
            inputs = torch.tensor(X, dtype=torch.float32).to(self.device)
            outputs = self.model(inputs)
            probs = torch.sigmoid(outputs)
        return probs.cpu().numpy()  

    def predict(self, X: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs > 0.5).astype(np.int8).squeeze()