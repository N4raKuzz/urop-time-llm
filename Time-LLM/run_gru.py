import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from models import GRU
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping

import argparse
import time
import random
import numpy as np
import os

parser = argparse.ArgumentParser(description='Time-LLM')
# basic config
parser.add_argument('--task_name', type=str, default='classification',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='TimeLLM',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, default='MOR', help='dataset type, options:[MOR, MIMIC]')
parser.add_argument('--root_path', type=str, default='./dataset/MIMIC/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='MIMICtable_261219.csv', help='data file')
parser.add_argument('--features', type=str, default='M',
                    help='forecasting task, options:[M, S, MS]; '
                         'M:multivariate predict multivariate, S: univariate predict univariate, '
                         'MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--loader', type=str, default='modal', help='dataset type')
parser.add_argument('--freq', type=str, default='h',
                    help='freq for time features encoding, '
                         'options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], '
                         'you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
parser.add_argument('--seq_len', type=int, default=8, help='input sequence length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')

# model define
parser.add_argument('--n_features', type=int, default=55, help='num of input features')
parser.add_argument('--input_size', type=int, default=1, help='input size of gru')
parser.add_argument('--hidden_size', type=int, default=27, help='number of gru units')
parser.add_argument('--output_size', type=int, default=1, help='output size of gru')
parser.add_argument('--num_layers', type=int, default=4, help='number of gru layers')

# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--loss', type=str, default='BCE', help='loss function')

args = parser.parse_args()


def evaluate(model, test_loader, device):
    model.eval()
    criterion = nn.BCELoss()
    predictions = []
    targets = []
    test_loss = []
    print('Evaluation on Test set...')
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            outputs = outputs.squeeze()  

            loss = criterion(outputs, batch_y)
            test_loss.append(loss.item())          

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())


    test_loss = np.average(test_loss)
    print("Test Loss: {0:.7f}".format(test_loss))
    auroc = roc_auc_score(targets, predictions)
    auprc = average_precision_score(targets, predictions)

    predictions = np.round(predictions).tolist()
    cm = confusion_matrix(targets, predictions)
    plot_confusion_matrix(cm=cm, labels=['Death','Survive'], save_path='plots/GRU3_MOR_cm.png')

    print("Test Loss: {0:.7f}".format(loss))
    print(f'AUROC: {auroc:.4f}')
    print(f'AUPRC: {auprc:.4f}')

def plot_confusion_matrix(cm, labels, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=labels, yticklabels=labels)
    
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Confusion matrix saved to {save_path}")

def vali(model, vali_loader, early_stopping, device):
    model.eval()
    # criterion = nn.BCELoss()
    pos_weight = torch.tensor([31.6]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    predictions = []
    targets = []
    vali_loss = []
    print('Validation on Vali set...')
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            optimizer.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            outputs = outputs.squeeze()  
            
            loss = criterion(outputs, batch_y)
            vali_loss.append(loss.item())          

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())


    vali_loss = np.average(vali_loss)
    early_stopping(vali_loss)

    return early_stopping.early_stop, vali_loss

def plot_loss_curves(train_loss, val_loss, save_path):
    # print("Generating loss curve..")
    epochs = range(1, len(train_loss) + 1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_loss, 'b-', label='Training Loss')
    plt.plot(epochs, val_loss, 'r-', label='Validation Loss')
    
    plt.title('Training and Validation Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.grid(True)
    plt.tight_layout()

    plt.savefig(save_path)
    print(f"Plot saved to {os.path.abspath(save_path)}")

# Load Data 
train_data, train_loader = data_provider(args, 'train')
test_data, test_loader = data_provider(args, 'test')
vali_data, vali_loader = data_provider(args, 'vali')
scaler = train_data.get_scaler()
columns_in, columns_out = train_data.get_columns()
print(f"Input Columns:{columns_in}")
print(f"Output Columns:{columns_out}")

# Initialize Device
device = torch.device("cuda")
print(f"Device name: {torch.cuda.get_device_name(device.index)}")
print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

early_stopping = EarlyStopping(patience=args.patience)
train_steps = len(train_loader)
time_now = time.time()

model = GRU.Model(args.hidden_size, args.num_layers, args.output_size, args.input_size).to(device)
# criterion = nn.BCELoss()
pos_weight = torch.tensor([31.57]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

train_loss = []
vali_loss = []
for epoch in range(args.train_epochs):
    iter_count = 0
    epoch_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
        iter_count += 1
        optimizer.zero_grad()

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device).view(-1)
        # print(f"targets: {batch_y.shape}")
        # print(batch_x, batch_y)
        
        outputs = model(batch_x).to(device)
        # print(outputs, batch_y)
        loss = criterion(outputs, batch_y)
        epoch_loss.append(loss.item())
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()
    
    is_early_stopping, loss = vali(model, vali_loader, early_stopping, device)
    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    epoch_loss = np.average(epoch_loss)
    vali_loss.append(loss)
    train_loss.append(epoch_loss)
    print("Epoch: {0} | Train Loss: {1:.7f}".format(epoch + 1, epoch_loss))

    # if(is_early_stopping):
    #     print("Early stopping")
    #     break


evaluate(model, test_loader, device)
plot_loss_curves(train_loss, vali_loss, 'plots/GRU3_MOR_loss_curve.png')
