import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import LLM_mimic, LSTM
from data_provider.data_factory import data_provider
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import EarlyStopping, adjust_learning_rate, load_content

parser = argparse.ArgumentParser(description='Time-LLM')

fix_seed = 2021
random.seed(fix_seed)
torch.manual_seed(fix_seed)
np.random.seed(fix_seed)

# basic config
parser.add_argument('--task_name', type=str, default='short_term_forecast',
                    help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
parser.add_argument('--is_training', type=int, default=1, help='status')
parser.add_argument('--model_id', type=str, default='test', help='model id')
parser.add_argument('--model_comment', type=str, default='none', help='prefix when saving test results')
parser.add_argument('--model', type=str, default='TimeLLM',
                    help='model name, options: [Autoformer, DLinear]')
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# data loader
parser.add_argument('--data', type=str, default='MIMIC', help='dataset type')
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

# forecasting task
parser.add_argument('--seq_len', type=int, default=8, help='input sequence length')
parser.add_argument('--label_len', type=int, default=48, help='start token length')
parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
parser.add_argument('--input_size', type=int, default=55, help='input size of mlp')
parser.add_argument('--hidden_size', type=int, default=2, help='hidden size of mlp')

# model define
parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--n_features', type=int, default=55, help='num of input features')
parser.add_argument('--d_model', type=int, default=16, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--d_ff', type=int, default=32, help='dimension of fcn')
parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
parser.add_argument('--factor', type=int, default=1, help='attn factor')
parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
parser.add_argument('--embed', type=str, default='timeF',
                    help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu', help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
parser.add_argument('--patch_len', type=int, default=8, help='patch length')
parser.add_argument('--stride', type=int, default=4, help='stride')
parser.add_argument('--prompt_domain', type=int, default=0, help='')
parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


# optimization
parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
parser.add_argument('--itr', type=int, default=1, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=10, help='train epochs')
parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
parser.add_argument('--batch_size', type=int, default=8, help='batch size of train input data')
parser.add_argument('--eval_batch_size', type=int, default=8, help='batch size of model evaluation')
parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test', help='exp description')
parser.add_argument('--loss', type=str, default='MSE', help='loss function')
parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--llm_layers', type=int, default=1)
parser.add_argument('--percent', type=int, default=100)

args = parser.parse_args()

def evaluate(model, test_loader, device):
    model.eval()
    # criterion = nn.BCELoss()
    pos_weight = torch.tensor([2.5]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    predictions = []
    targets = []
    test_loss = []
    print('Evaluation on Test set...')
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(test_loader)):
            model_optim.zero_grad()

            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)

            outputs = model(batch_x)
            outputs = outputs.squeeze()  

            loss = criterion(outputs, batch_y)
            test_loss.append(loss.item())          

            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    # print(predictions)
    test_loss = np.average(test_loss)
    print("Test Loss: {0:.7f}".format(test_loss))
    auroc = roc_auc_score(targets, predictions)
    auprc = average_precision_score(targets, predictions)

    predictions = np.round(predictions).tolist()
    cm = confusion_matrix(targets, predictions)
    plot_confusion_matrix(cm=cm, labels=['Decline','Incline'], save_path='plots/Llama2_cm.png')

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
    pos_weight = torch.tensor([2]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    
    predictions = []
    targets = []
    vali_loss = []
    print('Validation on Vali set...')
    with torch.no_grad():
        for i, (batch_x, batch_y) in tqdm(enumerate(vali_loader)):
            model_optim.zero_grad()

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

# Setting record of experiments
setting = '{}_{}_{}_{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_fc{}_eb{}_{}'.format(
    args.task_name,
    args.model_id,
    args.model,
    args.data,
    args.features,
    args.seq_len,
    args.label_len,
    args.pred_len,
    args.d_model,
    args.n_heads,
    args.e_layers,
    args.d_layers,
    args.d_ff,
    args.factor,
    args.embed,
    args.des)
print(f"Settings: {setting}")

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

# Initialize Model
model = LLM_mimic.Model(args).float().to(device)
# lstm = LSTM.Model(args.n_features,args.n_features,args.seq_len,args.n_features).to(device)
model.set_columns(columns_in)
time_now = time.time()

train_steps = len(train_loader)
trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)
early_stopping = EarlyStopping(patience=args.patience)                                
# criterion = nn.BCELoss()
pos_weight = torch.tensor([2]).to(device)
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

train_loss = []
vali_loss = []
for epoch in range(args.train_epochs):
    iter_count = 0
    epoch_loss = []

    model.train()
    # lstm.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
 
        # dec_inp = torch.zeros_like(batch_y[:, :]).float().to(device)
        # dec_inp = torch.cat([batch_y[:, :], dec_inp], dim=1).float().to(device)

        # print(f"llama_output: {llama_outputs.shape}")
        # lstm_outputs = lstm(llama_outputs).to(device)
        # print(f"lstm_output: {lstm_outputs.shape}")

        outputs = model(batch_x).to(device)
        loss = criterion(outputs, batch_y)
        epoch_loss.append(loss.item())

        loss.backward()
        model_optim.step()

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
plot_loss_curves(train_loss, vali_loss, 'plots/Llama2_loss_curve.png')