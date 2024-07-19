import argparse
import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from tqdm import tqdm

from models import LLM_mimic, LSTM
from data_provider.data_factory import data_provider
from sklearn.metrics import mean_absolute_error, mean_squared_error

import time
import random
import numpy as np
import os

os.environ['CURL_CA_BUNDLE'] = ''
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:64"

from utils.tools import del_files, EarlyStopping, adjust_learning_rate, vali, load_content

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

def evaluate(model, test_loader, criterion, columns, scaler, device):
    model.eval()
    total_loss = 0
    predictions = []
    targets = []
    print('Evaluation on Test set...')
    with torch.no_grad():

        for batch_x, batch_y in test_loader:
            batch_x = batch_x.float().to(device)
            batch_y = batch_y.float().to(device)
            
            outputs = model(batch_x).to(device)
            outputs = outputs[:, -args.pred_len:, list(range(3,51))]
            loss = criterion(outputs, batch_y)
            
            total_loss += loss.item() * batch_x.size(0)
            predictions.extend(outputs.cpu().numpy())
            targets.extend(batch_y.cpu().numpy())

    predictions = np.array(predictions).reshape(-1, 48)
    targets = np.array(targets).reshape(-1, 48)

    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)

    print(f'Overall MAE: {mae:.4f}')
    print(f'Overall MSE: {mse:.4f}')

    print('Scale Inverse transforming...')
    predictions = np.clip(predictions, 0, None)
    predictions_scaled = scaler.inverse_transform(predictions)
    targets_scaled = scaler.inverse_transform(targets)

    for i, column in enumerate(columns):
        feature_mae = mean_absolute_error(targets_scaled[:, i], predictions_scaled[:, i])
        print(f'{column} MAE: {feature_mae:.4f}')



# setting record of experiments
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

train_data, train_loader = data_provider(args, 'train')
test_data, test_loader = data_provider(args, 'test')

scaler = train_data.get_scaler()
columns_in, columns_out = train_data.get_columns()

print(f"Input Columns:{columns_in}")
print(f"Output Columns:{columns_out}")

device = torch.device("cuda")
print(f"Device name: {torch.cuda.get_device_name(device.index)}")
print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")

model = LLM_mimic.Model(args).float().to(device)
lstm = LSTM.Model(args.n_features,args.n_features,args.seq_len,args.label_len).to(device)
model.set_columns(columns_in)
time_now = time.time()

train_steps = len(train_loader)

trained_parameters = []
for p in model.parameters():
    if p.requires_grad is True:
        trained_parameters.append(p)

model_optim = optim.Adam(trained_parameters, lr=args.learning_rate)

if args.lradj == 'COS':
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(model_optim, T_max=20, eta_min=1e-8)
else:
    scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                        steps_per_epoch=train_steps,
                                        pct_start=args.pct_start,
                                        epochs=args.train_epochs,
                                        max_lr=args.learning_rate)

criterion = nn.MSELoss()
mae_metric = nn.L1Loss()

if args.use_amp:
    scaler = torch.cuda.amp.GradScaler()

for epoch in range(args.train_epochs):
    iter_count = 0
    train_loss = []

    model.train()
    epoch_time = time.time()
    for i, (batch_x, batch_y) in tqdm(enumerate(train_loader)):
        iter_count += 1
        model_optim.zero_grad()

        batch_x = batch_x.float().to(device)
        batch_y = batch_y.float().to(device)
 
        dec_inp = torch.zeros_like(batch_y[:, :]).float().to(device)
        dec_inp = torch.cat([batch_y[:, :], dec_inp], dim=1).float().to(device)

        llama_outputs = model(batch_x).to(device)
        print(f"llama_output: {llama_outputs.shape}")
        lstm_outputs = lstm(llama_outputs).to(device)
        lstm_outputs = lstm_outputs[:, -args.pred_len:, list(range(3,51))]
        batch_y = batch_y[:, -1:]
        
        loss = criterion(lstm_outputs, batch_y)
        train_loss.append(loss.item())

        loss.backward()
        model_optim.step()

        if (i + 1) % 100 == 0:
            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
            speed = (time.time() - time_now) / iter_count
            left_time = speed * ((args.train_epochs - epoch) * train_steps - i)
            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
            iter_count = 0
            time_now = time.time()

    # print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
    train_loss = np.average(train_loss)
    # print("Epoch: {0} | Train Loss: {1:.7f}".format(epoch + 1, train_loss))

    # if args.lradj != 'TST':
    #     if args.lradj == 'COS':
    #         scheduler.step()
    #         print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #     else:
    #         if epoch == 0:
    #             args.learning_rate = model_optim.param_groups[0]['lr']
    #             print("lr = {:.10f}".format(model_optim.param_groups[0]['lr']))
    #         adjust_learning_rate(model_optim, scheduler, epoch + 1, args, printout=True)

    # else:
    #     print('Updating learning rate to {}'.format(scheduler.get_last_lr()[0]))
  
evaluate(model, test_loader, criterion, columns_out, scaler, device)


