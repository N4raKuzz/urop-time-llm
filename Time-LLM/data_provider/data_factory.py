from data_provider.data_loader import Dataset_ETT_hour, Dataset_ETT_minute
from data_provider.mimic_loader import Dataset_MIMIC, Dataset_MOR
from torch.utils.data import DataLoader

data_dict = {
    'ETTh1': Dataset_ETT_hour,
    'ETTh2': Dataset_ETT_hour,
    'ETTm1': Dataset_ETT_minute,
    'ETTm2': Dataset_ETT_minute,
    'MIMIC': Dataset_MIMIC,
    'MOR': Dataset_MOR,
}


def data_provider(args, flag):
    Data = data_dict[args.data]

    if flag == 'test':
        shuffle_flag = False
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size
        freq = args.freq
        
    drop_last = False
    data_set = Data(
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        max_len=args.seq_len,
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last)

    return data_set, data_loader
