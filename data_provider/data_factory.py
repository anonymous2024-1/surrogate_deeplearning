from data_provider.data_loder import dPLDataset, surroDataset, surroDataset5, surroDataset7, \
    surroDataset_orig
from torch.utils.data import DataLoader
import numpy as np

def data_provider(args, flag):
    if flag == 'test' :
        shuffle_flag = False
        drop_last = False
        # batch_size = 1  # bsz=1 for evaluation
        batch_size = args.batch_size
    else:
        shuffle_flag = True
        drop_last = True
        batch_size = args.batch_size  # bsz for train and valid

    if args.dataset == 'dPL':
        data_set = dPLDataset(data_root=args.data_root,
                              size=[args.seq_len, args.label_len, args.pred_len],
                              scale=True,
                              flag=flag)
    if args.dataset == 'surro':
        data_set = surroDataset(data_root=args.data_root,
                                size=[args.seq_len, args.label_len, args.pred_len],
                                scale=True,
                                flag=flag)

    if args.dataset == 'surroLT':
        Ts_tuple = (np.arange(0, 100), np.arange(100, 120), np.arange(100, 120))
        data_set = surroDataset(data_root=args.data_root,
                                size=[args.seq_len, args.label_len, args.pred_len],
                                Ts_tuple=Ts_tuple,
                                scale=True,
                                flag=flag)
    # surro_deg0.5
    if args.dataset == 'surro5':
        data_set = surroDataset5(data_root=args.data_root,
                                size=[args.seq_len, args.label_len, args.pred_len],
                                 Ts_tuple=Ts_tuple,
                                scale=True,
                                flag=flag)
    # 7 params
    if args.dataset == 'surro7':
        data_set = surroDataset7(data_root=args.data_root,
                                size=[args.seq_len, args.label_len, args.pred_len],
                                scale=True,
                                flag=flag)
    #
    if args.dataset == 'surro_orig':
        data_set = surroDataset_orig(data_root=args.data_root,
                                    size=[args.seq_len, args.label_len, args.pred_len],
                                    scale=True,
                                    flag=flag)
    print(flag, len(data_set))
    data_loader = DataLoader(data_set,
                             batch_size=batch_size,
                             shuffle=shuffle_flag,
                             num_workers=args.num_workers,
                             drop_last=drop_last)

    return data_set, data_loader



