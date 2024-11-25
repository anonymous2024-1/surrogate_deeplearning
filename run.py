import argparse
import torch
import random
import numpy as np
from utils.print_args import print_args
from exp.exp_surrogate_DL import Exp_Forecast


if __name__ == '__main__':
    fix_seed = 2024
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    def str2bool(v):
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(description='DL surrogate model for PBM')
    parser.add_argument('--model', type=str, required=True, default='LSTMa', help='model name, options:[LSTM]')
    parser.add_argument('--model_opt', type=str, required=True, default='train', help='train, vali, test')

    # data loader
    parser.add_argument('--data_root', type=str, required=True, help='root path of the data file')
    parser.add_argument('--dataset', type=str, default='surro', help='which dataset, options:[dPL, surro, surroLT, surro5, surro7]')
    parser.add_argument('--scale', type=str2bool, default=True, help='normalization or not')
    parser.add_argument('--seq_len', type=int, default=30, help='input sequence length')
    parser.add_argument('--label_len', type=int, default=0, help='start token length')
    parser.add_argument('--pred_len', type=int, default=7, help='prediction sequence length')
    parser.add_argument('--inverse', action='store_true', default=False, help='inverse output data')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--features', type=str, default='MS',
                        help='forecasting task, options:[M, MS]; M:multivariate predict multivariate, MS:multivariate predict univariate')

    # model define
    parser.add_argument('--d_model', type=int, default=512, help='dimension of model')
    parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=39, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=1, help='output size')
    parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=1, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--factor', type=int, default=5, help='attn factor')
    parser.add_argument('--d_ff', type=int, default=None, help='dimension of fcn')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--output_attention', action='store_true', default=False, help='whether to output attention matrix in ecoder')

    # optimization
    parser.add_argument('--num_workers', type=int, default=6, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=1, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function, options:[MSE, MAE, NSE]')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate, options:[type1, type2, cosine]')
    parser.add_argument('--use_amp', action='store_true', default=False, help='use automatic mixed precision training')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--use_multi_gpu', action='store_true', default=False, help='use multiple gpus')
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    args = parser.parse_args()

    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]  # [0, 1, 2, 3]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print_args(args)
    print()

    if args.model_opt == 'train':
        for ii in range(args.itr):
            # setting record of experiments
            setting = '{}_{}_{}{}{}_dm{}h{}e{}d{}_{}'.format(
                args.dataset,
                args.model,
                args.seq_len,
                args.label_len,
                args.pred_len,
                args.d_model,
                args.n_heads,
                args.e_layers,
                args.d_layers, ii)

            exp = Exp_Forecast(args, setting)  # set experiments
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.train(setting)
            print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
            exp.test(setting)
            torch.cuda.empty_cache()
            print()
    else:
        ii = 0
        setting = '{}_{}_{}{}{}_dm{}h{}e{}d{}_{}'.format(
            args.dataset,
            args.model,
            args.seq_len,
            args.label_len,
            args.pred_len,
            args.d_model,
            args.n_heads,
            args.e_layers,
            args.d_layers, ii)

        exp = Exp_Forecast(args, setting)  # set experiments
        # exp = Exp_Forecast_testSA(args, setting)  # attention test
        print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.test(setting, test=1)
        torch.cuda.empty_cache()
        print()


