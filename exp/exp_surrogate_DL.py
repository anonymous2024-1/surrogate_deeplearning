from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
from utils.losses import mape_loss, mase_loss, smape_loss, nse_loss

import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np

warnings.filterwarnings('ignore')


class Exp_Forecast(Exp_Basic):
    def __init__(self, args, setting):
        super(Exp_Forecast, self).__init__(args, setting)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)  # device_ids=[0, 1, 2]
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, loss_name='MSE'):
        if loss_name == 'MSE':
            return nn.MSELoss()
        elif loss_name == 'MAE':
            return nn.L1Loss()
        elif loss_name == 'MAPE':
            return mape_loss()
        elif loss_name == 'MASE':
            return mase_loss()
        elif loss_name == 'SMAPE':
            return smape_loss()
        elif loss_name == 'NSE':
            return nse_loss()

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_params) in enumerate(vali_loader):
                pred, true = self._process_one_batch(batch_x, batch_y, batch_params)
                loss = criterion(pred.detach().cpu(), true.detach().cpu())
                total_loss.append(loss)

        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss


    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(self.args.loss)

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_params) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                pred, true = self._process_one_batch(batch_x, batch_y, batch_params)
                loss = criterion(pred, true)
                train_loss.append(loss.item())

                if (i + 1) % 5000 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        self.model.eval()

        if test:
            print('loading model ...')
            # model_fold = "surro_Transformer_270090_dm512h8e2d2_0"
            # print('model path:', model_fold)

            # setting = setting + "_OndPL"
            print('setting:', setting)

            # model_path = os.path.join('./checkpoints/' + model_fold, 'checkpoint.pth')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))

        preds, trues = [], []
        # folder_path = './test_results/' + setting + '/'
        # if not os.path.exists(folder_path):
        #     os.makedirs(folder_path)

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_params) in enumerate(test_loader):
                start_time = time.time()
                outputs, _ = self._process_one_batch(batch_x, batch_y, batch_params)
                if i == 0:
                    print("batch: {} cost time: {}".format(i + 1, time.time() - start_time))

                batch_x = batch_x.float()
                batch_y = batch_y.float()

                batch_y = batch_y[:, -self.args.pred_len:, :].to(self.device)
                outputs = outputs.detach().cpu().numpy()
                batch_y = batch_y.detach().cpu().numpy()

                if test_data.scale and self.args.inverse:
                    if self.args.features == 'MS':
                        outputs = (outputs + test_data.scaler.min_) / test_data.scaler.scale_  # MinMaxScaler
                    else:
                        shape = outputs.shape  # (B,7,target_num)
                        outputs = outputs.reshape(-1, shape[-1])
                        outputs = test_data.inverse_transform(outputs).reshape(shape)

                    shape = batch_y.shape  # (B, 7, 10)
                    batch_y = test_data.inverse_transform(batch_y.reshape(-1, shape[-1])).reshape((shape[0], shape[1], -1))

                outputs = outputs[:, :, 0:self.args.c_out]
                batch_y = batch_y[:, :, 7:8]
                preds.append(outputs)
                trues.append(batch_y)

        preds, trues = np.array(preds), np.array(trues)
        print('test shape:', preds.shape, trues.shape) #
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        mae, mse, nse, kge, corr = metric(preds, trues)
        print('mse:{}, mae:{}, nse:{}, kge:{}, corr:{}'.format(mse, mae, nse, kge, corr))

        # result save
        if not test:
            folder_path = './results/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            result_txt = 'result_' + setting + '.txt'
            with open(result_txt, 'a') as f:
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, nse:{}, kge:{}, corr:{}'.format(mse, mae, nse, kge, corr))
                f.write('\n')
                f.write('\n')

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, nse, kge, corr]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)
        else:
            folder_path = './result_test_surroGT/' + setting + '/'
            if not os.path.exists(folder_path):
                os.makedirs(folder_path)

            result_txt = folder_path + 'result_' + setting + '.txt'
            with open(result_txt, 'a') as f:
                f.write(setting + "  \n")
                f.write('mse:{}, mae:{}, nse:{}, kge:{}, corr:{}'.format(mse, mae, nse, kge, corr))
                f.write('\n')
                f.write('\n')

            np.save(folder_path + 'metrics.npy', np.array([mae, mse, nse, kge, corr]))
            np.save(folder_path + 'pred.npy', preds)
            np.save(folder_path + 'true.npy', trues)

        return


    def _process_one_batch(self, batch_x, batch_y, batch_params):
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_params = batch_params.float()

        enc_inp = torch.cat([batch_x[:, :, 0:7], batch_y[:, -self.args.pred_len:, 0:7]], dim=1).to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, 7:8]).float()
        dec_inp = torch.cat([batch_x[:, :, 7:8], dec_inp], dim=1)
        batch_params = torch.unsqueeze(batch_params, 1)

        batch_params = batch_params.repeat(1, self.args.seq_len + self.args.pred_len, 1)
        dec_inp = torch.cat([dec_inp, batch_params], dim=2).to(self.device)

        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.output_attention:
                    outputs = self.model(enc_inp, dec_inp)[0]
                else:
                    outputs = self.model(enc_inp, dec_inp)
        else:
            if self.args.output_attention:
                outputs = self.model(enc_inp, dec_inp)[0]
            else:
                outputs = self.model(enc_inp, dec_inp)

        batch_y = batch_y[:, -self.args.pred_len:, 7:8].to(self.device)

        return outputs, batch_y


    def _process_one_batch_grad(self, batch_x, batch_y, batch_params):
        batch_x = batch_x.float()
        batch_y = batch_y.float()
        batch_params = batch_params.float()

        enc_inp = torch.cat([batch_x[:, :, 0:7], batch_y[:, -self.args.pred_len:, 0:7]], dim=1).to(self.device)
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, 7:8]).float()
        dec_inp = torch.cat([batch_x[:, :, 7:8], dec_inp], dim=1)
        batch_params = torch.unsqueeze(batch_params, 1)

        batch_params = batch_params.repeat(1, self.args.seq_len + self.args.pred_len, 1)
        dec_inp = torch.cat([dec_inp, batch_params], dim=2).to(self.device)

        batch_y = batch_y[:, -self.args.pred_len:, 7:8].to(self.device)

        dec_inp = dec_inp.requires_grad_(True)
        outputs = self.model(enc_inp, dec_inp)
        outputs.backward(torch.ones_like(outputs))
        sensitivities = dec_inp.grad

        return outputs, batch_y, sensitivities

