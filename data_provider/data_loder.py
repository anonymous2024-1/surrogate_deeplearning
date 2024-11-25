import os
from pathlib import Path
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler



## data_dPL/deg0.5
class dPLDataset(Dataset):
    def __init__(self, data_root, size=None, Grd_tuple=None, Ts_tuple=None, scale=True, flag='train'):
        self.scaler = MinMaxScaler()
        self.data_root = data_root
        self.size = size
        self.scale = scale
        self.Ngrid = 4783
        self.Ntime = None
        self.grid_select = Grd_tuple
        self.time_select = Ts_tuple

        # select grids
        if self.grid_select == None:
            Ntrain = int(self.Ngrid * 0.7)
            Ntest = int(self.Ngrid * 0.1)
            Nvali = self.Ngrid - Ntrain - Ntest
            grid_train = np.arange(0, Ntrain)
            grid_test = np.arange(Ntrain, Ntrain + Ntest)
            grid_vali = np.arange(Ntrain + Ntest, self.Ngrid)
            self.grid_select = (grid_train, grid_vali, grid_test)

        if self.size == None:
            self.seq_len, self.label_len, self.pred_len = 30, 7, 7
        else:
            self.seq_len, self.label_len, self.pred_len = self.size[0], self.size[1], self.size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self._read_forcing_data()
        self._read_params_data()


    def _read_forcing_data(self):
        data_list = []
        for i in range(self.Ngrid):
            file = os.path.join(self.data_root, "forcing_new", str(i+1).zfill(4) + ".csv")
            data = pd.read_csv(file).drop(columns=['date']).to_numpy()  # drop 'date'
            data_list.append(data)

        data = np.array(data_list)  # (4783, 1096, 10)
        print('data shape:', data.shape)

        self.Ntime = data.shape[1]
        num_train = int(self.Ntime * 0.7)
        num_test = int(self.Ntime * 0.2)
        num_vali = self.Ntime - num_train - num_test

        border1s = [0, num_train - self.seq_len, self.Ntime - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.Ntime]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        x_list, y_list = [], []
        for i in range(data.shape[0]):
            data_1 = data[i]
            train_data = data[i, border1s[0]:border2s[0], :]

            if self.scale:
                self.scaler.fit(train_data)
                data_1 = self.scaler.transform(data_1)

            data_x = data_1[border1:border2]
            data_y = data_1[border1:border2]
            num_samples = len(data_x) - self.seq_len - self.pred_len + 1

            x_1, y_1 = [], []
            for j in range(num_samples):
                s_begin = j
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len

                x_1.append(data_x[s_begin:s_end])
                y_1.append(data_y[r_begin:r_end])

            x_list.append(np.array(x_1))
            y_list.append(np.array(y_1))

        x_allgrd, y_allgrd = np.array(x_list), np.array(y_list)
        print(x_allgrd.shape, y_allgrd.shape)  # (4783, 731, 30, 10) (4783, 731, 14, 10) include SM, ET, RUNOFF

        self.x, self.y = [], []
        grid_id_arr = self.grid_select[self.set_type]
        time_id_arr = np.arange(0, num_samples)
        if self.time_select is not None:
            time_id_arr = self.time_select[self.set_type]

        self.num_samples = len(time_id_arr)
        print('num_samples:', self.num_samples)

        ## get final sample list of different grid in the same date period
        for ti in time_id_arr:
            for id in grid_id_arr:
                self.x.append(x_allgrd[id, ti, :, :])
                self.y.append(y_allgrd[id, ti, :, :])

        self.x, self.y = np.array(self.x), np.array(self.y)
        print('x, y shape:', self.x.shape, self.y.shape)  # (N*731, 30, 10) (N*731, 14, 10)

        return self.x, self.y


    def _read_params_data(self):
        df_params = pd.read_csv(os.path.join(self.data_root, "params.csv"), dtype=float)
        scale_cols = ['Dsmax', 'depth_2', 'depth_3', 'expt_1', 'expt_2', 'expt_3', 'Ksat_1', 'Ksat_2', 'Ksat_3',
                      'init_moist_1', 'init_moist_2', 'init_moist_3',
                      'bubble_1', 'bubble_2', 'bubble_3',
                      'bulk_density_1', 'bulk_density_2', 'bulk_density_3',
                      'annual_prec', 'avg_T', 'elev']
        df_new = pd.DataFrame(df_params[scale_cols], columns=scale_cols)
        scaler1 = MinMaxScaler()
        scaler1.fit(df_new.values)
        params_scale = scaler1.transform(df_new.values)
        df_params[scale_cols] = params_scale

        grid_id_arr = self.grid_select[self.set_type]
        data_params = df_params.iloc[grid_id_arr].values

        self.x_params = [data_params for i in range(self.num_samples)]
        self.x_params = np.concatenate(self.x_params, axis=0)

        return self.x_params


    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_params[index]

    def __len__(self):
        return self.x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


## surro-GT, surro-LT
class surroDataset(Dataset):
    def __init__(self, data_root, size=None, Grd_tuple=None, Ts_tuple=None, scale=True, flag='train'):
        """
        :param data_root:
        :param size:
        :param Grd_tuple: grid points, tuple: (train_array1, vali_array2, test_array3)
        :param Ts_tuple: time periods, tuple: (train_array1, vali_array3, test_array2)
        :param scale:
        :param flag:
        """
        self.scaler = MinMaxScaler()
        self.data_root = data_root
        self.size = size
        self.scale = scale
        self.years = ['2016','2017', '2018', '2019', '2020']
        self.grid_select = Grd_tuple
        self.time_select = Ts_tuple
        self.Ngrid = 78496   # total grids
        self.Ntime = None  # total days is 1827

        ## select grids, if None, default to all grid points
        if self.grid_select == None:
            Ntrain = int(self.Ngrid * 0.7)
            Ntest = int(self.Ngrid * 0.1)
            Nvali = self.Ngrid - Ntrain - Ntest
            grid_train = np.arange(0, Ntrain)
            grid_test = np.arange(Ntrain, Ntrain + Ntest)
            grid_vali = np.arange(Ntrain + Ntest, self.Ngrid)
            self.grid_select = (grid_train, grid_vali, grid_test)

        if self.size == None:
            self.seq_len, self.label_len, self.pred_len = 30, 7, 7
        else:
            self.seq_len, self.label_len, self.pred_len = self.size[0], self.size[1], self.size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self._read_forcing_data()
        self._read_params_data()


    def _read_forcing_data(self):
        data = []
        for year in self.years:
            sub_data = np.load(os.path.join(self.data_root, "forcing/data_" + year + ".npy"))
            data.append(sub_data)
        data = np.concatenate(data, axis=1).astype(np.float16)  # (78496, 365, 8)  all:(78496, 1827, 8)  366,365,365, 365,366
        print('data shape: ', data.shape)

        self.Ntime = data.shape[1]
        num_train = int(self.Ntime * 0.6)
        num_test = int(self.Ntime * 0.2)
        num_vali = self.Ntime - num_train - num_test

        border1s = [0, num_train - self.seq_len, self.Ntime - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.Ntime]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        num_samples = border2 - border1 - self.seq_len - self.pred_len + 1

        x_list, y_list = [], []
        for i in self.grid_select[self.set_type]:
            data_1 = data[i]
            train_data = data[i, border1s[0]:border2s[0], :]

            if self.scale:
                self.scaler.fit(train_data)
            data_1 = self.scaler.transform(data_1)
            data_x = data_1[border1:border2]
            data_y = data_1[border1:border2]

            x_1, y_1 = [], []
            for j in range(num_samples):
                s_begin = j
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                x_1.append(data_x[s_begin:s_end])
                y_1.append(data_y[r_begin:r_end])

            x_list.append(np.array(x_1, dtype=np.float16))
            y_list.append(np.array(y_1, dtype=np.float16))

        x_allgrd, y_allgrd = np.array(x_list, dtype=np.float16), np.array(y_list, dtype=np.float16)
        print(x_allgrd.shape, y_allgrd.shape)
        x_list, y_list = [], []

        num_grid = len(self.grid_select[self.set_type])
        time_id_arr = np.arange(0, num_samples)
        if self.time_select is not None:
            time_id_arr = self.time_select[self.set_type]

        self.num_samples = len(time_id_arr)
        print('num_samples:', self.num_samples)

        self.x, self.y = [], []
        ## get final sample list of different grid in the same date period
        for ti in time_id_arr:
            for grd in range(num_grid):
                self.x.append(x_allgrd[grd, ti, :, :])
                self.y.append(y_allgrd[grd, ti, :, :])

        x_allgrd, y_allgrd = [], []
        self.x, self.y = np.array(self.x, dtype=np.float16), np.array(self.y, dtype=np.float16)
        print('x, y shape:', self.x.shape, self.y.shape)

        return self.x, self.y


    def _read_params_data(self):
        df_params = pd.read_csv(os.path.join(self.data_root, "soil_params/params_sampling1.csv"), dtype=np.float16)
        scale_cols = ['Dsmax', 'depth_2', 'depth_3', 'expt_1', 'expt_2', 'expt_3', 'Ksat_1', 'Ksat_2', 'Ksat_3',
                      'init_moist_1', 'init_moist_2', 'init_moist_3',
                      'bubble_1', 'bubble_2', 'bubble_3',
                      'bulk_density_1', 'bulk_density_2', 'bulk_density_3',
                      'annual_prec', 'avg_T', 'elev']
        df_new = pd.DataFrame(df_params[scale_cols], columns=scale_cols).astype(np.float16)

        scaler1 = MinMaxScaler()
        scaler1.fit(df_new.values)
        params_scale = scaler1.transform(df_new.values)
        df_params[scale_cols] = params_scale

        grid_id_arr = self.grid_select[self.set_type]
        data_params = df_params.iloc[grid_id_arr]

        self.x_params = [data_params for i in range(self.num_samples)]
        self.x_params = np.concatenate(self.x_params, axis=0).astype(np.float16)

        return self.x_params

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_params[index]

    def __len__(self):
        return self.x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)


# nldas: data_surro/deg0.5, lower resolution by sampling of dPL
class surroDataset5(Dataset):
    def __init__(self, data_root, size=None, Grd_tuple=None, Ts_tuple=None, scale=True, flag='train'):
        self.scaler = MinMaxScaler()
        self.data_root = data_root
        self.size = size
        self.scale = scale
        self.years = ['2016','2017', '2018', '2019', '2020']
        self.grid_select = Grd_tuple  # ([0,1,2], [3], [4])
        self.time_select = Ts_tuple  # ([0,30,60,90,120,150], [0], [0])
        self.Ngrid = 4783   # total grids
        self.Ntime = None  # total days is 1827

        ## select grids, if None, default to all grid points
        if self.grid_select == None:
            Ntrain = int(self.Ngrid * 0.7)
            Ntest = int(self.Ngrid * 0.1)
            Nvali = self.Ngrid - Ntrain - Ntest
            grid_train = np.arange(0, Ntrain)
            grid_test = np.arange(Ntrain, Ntrain + Ntest)
            grid_vali = np.arange(Ntrain + Ntest, self.Ngrid)
            self.grid_select = (grid_train, grid_vali, grid_test)

        if self.size == None:
            self.seq_len, self.label_len, self.pred_len = 30, 7, 7
        else:
            self.seq_len, self.label_len, self.pred_len = self.size[0], self.size[1], self.size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self._read_forcing_data()
        self._read_params_data()


    def _read_forcing_data(self):
        data = []
        for year in self.years:
            sub_data = np.load(os.path.join(self.data_root, "deg0.5/data_" + year + ".npy"))
            data.append(sub_data)
        data = np.concatenate(data, axis=1)  # (78496, 365, 8)  all:(78496, 1827, 8)  366,365,365, 365,366
        print('data shape: ', data.shape)

        self.Ntime = data.shape[1]
        num_train = int(self.Ntime * 0.6)
        num_test = int(self.Ntime * 0.2)
        num_vali = self.Ntime - num_train - num_test

        border1s = [0, num_train - self.seq_len, self.Ntime - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.Ntime]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        num_samples = border2 - border1 - self.seq_len - self.pred_len + 1

        x_list, y_list = [], []
        for i in range(data.shape[0]):
            data_1 = data[i]
            train_data = data[i, border1s[0]:border2s[0], :]

            if self.scale:
                self.scaler.fit(train_data)
            data_1 = self.scaler.transform(data_1)
            data_x = data_1[border1:border2]
            data_y = data_1[border1:border2]

            x_1, y_1 = [], []
            for j in range(num_samples):
                s_begin = j
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                x_1.append(data_x[s_begin:s_end])
                y_1.append(data_y[r_begin:r_end])

            x_list.append(np.array(x_1))
            y_list.append(np.array(y_1))

        x_allgrd, y_allgrd = np.array(x_list), np.array(y_list)
        print(x_allgrd.shape, y_allgrd.shape)  # (78496, 183, 30, 8) (78496, 183, 7, 8) include one target

        grid_id_arr = self.grid_select[self.set_type]
        time_id_arr = np.arange(0, num_samples)
        if self.time_select is not None:
            time_id_arr = self.time_select[self.set_type]

        self.num_samples = len(time_id_arr)
        print('num_samples:', self.num_samples)

        self.x, self.y = [], []
        ## get final sample list of different grid in the same date period
        for ti in time_id_arr:
            for grd in grid_id_arr:
                self.x.append(x_allgrd[grd, ti, :, :])
                self.y.append(y_allgrd[grd, ti, :, :])

        self.x, self.y = np.array(self.x), np.array(self.y)
        print('x, y shape:', self.x.shape, self.y.shape)
        return self.x, self.y

    def _read_params_data(self):
        df_params = pd.read_csv(os.path.join(self.data_root, "deg0.5/params_sampling1.csv"), dtype=float)

        scale_cols = ['Dsmax', 'depth_2', 'depth_3', 'expt_1', 'expt_2', 'expt_3', 'Ksat_1', 'Ksat_2', 'Ksat_3',
                      'init_moist_1', 'init_moist_2', 'init_moist_3',
                      'bubble_1', 'bubble_2', 'bubble_3',
                      'bulk_density_1', 'bulk_density_2', 'bulk_density_3',
                      'annual_prec', 'avg_T', 'elev']
        df_new = pd.DataFrame(df_params[scale_cols], columns=scale_cols)

        scaler1 = MinMaxScaler()
        scaler1.fit(df_new.values)
        params_scale = scaler1.transform(df_new.values)
        df_params[scale_cols] = params_scale

        grid_id_arr = self.grid_select[self.set_type]
        data_params = df_params.iloc[grid_id_arr].values

        self.x_params = [data_params for i in range(self.num_samples)]
        self.x_params = np.concatenate(self.x_params, axis=0)
        print('x_params:', self.x_params.shape)

        return self.x_params

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_params[index]

    def __len__(self):
        return self.x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



## 7 params only for our model on surro-0.5
class surroDataset7(Dataset):
    def __init__(self, data_root, size=None, Grd_tuple=None, Ts_tuple=None, scale=True, flag='train'):
        self.scaler = MinMaxScaler()
        self.data_root = data_root
        self.size = size
        self.scale = scale
        self.years = ['2016','2017', '2018', '2019', '2020']
        self.grid_select = Grd_tuple
        self.time_select = Ts_tuple

        self.Ngrid = 4783   # total grids
        self.Ntime = None  # total days

        ## select grids, if None, default to all grid points
        if self.grid_select == None:
            Ntrain = int(self.Ngrid * 0.7)
            Ntest = int(self.Ngrid * 0.1)
            Nvali = self.Ngrid - Ntrain - Ntest
            grid_train = np.arange(0, Ntrain)
            grid_test = np.arange(Ntrain, Ntrain + Ntest)
            grid_vali = np.arange(Ntrain + Ntest, self.Ngrid)
            self.grid_select = (grid_train, grid_vali, grid_test)

        if self.size == None:
            self.seq_len, self.label_len, self.pred_len = 30, 7, 7
        else:
            self.seq_len, self.label_len, self.pred_len = self.size[0], self.size[1], self.size[2]

        assert flag in ['train', 'val', 'test']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self._read_forcing_data()
        self._read_params_data()


    def _read_forcing_data(self):
        data = []
        for year in self.years:
            sub_data = np.load(os.path.join(self.data_root, "deg0.5/data_" + year + ".npy"))
            data.append(sub_data)
        data = np.concatenate(data, axis=1)

        self.Ntime = data.shape[1]
        num_train = int(self.Ntime * 0.6)
        num_test = int(self.Ntime * 0.2)
        num_vali = self.Ntime - num_train - num_test

        border1s = [0, num_train - self.seq_len, self.Ntime - num_test - self.seq_len]
        border2s = [num_train, num_train + num_vali, self.Ntime]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]
        num_samples = border2 - border1 - self.seq_len - self.pred_len + 1

        x_list, y_list = [], []
        for i in range(data.shape[0]):
            data_1 = data[i]
            train_data = data[i, border1s[0]:border2s[0], :]

            if self.scale:
                self.scaler.fit(train_data)
            data_1 = self.scaler.transform(data_1)
            data_x = data_1[border1:border2]
            data_y = data_1[border1:border2]

            x_1, y_1 = [], []
            for j in range(num_samples):
                s_begin = j
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                x_1.append(data_x[s_begin:s_end])
                y_1.append(data_y[r_begin:r_end])

            x_list.append(np.array(x_1))
            y_list.append(np.array(y_1))

        x_allgrd, y_allgrd = np.array(x_list), np.array(y_list)
        print(x_allgrd.shape, y_allgrd.shape)

        grid_id_arr = self.grid_select[self.set_type]
        time_id_arr = np.arange(0, num_samples)
        if self.time_select is not None:
            time_id_arr = self.time_select[self.set_type]

        self.num_samples = len(time_id_arr)
        print('num_samples:', self.num_samples)

        self.x, self.y = [], []
        ## get final sample list of different grid in the same date period
        for ti in time_id_arr:
            for grd in grid_id_arr:
                self.x.append(x_allgrd[grd, ti, :, :])
                self.y.append(y_allgrd[grd, ti, :, :])

        self.x, self.y = np.array(self.x), np.array(self.y)
        print('x, y shape:', self.x.shape, self.y.shape)

        return self.x, self.y


    def _read_params_data(self):
        # 7 params: ['infilt', 'Ds', 'Dsmax', 'Ws', 'depth_2', 'depth_3', 'expt_2']
        df_params = pd.read_csv(os.path.join(self.data_root, "deg0.5/params_sampling1.csv"), dtype=float) # [78496 rows x 38 columns]
        df_params = df_params[['infilt', 'Ds', 'Dsmax', 'Ws', 'depth_2', 'depth_3', 'expt_2']]

        scale_cols = ['Dsmax', 'depth_2', 'depth_3', 'expt_2']
        df_new = pd.DataFrame(df_params[scale_cols], columns=scale_cols)  #

        scaler1 = MinMaxScaler()
        scaler1.fit(df_new.values)
        params_scale = scaler1.transform(df_new.values)
        df_params[scale_cols] = params_scale

        grid_id_arr = self.grid_select[self.set_type]
        data_params = df_params.iloc[grid_id_arr].values

        self.x_params = [data_params for i in range(self.num_samples)]
        self.x_params = np.concatenate(self.x_params, axis=0)
        print('x_params:', self.x_params.shape)  # (N*731, 38)

        return self.x_params

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_params[index]

    def __len__(self):
        return self.x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)



## original params to drive vic, surroDataset_orig is only used to test our models
class surroDataset_orig(Dataset):
    def __init__(self, data_root, size=None, Grd_tuple=None, Ts_tuple=None, scale=True, flag='test'):
        self.scaler = MinMaxScaler()
        self.data_root = data_root
        self.size = size
        self.scale = scale

        self.Ngrid = 78496  # total grids
        self.Ntime = None  # total days is 1827
        self.grid_select = Grd_tuple
        self.time_select = Ts_tuple

        if self.size == None:
            self.seq_len, self.label_len, self.pred_len = 30, 7, 7
        else:
            self.seq_len, self.label_len, self.pred_len = self.size[0], self.size[1], self.size[2]

        assert flag in ['test']

        self._read_forcing_data()
        self._read_params_data()

    def _read_forcing_data(self):
        data = np.load(os.path.join(self.data_root, "forcing/data_original_2020.npy"))  # (78496, 366, 8)

        self.Ntime = data.shape[1]
        num_test = self.Ntime
        num_samples = num_test - self.seq_len - self.pred_len + 1

        x_list, y_list = [], []
        for i in range(data.shape[0]):
            data_1 = data[i]
            train_data = data[i]

            if self.scale:
                self.scaler.fit(train_data)
            data_1 = self.scaler.transform(data_1)
            data_x = data_1[0:num_test]
            data_y = data_1[0:num_test]

            x_1, y_1 = [], []
            for j in range(num_samples):
                s_begin = j
                s_end = s_begin + self.seq_len
                r_begin = s_end - self.label_len
                r_end = r_begin + self.label_len + self.pred_len
                x_1.append(data_x[s_begin:s_end])
                y_1.append(data_y[r_begin:r_end])

            x_list.append(np.array(x_1))
            y_list.append(np.array(y_1))

        x_allgrd, y_allgrd = np.array(x_list, dtype=np.float16), np.array(y_list, dtype=np.float16)
        print(x_allgrd.shape, y_allgrd.shape)  # (78496, 183, 30, 8) (78496, 183, 7, 8) include one target
        x_list, y_list = [], []

        time_id_arr = np.arange(0, num_samples)
        if self.time_select is not None:
            time_id_arr = self.time_select[0]

        self.num_samples = len(time_id_arr)
        print('num_samples:', self.num_samples)

        self.x, self.y = [], []
        ## get final sample list of different grid in the same date period
        for ti in time_id_arr:
            for grd in range(self.Ngrid):
                self.x.append(x_allgrd[grd, ti, :, :])
                self.y.append(y_allgrd[grd, ti, :, :])

        x_allgrd, y_allgrd = [], []
        self.x, self.y = np.array(self.x, dtype=np.float16), np.array(self.y, dtype=np.float16)
        print('x, y shape:', self.x.shape, self.y.shape)   # (N*num_samples, 30, 8) (N*num_samples, 7, 8)

        return self.x, self.y

    # read 38 params of params_GLO
    def _read_params_data(self):
        df_params = pd.read_csv(os.path.join(self.data_root, "soil_params/params_GLO.csv"), dtype=float) #
        # print(df_params) # [78496 rows x 38 columns]
        df_params_s = pd.read_csv(os.path.join(self.data_root, "soil_params/params_sampling1.csv"), dtype=float)

        scale_cols = ['Dsmax', 'depth_2', 'depth_3', 'expt_1', 'expt_2', 'expt_3', 'Ksat_1', 'Ksat_2', 'Ksat_3',
                      'init_moist_1', 'init_moist_2', 'init_moist_3',
                      'bubble_1', 'bubble_2', 'bubble_3',
                      'bulk_density_1', 'bulk_density_2', 'bulk_density_3',
                      'annual_prec', 'avg_T', 'elev']
        df_new = pd.DataFrame(df_params[scale_cols], columns=scale_cols)
        df_new_s = pd.DataFrame(df_params_s[scale_cols], columns=scale_cols)

        scaler1 = MinMaxScaler()
        scaler1.fit(df_new_s.values)
        params_scale = scaler1.transform(df_new.values)
        df_params[scale_cols] = params_scale

        grid_id_arr = np.arange(0, self.Ngrid)
        data_params = df_params.iloc[grid_id_arr].values

        self.x_params = [data_params for i in range(self.num_samples)]
        self.x_params = np.concatenate(self.x_params, axis=0)
        print('x_params:', self.x_params.shape)  # (N*731, 38)

        return self.x_params


    def __getitem__(self, index):
        return self.x[index], self.y[index], self.x_params[index]

    def __len__(self):
        return self.x.shape[0]

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)

