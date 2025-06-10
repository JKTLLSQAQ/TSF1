from exp.exp_basic import Exp_Basic
from models import FusedTimeModel
from data_provider.data_factory import data_provider
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import pandas as pd
from torch.optim import lr_scheduler
from tqdm import tqdm  # 添加tqdm导入

warnings.filterwarnings('ignore')


class Exp_Fused_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Fused_Forecast, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 模型前向
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                f_dim = -1 if self.args.features == 'MS' else 0
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                total_loss.append(loss.item())

                # 收集预测和真实值用于计算R²
                preds.append(outputs.detach().cpu().numpy())
                trues.append(batch_y.detach().cpu().numpy())

        total_loss = np.average(total_loss)

        # 计算验证集上的R²
        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        _, _, _, _, _ = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())

        self.model.train()
        return total_loss, r2

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        train_steps = len(train_loader)

        # 修改：设置早停patience为3，最大epoch为20
        early_stopping = EarlyStopping(patience=3, verbose=True)
        max_epochs = 20

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=model_optim,
            steps_per_epoch=train_steps,
            pct_start=self.args.pct_start,
            epochs=max_epochs,  # 使用20个epoch
            max_lr=self.args.learning_rate
        )

        # 添加：创建总的epoch进度条
        epoch_pbar = tqdm(range(max_epochs), desc="Training Epochs", unit="epoch")

        for epoch in epoch_pbar:
            iter_count = 0
            train_loss = []
            self.model.train()
            epoch_time = time.time()

            # 添加：为每个epoch的batch创建进度条
            batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{max_epochs}",
                              leave=False, unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(batch_pbar):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # 前向传播
                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                # 更新：batch进度条显示当前loss
                batch_pbar.set_postfix({'Loss': f"{loss.item():.6f}"})

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((max_epochs - epoch) * train_steps - i)
                    tqdm.write(f"\titers: {i + 1}, epoch: {epoch + 1} | loss: {loss.item():.7f}")
                    tqdm.write(f'\tspeed: {speed:.4f}s/iter; left time: {left_time:.4f}s')
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                model_optim.step()

                if self.args.lradj == 'TST':
                    # adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)
                    scheduler.step()

            batch_pbar.close()  # 关闭batch进度条

            epoch_cost_time = time.time() - epoch_time
            train_loss = np.average(train_loss)
            vali_loss, vali_r2 = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_r2 = self.vali(test_data, test_loader, criterion)

            # 更新：使用tqdm.write输出结果，避免与进度条冲突
            tqdm.write(f"Epoch: {epoch + 1} cost time: {epoch_cost_time:.2f}s")
            tqdm.write(
                f"Epoch: {epoch + 1}, Steps: {train_steps} | Train Loss: {train_loss:.7f} Vali Loss: {vali_loss:.7f} Test Loss: {test_loss:.7f}")
            tqdm.write(f"Vali R²: {vali_r2:.4f} Test R²: {test_r2:.4f}")

            # 更新：epoch进度条显示当前指标
            epoch_pbar.set_postfix({
                'Train_Loss': f"{train_loss:.6f}",
                'Vali_R2': f"{vali_r2:.4f}",
                'Test_R2': f"{test_r2:.4f}"
            })

            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                tqdm.write("Early stopping triggered!")
                tqdm.write(f"Training stopped at epoch {epoch + 1}")
                break

        epoch_pbar.close()  # 关闭epoch进度条

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        # 最终训练完成后计算训练集R²
        tqdm.write("Computing final training metrics...")
        final_train_loss, final_train_r2 = self.train_step_metrics(train_loader, criterion)
        tqdm.write(f"Final Training R²: {final_train_r2:.6f}")

        return self.model

    def train_step_metrics(self, train_loader, criterion):
        """计算训练集上的R²指标（可选调用）"""
        train_preds = []
        train_trues = []
        train_loss = []

        self.model.eval()
        with torch.no_grad():
            # 添加：为训练集指标计算添加进度条
            metric_pbar = tqdm(enumerate(train_loader), desc="Computing train metrics",
                               total=min(100, len(train_loader)), leave=False)

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in metric_pbar:
                if i >= 100:  # 只计算前100个batch，避免计算时间过长
                    break

                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                loss = criterion(outputs, batch_y)
                train_loss.append(loss.item())

                train_preds.append(outputs.detach().cpu().numpy())
                train_trues.append(batch_y.detach().cpu().numpy())

            metric_pbar.close()

        if train_preds:
            train_preds = np.concatenate(train_preds, axis=0)
            train_trues = np.concatenate(train_trues, axis=0)
            # 修复：使用与vali函数相同的R²计算方式
            _, _, _, _, _ = metric(train_preds, train_trues)
            from sklearn.metrics import r2_score
            train_r2 = r2_score(train_trues.flatten(), train_preds.flatten())
        else:
            train_r2 = 0.0

        self.model.train()
        return np.average(train_loss), train_r2

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))

        preds = []
        trues = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        with torch.no_grad():
            # 添加：为测试过程添加进度条
            test_pbar = tqdm(test_loader, desc="Testing", unit="batch")

            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_pbar):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs = self.model(batch_x, batch_x_mark, None, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                batch_y = batch_y[:, -self.args.pred_len:, f_dim:]

                pred = outputs.detach().cpu().numpy()
                true = batch_y.detach().cpu().numpy()

                preds.append(pred)
                trues.append(true)

            test_pbar.close()

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        from sklearn.metrics import r2_score
        r2 = r2_score(trues.flatten(), preds.flatten())
        print(f'mse:{mse:.6f}, mae:{mae:.6f}')
        print(f'rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}')
        print(f'R²:{r2:.6f}')

        # 保存结果到文件
        folder_path = './results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        preds_unscaled = test_data.scaler.inverse_transform(np.zeros((len(preds), test_data.data_x.shape[-1])))
        preds_unscaled[:, -1] = preds.flatten()
        preds_unscaled = preds_unscaled[:, -1]  # 仅提取反归一化后的预测值
        raw_df = test_data.raw_test_df
        num_preds = len(preds_unscaled)
        true_targets = raw_df[self.args.target].values[self.args.seq_len: self.args.seq_len + num_preds]

        cycle_data = raw_df['date'].values[self.args.seq_len: self.args.seq_len + num_preds]
        results_df = pd.DataFrame({
            'Cycle': cycle_data,
            'True_Target': true_targets,
            'Predicted_Target': preds_unscaled
        })
        results_csv_path = os.path.join(folder_path, 'forecast_results.csv')
        results_df.to_csv(results_csv_path, index=False)
        print(f"结果已保存至 {results_csv_path}")

        # 保存指标
        np.save(folder_path + 'metrics.npy', np.array([mae, mse, rmse, mape, mspe, r2]))
        np.save(folder_path + 'pred.npy', preds)
        np.save(folder_path + 'true.npy', trues)

        # 保存到文本文件
        f = open("result_fused_forecast.txt", 'a', encoding='utf-8')
        f.write(setting + "\n")
        f.write(f'mse:{mse:.6f}, mae:{mae:.6f}, rmse:{rmse:.6f}, mape:{mape:.6f}, mspe:{mspe:.6f}, R2:{r2:.6f}\n')
        f.write('\n')
        f.close()

        return