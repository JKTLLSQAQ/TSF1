from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
# 确保导入 matplotlib
import matplotlib.pyplot as plt
from utils.tools import EarlyStopping, adjust_learning_rate # 移除了 visual
from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
# 在其他 import 语句附近
from sklearn.metrics import r2_score
# ... 其他导入 ...
from utils.metrics import metric # 确保这个也在
from utils.util import AverageMeter # 假设它在 util.py 中
# ...
from models.PiTransformer import PiTransformerLoss
from plugin.Plugin.model import Plugin


warnings.filterwarnings('ignore')

class Exp_Long_Term_Forecast(Exp_Basic):
    def __init__(self, args):
        super(Exp_Long_Term_Forecast, self).__init__(args)

    def _build_model(self):
        # 检查模型名称是否在字典中，增加 robustness
        if self.args.model not in self.model_dict:
            raise KeyError(
                f"Model '{self.args.model}' not found in model_dict. Available models: {list(self.model_dict.keys())}")

        # 创建基础模型
        base_model = self.model_dict[self.args.model].Model(self.args).float()

        # 根据 flag 决定是否使用 Plugin
        if self.args.flag == 'Plugin':
            # 为 Plugin 准备参数
            plugin_args = type('', (), {})()
            plugin_args.dim = self.args.plugin_dim
            plugin_args.head_num = self.args.plugin_head_num
            plugin_args.dff = self.args.plugin_dff
            plugin_args.layer_num = self.args.plugin_layer_num
            plugin_args.dropout = self.args.plugin_dropout
            plugin_args.q = self.args.plugin_q
            plugin_args.hist_len = self.args.seq_len
            plugin_args.pred_len = self.args.pred_len

            # 确定通道数
            channel = self.args.c_out if hasattr(self.args, 'c_out') else self.args.enc_in

            # 创建包装类
            class EnhancedModel(nn.Module):
                def __init__(self, base_model, plugin):
                    super(EnhancedModel, self).__init__()
                    self.base_model = base_model
                    self.plugin = plugin

                def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
                    # 先用基础模型生成预测
                    base_output = self.base_model(x_enc, x_mark_enc, x_dec, x_mark_dec)

                    # 使用 Plugin 增强
                    if hasattr(self, 'output_attention') and self.output_attention:
                        # 如果模型输出带有注意力
                        enhanced_output = self.plugin(x_enc, x_mark_enc, base_output[0], x_mark_dec)
                        return enhanced_output, base_output[1]  # 返回增强后的结果和注意力
                    else:
                        # 标准情况
                        enhanced_output = self.plugin(x_enc, x_mark_enc, base_output, x_mark_dec)
                        return enhanced_output

            # 创建 Plugin 模块
            plugin_module = Plugin(plugin_args, channel)

            # 创建增强模型
            model = EnhancedModel(base_model, plugin_module)

            # 继承基础模型的属性
            if hasattr(base_model, 'output_attention'):
                model.output_attention = base_model.output_attention
            if hasattr(base_model, 'adaptive_weighting'):
                model.adaptive_weighting = base_model.adaptive_weighting
            if hasattr(base_model, 'get_kan_regularization_loss'):
                model.get_kan_regularization_loss = base_model.get_kan_regularization_loss
        else:
            # 使用标准模型
            model = base_model

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        return model.float()

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        if hasattr(self.model, 'adaptive_weighting'):
            criterion = PiTransformerLoss(self.model)
        else:
            criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        total_pde_loss = []  # Track PDE loss for monitoring
        total_physics_loss = []  # Track physics constraint loss

        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                # batch_y 应在后面截取后再移到 device
                batch_y_full = batch_y.float() # 保留完整 batch_y

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)


                # decoder input
                dec_inp = torch.zeros_like(batch_y_full[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_full[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                # 截取对应的真实值部分
                batch_y = batch_y_full[:, -self.args.pred_len:, f_dim:].to(self.device)

                pred = outputs.detach().cpu()
                true = batch_y.detach().cpu()

                loss = criterion(pred, true)

                total_loss.append(loss.item()) # 使用 .item() 获取标量值
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
        criterion = self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        # 获取 KAN 正则化强度参数 (即使不用也先获取)
        kan_reg_lambda = getattr(self.args, 'kan_reg_lambda', 0.001)
        use_kan_regularization = kan_reg_lambda > 0
        if use_kan_regularization:
            print(f"KAN regularization enabled with lambda = {kan_reg_lambda}")
            # ==============================

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_main_loss_meter = AverageMeter() # 使用 AverageMeter 记录损失
            train_kan_loss_meter = AverageMeter()

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                print(f"--- Iteration {i} ---")
                print(f"DataLoader raw output batch_x shape: {batch_x.shape}")
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y_full = batch_y.float() # 保留完整 batch_y
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_full[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_full[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                if batch_x.ndim == 2:
                    # 如果 batch_x 是二维 [B, T]，添加特征维度变为 [B, T, 1]
                    batch_x = batch_x.unsqueeze(-1)
                    # print(f"Adjusted batch_x shape in train: {batch_x.shape}") # (可选) 打印确认

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        if isinstance(criterion, PiTransformerLoss):
                            total_loss, loss_dict = criterion.compute_loss(batch_x, batch_y, outputs)

                            # 记录不同损失组件
                            train_loss.append(total_loss.item())

                            # 打印更详细的损失信息
                            if (i + 1) % 100 == 0:
                                print(f"\titers: {i + 1}, epoch: {epoch + 1}")
                                print(f"\tTotal loss: {total_loss.item():.7f}")
                                print(f"\tData loss: {loss_dict['data_loss']:.7f}")
                                print(f"\tPDE loss: {loss_dict['pde_loss']:.7f} (weight α: {loss_dict['alpha']:.4f})")
                                print(
                                    f"\tPhysics loss: {loss_dict['physics_loss']:.7f} (weight β: {loss_dict['beta']:.4f})")
                        else:

                            f_dim = -1 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y_full[:, -self.args.pred_len:, f_dim:].to(self.device)
                            main_loss = criterion(outputs, batch_y)

                        # --- 添加 KAN 正则化 ---
                        total_loss = main_loss
                        kan_loss_value = 0.0
                        if use_kan_regularization and hasattr(self.model, 'get_kan_regularization_loss'):
                            kan_reg_loss = self.model.get_kan_regularization_loss()
                            total_loss = total_loss + kan_reg_lambda * kan_reg_loss.to(total_loss.device)
                            kan_loss_value = kan_reg_loss.item()

                        train_loss.append(total_loss.item())
                        train_main_loss_meter.update(main_loss.item())
                        train_kan_loss_meter.update(kan_loss_value)

                else: # 非 AMP
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -1 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y_full[:, -self.args.pred_len:, f_dim:].to(self.device)
                    main_loss = criterion(outputs, batch_y)

                    # --- 添加 KAN 正则化 ---
                    total_loss = main_loss
                    kan_loss_value = 0.0
                    if use_kan_regularization and hasattr(self.model, 'get_kan_regularization_loss'):
                        kan_reg_loss = self.model.get_kan_regularization_loss()
                        total_loss = total_loss + kan_reg_lambda * kan_reg_loss.to(total_loss.device)
                        kan_loss_value = kan_reg_loss.item()
                    elif use_kan_regularization and i == 0 and epoch == 0: # 仅在第一次迭代警告
                         warnings.warn("KAN regularization lambda > 0 but model does not have 'get_kan_regularization_loss' method.")

                    train_loss.append(total_loss.item())
                    train_main_loss_meter.update(main_loss.item())
                    train_kan_loss_meter.update(kan_loss_value)


                # 打印信息
                if (i + 1) % 100 == 0:
                    print_loss = total_loss.item()
                    print_main_loss = main_loss.item()
                    print_kan_loss = kan_loss_value * kan_reg_lambda if use_kan_regularization else 0.0
                    print(f"\titers: {i + 1}, epoch: {epoch + 1} | total loss: {print_loss:.7f} (main: {print_main_loss:.7f}, kan_reg: {print_kan_loss:.7f})")

                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                # 反向传播
                if self.args.use_amp:
                    scaler.scale(total_loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    total_loss.backward()
                    model_optim.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss_avg = np.average(train_loss) # 这是总损失的平均值
            vali_loss = self.vali(vali_data, vali_loader, criterion) # 验证损失是主损失
            test_loss = self.vali(test_data, test_loader, criterion) # 测试损失是主损失

            # =====================================================
            # >>> 移除这里的绘图调用 <<<
            # if epoch % 1 == 0:
            #     print("Generating prediction plots...")
            #     # self._visualize_predictions(train_loader, phase='train', epoch=epoch) # 注释掉
            #     # self._visualize_predictions(vali_loader, phase='val', epoch=epoch) # 注释掉
            # =====================================================

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss_avg, vali_loss, test_loss))
            # (可选) 打印更详细的损失信息
            print(f"  Avg Train Main Loss: {train_main_loss_meter.avg:.7f}, Avg Train KAN Reg Loss: {train_kan_loss_meter.avg * kan_reg_lambda:.7f}")


            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        # 加载最佳模型状态
        try:
            self.model.load_state_dict(torch.load(best_model_path))
            print("Loaded best model state for testing.")
        except Exception as e:
            print(f"Warning: Failed to load best model state dict from {best_model_path}. Using last state. Error: {e}")


        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        if test:
            print('loading model')
            model_path = os.path.join('./checkpoints/' + setting, 'checkpoint.pth')
            if not os.path.exists(model_path):
                 print(f"Error: Model checkpoint not found at {model_path}")
                 return
            try:
                # 加载模型状态字典到当前模型实例
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            except Exception as e:
                print(f"Error loading model state_dict: {e}")
                return

        preds = []
        trues = []
        # folder_path_old = './test_results/' + setting + '/' # 旧的逐批次图片保存路径，不再需要
        # if not os.path.exists(folder_path_old):
        #     os.makedirs(folder_path_old)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y_full = batch_y.float() # 保留完整 batch_y
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y_full[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y_full[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    if self.args.output_attention:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                    else:
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, -self.args.pred_len:, :] # 保留所有特征维度，方便反标
                batch_y = batch_y_full[:, -self.args.pred_len:, :].to(self.device) # 对应截取真实值

                pred_batch = outputs.detach().cpu().numpy()
                true_batch = batch_y.detach().cpu().numpy()

                preds.append(pred_batch)
                trues.append(true_batch)


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('Test results concatenated shape:', preds.shape, trues.shape)

        # --- 反标准化 ---
        if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
             print("Inverse transforming test results...")
             original_shape_preds = preds.shape
             original_shape_trues = trues.shape
             try:
                 # 假设 inverse_transform 期望 (N, F) 输入
                 preds_2d = preds.reshape(-1, original_shape_preds[-1])
                 trues_2d = trues.reshape(-1, original_shape_trues[-1])

                 preds_inv = test_data.inverse_transform(preds_2d)
                 trues_inv = test_data.inverse_transform(trues_2d)

                 preds = preds_inv.reshape(original_shape_preds)
                 trues = trues_inv.reshape(original_shape_trues)
                 print("Inverse transform successful.")
             except Exception as e:
                 print(f"Error during inverse transform: {e}. Using standardized results for metrics and plotting.")
                 # 如果反标失败，指标和图将使用标准化结果

        # --- 在计算指标和绘图 *之前*，截取目标特征维度 ---
        # 无论是否反标成功，都需要截取目标维度用于评估和绘图
        preds_target = preds[:, :, f_dim:]
        trues_target = trues[:, :, f_dim:]
        print('Target shape for metrics/plot:', preds_target.shape, trues_target.shape)


        # --- 计算指标 (使用截取后的目标特征) ---
        mae, mse, rmse, mape, mspe = metric(preds_target, trues_target)
        try:
            # 将 3D (或 2D) 数组展平成 1D，以便计算整体 R²
            trues_flat = trues_target.reshape(-1)
            preds_flat = preds_target.reshape(-1)
            r2 = r2_score(trues_flat, preds_flat)
            print('mse:{}, mae:{}, r2:{}'.format(mse, mae, r2))  # 修改打印语句
        except Exception as e:
            print(f"Could not calculate R2 score: {e}")
            r2 = np.nan  # 或者设置为 None 或其他标记值
            print('mse:{}, mae:{}'.format(mse, mae))  # 保持原样打印

        # --- 结果保存 (保存目标特征的预测和真实值) ---
        folder_path_results = './results/' + setting + '/'
        if not os.path.exists(folder_path_results):
            os.makedirs(folder_path_results)

        f = open("result_long_term_forecast.txt", 'a')
        f.write(setting + "  \n")
        # 修改写入文件的内容以包含 R²
        f.write('mse:{}, mae:{}, r2:{}'.format(mse, mae, r2))
        f.write('\n')
        f.write('\n')
        f.close()

        # 修改保存的 metrics 文件以包含 R²
        # 注意：r2 变量可能因为异常处理变成 nan
        metrics_array = np.array([mae, mse, rmse, mape, mspe, r2])
        np.save(folder_path_results + 'metrics.npy', metrics_array)
        np.save(folder_path_results + 'pred_target.npy', preds_target)  # 保存目标预测
        np.save(folder_path_results + 'true_target.npy', trues_target)  # 保存目标真实值
        # (可选) 保存所有特征的预测/真实值
        # np.save(folder_path_results + 'pred_all.npy', preds)
        # np.save(folder_path_results + 'true_all.npy', trues)


        # =====================================================
        # >>> 绘制最终的对比图 (使用目标特征) <<<

        if preds_target.shape[1] == 1: # pred_len == 1 的情况
            plot_preds = preds_target.squeeze()
            plot_trues = trues_target.squeeze()
            if plot_preds.ndim == 0: # 如果只有一个样本点
                 plot_preds = np.array([plot_preds])
                 plot_trues = np.array([plot_trues])
            time_axis = np.arange(len(plot_preds))
            x_label = "Test Sample Index"
            plot_title = f"Prediction vs True (pred_len=1)"
        else: # pred_len > 1 的情况
            # 将所有窗口连接成一条曲线
            plot_preds = preds_target.reshape(-1, preds_target.shape[-1]).squeeze()
            plot_trues = trues_target.reshape(-1, trues_target.shape[-1]).squeeze()
            time_axis = np.arange(len(plot_preds))
            x_label = "Time Step (Concatenated)"
            plot_title = f"Prediction vs True (pred_len={self.args.pred_len}) - {setting}"

        plt.figure(figsize=(20, 8)) # 增大图像尺寸以便看得更清楚
        plt.plot(time_axis, plot_trues, color='blue', label='True Value', linewidth=1.5, marker='.', markersize=2, linestyle='-') # 可以加点标记
        plt.plot(time_axis, plot_preds, color='red', label='Prediction', linewidth=1.5, alpha=0.7, marker='.', markersize=2, linestyle='--') # 可以用虚线区分

        # 根据是否成功反标设置 Y 轴标签
        y_label = "Value"
        if hasattr(test_data, 'scale') and test_data.scale and self.args.inverse:
             # 假设反标成功了 (即使前面有 try-except，这里也这样假设)
             y_label = "Inverse-Transformed Value"
        else:
             y_label = "Standardized Value"


        plt.title(plot_title, fontsize=16)
        plt.xlabel(x_label, fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend(fontsize=10)
        plt.grid(True, linestyle=':', alpha=0.6)
        plt.tight_layout() # 调整布局防止标签重叠

        # 定义保存路径
        plot_save_path = os.path.join(folder_path_results, 'final_prediction_comparison.png')
        plt.savefig(plot_save_path)
        print(f"Final comparison plot saved to {plot_save_path}")
        plt.close() # 关闭图形，释放内存
        # =====================================================

        return # test 方法结束
