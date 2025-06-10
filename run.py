import argparse
import torch
from exp.exp_fused_forecast import Exp_Fused_Forecast
import random
import numpy as np
import os


def main():
    # 设置随机种子
    seed = 2021
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    parser = argparse.ArgumentParser(description='FusedTimeModel')

    # 基础配置
    parser.add_argument('--task_name', type=str, default='long_term_forecast')
    parser.add_argument('--is_training', type=int, default=1)
    parser.add_argument('--model_id', type=str, default='fused_test')
    parser.add_argument('--model', type=str, default='FusedTimeModel')

    # 数据配置
    parser.add_argument('--data', type=str, default='Custom')
    parser.add_argument('--root_path', type=str, default='./dataset/')
    parser.add_argument('--data_path', type=str, default='CS2.csv')
    parser.add_argument('--features', type=str, default='MS')
    parser.add_argument('--target', type=str, default='Target')
    parser.add_argument('--freq', type=str, default='d')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/')
    parser.add_argument('--seasonal_patterns', type=str, default=None, help='seasonal patterns for M4 dataset')

    # 预测任务配置
    parser.add_argument('--seq_len', type=int, default=30)
    parser.add_argument('--label_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=1)

    # 模型配置
    parser.add_argument('--enc_in', type=int, default=8)
    parser.add_argument('--c_out', type=int, default=1)
    parser.add_argument('--d_model', type=int, default=512)
    parser.add_argument('--d_ff', type=int, default=2048)
    parser.add_argument('--moving_avg', type=int, default=25)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--embed', type=str, default='timeF')
    parser.add_argument('--channel_independence', type=int, default=1)
    parser.add_argument('--use_norm', type=int, default=1)
    parser.add_argument('--down_sampling_layers', type=int, default=2)
    parser.add_argument('--down_sampling_window', type=int, default=2)

    # 训练配置
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--des', type=str, default='fused_test')
    parser.add_argument('--lradj', type=str, default='TST')
    parser.add_argument('--pct_start', type=float, default=0.3)

    # GPU配置
    parser.add_argument('--use_gpu', type=bool, default=True)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--use_multi_gpu', action='store_true', default=False)
    parser.add_argument('--devices', type=str, default='0,1')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = 'cuda' if args.use_gpu else 'cpu'

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    if args.is_training:
        for ii in range(args.itr):
            setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_{args.des}_{ii}'

            exp = Exp_Fused_Forecast(args)
            print(f'>>>>>>>start training : {setting}>>>>>>>>>>>>>>')
            exp.train(setting)

            print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<')
            exp.test(setting)
            torch.cuda.empty_cache()
    else:
        ii = 0
        setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_{args.des}_{ii}'

        exp = Exp_Fused_Forecast(args)
        print(f'>>>>>>>testing : {setting}<<<<<<<<<<<<<<<<<<')
        exp.test(setting, test=1)
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()

