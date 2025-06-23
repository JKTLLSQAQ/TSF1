import argparse
import torch
from exp.exp_fused_forecast import Exp_Enhanced_Forecast
import random
import numpy as np
import os
from datetime import datetime


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
    parser.add_argument('--model', type=str, default='Diffusion Denoising')

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
    parser.add_argument('--seq_len', type=int, default=45)
    parser.add_argument('--label_len', type=int, default=15)
    parser.add_argument('--pred_len', type=int, default=1)

    # 模型配置
    parser.add_argument('--enc_in', type=int, default=9)
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

    # ============ 新增：扩散模块配置 ============
    parser.add_argument('--diffusion_steps', type=int, default=15,
                        help='扩散过程的时间步数')
    parser.add_argument('--diffusion_scheduler', type=str, default='cosine',
                        choices=['cosine', 'linear'],
                        help='扩散噪声调度器类型：cosine(推荐) 或 linear')
    parser.add_argument('--use_enhanced_diffusion', type=bool, default=True,
                        help='是否使用增强的自适应扩散模块')
    parser.add_argument('--use_cross_attention_denoiser', type=bool, default=True,
                        help='是否在季节性分量中使用交叉注意力去噪器')

    # ============ 新增：融合机制配置 ============
    parser.add_argument('--use_dynamic_fusion', type=bool, default=False,
                        help='是否使用动态权重融合（实验性功能）')
    parser.add_argument('--fusion_type', type=str, default='static',
                        choices=['static', 'dynamic'],
                        help='融合类型：static(静态权重) 或 dynamic(动态权重)')

    # ============ 新增：KAN层配置 ============
    parser.add_argument('--kan_order_trend', type=int, default=3,
                        help='趋势分量KAN层的阶数')
    parser.add_argument('--kan_order_seasonal', type=int, default=5,
                        help='季节性分量KAN层的阶数')
    parser.add_argument('--kan_order_residual', type=int, default=4,
                        help='残差分量KAN层的阶数')

    # ============ 新增：去噪器配置 ============
    parser.add_argument('--denoiser_num_heads', type=int, default=4,
                        help='交叉注意力去噪器的注意力头数')
    parser.add_argument('--denoiser_num_layers', type=int, default=1,
                        help='交叉注意力去噪器的层数')

    # 训练配置
    parser.add_argument('--num_workers', type=int, default=10)
    parser.add_argument('--itr', type=int, default=1)
    parser.add_argument('--train_epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
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

    # 实验管理配置
    parser.add_argument('--experiment_name', type=str, default='TimeFuse',
                        help='自定义实验名称，会添加到时间戳前')
    parser.add_argument('--save_detailed_results', type=bool, default=True,
                        help='是否保存详细的实验结果')
    parser.add_argument('--auto_timestamp', type=bool, default=True,
                        help='是否自动添加时间戳')

    args = parser.parse_args()
    args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False
    args.device = 'cuda' if args.use_gpu else 'cpu'

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    # ============ 配置验证和自动调整 ============
    print("🔧 验证和调整配置...")

    # 根据融合类型设置动态融合
    if args.fusion_type == 'dynamic':
        args.use_dynamic_fusion = True
    elif args.fusion_type == 'static':
        args.use_dynamic_fusion = False

    # 验证扩散配置
    if args.diffusion_steps <= 0:
        print("⚠️  警告：diffusion_steps <= 0，重置为默认值 20")
        args.diffusion_steps = 20

    if args.diffusion_scheduler not in ['cosine', 'linear']:
        print("⚠️  警告：不支持的调度器类型，重置为 'cosine'")
        args.diffusion_scheduler = 'cosine'

    # 验证KAN阶数
    for component in ['trend', 'seasonal', 'residual']:
        order_attr = f'kan_order_{component}'
        order_value = getattr(args, order_attr)
        if order_value < 2:
            print(f"⚠️  警告：{order_attr} < 2，重置为 3")
            setattr(args, order_attr, 3)
        elif order_value > 10:
            print(f"⚠️  警告：{order_attr} > 10，可能导致过拟合，建议降低")

    # 实验信息输出
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print('=' * 100)
    print(f'🚀 FusedTimeModel 实验开始')
    print(f'⏰ 开始时间: {current_time}')
    print(f'📊 数据集: {args.data_path}')
    print(f'🤖 模型: {args.model}')
    print(f'📏 序列长度: {args.seq_len}, 预测长度: {args.pred_len}')
    print(f'🎯 特征类型: {args.features}, 目标变量: {args.target}')

    # 扩散配置信息
    print(f'\n🌀 扩散模块配置:')
    print(f'   - 启用增强扩散: {args.use_enhanced_diffusion}')
    print(f'   - 扩散步数: {args.diffusion_steps}')
    print(f'   - 调度器类型: {args.diffusion_scheduler}')
    print(f'   - 交叉注意力去噪: {args.use_cross_attention_denoiser}')

    # KAN配置信息
    print(f'\n🧠 KAN层配置:')
    print(f'   - 趋势分量阶数: {args.kan_order_trend}')
    print(f'   - 季节性分量阶数: {args.kan_order_seasonal}')
    print(f'   - 残差分量阶数: {args.kan_order_residual}')

    # 融合配置信息
    print(f'\n🔗 融合机制配置:')
    print(f'   - 融合类型: {args.fusion_type}')
    print(f'   - 动态权重: {args.use_dynamic_fusion}')

    if args.experiment_name:
        print(f'\n🏷️  实验名称: {args.experiment_name}')
    print('=' * 100)

    # 详细参数信息（可选输出）
    if args.is_training:
        print('\n📋 详细参数配置:')
        for key, value in sorted(vars(args).items()):
            if key.startswith(('diffusion_', 'kan_', 'denoiser_', 'fusion_', 'use_')):
                print(f'   {key}: {value}')

    print(f'\n🔧 完整参数:')
    print(args)

    if args.is_training:
        for ii in range(args.itr):
            # 构建实验设置名称
            base_setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_{args.des}_{ii}'

            # 添加扩散配置到设置名称
            diffusion_suffix = f'_diff{args.diffusion_steps}_{args.diffusion_scheduler}'
            if args.use_dynamic_fusion:
                diffusion_suffix += '_dynfusion'

            if args.experiment_name:
                setting = f'{args.experiment_name}_{base_setting}{diffusion_suffix}'
            else:
                setting = f'{base_setting}{diffusion_suffix}'

            exp = Exp_Fused_Forecast(args)
            print(f'\n🏃 开始训练: {setting}')
            print('>' * 80)
            exp.train(setting)

            print(f'\n🧪 开始测试: {setting}')
            print('>' * 80)
            exp.test(setting)

            # 输出实验完成信息
            finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print('=' * 100)
            print(f'✅ 实验完成!')
            print(f'⏰ 完成时间: {finish_time}')
            print(f'📁 结果文件夹: ./results/{setting}_{exp.experiment_timestamp}/')
            print(f'📊 扩散配置: {args.diffusion_steps}步 + {args.diffusion_scheduler}调度器')
            print(
                f'🧠 KAN配置: 趋势{args.kan_order_trend}/季节{args.kan_order_seasonal}/残差{args.kan_order_residual}阶')
            print(f'🔗 融合方式: {args.fusion_type}')
            print('=' * 100)

            torch.cuda.empty_cache()
    else:
        ii = 0
        base_setting = f'{args.task_name}_{args.model_id}_{args.model}_{args.data}_sl{args.seq_len}_pl{args.pred_len}_dm{args.d_model}_{args.des}_{ii}'

        diffusion_suffix = f'_diff{args.diffusion_steps}_{args.diffusion_scheduler}'
        if args.use_dynamic_fusion:
            diffusion_suffix += '_dynfusion'

        if args.experiment_name:
            setting = f'{args.experiment_name}_{base_setting}{diffusion_suffix}'
        else:
            setting = f'{base_setting}{diffusion_suffix}'

        exp = Exp_Fused_Forecast(args)
        print(f'\n🧪 开始测试: {setting}')
        print('>' * 80)
        exp.test(setting, test=1)

        finish_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print('=' * 100)
        print(f'✅ 测试完成!')
        print(f'⏰ 完成时间: {finish_time}')
        print(f'📁 结果文件夹: ./results/{setting}_{exp.experiment_timestamp}/')
        print('=' * 100)

        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()