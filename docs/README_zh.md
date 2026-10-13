# POS-Mamba 中文说明

POS-Mamba 用于根据多角度冠状动脉 CPR 图像评估直径狭窄程度。模型结合空间角度连续表示（SACR）、MambaVision 语义分支、Haar 频率分支、像素门控融合、六个临床先验任务头和概率序数监督。仓库提供预处理、模型、训练、评估及合成数据示例。

安装、数据格式和完整参数见[主 README](../README.md)。真实训练需要兼容的 CUDA/PyTorch 环境及 `mamba-ssm` 扩展，建议在 Linux 或兼容的 WSL2 环境配置。

```bash
python -m pos_mamba --csv /path/metadata.csv --image_root /path/images --group_col patient_id --test_groups_file /path/fixed_holdout.json --save_path outputs/run1 --gpu 0
```

CSV 每行对应一条血管，包含唯一的 `record_id`、跨血管一致的 `patient_id` 和 `stenosis_percentage`。图像按 `record_id_cpr_view0.png` 等命名并直接存放在图像目录。患者划分文件为 `{"test_groups": ["患者ID1", "患者ID2"]}`，只用于本地运行。

默认训练配置：MambaVision-T-1K、四级 Haar、最多 30 个视角、裁剪宽度 40、batch size 4、AdamW 学习率 1e-4、weight decay 1e-3、100 epochs、余弦退火、早停 patience 20、seed 42、sigma 10、概率监督权重 0.9。门控输出单个空间权重，在通道间广播；融合完成后再全局池化。

默认三个开发集划分复用同一个固定留出集。先完成各划分的训练，再根据最佳检查点的验证集 T1 macro-F1 选择主模型并保存选择记录，最后评估留出集。未提供留出患者清单时，程序按 seed 42 生成约 20% 的患者留出集。可通过 `--development_splits_file` 提供预定义开发集划分，每个条目包含 `train_groups` 和 `val_groups`，详见主 README。比较实验时应使用相同的患者划分。

T4–T6 的评估使用真实标签确定的适用子集。二分类 F1 同样采用 macro 平均；AUC 使用概率；患者级预测取已有血管预测的最大值。输出目录包含划分、检查点、训练历史、六任务预测、患者预测、主模型指标及三个检查点的均值和样本标准差。批处理会裁剪到该批次的最小图像尺寸，比较实验时应固定 batch size 和数据顺序。

```bash
python -m pip install -e '.[dev]'
python -m pytest -q
```

测试使用合成数据和小型卷积骨干，覆盖损失、梯度、预处理、患者隔离和完整导出流程，可在 CPU 上运行，无需下载模型。完整 MambaVision 训练需要上述 CUDA 扩展。仓库不包含临床数据、患者标识、训练检查点和私有实验输出。

项目自有代码使用 MIT。MambaVision 的代码及权重遵守上游 NVIDIA 非商业许可证，详见[第三方说明](../THIRD_PARTY_NOTICES.md)。
