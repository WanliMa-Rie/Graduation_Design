
本项目是本科毕业论文的代码库。本课题为：基于黎曼流形上生成模型的多目标分子优化。

v1

分为两个阶段：
- VAE pretraining
- Main iteration

## File Description
- model：储存模型
    - `VAE_model_zinc.pth`: 在ZINC上训练的模型。
    - `VAE_model.pth`: 非常古早的MCF7上训练的模型。
- data：存储数据
    - `zinc.txt`: ZINC raw数据集；
    - `mcf_train.csv`: MCF7数据集；

- results：储存结果
    - bo_idl_zinc_64d:
        - `all_generated_smiles.txt`: BO_IDL生成的全部分子
        - `final_pareto_smiles.txt`: 最终Pareto前沿上的点，以及SAS和QED值（仅包含有效值）
        - `final_pareto_Y.npy`: 所有Pareto前沿上的点的SAS和QED
        - `final_pareto_Z.npy`: 所有Pareto前沿上的点的潜向量
        - `observed_Z.npy`: 所有观测点的潜向量
        - `observed_Y.npy`: 所有观测点的SAS和QED
        - `hypervolume_improvement.png`: HVI值
        - `pareto_front_final.png`: Pareto前沿上点的可视化
        - `VAE_model_zinc_final_optimized.pth`: 最终优化后的模型
    - inerpolation_analysis
    - latent

- `check.ipynb`: 随便看看
- `dataset.py`: Dataloader
- `evaluation_gener.py`: 评估VAE采样的分子（运行：`python evaluation_gener.py`）
- `geolatent.py`: 可视化latent space
- `interpolate_latent.py`: 插值
- `main.py`: 主函数：
    - `python main.py --train_model`: 训练模型
    - `python main.py --optimize`: 用BO-IDL来训练模型
- `model.py`: VAE模型代码
- `bo_idl.py`: BO-IDL代码
- `train.py`: 训练代码（提供`train`函数）
- `utils.py`: 一堆代码
