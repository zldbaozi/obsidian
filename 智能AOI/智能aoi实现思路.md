# 阶段一：数据收集 (Cold Start)**

- AOI/PaDiM 输出所有疑似 NG 的图片。
- 人工 100% 复判。
- **关键动作**：人工在界面上点击 "Pass" (假 NG) 或 "Fail" (真 NG)。后台将这些图片分别存入 `dataset/train/false_ng` (其实是 OK) 和 `dataset/train/real_ng` 文件夹

# 阶段二：辅助判断 (Assisted Mode)**

- 当积累了少量数据（例如各 50 张）后，运行**复判模型训练**（见下文代码）。 
- 系统上线：对新的 AOI NG 图片，模型先给出一个**置信度 (Confidence)**。
- 如果模型说 "99.9% 是假 NG"，系统自动 Pass（此时已部分自主）。
- 如果模型模棱两可（50%~80%），仍交由人工复判。

# 阶段三：完全自主 (Autonomous Mode)**

- 随着人工复判的数据不断回流训练，模型越来越强。
- 设定高阈值（如 >0.98），绝大部分假 NG 被自动过滤。
- 人工只需处理极少数边缘情况。


1. **切分**：脚本读取 `01_Input_Big` -> 切成小图 -> 存入 `02_Process_Buffer`。
2. **筛选**：PaDiM 扫描 `02_Process_Buffer`。
    - OK 的图 -> **直接删除** (不占硬盘)。
    - NG 的图 -> **移动** 到 `03_Pending_Review`。
    - 生成 `report.json` 记录：“Board_001 有 3 个 NG，分别在(x1,y1)...”。
3. **复判**：ResNet 扫描 `03_Pending_Review`。
    - 读取 JSON 和图片。
    - 判断切片是真假 NG，更新 JSON 里的状态 (Status: OK/NG)。
4. **展示**：上位机读取更新后的 JSON -> 在 `01_Input_Big` 的原图上画框 -> 存入 `04_Output_Result`。
5. **闭环训练 (人工介入)**：
    - 每天下班前，人工把 `03` 里没判断对的图，拖进 `Dataset_Review` 里的 `real_ng` 或 `false_ng`。
    - 如果是被 PaDiM 误报的良品，还可以把原图拖进 `Dataset_PaDiM/good`（后续可以实现自动移动，如果一整张大图中都是误报的小图）


E:\Code\AOI_Integrated\
    ├── config\                 # [预留] 存放系统配置文件
    ├── data\                   # [仓库] 存放所有训练数据 (Training Data)
    │   ├── dataset_padim\      # PaDiM 训练集 (良品图片)
    │   └── dataset_review\     # 复判模型训练集
    │       ├── train\
    │       │   ├── false_ng\   # 放入PaDiM误报的良品
    │       │   └── real_ng\    # 放入因缺陷被挑出的坏品
    │
    ├── models\                 # [产出] 存放训练好的模型文件
    │   ├── padim\              # 存放 padim_backbone.onnx 等
    │   └── resnet\             # 存放 resnet18_review.onnx
    │
    ├── run_data\               # [现场] 运行时数据流 (已从 AOI_Project 迁移至此)
    │   ├── 01_Input\           # 输入图片放入此
    │   ├── 03_Pending_Review\  # 中间层: C++ 输出的 Pending 区
    │   └── 04_Output_Result\   # 结果层: Real_NG / False_Alarm / Manual_Review
    │
    └── src\
        ├── padim_infer\        # (建议) 将你的 C++ 代码移动到这里
        ├── padim_train\        # (建议) 将 PaDiM 训练脚本移动到这里
        └── review_system\      # [已更新] 存放 review_trainer.py 和 auto_judge_service.py



为你进行了第二轮深度优化：

### 核心问题诊断

之前的训练日志显示 `Loss` 长期在 `0.65-0.75` 之间横跳，准确率也忽高忽低。这在二分类问题中通常意味着**模型完全学不到有效特征**，一直在“瞎猜”边缘徘徊（随机猜测的 Loss 为 ln⁡(2)≈0.693ln(2)≈0.693）。

可能原因有两个：

1. **SGD 陷入局部鞍点/震荡**：小样本数据通常 loss 面极其崎岖，SGD 即使 Momentum 为 0.9 也容易震荡。
2. **分辨率矛盾**：
    - `600px`：ResNet18 的感受野跟不上，特征提取失效。
    - `224px`：对于细微的“真缺陷”（如微划痕、针孔），缩放过小导致特征彻底消失。

### 本次修正方案

我修改了 `review_trainer.py`，主要变动如下：

1. **分辨率折中 (448x448)**：既然 600 太大而 224 可能又太小，采用 `448x448`。这是 ResNet 标准输入的 2 倍，既能保留更多细节，又不至于让预训练权重完全失效。
2. **优化器换将 (Adam via SGD)**：将 SGD 替换为 **Adam (lr=1e-4)**。Adam 自适应调整学习率的能力对于这种极度不稳定的 loss 面通常有奇效，能更快地找到下降方向。
3. **移除干扰性增强**：去掉了 `RandomRotation` 和 `ColorJitter`。在模型还没学会“什么是缺陷”之前，过度的颜色抖动可能会把缺陷本身（通常只是颜色差异）给抹平了。
4. **增加训练集监控**：现在日志会同时打印 `TRAIN Loss/Acc` 和 `VAL Loss/Acc`。
    - 如果 `Train Loss` 仍然降不下去（保持 > 0.6），说明模型**根本看不清图**（欠拟合）。
    - 如果 `Train Loss` 很低但 `Val Loss` 很高，说明**数据太少**（过拟合）。


### 既然遇到了这类过拟合欠拟合的问题，那就顺便总结一下：
- #### 过拟合解决策略：模型对训练数据中的噪声过度敏感（死学习，不会变通）
	1. 数据层面：
		- 数据增强
		- 数据清洗
	2. 模型架构与正则化策略：
```C++
class RegularizedModel(nn.Module):
    """集成多种正则化技术的模型"""
    
    def __init__(self, base_model, config):
        super().__init__()
        self.base_model = base_model
        self.config = config
        
        # 添加Dropout层
        self.dropout_layers = nn.ModuleList()
        for i, layer in enumerate(self.get_fc_layers()):
            if config.get(f'dropout_{i}', 0.5) > 0:
                self.dropout_layers.append(
                    nn.Dropout(config[f'dropout_{i}'])
                )
        
        # 权重约束
        self.weight_constraint = config.get('weight_constraint', None)
        
    def forward(self, x):
        x = self.base_model(x)
        
        # 应用Dropout
        for dropout_layer in self.dropout_layers:
            x = dropout_layer(x)
        
        return x
    
    def apply_weight_constraints(self):
        """应用权重约束"""
        if self.weight_constraint == 'orthogonal':
            for param in self.parameters():
                if param.dim() >= 2:
                    nn.init.orthogonal_(param)
        elif self.weight_constraint == 'spectral_norm':
            for name, module in self.named_modules():
                if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                    nn.utils.spectral_norm(module)
    
    def get_fc_layers(self):
        """获取全连接层"""
        fc_layers = []
        for module in self.base_model.modules():
            if isinstance(module, nn.Linear):
                fc_layers.append(module)
        return fc_layers
		
		```
		
```C++
class AdvancedRegularization:
    """高级正则化技术集合"""
    
    @staticmethod
    def label_smoothing_loss(pred, target, smoothing=0.1, num_classes=None):
        """标签平滑损失"""
        if num_classes is None:
            num_classes = pred.size(1)
        
        smoothed_targets = torch.zeros_like(pred).scatter_(
            1, target.unsqueeze(1), 1 - smoothing
        )
        smoothed_targets += smoothing / num_classes
        
        log_preds = F.log_softmax(pred, dim=1)
        loss = -torch.sum(smoothed_targets * log_preds, dim=1)
        return loss.mean()
    
    @staticmethod
    def stochastic_depth(layer, p, mode='train'):
        """随机深度（用于ResNet等）"""
        if mode == 'train' and torch.rand(1).item() < p:
            return nn.Identity()
        return layer
    
    @staticmethod
    def weight_decay_skip_bn(model, weight_decay=1e-4):
        """跳过BatchNorm的权重衰减"""
        decay = []
        no_decay = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            if 'bn' in name or 'bias' in name:
                no_decay.append(param)
            else:
                decay.append(param)
        
        return [
            {'params': decay, 'weight_decay': weight_decay},
            {'params': no_decay, 'weight_decay': 0.0}
        ]
    
    @staticmethod
    def gradient_clipping(model, max_norm=1.0, norm_type=2):
        """梯度裁剪"""
        torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            max_norm=max_norm, 
            norm_type=norm_type
        )
```

	3. 训练策略优化：
```C++
class AdaptiveLearningRateScheduler:
    """自适应学习率调度器"""
    
    def __init__(self, optimizer, config):
        self.optimizer = optimizer
        self.config = config
        self.schedulers = []
        
        # 组合多个调度器
        if config.get('use_warmup', True):
            self.schedulers.append(
                self._get_warmup_scheduler()
            )
        
        if config.get('use_cosine', True):
            self.schedulers.append(
                self._get_cosine_scheduler()
            )
        
        if config.get('use_plateau', True):
            self.schedulers.append(
                self._get_plateau_scheduler()
            )
    
    def _get_warmup_scheduler(self):
        """预热学习率"""
        from torch.optim.lr_scheduler import LambdaLR
        
        def warmup_lambda(epoch):
            if epoch < self.config.get('warmup_epochs', 5):
                return (epoch + 1) / self.config['warmup_epochs']
            return 1.0
        
        return LambdaLR(self.optimizer, warmup_lambda)
    
    def _get_cosine_scheduler(self):
        """余弦退火"""
        from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
        
        return CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.get('T_0', 10),
            T_mult=self.config.get('T_mult', 2),
            eta_min=self.config.get('eta_min', 1e-6)
        )
    
    def _get_plateau_scheduler(self):
        """基于验证损失的调度"""
        from torch.optim.lr_scheduler import ReduceLROnPlateau
        
        return ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=self.config.get('plateau_patience', 5),
            min_lr=1e-6,
            verbose=True
        )
    
    def step(self, val_loss=None):
        """执行调度步骤"""
        for scheduler in self.schedulers:
            if isinstance(scheduler, ReduceLROnPlateau):
                if val_loss is not None:
                    scheduler.step(val_loss)
            else:
                scheduler.step()
```

```C++
# 优化器的选择与配置
def get_optimizer(model, config):
    """获取优化器（针对过拟合优化）"""
    optimizer_type = config.get('optimizer', 'adamw')
    lr = config.get('lr', 1e-3)
    weight_decay = config.get('weight_decay', 1e-4)
    
    # 分离BatchNorm参数（不应用权重衰减）
    params = AdvancedRegularization.weight_decay_skip_bn(model, weight_decay)
    
    if optimizer_type == 'adamw':
        return torch.optim.AdamW(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'sgd':
        return torch.optim.SGD(
            params,
            lr=lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=weight_decay
        )
    elif optimizer_type == 'lamb':
        # 需要安装第三方库：pip install pytorch-lamb
        from pytorch_lamb import Lamb
        return Lamb(
            params,
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=weight_decay
        )
```

	3. 集成学习与模型选择：
	4. 监控与评估系统：

- #### 欠拟合解决策略：老师给的题都做不好，更别谈去举一反三
	1. 模型架构优化策略:
	2. 训练策略优化：








