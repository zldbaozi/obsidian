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
		- 模型复杂度提升框架：
```C++
class ModelComplexityEnhancer:
    """模型复杂度增强器"""
    
    @staticmethod
    def enhance_cnn_model(base_model, enhancement_level='moderate'):
        """增强CNN模型复杂度"""
        import torch.nn as nn
        
        class EnhancedCNN(nn.Module):
            def __init__(self, base_model):
                super().__init__()
                self.base_model = base_model
                
                if enhancement_level == 'light':
                    # 轻度增强：增加层宽度
                    self.enhancement = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    
                elif enhancement_level == 'moderate':
                    # 中度增强：增加深度和宽度
                    self.enhancement = nn.Sequential(
                        nn.Conv2d(256, 512, kernel_size=3, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 512, kernel_size=3, padding=1),
                        nn.BatchNorm2d(512),
                        nn.ReLU(inplace=True),
                        nn.Conv2d(512, 1024, kernel_size=3, padding=1),
                        nn.BatchNorm2d(1024),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                    
                elif enhancement_level == 'strong':
                    # 强力增强：残差连接+注意力机制
                    self.enhancement = nn.Sequential(
                        ResidualBlock(256, 512),
                        ResidualBlock(512, 512),
                        SEBlock(512),
                        ResidualBlock(512, 1024),
                        SEBlock(1024),
                        nn.AdaptiveAvgPool2d((1, 1))
                    )
                
                # 增强分类头
                self.classifier = nn.Sequential(
                    nn.Linear(1024, 512),
                    nn.BatchNorm1d(512),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.3),
                    nn.Linear(512, 256),
                    nn.BatchNorm1d(256),
                    nn.ReLU(inplace=True),
                    nn.Linear(256, base_model.num_classes)
                )
            
            def forward(self, x):
                x = self.base_model.features(x)
                x = self.enhancement(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return EnhancedCNN(base_model)
    
    @staticmethod
    def add_attention_mechanism(model, attention_type='cbam'):
        """添加注意力机制"""
        import torch.nn as nn
        
        class CBAM(nn.Module):
            """Convolutional Block Attention Module"""
            def __init__(self, channels, reduction=16):
                super().__init__()
                self.channel_attention = nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(channels, channels // reduction, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(channels // reduction, channels, 1),
                    nn.Sigmoid()
                )
                self.spatial_attention = nn.Sequential(
                    nn.Conv2d(2, 1, kernel_size=7, padding=3),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                # 通道注意力
                ca = self.channel_attention(x)
                x = x * ca
                
                # 空间注意力
                avg_out = torch.mean(x, dim=1, keepdim=True)
                max_out, _ = torch.max(x, dim=1, keepdim=True)
                sa_input = torch.cat([avg_out, max_out], dim=1)
                sa = self.spatial_attention(sa_input)
                x = x * sa
                
                return x
        
        # 在模型的特定层后添加注意力
        def insert_attention(module, name):
            for child_name, child_module in module.named_children():
                if isinstance(child_module, nn.Conv2d):
                    # 在卷积层后添加注意力
                    setattr(module, child_name, 
                           nn.Sequential(child_module, CBAM(child_module.out_channels)))
                else:
                    insert_attention(child_module, child_name)
        
        insert_attention(model, 'model')
        return model
    
    @staticmethod
    def increase_model_capacity(model, multiplier=2.0):
        """按比例增加模型容量"""
        def increase_layer_capacity(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Conv2d):
                    # 增加卷积层通道数
                    new_out_channels = int(child.out_channels * multiplier)
                    new_layer = nn.Conv2d(
                        child.in_channels, new_out_channels,
                        kernel_size=child.kernel_size,
                        stride=child.stride,
                        padding=child.padding
                    )
                    setattr(module, name, new_layer)
                elif isinstance(child, nn.Linear):
                    # 增加全连接层神经元数
                    new_out_features = int(child.out_features * multiplier)
                    new_layer = nn.Linear(child.in_features, new_out_features)
                    setattr(module, name, new_layer)
                else:
                    increase_layer_capacity(child)
        
        increase_layer_capacity(model)
        return model
```

		 - 残差连接与密集连接：
```C++
class AdvancedConnections:
    """高级连接策略"""
    
    @staticmethod
    def add_residual_connections(model):
        """为模型添加残差连接"""
        import torch.nn as nn
        
        class ResidualWrapper(nn.Module):
            def __init__(self, block):
                super().__init__()
                self.block = block
                if hasattr(block, 'out_channels'):
                    self.shortcut = nn.Conv2d(block.in_channels, block.out_channels, 
                                            kernel_size=1, stride=block.stride)
                else:
                    self.shortcut = nn.Identity()
            
            def forward(self, x):
                identity = self.shortcut(x)
                out = self.block(x)
                out += identity
                return nn.ReLU(inplace=True)(out)
        
        def wrap_residual(module):
            for name, child in module.named_children():
                if isinstance(child, nn.Sequential) and len(child) >= 2:
                    # 包装序列模块
                    setattr(module, name, ResidualWrapper(child))
                else:
                    wrap_residual(child)
        
        wrap_residual(model)
        return model
    
    @staticmethod
    def create_dense_block(in_channels, growth_rate=32, num_layers=4):
        """创建密集连接块"""
        import torch.nn as nn
        
        layers = []
        for i in range(num_layers):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_channels + i * growth_rate),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels + i * growth_rate, growth_rate, 
                         kernel_size=3, padding=1),
                nn.Dropout2d(0.2)
            ))
        
        class DenseBlock(nn.Module):
            def __init__(self):
                super().__init__()
                self.layers = nn.ModuleList(layers)
            
            def forward(self, x):
                features = [x]
                for layer in self.layers:
                    new_features = layer(torch.cat(features, dim=1))
                    features.append(new_features)
                return torch.cat(features, dim=1)
        
        return DenseBlock()
```

	2. 训练策略优化：
		- 动态学习率：
```C++
class AdaptiveLearningRateOptimizer:
    """自适应学习率优化器"""
    
    @staticmethod
    def get_optimizer_with_warmup(model, config):
        """带热身的优化器配置"""
        import torch.optim as optim
        from torch.optim.lr_scheduler import LambdaLR
        
        # 分离参数组
        param_groups = []
        
        # 卷积层参数（通常需要不同的学习率）
        conv_params = []
        # 全连接层参数
        fc_params = []
        # BatchNorm参数（通常需要较小的学习率）
        bn_params = []
        
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            if 'conv' in name or 'features' in name:
                conv_params.append(param)
            elif 'fc' in name or 'classifier' in name:
                fc_params.append(param)
            elif 'bn' in name or 'norm' in name:
                bn_params.append(param)
            else:
                conv_params.append(param)  # 默认
        
        # 为不同层设置不同的学习率
        param_groups = [
            {'params': conv_params, 'lr': config.get('conv_lr', 1e-3)},
            {'params': fc_params, 'lr': config.get('fc_lr', 1e-2)},
            {'params': bn_params, 'lr': config.get('bn_lr', 1e-4)}
        ]
        
        optimizer = optim.AdamW(
            param_groups,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=config.get('weight_decay', 1e-4)
        )
        
        # 学习率热身
        def warmup_lambda(epoch):
            if epoch < config.get('warmup_epochs', 5):
                # 线性热身
                return (epoch + 1) / config['warmup_epochs']
            else:
                # 余弦退火
                progress = (epoch - config['warmup_epochs']) / \
                          (config.get('total_epochs', 100) - config['warmup_epochs'])
                return 0.5 * (1 + math.cos(math.pi * progress))
        
        scheduler = LambdaLR(optimizer, warmup_lambda)
        
        return optimizer, scheduler
    
    @staticmethod
    def cyclical_learning_rate(optimizer, base_lr=1e-4, max_lr=1e-2, 
                              step_size=2000, mode='triangular'):
        """循环学习率策略"""
        from torch.optim.lr_scheduler import _LRScheduler
        
        class CyclicLR(_LRScheduler):
            def __init__(self, optimizer, base_lr, max_lr, step_size, mode='triangular'):
                self.base_lr = base_lr
                self.max_lr = max_lr
                self.step_size = step_size
                self.mode = mode
                self.cycle = 0
                self.step_count = 0
                super().__init__(optimizer)
            
            def get_lr(self):
                self.step_count += 1
                cycle = math.floor(1 + self.step_count / (2 * self.step_size))
                x = abs(self.step_count / self.step_size - 2 * cycle + 1)
                
                if self.mode == 'triangular':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))
                elif self.mode == 'triangular2':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) / (2 ** (cycle - 1))
                elif self.mode == 'exp_range':
                    lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x)) * (0.999 ** self.step_count)
                
                return [lr for _ in self.optimizer.param_groups]
        
        return CyclicLR(optimizer, base_lr, max_lr, step_size, mode)
    
    @staticmethod
    def one_cycle_policy(optimizer, max_lr=1e-2, total_steps=10000, 
                        pct_start=0.3, div_factor=25.0, final_div_factor=10000.0):
        """One Cycle策略"""
        from torch.optim.lr_scheduler import OneCycleLR
        
        scheduler = OneCycleLR(
            optimizer,
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        
        return scheduler
```
		- 梯度优化与二阶优化:
```C++
class GradientOptimization:
    """梯度优化策略"""
    
    @staticmethod
    def gradient_accumulation(model, train_loader, accumulation_steps=4):
        """梯度累积：模拟更大的batch size"""
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        
        model.train()
        optimizer.zero_grad()
        
        for batch_idx, (data, target) in enumerate(train_loader):
            output = model(data)
            loss = F.cross_entropy(output, target)
            
            # 归一化损失
            loss = loss / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                # 累积多个batch后更新
                optimizer.step()
                optimizer.zero_grad()
        
        return model
    
    @staticmethod
    def gradient_clipping_with_warmup(model, max_norm=1.0, warmup_steps=1000):
        """带热身的梯度裁剪"""
        total_norms = []
        
        def clip_grad_hook(grad):
            # 计算梯度范数
            total_norm = grad.norm(2).item()
            total_norms.append(total_norm)
            
            # 热身阶段逐渐增加裁剪阈值
            if len(total_norms) < warmup_steps:
                current_max_norm = max_norm * (len(total_norms) / warmup_steps)
            else:
                current_max_norm = max_norm
            
            # 应用梯度裁剪
            clip_coef = current_max_norm / (total_norm + 1e-6)
            if clip_coef < 1:
                grad.mul_(clip_coef)
            
            return grad
        
        # 为所有参数注册hook
        for param in model.parameters():
            if param.requires_grad:
                param.register_hook(clip_grad_hook)
        
        return model, total_norms
    
    @staticmethod
    def lookahead_optimizer(base_optimizer, k=5, alpha=0.5):
        """Lookahead优化器"""
        class Lookahead:
            def __init__(self, base_optimizer, k=5, alpha=0.5):
                self.base_optimizer = base_optimizer
                self.k = k
                self.alpha = alpha
                self.param_groups = self.base_optimizer.param_groups
                self.state = defaultdict(dict)
                self.fast_state = self.base_optimizer.state
                
                for group in self.param_groups:
                    group["counter"] = 0
            
            def step(self, closure=None):
                loss = None
                if closure is not None:
                    loss = closure()
                
                self.base_optimizer.step()
                
                for group in self.param_groups:
                    group["counter"] += 1
                    if group["counter"] >= self.k:
                        group["counter"] = 0
                        
                        for p in group["params"]:
                            if p.grad is None:
                                continue
                            
                            param_state = self.state[p]
                            if "slow_param" not in param_state:
                                param_state["slow_param"] = torch.zeros_like(p.data)
                                param_state["slow_param"].copy_(p.data)
                            
                            slow = param_state["slow_param"]
                            slow.add_(self.alpha, p.data - slow)
                            p.data.copy_(slow)
                
                return loss
            
            def zero_grad(self):
                self.base_optimizer.zero_grad()
        
        return Lookahead(base_optimizer, k, alpha)
```








