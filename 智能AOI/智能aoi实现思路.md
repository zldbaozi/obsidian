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






