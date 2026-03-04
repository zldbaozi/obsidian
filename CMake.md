# 删除不想要的CMake项目以及增添新的项目：

首先确保你现在的cmake是干净的，没有build目录
1. **打开命令面板**（`Ctrl+Shift+P` 或 `Cmd+Shift+P`）
2. 输入并执行：**`CMake: Delete Build Directory`**
3. 然后执行：**`CMake: Clean Reconfigure`**

去.vscode里面找到setting.json中的CMake配置
**更改内容：**
- `cmake.sourceDirectory`: `E:/Code/PaDiM_cpp` → `E:/Code/AOI_Integrated/src/infer_cpp`
- `cmake.buildDirectory`: `${workspaceFolder}/PaDiM_cpp/build` → `${workspaceFolder}/AOI_Integrated/src/infer_cpp/build`
以及一些运行所需要的参数并在你的项目文件夹下创建build

然后再在CMake Tool里面删除不需要的项目
1. **按 `Ctrl+Shift+P`** 打开命令面板
2. 输入 `CMake: Remove Folder from Workspace` 或 `CMake: 从工作区删除文件夹`
3. 选择 `e:\Code\PaDiM_cpp\build`

最后是清理CMake旧缓存，可以使用清理脚本



