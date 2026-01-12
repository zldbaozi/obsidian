```
清理未引用的对象
git gc --prune=now --aggressive
```

```
查看暂存区状态
git status
从暂存区移除文件
git reset file1.txt
列出暂存区的文件及其大小
git diff --cached --name-only | xargs -I{} du -h {}
```

```
查看当前提交的文件
git show --name-only
查看当前提交的对象大小
git diff --cached --numstat | awk '{sum += $1} END {print sum " KB"}'
检查整个仓库大小
git count-objects -vH
列出历史记录中最大的文件
git rev-list --objects --all | git cat-file --batch-check='%(objectname) %(objecttype) %(rest)' | sort -k 3 -n | tail -10
删除大文件
git filter-repo --path （文件名） --invert-paths
删除git文件夹
rm -rf .git 
```

```
合并本地和远程历史
git pull padim master --allow-unrelated-histories
查看当前远程库
git remote -v
删除指定的远程库
git remote remove padim
```

```
将本地 Git 仓库中当前分支的「所有文件」，从 Git 版本库（.git 目录）「检出 / 恢复」到本地工作区
git checkout . 
克隆远程仓库的【windows分支】到本地depend文件夹
git clone -b windows git@192.168.101.149:Temp/Depend.git depend
在遇到克隆大仓库报错内存不足时，选用指定分支+浅克隆的方式组合克隆；--depth=1 是「浅克隆」，只下载远程分支的最新版本代码，不下载历史提交记录，压缩包体积从几十 G → 几 G，解压内存占用直接降到最低，永远不会内存不足。
```

```
【拆分流程 / 解压减负】`手动分步克隆(fetch+checkout)` 「专治解压失败」
# 步骤1：新建文件夹+初始化空仓库 mkdir depend && cd depend && git init
# 步骤2：关联远程仓库地址 git remote add origin git@192.168.101.149:Temp/Depend.git 
# 步骤3：只下载远程windows分支数据（不解压，零内存压力） git fetch origin windows 
# 步骤4：本地检出代码（分片解压，写入文件） git checkout -b windows FETCH_HEAD
```
