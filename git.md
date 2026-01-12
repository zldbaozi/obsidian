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
```

