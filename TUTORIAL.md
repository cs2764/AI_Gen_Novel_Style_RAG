# Style-RAG 使用教程

本教程介绍如何使用 Style-RAG 系统管理文章索引。

---

## 快速开始

### 1. 激活虚拟环境

```powershell
# Windows PowerShell
.\.venv\Scripts\activate
```

### 2. 运行管理程序

```bash
python run.py
```

这将启动交互式菜单，可以选择所有操作。

---

## 交互式菜单

运行 `python run.py` 后显示：

```
╔══════════════════════════════════════════════════════════╗
║                  Style-RAG 管理系统                      ║
╚══════════════════════════════════════════════════════════╝

  📁 输入目录: ./input (待处理文件数)
  🗄️  索引数据: ./rag_index (片段数)

  1. 创建索引      从input目录创建新索引
  2. 添加文件      将新文件添加到现有索引
  3. 查询索引      输入关键词进行检索
  4. 检索测试      运行自动检索测试
  5. 查看状态      显示详细索引统计
  6. 导出索引      备份索引到exports目录
  7. 导入索引      从备份恢复索引
  8. 重置索引      清空所有索引数据
  0. 退出
```

---

## 独立脚本

如果你更喜欢命令行，可以直接运行以下脚本：

| 脚本 | 说明 | 用法 |
| ---- | ---- | ---- |
| `create_index.py` | 创建新索引 | `python create_index.py` |
| `add_files.py` | 添加文件到索引 | `python add_files.py` |
| `reset_index.py` | 重置/清空索引 | `python reset_index.py` |
| `check_status.py` | 查看索引状态 | `python check_status.py` |
| `test_retrieval.py` | 检索测试 | `python test_retrieval.py --auto` |
| `query_index.py` | 查询索引 | `python query_index.py "关键词"` |
| `export_index.py` | 导出索引备份 | `python export_index.py` |
| `import_index.py` | 导入索引备份 | `python import_index.py --input [备份路径]` |
| `start_server.py` | 启动HTTP服务 | `python start_server.py` |

---

## 目录结构

```
f:\AI_Gen_Novel_rag\
├── input/          # 放入要索引的文本文件
├── rag_index/      # 索引数据库（自动创建）
├── exports/        # 导出备份目录
├── run.py          # 交互式管理程序
└── *.py            # 独立脚本
```

---

## 常用操作

### 创建索引

1. 将 `.txt` 或 `.md` 文件放入 `input/` 目录
2. 运行 `python run.py` 选择 "1. 创建索引"
3. 或直接运行 `python create_index.py`

**选项：**
- `--clear` 先清空再索引
- `--delete` 索引后删除源文件

### 添加更多文件

将新文件放入 `input/` 目录后：

```bash
python add_files.py
# 或
python add_files.py --delete  # 添加后删除源文件
```

### 查询索引

```bash
python query_index.py "战斗场景"
python query_index.py "月下相遇" -k 5  # 返回5条结果
```

### 重置索引

```bash
python reset_index.py           # 清空数据
python reset_index.py --delete-db  # 删除整个数据库
```

### 导出备份

```bash
python export_index.py
# 导出到 ./exports/rag_backup_日期时间/
```

### 导入备份

```bash
python import_index.py --input ./exports/rag_backup_20260117_200000
```

---

## Python API 使用

```python
from style_rag import StyleRAGClient

# 初始化
client = StyleRAGClient(db_path="./rag_index")

# 索引目录
client.index_directory("./input")

# 检索
results = client.search("战斗场面", top_k=5, min_similarity=0.3)

# 查看统计
print(client.get_stats())

# 清空索引
client.clear_index()
```

---

## 编码支持

系统自动识别以下中文编码：

- UTF-8, UTF-8 BOM, UTF-16
- GB18030, GBK, GB2312
- Big5, Big5-HKSCS (繁体中文)
- 其他常见编码

---

## 常见问题

**Q: 索引后搜索不到结果？**

A: 尝试降低相似度阈值：
```bash
python query_index.py "关键词" --min-sim 0.2
```

**Q: 文件太多导致内存不足？**

A: 使用 `--delete` 选项边索引边删除源文件。

**Q: 如何切换嵌入模型？**

A: 使用 `--embedding-model` 参数：

```bash
python create_index.py --embedding-model "Qwen/Qwen3-Embedding-0.6B"
```

---

## GPU 加速设置

如果你有 NVIDIA GPU，可以安装 CUDA 版本的 PyTorch 来加速模型推理：

```bash
# 激活虚拟环境
.\.venv\Scripts\activate

# 安装 CUDA 版本的 PyTorch (CUDA 12.9)
uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

> **注意**: 请确保已安装 NVIDIA 驱动和 CUDA Toolkit。

**验证 GPU 是否可用：**

```python
import torch
print(torch.cuda.is_available())  # 应输出 True
print(torch.cuda.get_device_name(0))  # 显示 GPU 型号
```

---

## 模型存储

嵌入模型会自动下载并保存到项目本地的 `./models` 目录，无需重复下载。

```
f:\AI_Gen_Novel_rag\
├── models/         # 嵌入模型缓存（首次运行自动创建）
├── input/          # 输入文件
├── rag_index/      # 索引数据库
└── ...
```
