# STSECL & POIRec 推荐系统演示平台

基于 Flask 的推荐系统演示网页，集成了两个图神经网络推荐模型：

| 模型 | 任务 | 可运行数据集 |
|------|------|--------------|
| **STSECL** | 朋友推荐 | BER、CHI、NYC、JK、KL、SP |
| **POIRec** | 兴趣点（POI）推荐 | BER、CHI |

POIRec 的推荐结果会在 **Leaflet 地图**上标出目标用户（红点，居中）与推荐地点（蓝点）。

## 快速开始

```bash
# 1. 安装基础依赖
pip install -r requirements.txt

# 2. 安装 PyTorch + 图神经网络库（版本必须严格匹配，详见 部署说明.md）
#    CPU 版：
pip install torch==1.12.0+cpu -f https://download.pytorch.org/whl/torch_stable.html
pip install torch-scatter==2.1.0 -f https://data.pyg.org/whl/torch-1.12.0+cpu.html
pip install torch-geometric==2.2.0

# 3. 启动网页
python app.py
```

浏览器打开 **http://127.0.0.1:5000/** 即可使用。

> 完整部署与使用说明见 [部署说明.md](部署说明.md)。

## 数据说明

- 源数据已包含在仓库中：`STSECL/data/*/` 下的 `.pkl` 文件、`POIRec/data/*/` 下的原始 CSV。
- 预处理产物（`*.pt`）、训练权重（`save_user_embedding/`）、推荐结果（`*.db`、`recommendations.tsv`）等**在运行时自动生成**，已通过 `.gitignore` 排除。
- 因此从 GitHub clone 后首次运行会自动重新预处理并训练（耗时视数据量与机器而定）。

## 目录结构

```
RecommendationWeb/
├── app.py                # Flask 后端入口
├── templates/index.html  # 单页前端
├── static/               # app.js（含地图渲染）/ style.css
├── STSECL/               # 朋友推荐模型
└── POIRec/               # 兴趣点推荐模型
```
