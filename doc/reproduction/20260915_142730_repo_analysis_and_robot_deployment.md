# MegaLoc 仓库分析:模型、输入输出、复现结果与机器人(前视相机+IMU+轮速)部署方案

- 时间戳: 2026-09-15 14:27 (CST)
- 仓库: /data/code/reLocation-all/MegaLoc (main, 已含本仓复现提交 fc7e9bc)
- 结论先行: 本仓库是 **推理专用** 的视觉位置识别(VPR)模型发布仓,无训练代码。机器人部署路径明确可行: 前视相机正好是模型的输入;IMU+轮速不参与模型,用于建图时的里程计位姿和运行时的预测/门控,模型负责"我在哪(地图节点)",里程计负责"两帧之间怎么动"。

---

## 1. 推理用的模型是什么

MegaLoc (CVPR-W 2025, "MegaLoc: One Retrieval to Place Them All", arXiv:2502.17237),单图 → 全局描述子的图像检索模型。结构 (megaloc_model.py:400 `MegaLoc`):

1. 骨干: DINOv2 ViT-B/14 (megaloc_model.py:270 `DINOv2`), patch=14, embed=768, 12 层, 位置编码可插值到任意 14 倍数分辨率。
2. 聚合头: SALAD 式最优传输聚合 (megaloc_model.py:67 `FeatureAggregator`, Sinkhorn 3 轮): 把 patch 特征聚合为 64 簇 × 256 维 + 256 维全局 token = 16640 维,再过一层 `nn.Linear` 投到 8448 维 (Aggregator, megaloc_model.py:389)。

	疑问：这里的聚合头在做什么？代码中如何实现的？对于整个推理过程，请再次详细给出，并给出流程图。然后，结合代码进行详细讲解。

1. 输出 L2 归一化,余弦相似度 = 点积。

权重: HF `gberton/MegaLoc` 的 `model.safetensors` (~872MB, fp32), 本地缓存在
`/data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors`。
加载入口: `hubconf.py:get_trained_model()` 或 `scripts/reproduce_inference.py:load_model()`。
仓库内无训练代码(官方声明不发布),只有推理 + 本仓复现脚本。

## 2. 输入 / 输出

输入 (README 评测配方, 复现脚本一致):

- RGB 图像, `ToTensor` → ImageNet 归一化 (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]) → Resize 322×322 (antialias)。
- 张量 [B, 3, 322, 322]; 任意 14 倍数尺寸也可(model.forward 自动对齐)。

疑问：这里的张量 [B, 3, 322, 322]是什么？

输出:

- 全局描述子 [B, 8448], float32, 已 L2 归一 (fp32 约 33 KB/图)。
- 相似度 = `q @ g.T` (余弦)。检索 top-K 最近邻地图图像即"最可能的位置"。
- 注意: **不输出位姿**。模型只给"哪张地图图像最像",位姿来自该图像建图时记录的 GPS/UTM 或里程计位姿。

## 3. 测试数据集上的结果(本仓复现 + 论文 Table 1 对照)

来源: doc/reproduction/20260915_134347_COMPARE_TABLE1.md、各 outputs/*/metrics.json。
判据: 25m UTM 半径(标准 VPR 协议)或帧序差 ≤25(Nordland 子集协议)。

| 数据集                                         | 协议           | 复现 R@1 / R@10           | 论文 R@1 / R@10     | 一致性                    |
| ---------------------------------------------- | -------------- | ------------------------- | ------------------- | ------------------------- |
| Pitts30k test (db=10000, q=6816)               | 25m UTM        | 94.07 / 98.24             | 94.1 / 98.2         | ✅ 吻合                   |
| Tokyo247 (db=75984, q=315)                     | 25m UTM        | 96.51 / 99.68             | 96.5 / 99.4         | ✅ 吻合                   |
| Nordland-Part-2020 (冬查询→夏画廊, 1750/1750) | \|帧序差\|≤25 | 65.94 / 80.11 (R@5 75.09) | —(非 Table 1 协议) | ⚠️ 季节/协议不同,仅参考 |
| MSLS-val                                       | 25m            | 未跑(磁盘策略取消)        | 91.0 / 95.8         | —                        |
| smoke 200 连续帧                               | \|Δidx\|≤25  | 100 / 100                 | —                  | ✅ 链路 sanity            |

结果文件: `outputs/table1_repro/{pitts30k,tokyo247}/metrics.json`、`outputs/inference_repro/metrics.json`(含可视化 `viz_*.png`、描述子 `descriptors.pt/npz`)。
耗时参考(RTX GPU, batch 32): Pitts30k 16.8k 图 ~11 min; Tokyo247 76.3k 图 ~54 min。

## 4. 机器人部署: 前视相机 + IMU + 轮速

### 4.1 角色划分(先想清楚这个)

- MegaLoc = **重定位/回路检测器**: 回答"当前前视图像对应地图上哪个已知点"。单目前视恰好是它的标准输入域(街景/车载前视训练),不需要深度、不需要多相机。
- IMU + 轮速 = **里程计**: 提供帧间相对运动(短时精确、长时漂移)。它们不进模型,和模型输出在**滤波层**融合。
- 模型给绝对约束(位置节点),里程计给连续估计;两者互补: 里程计填两次识别之间的空档,识别结果修正里程计漂移。

### 4.2 阶段一: 建图(离线, 一次)

1. 人工/遥控跑一遍目标区域, 采集前视帧 + 位姿来源二选一:

   - 室外有 GPS: 记 UTM 位姿(本仓评测即此方式)。
   - 无 GPS: 用 IMU+轮速里程计(如差速模型+互补/EKF)输出每帧位姿。等间隔采样(如每 1~2 m 或每 ~0.5 s 一帧)。
2. 对每帧跑 MegaLoc 提描述子(本仓脚本逻辑直接可用, 322×322 配方)。
3. 存库: `descriptors.npy [N,8448] float32 + poses.npy [N,3] (x,y,θ) + 图像缩略图/路径`。

   - 检索用暴力矩阵乘即可: N=1万时单次查询 8448×10000 ≈ 0.08 GFLOP, 机器人端 CPU 也 <10 ms; 更大库再考虑 faiss(`pip install faiss-cpu`, IVF/HNSW)。别提前上向量数据库, YAGNI。

   疑问：建图时还会存储缩略图吗？它的作用是什么？

### 4.3 阶段二: 运行时定位(在线)

低配版(推荐先做这个, 覆盖 90% 需求):

```
每帧(里程计, 高频):  x_pred = x_last + Δodometry(IMU+轮速)     # 连续位姿输出
每 T 秒/T 米(低频):  q = MegaLoc(前视帧)                       # [8448]
                     s = q @ DB.T;  top-k
                     若 s[top1] > τ (如 0.3~0.5, 现场标定):
                         若 |DB_pose[top1] − x_pred| < R_gate (如 10~20 m):
                             x_last ← DB_pose[top1 的聚类质心]   # 重锚定
                         否则: 接受为"检测到回路/重定位成功"后再锚定
```

要点:

- **τ 和 R_gate 是两个必须的标定旋钮**: 感知混叠(相似路口)靠 R_gate(用里程计预测筛掉远处假阳性)解决; 特征退化(空旷/逆光)靠 τ 拒绝低置信。跑一遍记录 s 分布再定阈值。
- 低频即可: 重锚定 0.2~1 Hz 足够,描述子提取是主要开销。
- top-k 投票/聚类比取 top-1 更稳(k=5~10,取位姿中位数)。
- 起步用"跳到最近匹配"这种非融合更新即可;漂移表现为位置跳变,够用再说。

升级路径(出现以下需求再做):

- 需要连续平滑的绝对修正 → 把重定位结果作为量测进 EKF/因子图(如 GTSAM),同时保留回路约束做 pose-graph 优化。
- 需要朝向精度 → 单帧不够,用 top-k 命中节点的位姿做投票,或用里程计段与地图段轨迹匹配。

### 4.4 算力落地(RTX 级工作站 → 嵌入式)

- fp32 ViT-B/14 @322×322: 独显毫秒级;Jetson Orin NX 级 fp16/TensorRT 可到 ~10-30 ms/帧,满足 1 Hz 重定位绰绰有余。
- 最省事的导出: `torch.onnx.export`(输入固定 1×3×322×322)→ TensorRT / ONNX Runtime。模型纯静态图(Sinkhorn 固定 3 轮迭代),导出无障碍。
- 内存: 权重 ~872 MB(fp32)/ ~436 MB(fp16); 描述子库 10k×8448×4B ≈ 340 MB,可放 RAM 或 fp16 减半。
- 先跑通验证(在机器人同环境或本仓即可):

```bash
# 未来命令(本仓已验证过的复现入口):
/data/software/conda/anaconda3/envs/mv2d/bin/python scripts/reproduce_inference.py \
    --max-images 200 --batch-size 16 --out outputs/inference_repro_smoke200
```

### 4.5 边界与风险(如实说)

- 单目 VPR 给的是**地图节点级位置**,不是度量位姿;精度上限 = 建图采样的间距 + GPS/里程计标定质量(典型 1~5 m,非厘米级)。
- 外观剧变(昼夜/季节/施工)会掉召回 —— Nordland 冬→夏 R@1 66% 就是量级参考;同季节往返则接近 Pitts/Tokyo 的 90%+。
- 开阔无特征区域(白墙、空旷地)检索失效,此时只剩里程计,靠离开该区域后的下一次成功匹配重新锚定。
- 轮速打滑/IMU 零偏造成的里程计漂移必须小于 R_gate,否则真阳性会被门掉 —— 阈值按最大可信漂移设。

## 5. 关键文件索引

| 内容                 | 路径                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| 模型定义             | megaloc_model.py (MegaLoc / DINOv2 / FeatureAggregator)                   |
| Hub 入口             | hubconf.py (get_trained_model)                                            |
| Nordland 复现脚本    | scripts/reproduce_inference.py                                            |
| Pitts/Tokyo 评测脚本 | scripts/eval_table1_dataset.py                                            |
| 权重                 | /data/code/reLocation/.hf_cache/gberton_MegaLoc/model.safetensors         |
| 指标                 | outputs/inference_repro/metrics.json, outputs/table1_repro/*/metrics.json |
| 对照表               | doc/reproduction/20260915_134347_COMPARE_TABLE1.md                        |
