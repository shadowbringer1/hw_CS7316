---
marp: true
size: 16:9
theme: am_red
paginate: true
headingDivider: [2,3]
---

<!-- _class: cover_e 
<!-- _paginate: "" --> 
<!-- _footer: ![](https://vi.sjtu.edu.cn/uploads/files/22fc9c46e0998e2454d64f7eda901595-d2c02ba527abb94ad97653e36bbac8a1.png) -->
<!-- _header: ![](https://vi.sjtu.edu.cn/uploads/files/caf2f5045c47308250fab3812dfe2003-6896b91594f238b24e67696224948251.png) -->
# 基于流行度感知的元学习在线物品冷启动推荐

汇报人：夏卓；舒佳；苏畅；郭珺琨
汇报日期：2025 年 5 月 30 日

##

<!-- _header: CONTENTS-->
<!-- _class: toc_a -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

- [引言](#3)
- [背景知识](#7) 
- [方法介绍](#11)
- [实验结果](#17)
- [总结与反思](#24)

## 1. 引言

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 研究动机
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ **引言** *背景知识* *方法介绍* *实验结果* *总结与反思*-->

<div class=ldiv>

**在线推荐系统的兴起**

- 对电子商务、社交媒体、视频平台至关重要。

- 关键特征：

  - 流式数据：用户交互（浏览、购买、评分）和新物品信息随时间持续到达。

  - 动态更新：实时捕捉用户动态兴趣。

  - 计算效率：必须以低延迟处理大量数据和请求。

**持续存在的挑战：物品冷启动**

- 新物品几乎没有交互数据。

- 难以进行有效推荐。

- 影响用户体验和物品发现。

</div>

<div class=rimg>

![#c](https://www.prefixbox.com/blog/wp-content/uploads/2023/05/Ecommerce-Product-Recommendations-2-768x664.png)
</div>

## 核心问题
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ **引言** *背景知识* *方法介绍* *实验结果* *总结与反思*-->

**在线流式场景下的物品冷启动**：
- 为系统中新加入的、交互数据极少或没有的物品进行推荐。
- 在线系统特点：

  - 数据以连续流的形式到达。
  - 模型需要频繁更新。
  - 计算开销和时间限制至关重要。

**现有解决方案的局限性**：

- 许多离线冷启动方法（如微调、知识迁移）在处理在线流式数据时并不可行，原因包括：

  - 训练范式不同。
  - 计算成本高昂。
  - 延迟问题。

## 本文方法简介 (PAM)
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ **引言** *背景知识* *方法介绍* *实验结果* *总结与反思*-->

基于流行度感知的元学习 (Popularity-Aware Meta-learning, PAM)：

- 一种模型无关的推荐算法（意味着它可以应用于不同的基础推荐模型）。

- 专为流式数据场景下的物品冷启动问题设计，同时避免了在线服务期间额外的微调开销。

- 核心思想：

  - 通过预定义的流行度阈值将项目划分为不同的元学习任务，使得模型能够为不同流行度级别的项目学习到更具针对性的推荐策略。

  - 通过元学习机制，使得模型能够从历史数据和热门项目中学习到可迁移的“元知识”，并快速适应到数据稀疏的冷启动项目中。

  - 通过引入针对低流行度任务的数据增强和额外的自监督损失，有效缓解冷启动物品反馈稀疏的问题。

  - 通过固定的任务划分和特定的训练服务机制，降低计算和存储开销，满足在线系统对效率的要求。

## 2. 背景知识

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 项目冷启动
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* **背景知识** *方法介绍* *实验结果* *总结与反思*-->
当系统中出现新项目（如新商品、新视频）时，由于这些项目缺乏足够的用户交互历史，系统难以准确判断用户对这些新项目的偏好，从而无法进行有效推荐 。

**交互稀疏性**:

- 低流行度物品天生用户交互就很少，这使得依赖用户-物品交互模式的协同过滤方法难以发挥作用。

**长尾分布**:

- 少数物品非常受欢迎，而绝大多数物品是小众的或新的，交互很少。冷启动物品就位于这个长尾中。

**马太效应**:

- 受欢迎的物品倾向于被更多地推荐，从而收集更多的交互数据，变得更加受欢迎。

- 这进一步抑制了新的/不受欢迎物品的可见性和数据积累。

## 元学习
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* **背景知识** *方法介绍* *实验结果* *总结与反思*-->

**“学习如何学习” (Learning to Learn)**:

- 旨在训练一个模型，使其能够用少量样本快速适应新任务。

- 非常适合冷启动问题（一个新物品就像一个新的“任务”）。

**训练过程**：

- 将每个任务的数据集划分为支持集 (Support Set) 和查询集 (Query Set)。
- 在支持集上进行学习（例如，通过几步梯度下降更新参数）。
- 在查询集上评估其性能，并据此更新元学习器（即学习如何调整学习过程的那个模型）。

**在线场景的挑战**： 

传统的按物品定义任务的方式成本可能过高。PAM 提出基于流行度的固定任务划分以提高效率。


## 基础推荐模型（双塔结构）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* ***方法介绍** *实验结果* *总结与反思*-->

<div class=ldiv>

输入数据 $D_t$ （在每个时间段 t）:
  - $u$: 用户侧特征，包括用户 ID 与交互历史
  - $i$ : 物品侧特征，包括物品 ID 与内容特征
  - $v_i$: 用户浏览次数（用于划分元学习任务）

推荐系统（同样随时间更新）：
- 嵌入矩阵(feature embedding)参数 $\Phi_t$
- 网络权重(DNN)参数 $\Omega_t$ 及其初始化值 $\Theta_t$

数据流变化：
- 嵌入层：$(u, i) \to e_u, e_i$
- 隐藏层：$e_u, e_i → z_u, z_i$
- 输出层：$\hat{y_{ui}}=\frac{exp(z_u\cdot z_i/\tau)}{\sum_{u'\in D_t}exp(z_u'\cdot z_i/\tau)}$

</div>

<div class=rimag>

![#c h:450](https://i-blog.csdnimg.cn/blog_migrate/f0f67afac0d97f217e643ea132be9c9a.png)

</div>

## 3. 方法介绍

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## PAM 方法总体架构（Figure 1）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

<div class=ldiv>

- **模型目标：**  
  提升冷启动推荐性能，兼顾热门项推荐效果。

- **核心思想：**  
  - 基于 dual-tower 的基础推荐模型；  
  - 结合 task-fixed 的 popularity-aware meta-learning；  
  - 冷启动任务与热门任务共享低层特征，分任务更新高层特征。
</div>

<div class=rdiv>

- **图示说明：**  
  - 左侧为冷启动增强模块；  
  - 中部为 Task-Fixed Meta-Learning 模块（不同任务分支）；  
  - 上部参数 Φ 和 Θ 为任务共享；  
  - 下部各任务参数 Ω 专用于特定受欢迎度分段的任务。

</div>

## PAM 方法总体架构（Figure 1）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

<div style="position: absolute; top: 25%; left: 8%; width: 90%;">
  <img 
    src="https://arxiv.org/html/2411.11225v3/x1.png" 
    alt="Figure 1"
    style="width: 100%; height: auto; box-shadow: 0 12px 32px rgba(0,0,0,0.25); border: 1px solid #ccc;" />
</div>


## 基础推荐模型（Dual-Tower）

<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
<div class=ldiv>

**1. Embedding Layer**

- 输入用户 u 和物品 i 的多维特征；  
- 多个 embedding 矩阵提取特征：

  $$
  \mathbf{e}_u = [E_U^1 c_u^1, ..., E_U^P c_u^P],\quad \mathbf{e}_i = [E_I^1 c_i^1, ..., E_I^Q c_i^Q]
  $$

- 得到稠密向量 $\mathbf{e}_u, \mathbf{e}_i$

**2. Hidden Layer**

- 分别将 $\mathbf{e}_u, \mathbf{e}_i$ 输入各自 MLP：

  $$
  \mathbf{z}_u = f_u^L \circ ... \circ f_u^1(\mathbf{e}_u),\quad \mathbf{z}_i = f_i^L \circ ... \circ f_i^1(\mathbf{e}_i)
  $$

- 每层结构：ReLU 激活 $\sigma(W_l e + b_l)$
</div>

<div class=rdiv>

**3. Output Layer**

- 使用 InfoNCE loss 计算打分：

  $$
  \hat{y}_{ui} = \frac{ \exp(\mathbf{z}_u \cdot \mathbf{z}_i / \tau) }{ \sum_{u'} \exp(\mathbf{z}_{u'} \cdot \mathbf{z}_i / \tau) }
  $$

</div>


## Popularity-Aware Meta-Learning（动机与思路）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

- **问题：**  
  - 冷启动样本占比小，更新影响力低；  
  - 直接忽略热门项会导致整体性能下降。

- **解决方案：**  
  - 固定按 item 受欢迎度将样本划分成多个任务；  
  - 使用元学习对不同任务进行更新；  
  - 冷启动任务重内容特征，热门任务重历史行为。


## 固定任务划分策略
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

<div class=ldiv>

- 基于 item 受欢迎度（点击数/销量）分段：

  $$
  F(v_i) : v_i \in \mathbb{N} \rightarrow T_n \in \{T_1, ..., T_N\}
  $$

- 每个交互样本 (u, i) 被唯一分配到任务 $T_n$  
- 任务划分在整个训练过程固定，不变

</div>

<div class=rimag>

![#c w:600](https://i.postimg.cc/1RYwxcKr/local-weight.png)

</div>


## 局部任务特化更新（Local Updates）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

<div class=ldiv>

- 每批数据 $D_t$ → 任务分割 $\{D_t^1, ..., D_t^N\}$  
- 每个任务内划分支持集（Support Set）和查询集（Query Set）

- 支持集用于局部更新：

  $$
  \Omega_t^n \leftarrow \Theta_t - \alpha \nabla_{\Theta_t} \mathcal{L}_{T_n}(\Theta_t | D_t^{n,S})
  $$

- 仅更新网络权重，不更新 embedding，避免参数重叠
</div>

<div class=rimag>

![#c w:600](https://i.postimg.cc/1RYwxcKr/local-weight.png)

</div>


## 共享参数的全局元优化（Global Updates）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
<div class=ldiv>

- 基于所有任务查询集，反向更新共享参数 $\Theta_t, \Phi_t$：

  $$
  \{\Phi_{t+1}, \Theta_{t+1}\} \leftarrow \{\Phi_t, \Theta_t\} - \beta \nabla_{\Phi_t, \Theta_t} \mathcal{L}_M
  $$

- 总体元损失：

  $$
  \mathcal{L}_M = \sum_{n=1}^N \lambda_n \mathcal{L}_{T_n}(\Omega_t^n | D_t^{n,Q})
  $$

- 可为冷启动任务设更高权重 $\lambda_n$
</div>

<div class=rimag>

![#c w:600](https://i.postimg.cc/152GFcn3/shared-weights.png)

</div>

## 冷启动任务增强器（Cold-start Task Enhancer）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

## 1. ID嵌入(ID Embeddings)
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
<div class=ldiv>

- **定义**  
  基于物品的唯一标识符（如物品ID）生成的嵌入向量。
- **关键作用**  
  捕捉物品的**个体独特性**（如“电影A” vs “电影B”）  
  通过用户交互动态优化（依赖历史数据）
- **冷启动适用性**  
  低：冷启动物品因交互稀疏，ID嵌入难以充分训练。
</div>
<div class=rdiv>

![](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_25-5-2025_13526_.jpeg)
</div>



## 2. 序列嵌入(Sequential Embeddings)
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
<div class=ldiv>

- **定义**
  基于物品的历史交互序列（如点击、购买顺序）生成的嵌入。
- **关键作用**  
  反映**上下文关联性**（如“用户点击A后常点击B”）  
  捕捉用户兴趣的**动态演化**（如偏好从“科幻”迁移到“动作”）
- **冷启动适用性**  
  低：依赖历史行为数据，冷启动物品数据不足。
</div>
<div class=rdiv>

![](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_25-5-2025_13526_.jpeg)
</div>



## 3. 内容嵌入(Content-based Embeddings)
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
<div class=ldiv>

- **定义**  
  基于物品的元数据或内容特征（如文本、图像、类别）生成的嵌入。
- **关键作用**  
  **用于冷启动**：直接利用物品属性（如类型、导演）  
  提升跨任务**泛化能力**（如“喜剧”类型共享特征）
- **生成方式**  
  文本特征：BERT、Word2Vec  
  图像特征：CNN、预训练视觉模型
</div>
<div class=rdiv>

![](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_25-5-2025_13526_.jpeg)
</div>



## 冷启动任务增强器（Cold-start Task Enhancer）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
**利用热门物品信息生成冷启动任务数据**
![](https://gitee.com/guojunkun/picture/raw/master/1.jpeg)

## 数据增强
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
**物品可能从冷门成长为热门**
**旧的行为嵌入 + 当前内容嵌入:**  $\hat{e}_i = [\hat{e}_i^{ID}, \hat{e}_{i2}^{seq}, \cdots, \hat{e}_{ik}^{seq}, e_{ik+1}^{con}, \cdots, e_{iQ}^{con}]$
![h:300](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_24-5-2025_21476_.jpeg)
通过复用热门物品的早期嵌入，为冷启动任务生成更多训练样本，缓解数据稀疏性


## Training process
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->
**数据增强损失**将伪样本 $\hat{D}{t}^{cold}$ 划分为支持集 $\hat{D}{t}^{coldS}$ 和查询集 $\hat{D}_{t}^{coldQ}$，用于元学习的优化。
$$ \mathcal{L}_t^A = \mathcal{L}(\hat{\Omega}_t^{cold} \mid \hat{D}_t^{cold,\mathcal{Q}}) $$
**自监督学习损失**通过自监督信号，强制冷启动任务网络学习与热门阶段一致的ID嵌入。
<!--提升模型对冷启动物品内容特征的利用效率-->

$$ \mathcal{L}_t^{S} = \frac{1}{N} \left\| \hat{z}^{\mathrm{ID}}, e^{\mathrm{ID}} \right\|_2^2 $$
![](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_24-5-2025_214721_.jpeg)


## 训练过程设计

<!-- _class: navbar fixedtitleA cols-2-37 -->
<!-- _header: \ *引言* *背景知识* **方法介绍** *实验结果* *总结与反思*-->

<div class=ldiv>

#### **总损失函数**
$$\mathcal{L}_t^T = \gamma_M \mathcal{L}_t^M + \gamma_S \mathcal{L}_t^S + \gamma_A \mathcal{L}_t^A$$  
**多任务联合优化**：  
- $\gamma_M$：主元学习损失权重  
- $\gamma_S$：自监督损失权重  
- $\gamma_A$：数据增强损失权重  

#### **参数更新方法**  
$$\{\Phi_{t+1}, \Theta_{t+1}, f_{t+1}^{Sup}\} \leftarrow \{\Phi_t, \Theta_t, f_t^{Sup}\} - \beta \nabla \mathcal{L}_t^T$$
</div>

<div class=rimg>

![h:600](https://gitee.com/guojunkun/picture/raw/master/屏幕截图_25-5-2025_135150_.jpeg)
</div>

## 4. 实验结果

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 实验设置与评估指标
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
- **数据集**：
  - MovieLens、Yelp、Book（带时间戳，模拟流数据）
- **评价指标**：
  - Recall@K, NDCG@K
  - K=5/10/20，聚焦冷启动推荐效果
- **冷启动划分**：
  - 按物品流行度最低5%划分冷启动任务
- **训练设置**：
  - 无预训练，严格模拟在线流推荐环境
  - 所有模型基于双塔结构统一实现


## 实验结果：冷启动推荐效果（RQ1）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
- PAM在所有数据集上**显著优于**现有方法（PF, SML, IncCTR等）
- 对Recall@5、NDCG@5等小K指标提升更显著
- 说明PAM更精准地把冷启动物品推荐给真正感兴趣的用户
<p align="center">
  <img src="https://github.com/shadowbringer1/hw_CS7316/raw/main/presentation/figure/fig1.jpg" width="750">
</p>

> 🔼 表明PAM能有效缓解长尾分布问题，并提升冷启动推荐质量

## 消融实验：各模块贡献分析（RQ2）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
- **PAM-M**：移除增强模块，性能下降
- **PAM-S**：仅加自监督模块
- **PAM-A**：仅加数据增强模块
- **PAM-F**：完整增强模块组合
<p align="center">
  <img src="https://github.com/shadowbringer1/hw_CS7316/raw/main/presentation/figure/fig2.jpg" width="750">
</p>

## 任务个性化能力分析（RQ3）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
- 对比冷启动任务和热门任务在屏蔽不同embedding后的表示变化：

<p align="center">
  <img src="https://github.com/shadowbringer1/hw_CS7316/raw/main/presentation/figure/fig3.jpg" width="500">
</p>

🧊 冷启动任务：表示主要依赖**内容特征**，行为特征权重低

🔥 热门任务：表示更依赖**行为特征**（如ID embedding）


> 🔼  表明PAM能为不同任务学习到**任务定制化的参数表示**，提升推荐精准度

## 超参数敏感性分析（RQ4）
<!-- _class: navbar fixedtitleA cols-2 -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
<div class=ldiv>

- 冷启动任务权重设置范围需合适：

  - **太低** ➝ 无法优化冷启动参数
  - **太高** ➝ 损失热门信息，整体退化

- 自监督和数据增强模块的损失权重对性能稳定性影响小


> 🔁 PAM参数设置具备良好鲁棒性，实际部署中易于调优

</div>

<div class=rimg>

![#c](https://github.com/shadowbringer1/hw_CS7316/raw/main/presentation/figure/fig4.jpg)

</div>

## 在线部署效果（RQ5）
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* **实验结果** *总结与反思*-->
📈 在真实商业推荐系统中的A/B测试结果：
<p align="center">
  <img src="https://github.com/shadowbringer1/hw_CS7316/raw/main/presentation/figure/fig5.jpg" width="500">
</p>

- **推荐展示率（Show%）**：+41.39%
- **用户点赞率（LTR）**：+60.45%
- **评论率**、**收藏率**也有显著提升


## 5. 总结与反思

<!-- _class: trans -->
<!-- _footer: "" -->
<!-- _paginate: "" -->

## 总结
<!-- _class: navbar fixedtitleA -->
<!-- _header: \ *引言* *背景知识* *方法介绍* *实验结果* **总结与反思**-->

- **研究动机**

  - 现有推荐方法难以应对流式推荐下的冷启动问题；特别是长尾分布使得冷门物品很难被有效推荐。

- **方法设计：PAM 框架**

  - 构造基于物品流行度的元学习任务；在任务间共享知识，在新任务中无需在线微调即可做推荐。
 
- **增强模块：Enhancer**

  - 利用热门物品信息增强冷门物品表示；提高冷门物品被精准推荐的概率。

- **实验验证与部署成效**

  - 在多个数据集上 Recall@5、NDCG@5 均显著提升；在真实系统部署中表现出良好冷启动适应性。

- **未来工作方向**

  - 探索更高效的在线冷启动优化方案；提升模型在大规模场景下的 泛化能力与实用性。

---

<!-- _class: lastpage -->
<!-- _footer: "" -->
###### 汇报完毕，恳请指正
汇报人：夏卓；舒佳；苏畅；郭珺琨
