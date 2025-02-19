# CCL2025-Chinese-Hate-Speech-Detection
# 中国计算语言学大会技术评测-中文仇恨言论检测

随着社交媒体的普及，用户生成内容呈现出爆炸性增长的态势，也滋生了仇恨言论的传播。仇恨言论是基于种族、宗教、性别、地域、性取向、生理等特征对特定个体或群体表达仇恨、煽动伤害的有害言论。相比于其他有害言论，仇恨言论通常更具有强迫性、欺凌性和煽动性的特点，给个体乃至整个社会带来了严重的危害。在《中华人民共和国治安管理处罚法》以及《互联网信息服务管理办法》等多部法律法规中，均有禁止民族歧视或仇恨言论的规定。如何有效检测仇恨言论已经成为自然语言处理领域研究者广受关注的问题。本次评测旨在推动中文仇恨言论检测技术的发展，加强对不良网络行为的管控，助力文明网络的建设。

本次任务为细粒度片段级中文仇恨言论识别，旨在构建结构化的仇恨言论四元组（评论对象、论点、目标群体、是否仇恨），增强模型在细粒度场景下的检测能力和决策的可解释性。



## 任务介绍

**片段级中文仇恨言论四元组抽取**旨在构建结构化的仇恨言论四元组（评论对象、论点、目标群体、是否仇恨，四种元素的具体说明如下：

**评论对象（Target）：** 帖子的评述对象，如一个人或一个群体。当实例无具体目标时设为NULL，例如，保持安全!

**论点（Argument）：** 包含对评论目标关键论点的信息片段。

**目标群体（Targeted Group）：** 指包含仇恨信息的评论对象-论点对涉及的目标群体。标注的目标群体包括“地域”、“种族”、“性别”、“LGBTQ”、“其他”共5类。如样例1中包含了对LGBTQ群体和艾滋病群体的仇恨信息。

**是否仇恨（Hateful）：** 评论对象-论点对是否构成了对某些群体的仇恨言论。

对于非仇恨文本以及不包含特定群体的一般攻击性言论，同样需要对目标群体和观点进行抽取，并设为non-hate。由于样本中可能有多个评论对象，因此可以包含多个四元组。

## 评测数据

本次评测使用的中文仇恨言论四元组抽取数据集收集了贴吧、知乎等国内社交媒体平台的用户评论数据，为每条样本提供了高质量的二元分类标签，并对句子中的评论对象、论点和目标群体进行片段级标注。该数据集总计**8000**条中文数据，其中仇恨言论为**4935**，非仇恨言论为**3065**条。每条语句均包含一个或多个中文仇恨言论四元组，共计**9405**个，其中仇恨四元组**5949**个，非仇恨四元组**3456**个。

**数据集的所有权归大连理工大学信息检索研究室所有。数据集包含有害违规内容示例，均不代表本团队立场。所有资源仅供科学研究使用，严禁商用。**

## 数据样例

任务的输入为社交媒体文本，输出为仇恨四元组，顺序依次为Target、Argument、Targeted Group、Hateful。每个四元组中各个元素之间用"|"分割，并利用[END]结尾；如果一条样本中包含多个四元组，不同四元组之间利用[SEP]分割。

>**输入**：你可真是头蠢驴，这都做不好。
>
>**输出**：你 | 蠢驴 | non-hate | non-hate [END]

>**输入**：老黑我是真的讨厌，媚黑的还倒贴。
>
>**输出**：老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]

## 评价指标：

**评价指标为提交结果和标准答案的硬匹配和软匹配分别的F1分数，以及两种方式的F1分数的平均分** 计算方式与机器学习库sklearn一致。具体的计算公式如下：

F1-score:
${F1}=2\times\frac{P\times R}{P+R}$


**硬匹配：** 当且仅当预测四元组的每一个元素都与答案中对应元素**完全一致**才判断为正确抽取的四元组。

**软匹配：** 当且仅当预测四元组的 **Targeted Group , Hateful** 两个元素和标准答案中相对应的两个元素**完全一致**，并且预测四元组的 **Target ，Argument** 两个元素和标准答案中相对应的两个元素的字符串匹配程度超过50% 才判断为正确抽取的四元组。（计算方式为Python 标准库 difflib 模块中的 SequenceMatcher 函数一致）。具体计算如下：

$len_{pred}$:预测四元组长度

$len_{gold}$:标准答案长度

M：预测四元组和标准答案之间的最长公共子序列长度


Similarity：
${Similarity}=\frac{M\times 2}{len_{pred}+len_{gold}}$

**说明：在软匹配指标计算过程中最长公共子序列对文本的顺序有要求，只有字符正确并目字符顺序正确才会被计算为最长公共子序列。**

## 评测赛程及提交方式

**本次比赛分为报名认证、初赛、复赛、专家评审、颁奖典礼等阶段。**

•	注册与实名验证：2025年3月1日-3月31日

•	初赛阶段：2025年3月15日-2025年4月25日

•	初赛结果评估：2025年4月26日-4月30日

•	复赛阶段：2025年5月1日-5月10日

•	复赛选手模型代码、技术方案提交：2025年5月11日-5月13日

•	结果评估：2025年5月14日-5月20日

•	完赛与颁奖：2025年8月（具体时间另行通知）


**提交方式：** 利用天池平台提交结果，具体链接将于近日更新

## 奖励设置

•	本届评测将设置一等奖一名、二等奖两名、三等奖三名，采用“宁缺勿滥”的原则，由中国中文信息学会提供荣誉证书。

•	最终名次前六名的团队将被邀请提交参赛报告论文。评测委员会将组织专家进行双盲审稿，评测中内容和写作质量均佳的评测报告（中英文）将有机会被 **CCL Anthology** 和 **ACL Anthology** 收录。

## 官方答疑群

<img src="https://github.com/DUTIR-Emotion-Group/CCL2025-Chinese-Hate-Speech-Detection/blob/main/images/group.jpg" width="300">

## 组织机构

任务组织者：林鸿飞（大连理工大学）、杨亮（大连理工大学）、卢俊宇（大连理工大学博士生）、白泽文（大连理工大学博士生）、尹圣迪（大连理工大学硕士生）

**联系邮箱：dlutbzw@mail.dlut.edu.cn**

## 伦理声明

**数据集的所有权归大连理工大学信息检索研究室所有。数据集包含有害违规内容示例，均不代表本团队立场。**

**所有资源仅供科学研究使用，严禁商用。**
