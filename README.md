# CCL2025-Chinese-Hate-Speech-Detection
# 中国计算语言学大会技术评测-中文仇恨言论检测

随着社交媒体的普及，用户生成内容呈现出爆炸性增长的态势，也滋生了仇恨言论的传播。仇恨言论是基于种族、宗教、性别、地域、性取向、生理等特征对特定个体或群体表达仇恨、煽动伤害的有害言论。相比于其他有害言论，仇恨言论通常更具有强迫性、欺凌性和煽动性的特点，给个体乃至整个社会带来了严重的危害。在《中华人民共和国治安管理处罚法》以及《互联网信息服务管理办法》等多部法律法规中，均有禁止民族歧视或仇恨言论的规定。如何有效检测仇恨言论已经成为自然语言处理领域研究者广受关注的问题。本次评测旨在推动中文仇恨言论检测技术的发展，加强对不良网络行为的管控，助力文明网络的建设。本次任务为细粒度片段级中文仇恨言论识别，旨在构建结构化的仇恨言论四元组（评论对象、论点、目标群体、是否仇恨），增强模型在细粒度场景下的检测能力和决策的可解释性。



## 任务介绍：

**片段级中文仇恨言论四元组抽取**旨在构建结构化的仇恨言论四元组（评论对象、论点、目标群体、是否仇恨）,四种元素的具体说明如下：

**评论对象（Target）：** 帖子的评述对象，如一个人或一个群体。当实例无具体目标时设为NULL，例如，保持安全!

**论点（Argument）：** 包含对评论目标关键论点的信息片段。

**目标群体（Targeted Group）：** 指包含仇恨信息的评论对象-论点对涉及的目标群体。标注的目标群体包括“地域”、“种族”、“性别”、“LGBTQ”、“其他”共5类。如样例1中包含了对LGBTQ群体和艾滋病群体的仇恨信息。

**是否仇恨（Hateful）：** 评论对象-论点对是否构成了对某些群体的仇恨言论。

对于非仇恨文本（包括不包含特定群体的一般攻击性言论），同样需要对目标群体和观点进行抽取，并设为non-hate。

## 数据样例：

输入：你可真是头蠢驴，这都做不好。
输出：你 | 蠢驴 | non-hate | non-hate [END]

输入：老黑我是真的讨厌，媚黑的还倒贴。
输出：老黑 | 讨厌 | Racism | hate [SEP] 媚黑的 | 倒贴 | Racism | hate [END]

## 评价指标：

**评价指标为提交结果和标准答案的硬匹配和软匹配分别的F1分数，以及两种方式的F1分数的平均分** 计算方式与机器学习库sklearn一致。具体的计算公式如下：

F1-score:
${F1}=2\times\frac{P\times R}{P+R}$


**硬匹配：** 当且仅当预测四元组的每一个元素都与答案中对应元素**完全一致**才判断为正确抽取的四元组。

**软匹配：** 当且仅当预测四元组的 **Targeted Group , Hateful** 两个元素和标准答案中相对应的两个元素**完全一致**，并且预测四元组的 **Target ，Argument** 两个元素和标准答案中相对应的两个元素的字符串匹配程度超过50% 才判断为正确抽取的四元组。（计算方式为Python 标准库 difflib 模块中的 SequenceMatcher 函数一致）。具体计算如下：

len(a):预测四元组长度

len(b):标准答案长度

M：预测四元组和标准答案之间的最长公共子序列长度


Similarity：
${Similarity}=\frac{M\times 2}{len(a)+len(b)}$

**说明：在软匹配指标计算过程中最长公共子序列对文本的顺序有要求，只有字符正确并目字符顺序正确才会被计算为最长公共子序列。**

## 评测赛程及提交方式

待公布。

## 伦理声明

**数据集的所有权归大连理工大学信息检索研究室所有。数据集包含有害违规内容示例，均不代表本团队立场。**

**所有资源仅供科学研究使用，严禁商用。**
