---
permalink: /
title: ""
excerpt: ""
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---
{% if site.google_scholar_stats_use_cdn %}
{% assign gsDataBaseUrl = "https://cdn.jsdelivr.net/gh/" | append: site.repository | append: "@" %}
{% else %}
{% assign gsDataBaseUrl = "https://raw.githubusercontent.com/" | append: site.repository | append: "/" %}
{% endif %}
{% assign url = gsDataBaseUrl | append: "google-scholar-stats/gs_data_shieldsio.json" %}

<span class='anchor' id='about-me'></span>
大家好！

我是 **程子洋**，目前就读于 [武汉大学](https://www.whu.edu.cn/) 网络空间安全专业本科。

我对网络安全和人工智能充满热情，自入学以来，积极参与各类科研项目和学术竞赛。在学术方面，我的GPA为3.86，曾获得雷军计算机奖学金和多项校内外奖项。在科研方面，我参与并主导了多个项目，包括基于CLIP的推荐系统、高层建筑智能监测系统等，这些项目不仅提升了我的专业技能，也让我积累了丰富的实践经验。

我熟练掌握C++、Python等编程语言，熟悉深度学习框架如PyTorch，能够独立完成从数据处理到模型训练的整个流程。同时，我也积极参与各种学术交流活动，与来自不同领域的专家学者进行讨论和合作，不断拓宽自己的知识面。

我希望能与更多对网络安全和人工智能感兴趣的同仁交流合作，共同探讨前沿技术的发展。如果您对我的研究工作感兴趣，或者有合作意向，请随时联系我。

# 🎓 Education Background

- **武汉大学**
  - **地点:** 武汉
  - **专业:** 网络空间安全
  - **学位:** 本科
  - **入学时间:** 2021.9
  - **GPA:** 3.86/4.0
  - **加权平均分:** 90.55/100
  - **排名:**

    - 综合素质测评排名: 1/132 (0.7%)
    - 平均学分绩点排名: 15/132 (11.3%)
  - **语言能力:**

    - 已通过 CET4 和 CET6，CET4 成绩为 662
    - 具备良好的英语文献阅读和写作能力
  - **核心课程得分:**

    - 概率论与数理统计 (96)
    - 高等数学 (93)
    - 数据结构 (93, 93)
    - 算法设计与分析 (90)
    - 机器学习 (95, 95)
    - 自然语言处理 (95)
    - 舆情分析 (93)
    - 社会计算 (91)
    - 数据库系统 (94)
  - **编程能力:**

    - 熟练掌握 C++ 和 Python
    - 熟悉算法与数据结构，抽象能力强，代码风格良好
    - 熟练掌握 PyTorch、Numpy、Pandas，熟悉各类深度学习模型
  - **开发工具:**

    - VS Code, PyCharm, Jupyter Notebook, LaTeX (Overleaf), Git
  - **在校荣誉:**

    - 武汉大学雷军计算机奖学金 (￥10000，全校共60人)
    - 武汉大学甲等奖学金（5%）
    - 武汉大学三好学生
    - 武汉大学青年志愿者优秀个人奖

# 🧪 Research Experience

## **CLIP4Rec - 基于 CLIP 的通用推荐框架研究**

- **时间:** 2023.9 - 2024.7
- **角色:** 推荐系统，多模态 (个人)
- **摘要:**
  - 识别出当前推荐系统以 ID 为主流，难以跨域迁移的问题。
  - 提出了基于图像的推荐关系数据集和通用数据表示方式，通过五个核心步骤设计和优化模型。
  - 初步计划投稿 NeurIPS 2024，但在投稿前决定改进后投稿 AAAI 2024。

## **AeroSentry - 高层建筑智能监测系统**

- **时间:** 2023.1 - 2024.5
- **角色:** Geo AI，计算机视觉 (团队)
- **摘要:**
  - 解决现有高层建筑监测系统应用范围小、性能不理想的问题。
  - 引入轻量化的语义分割网络和知识蒸馏技术，提高检测准确性。
  - 项目成功申请为省级大学生创新创业项目，并申请了一项国家发明专利和两项软件著作权。
  - 带队参加第17届中国大学生计算机设计大赛，中南地区赛人工智能应用组第一名。

## **FuzzLLM - 一种主动发现大型语言模型中越狱漏洞的新型通用模糊测试框架**

- **时间:** 2023.6 - 2023.11
- **角色:** 大语言模型，人工智能安全 (团队)
- **摘要:**
  - 解决 LLMs 引入的安全风险，尤其是越狱漏洞问题。
  - 提出 FuzzLLM，通过自动生成大量随机但结构化的输入提示来测试模型。
  - 与 EasyJailbreak 框架合作，申请了一项国家发明专利和一项软件著作权。
  - 参加第17届中国大学生计算机设计大赛，中南地区赛一等奖。

## **UnVC - 面向生成式伪造语音欺骗的鲁棒主动防御系统**

- **时间:** 2024.1 - 2024.5
- **角色:** 多媒体信息内容安全 (团队)
- **摘要:**
  - 解决零样本语音转换工具的普及带来的伪造语音信息泛滥问题。
  - 提出声纹匿名化和声学特征破坏技术，实现对合法说话人语音的保护。
  - 研究对抗攻击技术的鲁棒性，确保在传输过程中稳定。
  - 参加2024年全国大学生信息安全竞赛。

## **DtFormer - 基于自训练范式的恶劣天气场景鲁棒语义分割**

- **时间:** 2023.8 - 2024.1
- **角色:** 计算机视觉，域适应 (团队)
- **摘要:**
  - 解决语义分割模型在恶劣天气下表现不佳的问题。
  - 采用多层次教师自训练方法，逐步完成雾场景的语义分割。
  - 项目完成后，获得第17届中国大学生计算机设计大赛中南地区赛三等奖。

## **话题-观点图谱（数据集构建和模型微调）**

- **时间:** 2023.9 - 2023.12
- **角色:** 自然语言处理，舆情分析 (团队)
- **摘要:**
  - 收集并分析推特及主流新闻平台上的内容，构建话题-观点图谱。
  - 采用级联策略识别文本中的实体关系三元组。
  - 使用 Flask 和 D3.js 构建动态话题-观点知识图谱。
  - 被评为 Best Project（课程得分: 93/100）。

## **从“堆盒子”到动态规划**

- **时间:** 2022.1 - 2022.05
- **角色:** 算法教学，算法可视化 (团队)
- **摘要:**
  - 针对算法学习的抽象性障碍，采用可视化方法。
  - 使用 Python 的 Manim 引擎制作数学动画，编写了3000余行代码。
  - 在学生和教育者中获得良好反馈，推广应用于各级教育。

# 🏆 Competition Awards

- **中国大学生计算机设计大赛：已晋级/国赛ing** *国家级* 2024
- **“蓝桥杯”数字科技创新赛：已晋级/国赛ing** *国家级* 2024
- **全国大学生数学竞赛：二等奖** *国家级* 2023
- **“华中杯”大学生数学建模挑战赛：特等奖 (2 out of 2030 teams)** *省部级* 2023
- **中国大学生计算机设计大赛 (中南地区赛)：一等奖 (第一名)** *省部级* 2024
- **中国大学生计算机设计大赛 (中南地区赛)：一等奖** *省部级* 2024
- **中国大学生计算机设计大赛 (中南地区赛)：二等奖** *省部级* 2024
- **中国大学生计算机设计大赛 (中南地区赛)：三等奖** *省部级* 2024
- **中国大学生计算机设计大赛 (中南地区赛)：三等奖** *省部级* 2024
- **中国大学生计算机设计大赛 (中南地区赛)：三等奖** *省部级* 2023
- **“蓝桥杯”数字科技创新赛 (全国选拔赛)：一等奖** *省部级* 2024
- **“蓝桥杯”软件赛道 C++ 程序设计赛 (湖北赛区)：三等奖** *省部级* 2023
- **“蓝桥杯”软件赛道 Python 程序设计赛 (湖北赛区)：三等奖** *省部级* 2024

# 🎖 Scholarships and Honors

- *2023.10* **雷军计算机本科生奖学金** (获奖率: 20/1213=1.6%) *武汉大学&小米科技有限公司*
- *2023.10* **三好学生** (获奖率: 全校10%) *武汉大学*
- *2023.10* **武汉大学甲等学业奖学金** (获奖率: 全校5%) *武汉大学*
- *2023.9* **武汉大学青年志愿者优秀个人奖** *武汉大学*
- 2024.1 **武汉大学国家网络安全学院科创先进个人** 武汉大学国家网络安全学院
- 2024.9 **武汉大学优秀共青团员 武汉大学**

# 📖 Educations

- *2021.9 - *, 本科生, 武汉大学国家网络安全学院  专业: 网络空间安全.

<!--## 📚 Textbooks

At present, I have no time to upload all the textbooks. If you need more, please send me an email (of course you need attach your grade, class and name).

- *[高等数学（下）-武汉大学](https://github.com/1NormalGuy/1normalguy.github.io/raw/main/docs\高等数学(上).pdf)*
-->

<!--
- 计算机设计大赛经验分享, Spring 2023. \[[Slides](https://github.com/AntigoneRandy/antigonerandy.github.io/raw/main/docs/ComputerDeignCompetition.pdf)\]

- 竞赛经验漫谈, Fall 2022. \[[Slides](https://github.com/AntigoneRandy/antigonerandy.github.io/raw/main/docs/Competitions-2022Fall.pdf)\]

- 新老生经验交流会, Fall 2021. \[[Slides and Other Materials](https://github.com/AntigoneRandy/antigonerandy.github.io/raw/main/docs/ExperienceSharing2021Winter.zip)\]
-->

<!--
$^\dagger$: equal contribution, $^*$: corresponding author
-->

<!-- ## 🛰️ Geoinformatics & Remote Sensing
- [Optimized Design Method for Satellite Constellation Configuration Based on Real-time Coverage Area Evaluation](https://ieeexplore.ieee.org/document/9963835)   
Jiahao Zhou, **Boheng Li**, Qingxiang Meng   
*The 29th International Conference on Geoinformatics (CPGIS), 2022*

- [Comprehensive Evaluation of Emergency Shelters in Wuhan City Based on GIS](https://ieeexplore.ieee.org/document/9963810)   
Tingyu Luo, **Boheng Li**, Jiahao Zhou, Qingxiang Meng   
*The 29th International Conference on Geoinformatics (CPGIS), 2022* -->

<!-- ## 🤖️ AI Security, Privacy & Intellectual Property (IP) Protection -->

<!--
- [What can Discriminator do? Towards Box-free Ownership Verification of Generative Adversarial Networks](https://arxiv.org/abs/2307.15860)   
Ziheng Huang$^\dagger$, **Boheng Li**$^\dagger$, Yan Cai, Run Wang, Shangwei Guo, Liming Fang, Jing Chen, Lina Wang   
*International Conference on Computer Vision (ICCV), 2023*

- [Free Fine-tuning: A Plug-and-Play Watermarking Scheme for Deep Neural Networks](https://arxiv.org/abs/2210.07809)   
Run Wang, Jixing Ren, **Boheng Li**, Tianyi She, Wenhui Zhang, Liming Fang, Jing Chen, Lina Wang  
*ACM Multimedia (MM), 2023*

- [Dual-level Interaction for Domain Adaptive Semantic Segmentation](https://arxiv.org/abs/2307.07972)   
Dongyu Yao, **Boheng Li**$^\*$   
*ICCV Workshop on Uncertainty Quantification for Computer Vision (UnCV), 2023*


Other 2 papers regarding IP protection of DL have currently been submitted to CCF-A tier conferences.
<!-- ## 🖨️ Preprints & In Submission
-->

<!-- # 💻 Internships
To be updated. -->

<!-- # 🔗 Useful Links

## Courses

- [Linear Algebra (Hung-yi Lee, NTU, 2018)](https://www.youtube.com/watch?v=uUrt8xgdMbs&list=PLJV_el3uVTsNmr39gwbyV-0KjULUsN7fW)

- [CS229: Machine Learning](https://cs229.stanford.edu/)

- [CS230 Deep Learning](https://cs230.stanford.edu/)

- [CS231n Deep Learning for Computer Vision](http://cs231n.stanford.edu/)

- [CS224n: Natural Language Processing with Deep Learning](http://web.stanford.edu/class/cs224n/)

- [CS131 Computer Vision: Foundations and Applications](http://vision.stanford.edu/teaching/cs131_fall2223/index.html)

- [北京邮电大学鲁鹏-计算机视觉 清晰版 国家级精品课程](https://www.bilibili.com/video/BV1VW4y1v7Ph/)

- [火炉课堂-深度学习 (厦门大学)](https://www.bilibili.com/video/BV1qq4y1f7Fm)

- [中科大-凸优化](https://www.bilibili.com/video/av40868517)

- [The Next Step for Machine Learning (Hung-yi Lee, NTU, 2019)](https://www.youtube.com/watch?v=XnyM3-xtxHs&list=PLJV_el3uVTsOK_ZK5L0Iv_EQoL1JefRL4)

- [人工智能的数学基础（清华出版社）](https://www.bilibili.com/video/BV15N4y1w7e1/)

- [理解机器学习](https://www.bilibili.com/video/BV1hg411h7ys)

## Writing

- 英文学术论文写作指南 \[[link](https://www.bilibili.com/video/BV1aa411H757/)\]

- 学术规范与论文写作-南开大学程明明 \[[link](https://www.bilibili.com/video/BV18F411M7YL/)\]

- [Matplotlib cheatsheets and handouts](https://matplotlib.org/cheatsheets/)

- [十分钟掌握Seaborn，进阶Python数据可视化分析](https://zhuanlan.zhihu.com/p/49035741)

- [科学写作与哲学](https://zhuanlan.zhihu.com/p/433168083)

- [绘图软件/编程大全](https://www.bilibili.com/video/BV1gR4y1y76U)

- [如何进行高质量科研论文的写作：Shui Yu 悉尼科技大学](https://www.bilibili.com/video/BV1a8411s7Nr?p=1)

## 💻 Coding Skills

- Python最佳实践指南 \[[link](http://itpcb.com/docs/pythonguide/)\]

- Python Cookbook 3rd Edition Documentation \[[link](http://itpcb.com/docs/python3cookbook/)\]

- 🥡 Git 菜单 \[[link](http://itpcb.com/docs/gitrecipes/)\]

- Linux 基础与工具教程 \[[link](http://itpcb.com/docs/linuxtools/base/index.html)\]

## 🤖️ Artificial Intelligence & Deep Learning

- 新手如何入门pytorch？ \[[link](https://www.zhihu.com/question/55720139/answer/2788304721)\]

- 人工智能与Pytorch深度学习 \[[link](https://space.bilibili.com/100682193/channel/collectiondetail?sid=689091)\]

- [A PyTorch Tools, best practices & Styleguide](https://github.com/IgorSusmelj/pytorch-styleguide)

## Roadmap

- [科研人必看！盘点那些最好用的 AI 学术科研工具](https://zhuanlan.zhihu.com/p/153279496)

- [本科生如何自学机器学习？](https://www.zhihu.com/question/332726203/answer/737596538)

- [计算机视觉中的对抗样本 (Adversarial example)](https://zhuanlan.zhihu.com/p/352456539)

- [简单梳理一下机器学习可解释性 (Interpretability)](https://zhuanlan.zhihu.com/p/141013178)

## Misc

- [网络安全领域的科学研究和论文发表 美国西北大学 Xinyu Xing](https://www.bilibili.com/video/BV1Le4y1S7uw)

- [CVPR 9999 Best Paper——《一种加辣椒的番茄炒蛋》](https://zhuanlan.zhihu.com/p/433237905)

- [深度学习理论与实践---深度学习中的信息论：熵、最短编码、交叉熵与互信息](https://zhuanlan.zhihu.com/p/565412701)

- [Pytorch实验代码的亿些小细节](https://github.com/ahangchen/windy-afternoon/blob/master/ml/pratice/torch_best_practice.md)

- [【万字长文详解】Python库collections，让你击败99%的Pythoner](https://zhuanlan.zhihu.com/p/343747724)

- [记一次神奇的 Rebuttal 经历](https://zhuanlan.zhihu.com/p/353761920)

- [精美的终端工具 - Rich](https://www.zhihu.com/question/317758961/answer/2627662722)

- [有没有什么可以节省大量时间的 Deep Learning 效率神器？-深度学习可视化中间变量的神器Visualizer](https://www.zhihu.com/question/384519338/answer/2620414587)

- [AI-research-tools](https://github.com/bighuang624/AI-research-tools/blob/master/README.md#ai-research-tools)

- [自动超参数搜索工具optuna](https://github.com/optuna/optuna)

- [科研写作技巧](https://www.zhihu.com/question/528654768/answer/2452424449) -->
