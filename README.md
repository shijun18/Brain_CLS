# Brain_CLS

- 题名：[脑PET图像分析和疾病预测挑战赛](http://challenge.xfyun.cn/topic/info?type=PET)

- 举办：科大讯飞 + 安徽大学

- 赛制：初赛 + 复赛

- 时间：6.22 - 8.21 初赛、8.21 - 10.24 复赛

- 任务：初赛：AD和CN的二分类（AD：阿尔茨海默综合症，CN：健康）/ 复赛：AD、CN和MCI的三分类（MCI：轻度认知障碍）

  



## 初赛

- 训练集：2000张脑部PET图像，格式为png，AD:CN=1000:1000，各1000张
- 评价标准：precision、recall和F1-score，最终结果以**F1-score**为主
- 测试集：1000张脑部PET图像，格式为png
- 提交形式：.csv文件，uuid + label (AD or CN)



### 复赛
- 训练集：10000张脑部PET图像，格式为png，AD:CN:MCI=3000:3000:4000
- 评价标准：precision、recall和F1-score，最终结果以**F1-score**为主
- 测试集：2000张脑部PET图像，格式为png
- 提交形式：.csv文件，uuid + label (AD or CN or MCI)

  
