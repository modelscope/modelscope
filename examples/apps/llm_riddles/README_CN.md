# 完蛋！我被LLM包围了！(LLMRiddles)

## 项目简介
《完蛋！我被LLM包围了！》是一款智力挑战游戏。该项目利用gpt4, 基于ModelScope社区内现有的LLM对话Gradio应用程序代码，结合知乎文章[《如何用“不可能”完成任务》](https://zhuanlan.zhihu.com/p/665393240)中的预设问题，自动生成了对应的游戏代码，创造了一个独特的游戏体验。在这个游戏中，玩家需要巧妙构造问题，挑战LLM给出满足特定条件的回答。


## 更新
2023.11.7 发布初版demo🔥
2023.11.8 拆分关卡模块和llm，支持关卡独立接入，llm独立接入， 欢迎PR 🔥 🔥

## 开始游戏

### 在线体验

[LLMRiddles](https://modelscope.cn/studios/LLMRiddles/LLMRiddles/summary)

### 本地运行
要开始游戏，请按照以下步骤操作：

1. 克隆项目代码：
   ```
   git clone https://github.com/modelscope/modelscope.git
   ```
2. 进入到`examples/apps/llm_riddles`目录。
3. 安装所需的Python依赖`pip install -r requirements.txt`。
4. 执行启动命令`python app.py`.

## RoadMap
- [x] 初版本源码和创空间体验ready
- [ ] 支持自定义问题和验证逻辑接入
- [ ] 扩充到9个大关卡，每个关卡9个问题
- [ ] 支持更多开源模型
- [ ] 支持云端API和本地推理切换

## 贡献指南
我们欢迎大家为《完蛋！我被LLM包围了！》做出贡献，包括提出更多好玩的问题，修复validator的corner case，以及提供更多的玩法。请按以下步骤操作：

1. 访问项目地址 [ModelScope](https://github.com/modelscope/modelscope) 并fork项目。
2. 在你的本地环境中创建你的特性分支 (`git checkout -b feature/AmazingFeature`)。
3. 提交你的改动 (`git commit -m 'Add some AmazingFeature'`)。
4. 将你的改动推送到分支上 (`git push origin feature/AmazingFeature`)。
5. 在原项目下发起一个Pull Request。

## 社区贡献者
我们诚挚感谢所有对本项目做出贡献的社区成员，特别是：

- idea来源: [haoqiangfan](https://www.zhihu.com/people/haoqiang-fan)
- 代码大部分来自于GPT4自动生成

## 支持
如果你在游戏过程中遇到任何问题或需要帮助，请通过项目的[Issues页面](https://github.com/modelscope/modelscope/issues)提交你的问题。

## 版权和许可
本项目采用APACHE License许可证。请查看项目中的[LICENSE](https://github.com/modelscope/modelscope/blob/main/LICENSE)文件了解更多信息。
