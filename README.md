# MGA-Net-Multi-Granularity-Attribute-Network-with-Dynamic-Feedback-
本项目基于PyTorch实现多粒度属性网络嵌入（MGA），结合动态权重分配与历史反馈机制，用于图数据的高效特征学习。核心功能包括：  多粒度属性嵌入层：融合网络拓扑与节点属性，生成低维语义表征。 动态权重分配矩阵：通过Softmax概率映射实现权重自适应调整。 历史反馈集成模块：调用历史权重状态优化当前训练过程。 适用于社交网络分析、推荐系统、知识图谱补全等场景。

embedding_layer.py是MGA属性嵌入层实现（PyTorch自定义层）

dynamic_weights.py是动态权重分配模块（Softmax概率映射）

history_feedback.py是历史权重反馈机制（集成MGA模块）
