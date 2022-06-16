# OHEM implementation 

OHEM 通过减少计算成本来选择难例，提高网络性能，主要用于目标检测。假设你想训练一个汽车检测器，并且你有正样本图像（图像中有汽车）和负样本图像（图像中没有汽车）。现在你想训练你的网络。实际上，你会发现负样本数量会远远大于正样本数量，而且大量的负样本是比较简单的。因此，比较明智的做法是选择一部分对于网络最有帮助的负样本（比较有难度的，容易被识别为正样本的）参与训练，难例挖掘就是用于选择对网络最有帮助的负样本的。

通常来说，通过对网络训练进行一定的迭代后得到临时的模型，使用临时模型对所有的负样本进行测试，便可以发现那些 loss 很大的负样本实例，这些实例就是所谓的难例。但是这种查找难例的方法需要很大的计算量，因为负样本图像可能会很多。外这一方法可能是次优的，当你进行难例挖掘的时候，模型的权重是固定的，当前权重下的难例未必适用于接下了的迭代

OHEM 通过批量难例选择来解决上述两个问题，给定 batch-size K，前向传播保持不变，计算损失。然后选择 M（M<K）个高损失的实例，仅使用这 M 个实例的损失进行反向传播。