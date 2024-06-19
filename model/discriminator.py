import torch.nn as nn
import torch.nn.functional as F


class FCDiscriminator(nn.Module):

	def __init__(self, num_classes, ndf = 64):
		super(FCDiscriminator, self).__init__()

		# 第一个卷积层，输入通道数为num_classes，输出通道数为ndf，卷积核大小为4，步长为2，填充为1
		self.conv1 = nn.Conv2d(num_classes, ndf, kernel_size=4, stride=2, padding=1)
		# 第二个卷积层，输入通道数为ndf，输出通道数为ndf*2，卷积核大小为4，步长为2，填充为1
		self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
		# 第三个卷积层，输入通道数为ndf*2，输出通道数为ndf*4，卷积核大小为4，步长为2，填充为1
		self.conv3 = nn.Conv2d(ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
		# 第四个卷积层，输入通道数为ndf*4，输出通道数为ndf*8，卷积核大小为4，步长为2，填充为1
		self.conv4 = nn.Conv2d(ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
		# 分类层，输入通道数为ndf*8，输出通道数为1，卷积核大小为4，步长为2，填充为1
		self.classifier = nn.Conv2d(ndf*8, 1, kernel_size=4, stride=2, padding=1)

		# LeakyReLU激活函数，负斜率为0.2，inplace为True
		self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
		#self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
		#self.sigmoid = nn.Sigmoid()


	def forward(self, x):
		# 通过第一个卷积层并激活
		x = self.conv1(x)
		x = self.leaky_relu(x)

		# 通过第二个卷积层并激活
		x = self.conv2(x)
		x = self.leaky_relu(x)

		# 通过第三个卷积层并激活
		x = self.conv3(x)
		x = self.leaky_relu(x)

		# 通过第四个卷积层并激活
		x = self.conv4(x)
		x = self.leaky_relu(x)

		# 通过分类层
		x = self.classifier(x)
		#x = self.up_sample(x)
		#x = self.sigmoid(x)

		return x
