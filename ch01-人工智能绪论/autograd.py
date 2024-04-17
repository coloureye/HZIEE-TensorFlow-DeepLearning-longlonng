"""
该代码主要作用是使用TensorFlow库进行自动微分。主要计算过程可以用以下数学公式表示：
假设有一个二次函数 y = a * w**2 + b * w + c，其中 a = 1, b = 2, c = 3, w = 4。
想要计算函数 y 关于 w 的导数，即 dy/dw。根据二次函数的导数公式，dy/dw = 2*a*w + b
将 a = 1, b = 2, w = 4 代入上述公式，得到：dy/dw = 2*1*4 + 2 = 10 这就是主要计算过程。
使用TensorFlow的 tf.GradientTape() 上下文和 tape.gradient() 函数自动完成了这个导数计算过程。
在脚本的最后，它打印出了计算得到的导数值 dy/dw = 10。
"""

import tensorflow as tf

# 创建4个张量
a = tf.constant(1.) #python中1.和1.0都表示相同的浮点数值1.0。写成1.的方式是一种简写形式，
b = tf.constant(2.)
c = tf.constant(3.)
w = tf.constant(4.)

# 创建了一个tf.GradientTape上下文。TensorFlow使用这个上下文来自动记录涉及可训练变量的所有操作
with tf.GradientTape() as tape:# 构建梯度环境
	tape.watch([w]) # 将w加入梯度跟踪列表，只有加入跟踪列表中的变量才能求导
	# 构建计算过程
	y = a * w**2 + b * w + c #y是w的二次函数

# 计算y关于w的梯度，并将结果赋值给dy_dw。这是通过调用tape.gradient函数实现的，该函数接受两个参数：
# 想要微分的值（在这里是y），以及想要对其进行微分的变量列表（在这里是[w]）
[dy_dw] = tape.gradient(y, [w])
print(dy_dw)

