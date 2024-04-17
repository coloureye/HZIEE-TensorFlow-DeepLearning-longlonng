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

