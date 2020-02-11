TensorFlow
1.	Model
1.1.	TensorFlow2.0基于tf.keras.Model：
1.1.1.	__init__():用于构建模型所用的网络层；
1.1.2.	 call(self,input): 构建前向传播的顺序。
1.	class MyModel(tf.keras.Model):
2.	    def __init__(self):
3.	        super().__init__()     
4.	        # Python 2 下使用 super(MyModel, self).__init__()
5.	        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
6.	        # layer1 = tf.keras.layers.BuiltInLayer(...)
7.	        # layer2 = MyCustomLayer(...)
8.	
9.	    def call(self, input):
10.	  # 此处添加模型调用的代码（处理输入并返回输出），例如
11.	  # x = layer1(input)
12.	  # output = layer2(x)
13.	  return output
1.1.3.	此方法级联网络（例如ResNet, Channel Attention）
1.	class Model_1(tf.keras.Model):
2.	    def __init__(self):
3.	        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
4.	        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
5.	        # layer1 = tf.keras.layers.BuiltInLayer(...)
6.	        # layer2 = MyCustomLayer(...)
7.	    def call(self, input):
8.	        # 此处添加模型调用的代码（处理输入并返回输出），例如
9.	        # x = layer1(input)
10.	       # output = layer2(x)
11.	       return output
12.	class Model_2(tf.keras.Model):
13.	    def __init__(self):
14.	        super().__init__()     # Python 2 下使用 super(MyModel, self).__init__()
15.	        # 此处添加初始化代码（包含 call 方法中会用到的层），例如
16.	        # layer0 = MyModel_1(…)
17.	        # layer1 = tf.keras.layers.BuiltInLayer(...)
18.	        # layer2 = MyCustomLayer(...)
19.	    def call(self, input):
20.	        # 此处添加模型调用的代码（处理输入并返回输出），例如
21.	        # x = layer0(input)
22.	        # output = layer1(x)
23.	        return output

1.2.	TensorFlow1.x—基于with tf.vari_scope(“model_1_name, reuse=reuse): 
（此方法给范围内取名，可构建多模型）
PS. W_init/b_init/gamma_init: 为参数初始化
1.	with tf.variable_scope('discriminator', reuse=reuse):
2.	    layers.set_name_reuse(reuse)   # 此处注意：要保证layers名称可复用
3.	    net = InputLayer(inputs=input_image, name='input')
4.	    for i in range(5):
5.	        n_channels = df_dim * 2 ** i
6.	        net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(1, 1),  act=None,
padding='SAME', W_init=w_init, b_init=b_init, name='n%ds1/c' % n_channels)
7.	        net = BatchNormLayer(layer=net, act=lrelu, is_train=is_train, gamma_init=g_init,
name='n%ds1/b' % n_channels)
8.	        net = Conv2d(net=net, n_filter=n_channels, filter_size=(3, 3), strides=(2, 2), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n%ds2/c' % n_channels)
9.	        net = BatchNormLayer(layer=net, act=lrelu, is_train=is_train, gamma_init=g_init,
                             name='n%ds2/b' % n_channels)
10.	    net = FlattenLayer(layer=net, name='flatten')
11.	    net = DenseLayer(layer=net, n_units=1024, act=lrelu, name='fc2014')
12.	    net = DenseLayer(net, n_units=1, name='output') 	# 最后一层命名为output
13.	    logits = net.outputs
14.	    net.outputs = tf.nn.sigmoid(net.outputs)  # 最后一层命名为output
15.	    return net, logits  # 返回的为整个模型
PS. 在多模型级联构建时
1.	with tf.variable_scope(“model-1”, reuse=reuse):
2.	    layer.set_name_reuse(reuse) 
3.	    net = InputLayers(inputs=inputs_data, name=’input’)
4.	    net = Conv2d(net=net, n_filter=64, n_filter=n_channels, filter_size=(3, 3),    
    strides=(1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=’c1’)
5.	    net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='b1’)
6.	    net = FlattenLayer(layer=net, name=’flatter’)
7.	    model_1 = DenseLayer(layer=net, name=’output’)		
8.	    # 网络的输出不要命名为net，以便级联
9.	with tf.variable_scope(“model-2”, reuse=reuse):
10.	    layer.set_name_reuse(reuse)
11.	    net = InputLayers(inputs=model_1, name='input')  # 级联即：输入为上层网络的输出
12.	    net = Conv2d(net=net, n_filter=64, n_filter=n_channels, filter_size=(3, 3),    
    strides=(1,1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name=’c1’)
13.	    net = BatchNormLayer(layer=net, is_train=is_train, gamma_init=g_init, name='b1’)
14.	    net = FlattenLayer(layer=net, name=’flatter’)
15.	    Model-2 = DenseLayer(layer=net, name=’output’)	
16.	    return model-2, model-2.output # 返回整个结构及其输出
2.	Train
2.1.	TensorFlow 1.X
2.1.1.	定义超参数：
	num_epochs = 5
	batch_size = 50
	num_batches = int (num_files / batch_size)
	learning_rate：随着训练的次数下降：
1.	learning_rate = 0.1 # 学习率初始化
2.	global_step = tf.Variable(0, trainable=False, name='global_step')
	#全局步数初始化，随着训练增长
3.	decayed_learning_rate=tf.train.exponential_decay(learning_rate=learning_rat,
				global_step=global_step,
decay_steps=max(num_epochs * num_batches / 2, 1), 
decay_rate=.1,staircase=True)
			# 指数衰减学习率计算：
		# decayed_learning_rate = 
learning_rate * decay_rate ^ (global_step / decay_steps)
2.1.2.	输入节点（占位符）
	初始化超参数：input_size
	self.input = tf.placeholder(dtype=tf.float32,
				shape=[batch_size, input_size, input_size, 3])
	self.ground_truth = tf.placeholder(dtype=tf.float32, 
				shape=[batch_size, input_size * 4, input_size * 4, 3]) 
2.1.3.	定义损失函数：
		loss_reconstruction = tf.reduce_mean(tf.abs(self.net_srntt.outputs
								 - self.ground_truth))
		# self.net_srntt.output = self.model(self.input, self.maps)
		# 由预先定义的模型得到，此时输入值为placeholder
（loss_perceptual/loss_texture/loss_adversarial）
2.1.4.	定义需要优化的变量：
1.	trainable_vars = tf.trainable_variables() # 获取计算图中所有变量
2.	var_g = [v for v in trainable_vars if 'texture_transfer' in v.name] 
# 生成器的变量（纹理迁移网络）
3.	var_d = [v for v in trainable_vars if 'discriminator' in v.name]    
# 判别器的变量
2.1.5.	定义优化器：
1.	optimizer = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate, beta1=beta1).minimize(loss, var_list=var_g, global_step=global_step) 
# 用于训练生成器
2.	optimizer_d = tf.train.AdamOptimizer(learning_rate=decayed_learning_rate,beta1=beta1).minimize(loss_d,var_list=var_d, global_step=global_step) # 用于训练判别器
 
2.1.6.	迭代训练：
1.	with tf.Session(config=config) as sess:	 				第一步开启Sess对话
  logging.info('Loading models ...')
2.	     tf.global_variables_initializer().run()  			    第二步变量初始化
	idx = np.arange(num_files)
3.		 for epoch in xrange(num_epochs):                 第三步epoch开始
   np.random.shuffle(idx)
4.	          for n_batch in xrange(num_batches):     第四步batch开始
5.	 第五步：Input/Label批次化处理
        sub_idx = idx[n_batch * batch_size:n_batch * batch_size + batch_size]
        batch_imgs = [imread(files_input[i], mode='RGB') for i in sub_idx]
        batch_truth = [img.astype(np.float32) for img in batch_imgs]
        batch_input = [imresize(img, .25,interp='bicubic').for img inbatch_imgs] 
6.	 第六步：sess.run()注入数据：
		其中fetches表示需要输出数据，feed_dict以字典形式注入数据
		# train with reference
        for _ in xrange(2):
            _ = sess.run(fetches=[optimizer_d], # fetches:表示输出值
feed_dict={self.input: batch_input, 
self.ground_truth: batch_truth })
			 _, _, l_rec, l_per, l_tex, l_adv, l_dis, l_bp, =                   sess.run(fetches = [optimizer, optimizer_d, loss_reconst,
loss_percep, loss_texture, loss_g, loss_d, loss_bp]
feed_dict = {self.input: batch_input,                               self.ground_truth: batch_truth, })
# train with truth
_, _, l_rec, l_per, l_tex, l_adv, l_dis, l_bp = 
sess.run(fetches=[optimizer, optimizer_d, loss_reconst, loss_percep, loss_texture, loss_g, loss_d, loss_bp],
   feed_dict={self.input: batch_input, self.ground_truth: batch_truth})
7.	格式化输出
logging.info('Epoch [%02d/%02d] Batch [%03d/%03d]\tETA: %s\n' 
'\tl_rec = %.4f\tl_bp  = %.4f\n' '\tl_per = %.4f\tl_tex = %.4f\n' 
'\tl_adv = %.4f\tl_dis = %.4f' % (epoch + 1, num_epochs, n_batch + 1, 
num_batches, eta_str, weights[4] * l_rec, weights[3] * l_bp, weights[0] * l_per, 
weights[1] * l_tex, weights[2] * l_adv, l_dis)) 
8.	每个epoch都保存模型
files.save_npz( save_list=self.net_srntt.all_params, # 需要保存的内容
name=join(self.save_dir, MODEL_FOLDERSRNTT_MODEL_NAMES['condit ional_texture_transfer']), sess=sess)
		 files.save_npz(save_list=self.net_d.all_params,
				name=join(self.save_dir, MODEL_FOLDER, SRNTT_MODEL_NAMES['discriminator']), sess=sess)

 
2.2.	TensorFlow 2.0 
2.2.1.	定义超参数：
	num_epochs = 5
	batch_size = 50
	num_batches = int (num_files / batch_size)
	learning_rate：随着训练的次数下降：
4.	learning_rate = 0.1 # 学习率初始化
5.	global_step = tf.Variable(0, trainable=False, name='global_step')
	#全局步数初始化，随着训练增长
6.	decayed_learning_rate =tf.train.exponential_decay(learning_rate=learning_rate,
global_step=global_step,
decay_steps=max(num_epochs * num_batches / 2, 1), 
decay_rate=.1,staircase=True)
			# 指数衰减学习率计算：
		# decayed_learning_rate = learning_rate * decay_rate ^ (global_step / decay_steps)
2.2.2.	实例化模型和优化器
	model = MyModel()
	optimizer = tf.keras.optimizers.Adam(learning_rate)
2.2.3.	定义损失函数（根据不同任务设置对应的损失函数，2.0中在外写function）
Loss = tf.keras.losses.sparse_categorical_crossentropy(y_true, y_pred)
# 常用交叉熵损失函数，在2.0中放在训练过程中
loss = tf.reduce_mean(loss) 
2.2.4.	迭代训练
3.	for batch_index in range(num_batches):
4.	        X, y = data_loader.get_batch(batch_size)
5.	        with tf.GradientTape() as tape:
6.	            y_pred = model(X)
7.	            loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y,
y_pred=y_pred)
8.	            loss = tf.reduce_mean(loss)
9.	            print("batch %d: loss %f" % (batch_index, loss.numpy()))
10.	        grads = tape.gradient(loss, model.variables)
11.	        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
 
3.	常用模块（基于tensorlayers模块）
3.1.	输入层：
	InputLayer(inputs=inputs, name='input')
#	inputs: The input tensor data (a placeholder or tensor)
#	name：An optional name to attach to this layer.

3.2.	卷积层：
	Conv2d(net=net_, n_filter=64, filter_size=(3, 3), strides=(1, 1), act=None,
padding='SAME', W_init=w_init, b_init=b_init, name=’convolution’)
#	net : TensorLayer layer.
#	n_filter : number of filter.
#	filter_size : tuple (height, width) for filter sizact : activation function
#	padding : a string from: "SAME", "VALID".( The type of padding algorithm to use.)
#	act : activation function
#	W_init : weights initializer, The initializer for initializing the weight matrix.
#	b_init : biases initializer or None, The initializer for initializing the bias vector. If None, skip biases. 
#	W_init_args : dictionary, The arguments for the weights tf.get_variable().
#	b_init_args : dictionary, The arguments for the biases tf.get_variable().
#	use_cudnn_on_gpu : an optional string from: "NHWC", "NCHW". Defaults to "NHWC".
#	data_format : an optional bool. Defaults to True.
#	name : An optional name to attach to this layer.

3.3.	BatchNormaliztion层
	BatchNormLayer(layer=net_, act=tf.nn.relu, is_train=is_train,
                      gamma_init=g_init, name='large/resblock_%d/bn1' % i)
#	layer: The `Layer` class feeding into this layer.
#	act : activation function
#	is_train : Boolean, Whether train or inference.
#	beta_init :beta initializer, The initializer for initializing beta 
#	gamma_init : gamma initializer, The initializer for initializing gamma
#	name : An optional name to attach to this layer.
3.4.	池化层((MaxPool2d/ MeanPool2d/MaxPool3d/MeanPool3d)
		MaxPool1d(net=net, filter_size, strides, padding='valid', 
data_format='channels_last', name=None)
#	data_format : A string, one of channels_last (default) or channels_first. 


3.5.	反卷积层
DeConv2d(net, n_out_channel = 32, filter_size=(3, 3),  out_size = (30, 30), 
strides = (2, 2), padding = 'SAME', batch_size = None, act = None, 
W_init = tf.truncated_normal_initializer(stddev=0.02), 
b_init = tf.constant_initializer(value=0.0), W_init_args = {},
b_init_args = {}, name ='decnn2d')
#	net : TensorLayer layer.
#	batch_size : int or None, batch_size.
#	filter_size : tuple (height, width) for filter sizact : activation function.
#	n_out_channel : int or None, the number of output channels.
#	out_size :  tuple of (height, width) of output.
#	act : activation function
#	padding : a string from: "SAME", "VALID".( The type of padding algorithm to use.)
#	W_init : weights initializer, The initializer for initializing the weight matrix.
#	b_init : biases initializer or None, The initializer for initializing the bias vector. If None, skip biases. 
#	W_init_args : dictionary, The arguments for the weights tf.get_variable().
#	b_init_args : dictionary, The arguments for the biases tf.get_variable().
#	use_cudnn_on_gpu : an optional string from: "NHWC", "NCHW". Defaults to "NHWC".
#	data_format : an optional bool. Defaults to True.
#	name : An optional name to attach to this layer.

3.6.	子像素卷积层
	SubpixelConv2d(net=net, scale=2, n_out_channel=None, act=tf.nn.relu,
 name='subpixel')
#	net : TensorLayer layer.
#	n_out_channel : int or None, the number of output channels.
#	act : activation function
#	scale : int, upscaling ratio,
#	name : An optional name to attach to this layer
3.7.	ElementwiseLayer层
	ElementwiseLayer(layer=[net, net_], combine_fn=tf.add, name='resblock' )
ElementwiseLayer class combines multiple `Layer` which have the same output shapes by a given elemwise-wise operation.
#	layer: a list of `Layer` class feeding into this layer. 
#	combine_fn : a TensorFlow elemwise-merge function
#	name : An optional name to attach to this layer
3.8.	ConcatLayer层（连接/合并层）
	ConcatLayer(layer=[map_in, map_ref], concat_dim=-1, name='concatenation2')
ConcatLayer class is layer which concat (merge) two or more DenseLayer to a single DenseLayer`
#	layer: a list of `Layer` class feeding into this layer. 
#	concat_dim : int，Dimension along which to concatenate.
#	name : An optional name to attach to this layer
3.9.	全连接层
	DenseLayer(layer=net, n_units=1024, act=lrelu, name='fc2014')
#	layer: The `Layer` class feeding into this layer.
#	n_units: int,  The number of units of the layer.
#	act : activation function
#	W_init : weights initializer, The initializer for initializing the weight matrix.
#	b_init : biases initializer or None, The initializer for initializing the bias vector. If None, skip biases. 
#	W_init_args : dictionary, The arguments for the weights tf.get_variable().
#	b_init_args : dictionary, The arguments for the biases tf.get_variable().
#	name : An optional name to attach to this layer. 
3.10.	Flatten层
	FlattenLayer(layer=net, name='flatten')
#	layer: The `Layer` class feeding into this layer.
#	name : An optional name to attach to this layer. 

------------------------------------更多简化应用的Layer见代码，详细注释--------------------------------
P.S. TensorFlow的API十分详细不再赘述
