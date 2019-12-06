import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"
import tensorflow as tf
from tensorflow.keras import Model
import numpy as np
from matplotlib import pyplot as plt
# tf.__version__

# 检查点保存至的目录
checkpoint_dir = './ckpt99'
checkpoint_dir = os.path.join(os.getcwd(),'server/model/ckpt99')
# 检查点的文件名
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)

tf.train.latest_checkpoint(checkpoint_dir)

def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.GRU(rnn_units,
                        return_sequences=True,
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

vocab=['\t', '\n','\r', ' ', '!', '"', '$', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.',
 '/','0', '1', '2', '3', '4', '5', '6', '7', '8', '9',':', ';', '<', '=','>', '?', '@', 'A', 'B',
 'C', 'D', 'E', 'F','G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
 '[','\\', ']','^', '_', '`', 'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o',
 'p', 'q', 'r', 's','t', 'u','v', 'w', 'x', 'y', 'z', '{', '|', '}',
 '~', '£', 'å', 'é', 'î', 'ñ', 'ø', 'ı','准', '合', '回', '大', '小', '总', '报', '数', '时', '测', '率', '矩', '确', '算', '练',
 '训', '证', '试', '运','间', '阵', '验', '가', '힣']

# 创建从非重复字符到索引的映射
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

# 词集的长度
vocab_size = 128

# 嵌入的维度
embedding_dim = 256

# RNN 的单元数量
rnn_units = 1024

model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

model.build(tf.TensorShape([1, None]))

def generate_text( start_string):
  # 评估步骤（用学习过的模型生成文本）

  # 要生成的字符个数
  num_generate = 10

  # 将起始字符串转换为数字（向量化）
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # 空字符串用于存储结果
  text_generated = []

  # 低温度会生成更可预测的文本
  # 较高温度会生成更令人惊讶的文本
  # 可以通过试验以找到最好的设定
  temperature = 1.0

  # 这里批大小为 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # 删除批次的维度
      predictions = tf.squeeze(predictions, 0)

      # 用分类分布预测模型返回的字符
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))

#这句放到completion里，全局变量model和generate_text
if __name__ == "__main__":
  print(generate_text(start_string=u"def "))