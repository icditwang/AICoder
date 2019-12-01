import os
import tensorflow as tf
import numpy as np
import os
import time
def Completion(content):
    path_to_file = tf.keras.utils.get_file('shakespeare.txt',
                                           'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')
    text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
    vocab = sorted(set(text))
    char2idx = {u: i for i, u in enumerate(vocab)}
    idx2char = np.array(vocab)
    text_as_int = np.array([char2idx[c] for c in text])
    seq_length = 100
    examples_per_epoch = len(text) // seq_length
    char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)
    sequences = char_dataset.batch(seq_length + 1, drop_remainder=True)
    def split_input_target(chunk):
        input_text = chunk[:-1]
        target_text = chunk[1:]
        return input_text, target_text

    dataset = sequences.map(split_input_target)
    BATCH_SIZE = 64
    BUFFER_SIZE = 10000
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)
    dataset
    vocab_size = len(vocab)
    embedding_dim = 256
    rnn_units = 1024
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
    model = build_model(
        vocab_size=len(vocab),
        embedding_dim=embedding_dim,
        rnn_units=rnn_units,
        batch_size=BATCH_SIZE)
    for input_example_batch, target_example_batch in dataset.take(1):
        example_batch_predictions = model(input_example_batch)
        # print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")
    sampled_indices = tf.random.categorical(example_batch_predictions[0], num_samples=1)
    sampled_indices = tf.squeeze(sampled_indices, axis=-1).numpy()
    sampled_indices
    def loss(labels, logits):
        return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)

    example_batch_loss = loss(target_example_batch, example_batch_predictions)
    model.compile(optimizer='adam', loss=loss)
    # 检查点保存至的目录
    checkpoint_dir = os.path.join(os.getcwd(),'server/model/training_checkpoints' )
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 6

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)
    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

    # model.load_weights(checkpoint_dir)
    model.build(tf.TensorShape([1, None]))

    def generate_text(model, start_string):
        
        # 要生成的字符个数
        num_generate = 20

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
            predicted_id = tf.random.categorical(predictions, num_samples=1)[-1, 0].numpy()

            # 把预测字符和前面的隐藏状态一起传递给模型作为下一个输入
            input_eval = tf.expand_dims([predicted_id], 0)

            text_generated.append(idx2char[predicted_id])

        return (start_string + ''.join(text_generated))
    return generate_text(model, start_string=content)


if __name__ == "__main__":
    # a1 = "from pygls.features import COMPLETION\nfrom pygls.server import LanguageServer\nfrom pygls.types import CompletionItem, CompletionList, CompletionParams\n\nserver = LanguageServer()\n\n@server.feature(COMPLETION, trigger_characters=[','])\ndef completions(params: CompletionParams):\n    \"\"\"Returns completion items.\"\"\"\n    print(\"completions.....\")\n    return CompletionList(False, [\n        CompletionItem('hello',kind=2,data=1),\n        CompletionItem('world',kind=2,data=2),\n        CompletionItem('testpygls',kind=2,data=2)\n    ])\ntxt = \"a\\nb\"\ntest\nprint(\"start_tcp.....\")\nserver.start_tcp('127.0.0.1', 2087)"
    # a = "test AICoder hello world"
    print(Completion(u"text"))   
    # print(os.path.abspath('.'))
