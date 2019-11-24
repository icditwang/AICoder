############################################################################
# Copyright(c) Open Law Library. All rights reserved.                      #
# See ThirdPartyNotices.txt in the project root for additional notices.    #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License")           #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http: // www.apache.org/licenses/LICENSE-2.0                         #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
############################################################################
import asyncio
import time
import uuid

from pygls.features import (COMPLETION, TEXT_DOCUMENT_DID_CHANGE,
                            TEXT_DOCUMENT_DID_CLOSE, TEXT_DOCUMENT_DID_OPEN)
from pygls.server import LanguageServer
from pygls.types import (CompletionItem, CompletionList, CompletionParams,
                         ConfigurationItem, ConfigurationParams, Diagnostic,
                         DidChangeTextDocumentParams,
                         DidCloseTextDocumentParams, DidOpenTextDocumentParams,
                         MessageType, Position, Range, Registration,
                         RegistrationParams, Unregistration,
                         UnregistrationParams)

# from __future__ import absolute_import, division, print_function, unicode_literals
from .model.text_generation import Completion

"""
# import tensorflow as tf
import tensorflow as tf
import numpy as np
import os
import time
from .text_generation import Completion
"""
COUNT_DOWN_START_IN_SECONDS = 10
COUNT_DOWN_SLEEP_IN_SECONDS = 1


class AILanguageServer(LanguageServer):
    CMD_COUNT_DOWN_BLOCKING = 'countDownBlocking'
    CMD_COUNT_DOWN_NON_BLOCKING = 'countDownNonBlocking'
    CMD_REGISTER_COMPLETIONS = 'registerCompletions'
    CMD_SHOW_CONFIGURATION_ASYNC = 'showConfigurationAsync'
    CMD_SHOW_CONFIGURATION_CALLBACK = 'showConfigurationCallback'
    CMD_SHOW_CONFIGURATION_THREAD = 'showConfigurationThread'
    CMD_UNREGISTER_COMPLETIONS = 'unregisterCompletions'

    CONFIGURATION_SECTION = 'AICoderServer'

    def __init__(self):
        super().__init__()


AI_coder = AILanguageServer()

def _validate(ls, params):
    ls.show_message_log('Validating file...')

    text_doc = ls.workspace.get_document(params.textDocument.uri)
    
    source = text_doc.source
    print("source",source)
    # diagnostics = _validate_AICoder(ls,source) if source else []

    # ls.publish_diagnostics(text_doc.uri, diagnostics)



context = ""
check = 0
# def test_Completion(content):

@AI_coder.feature(COMPLETION, trigger_characters=[','])
def completions(ls,params: CompletionParams = None):
    """Returns completion items."""
    print("completions")
    ls.show_message_log("completions")
    global context
    global check
    # if check == 1:   
    # context = Completion("test")
    print("check_contexts:{}".format(context))
    ls.show_message_log("check_contexts:{}".format(context))
    return CompletionList(False, [
        #CompletionItem(context),
        CompletionItem('aicoder'),
        # CompletionItem('world'),
        CompletionItem('test_AICoder'),
        CompletionItem(context)
    ])


@AI_coder.command(AILanguageServer.CMD_COUNT_DOWN_BLOCKING)
def count_down_10_seconds_blocking(ls, *args):
    """Starts counting down and showing message synchronously.
    It will `block` the main thread, which can be tested by trying to show
    completion items.
    """
    for i in range(COUNT_DOWN_START_IN_SECONDS):
        ls.show_message('Counting down... {}'
                        .format(COUNT_DOWN_START_IN_SECONDS - i))
        time.sleep(COUNT_DOWN_SLEEP_IN_SECONDS)


@AI_coder.command(AILanguageServer.CMD_COUNT_DOWN_NON_BLOCKING)
async def count_down_10_seconds_non_blocking(ls, *args):
    """Starts counting down and showing message asynchronously.
    It won't `block` the main thread, which can be tested by trying to show
    completion items.
    """
    for i in range(COUNT_DOWN_START_IN_SECONDS):
        ls.show_message('Counting down... {}'
                        .format(COUNT_DOWN_START_IN_SECONDS - i))
        await asyncio.sleep(COUNT_DOWN_SLEEP_IN_SECONDS)


def get_change_word(txt,i):
    p = i
    while p>=0 and txt[p] != " ":
        p -= 1
    return txt[p:i]

@AI_coder.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls, params: DidChangeTextDocumentParams):
    """Text document did change notification."""
    print("did_change")
    ls.show_message_log("did_change")
    # a = Object(textDocument=Object(uri='file:///Users/wangchong/Desktop/AI%E4%BB%A3%E7%A0%81%E8%A1%A5%E5%85%A8/test.py', version=10), contentChanges=[Object(range=Object(start=Object(line=1, character=24), end=Object(line=1, character=26)), rangeLength=2, text='te')])
    # ls.show_message_log("param_is: {}".format(params))
    file_change_content = params.contentChanges[0].text
    change_start_position = []
    change_end_position = []
    change_start_position.append(params.contentChanges[0].range.start.line)
    change_start_position.append(params.contentChanges[0].range.start.character)
    change_end_position.append(params.contentChanges[0].range.end.line)
    change_end_position.append(params.contentChanges[0].range.end.character)
    text_doc = ls.workspace.get_document(params.textDocument.uri)
    file_content = text_doc.source
    split = file_content.split('\n')
    changed_line_content = split[change_start_position[0]]
    change_world = get_change_word(changed_line_content,change_start_position[1])
    # ls.show_message_log("change_position_is: {},{},{},{}".format(change_start_position[0],change_start_position[1],change_end_position[0],change_end_position[1]))
    # ls.show_message_log("file_change_content_is: {}".format(file_change_content))
    # ls.show_message_log("file_content: {}".format(file_content))
    
    # ls.show_message_log("changed_line_content: {}".format(changed_line_content))
    ls.show_message_log("change_world---:{}".format(change_world))
    print("change_world---:{}".format(change_world))
    global context
    # global check
    # check = 0
    context = Completion(change_world)
    # check = 1
    print("contexts---:{}".format(context))
    ls.show_message_log("contexts---:{}".format(context))

    # _validate(ls, params)


@AI_coder.feature(TEXT_DOCUMENT_DID_CLOSE)
def did_close(server: AILanguageServer, params: DidCloseTextDocumentParams):
    """Text document did close notification."""
    server.show_message('Text Document Did Close')


@AI_coder.feature(TEXT_DOCUMENT_DID_OPEN)
async def did_open(ls, params: DidOpenTextDocumentParams):
    """Text document did open notification."""
    ls.show_message('Text Document Did Open')
    # _validate(ls, params)


@AI_coder.command(AILanguageServer.CMD_REGISTER_COMPLETIONS)
async def register_completions(ls: AILanguageServer, *args):
    """Register completions method on the client."""
    params = RegistrationParams([Registration(str(uuid.uuid4()), COMPLETION,
                                              {"triggerCharacters": "[':']"})])
    response = await ls.register_capability_async(params)
    if response is None:
        ls.show_message('Successfully registered completions method')
    else:
        ls.show_message('Error happened during completions registration.',
                        MessageType.Error)


@AI_coder.command(AILanguageServer.CMD_SHOW_CONFIGURATION_ASYNC)
async def show_configuration_async(ls: AILanguageServer, *args):
    """Gets exampleConfiguration from the client settings using coroutines."""
    try:
        config = await ls.get_configuration_async(ConfigurationParams([
            ConfigurationItem('', AILanguageServer.CONFIGURATION_SECTION)
        ]))

        example_config = config[0].exampleConfiguration

        ls.show_message(
            'AICoderServer.exampleConfiguration value: {}'.format(example_config)
        )

    except Exception as e:
        ls.show_message_log('Error ocurred: {}'.format(e))


@AI_coder.command(AILanguageServer.CMD_SHOW_CONFIGURATION_CALLBACK)
def show_configuration_callback(ls: AILanguageServer, *args):
    """Gets exampleConfiguration from the client settings using callback."""
    def _config_callback(config):
        try:
            example_config = config[0].exampleConfiguration

            ls.show_message(
                'AICoderServer.exampleConfiguration value: {}'
                .format(example_config)
            )

        except Exception as e:
            ls.show_message_log('Error ocurred: {}'.format(e))

    ls.get_configuration(ConfigurationParams([
        ConfigurationItem('', AILanguageServer.CONFIGURATION_SECTION)
    ]), _config_callback)


@AI_coder.thread()
@AI_coder.command(AILanguageServer.CMD_SHOW_CONFIGURATION_THREAD)
def show_configuration_thread(ls: AILanguageServer, *args):
    """Gets exampleConfiguration from the client settings using thread pool."""
    try:
        config = ls.get_configuration(ConfigurationParams([
            ConfigurationItem('', AILanguageServer.CONFIGURATION_SECTION)
        ])).result(2)

        example_config = config[0].exampleConfiguration

        ls.show_message(
            'AICoderServer.exampleConfiguration value: {}'.format(example_config)
        )

    except Exception as e:
        ls.show_message_log('Error ocurred: {}'.format(e))


@AI_coder.command(AILanguageServer.CMD_UNREGISTER_COMPLETIONS)
async def unregister_completions(ls: AILanguageServer, *args):
    """Unregister completions method on the client."""
    params = UnregistrationParams([Unregistration(str(uuid.uuid4()), COMPLETION)])
    response = await ls.unregister_capability_async(params)
    if response is None:
        ls.show_message('Successfully unregistered completions method')
    else:
        ls.show_message('Error happened during completions unregistration.',
                        MessageType.Error)

if __name__ == '__main__':
    text = Completion("text")
    print(text)
"""
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
    checkpoint_dir = './training_checkpoints'
    # 检查点的文件名
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_prefix,
        save_weights_only=True)

    EPOCHS = 6

    model = build_model(vocab_size, embedding_dim, rnn_units, batch_size=1)

    model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))

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
"""
