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
# modal

# modal
# from __future__ import absolute_import, division, print_function, unicode_literals
# from .model.text_generation import Completion
from .model.RNNmodel import generate_text
COUNT_DOWN_START_IN_SECONDS = 10
COUNT_DOWN_SLEEP_IN_SECONDS = 1
context = ""
check = 0
# modal


# modal
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

@AI_coder.feature(COMPLETION, trigger_characters=[chr(i) for i in range(33,127)])
def completions(ls,params: CompletionParams = None):
    """Returns completion items."""
    print("completions")
    ls.show_message_log('completions')
    global context
    global check
    print("check_contexts:{}".format(context))
    return CompletionList(False, [
        #CompletionItem(context),
        CompletionItem(label='aicoder',detail="AICoder",documentation='aicoder'),
        # CompletionItem('world'),
        CompletionItem('test_AICoder',detail="AICoder",documentation='test_AICoder'),
        CompletionItem(label=context,detail="AICoder",documentation=context)
    ])
def get_change_word(txt,i):
    p = i
    while p>=0 and txt[p] != " ":
        p -= 1
    return txt[p+1:i+1]
def test(word):
    return word+"test"
@AI_coder.feature(TEXT_DOCUMENT_DID_CHANGE)
def did_change(ls, params: DidChangeTextDocumentParams):
    """Text document did change notification."""
    print("did_change")
    ls.show_message_log('did_change')
    file_change_content = params.contentChanges[0].text
    change_start_position = []
    change_end_position = []
    change_start_position.append(params.contentChanges[0].range.start.line)
    change_start_position.append(params.contentChanges[0].range.start.character)
    change_end_position.append(params.contentChanges[0].range.end.line)
    change_end_position.append(params.contentChanges[0].range.end.character)
    text_doc = ls.workspace.get_document(params.textDocument.uri)
    file_content = text_doc.source
    print("change_position: {},{},{},{}".format(change_start_position[0],change_start_position[1],change_end_position[0],change_end_position[1]))
    split = file_content.split('\n')
    changed_line_content = split[change_start_position[0]]
    print("changed_line_content: {}".format(changed_line_content))
    if change_start_position[1] < change_end_position[1]:
        p = change_start_position[1]-1
    else:
        p = change_start_position[1]
    change_world = get_change_word(changed_line_content,p)

    # ls.show_message_log("file_change_content_is: {}".format(file_change_content))
    # ls.show_message_log("file_content: {}".format(file_content))
    print("change_world---:{}".format(change_world))
    global context
    # context = Completion(change_world)
    context = generate_text(change_world)
    ls.show_message_log("contexts---:{}".format(context))

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
    text = generate_text("text")
    print(text)
