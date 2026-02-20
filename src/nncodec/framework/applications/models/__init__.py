'''
The copyright in this software is being made available under the Clear BSD
License, included below. No patent rights, trademark rights and/or
other Intellectual Property Rights other than the copyrights concerning
the Software are granted under this license.

The Clear BSD License

Copyright (c) 2019-2025, Fraunhofer-Gesellschaft zur Förderung der angewandten Forschung e.V. & The NNCodec Authors.
All rights reserved.

Redistribution and use in source and binary forms, with or without modification,
are permitted (subject to the limitations in the disclaimer below) provided that
the following conditions are met:

     * Redistributions of source code must retain the above copyright notice,
     this list of conditions and the following disclaimer.

     * Redistributions in binary form must reproduce the above copyright
     notice, this list of conditions and the following disclaimer in the
     documentation and/or other materials provided with the distribution.

     * Neither the name of the copyright holder nor the names of its
     contributors may be used to endorse or promote products derived from this
     software without specific prior written permission.

NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER
IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
POSSIBILITY OF SUCH DAMAGE.
'''

from .ResNet_CIFAR import resnet20, resnet20_client_split, resnet20_server_split, resnet56, resnet56_client_split, resnet56_server_split
from .tinyllama import Transformer, ModelArgs
# from .tokenizer import Tokenizer

__all__ = ['resnet20', 'resnet56', 'tinyllama']

def init_model(model_name, num_classes=100, pretrained=False, parser_args=None):
    ##################################
    if model_name == 'resnet20':
        model = resnet20()
    elif model_name == 'resnet20_client':
        model = resnet20_client_split()
    elif model_name == 'resnet20_server':
        model = resnet20_server_split()
    ##################################
    elif model_name == 'resnet56':
        model = resnet56(class_num=num_classes)
    elif model_name == 'resnet56_client':
        model = resnet56_client_split()
    elif model_name == 'resnet56_server':
        model = resnet56_server_split(class_num=num_classes)
    ##################################
    elif model_name =='tinyllama':
        assert parser_args is not None, ("For tinyllama models please provide parser_args argument including args.TLM_size"
                                         " args.tokenizer_path and optionally args.max_seq_len")
        multiple_of = 32
        if parser_args.TLM_size == 0:
            dim = 64
            n_layers = 8 # 5#10#5
            n_heads = 8
            n_kv_heads = 4
            max_seq_len = 512
        elif parser_args.TLM_size == 1:
            dim = 288
            n_layers = 10  # 6# 10#6
            n_heads = 6
            n_kv_heads = 6
            max_seq_len = 256
        elif parser_args.TLM_size == 2:
            dim = 512
            n_layers = 10  # 8
            n_heads = 8
            n_kv_heads = 8
            max_seq_len = 1024
        elif parser_args.TLM_size == 3:
            dim = 768
            n_layers = 12
            n_heads = 12
            n_kv_heads = 12
            max_seq_len = 1024
        else:
            assert 0, "TLM architecture not implemented"

        if parser_args.max_seq_len is not None:
            max_seq_len = parser_args.max_seq_len

        enc = Tokenizer(tokenizer_model=parser_args.tokenizer_path)
        vocab_size = enc.sp_model.get_piece_size()

        model_args = dict(
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
            n_kv_heads=n_kv_heads,
            vocab_size=vocab_size,
            multiple_of=multiple_of,
            max_seq_len=max_seq_len,
            dropout=parser_args.dropout if "dropout" in parser_args else 0.0,
        )
        conf = ModelArgs(**model_args)
        model = Transformer(conf), enc

    return model
