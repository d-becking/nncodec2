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
import os
import torch
import numpy as np
from nncodec import nnc
from nncodec.framework.applications.utils.sparsification import apply_struct_spars_v2, apply_unstruct_spars_v2, get_sparsity

nncargs = {'approx_method': 'uniform',
            'bitdepth': None,
            'compress_differences': False,
            'cuda_device': None,
            'err_accumulation': False,
            'job_identifier': 'TAIMP_coding',
            'nonweight_qp': -75,
            'opt_qp': False,
            'qp': -32,
            'quantize_only': False,
            'results': '.',
            'row_skipping': True,
            'sparsity': 0.0,
            'struct_spars_factor': 0.9,
            'tca': False,
            'pre_signalling': False,
            'tensor_id': '0',
            'tensor_path': None,
            'use_dq': True,
            'param_opt': True,
            'qp_per_tensor': None,  # dict containing one qp value per parameter {Tensor1: -32, Tensor2: -40}
            'verbose': True
           }

def encode(tensor, args=None, approx_param_base=None, quantize_only=False):

    if args == None:
        args = nncargs.copy()
    elif isinstance(args, dict):
        args = {**nncargs, **args}

    if args["bitdepth"]:
        assert args["bitdepth"] < 32 and args["bitdepth"] > 0, "Selected bitdepth outside the suitable range of [1, 31] bit."

    if tensor is None and args["tensor_path"] and os.path.exists(args["tensor_path"]):
        tensor = torch.load(args["tensor_path"])

    type_list_int = ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64']
    if isinstance(tensor, torch.Tensor):
        nnc_tensor = {f'{args["tensor_id"]}': np.int32(tensor.data.cpu().detach().numpy()) if tensor.data.cpu().detach().numpy().dtype in type_list_int
                                                                        else tensor.data.cpu().detach().numpy()}
    elif isinstance(tensor, np.ndarray):
        nnc_tensor = {f'{args["tensor_id"]}': np.int32(tensor) if tensor.dtype in type_list_int else np.float32(tensor)}

    if args["sparsity"] > 0:
        print(f"Tensor inherent sparsity: {get_sparsity(nnc_tensor) * 100:.2f}%")
        sparse_params = apply_unstruct_spars_v2(args["qp"], nnc_tensor, target_sparsity=args["sparsity"])
        nnc_tensor = apply_struct_spars_v2(sparse_params, gain=args["struct_spars_factor"]) if args["struct_spars_factor"] > 0 else sparse_params
        print(f"Sparsity after sparsification: {get_sparsity(nnc_tensor) * 100:.2f}%")

    bs_path = f'{args["results"]}/{args["job_identifier"]}_{"qp_" + str(args["qp"]) if args["bitdepth"] is None else str(args["bitdepth"]) + "bit"}_bitstream.nnc'

    bs = nnc.compress(nnc_tensor,
                      bitstream_path=bs_path,
                      codebook_mode=2 if args["approx_method"] == 'codebook' else 0,
                      qp=args["qp"],
                      qp_per_tensor=args["qp_per_tensor"],
                      use_dq=args["use_dq"],
                      opt_qp=args["opt_qp"],
                      param_opt=args["param_opt"],
                      row_skipping=args["row_skipping"],
                      tca=args["tca"],
                      pre_signalling=args["pre_signalling"],
                      verbose=args["verbose"],
                      return_bitstream=True,
                      approx_param_base=approx_param_base,
                      device_id=0,
                      compress_differences=False,
                      int_quant_bw=args["bitdepth"],
                      quantize_only=quantize_only
                      )
    return bs

def decode(bitstream, tensor_id='0', approx_param_base=None, return_hls=False):
    update_base = approx_param_base is not None
    dec_nnc_tensor = nnc.decompress(bitstream, approx_param_base=approx_param_base, update_base_param=update_base, return_hls=return_hls)

    if return_hls:
        return dec_nnc_tensor[0][tensor_id], dec_nnc_tensor[1]
    else:
        return dec_nnc_tensor[tensor_id]
