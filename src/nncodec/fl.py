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
import numpy as np
from nncodec import nnc, nnc_core
from nncodec.framework.applications.utils.sparsification import apply_struct_spars_v2, apply_unstruct_spars_v2, get_sparsity
from nncodec.framework.applications.utils.flwr import NNClient, NNCFedAvg


nncargs = {'approx_method': 'uniform',
           'bitdepth': None,
           'bnf': False,
           'compress_differences': True,
           'compress_downstream': True,
           'compress_upstream': True,
           'diff_qp': -26,
           'err_accumulation': True,
           'job_id': '',
           'lsa': False,
           'nonweight_qp': -75,
           'opt_qp': False,
           'param_opt': True,
           'qp': -26,
           'qp_density': 2,
           'results': '.',
           'row_skipping': False,
           'sparsity': 0.6,
           'struct_spars_factor': 0.9,
           'tca': True,
           'use_dq': False,
           'verbose': False
           }

def encode(model, args=None, epoch=0, nnc_mdl=None, model_executer=None, blkid_ptypes=None, approx_param_base=None, device_id=None):
    if args == None:
        args = nncargs.copy()
    elif isinstance(args, dict):
        args = {**nncargs, **args}

    if args["sparsity"] > 0:
        sparse_params = apply_unstruct_spars_v2(args["qp"], model, target_sparsity=args["sparsity"],
                                                qp_density=np.int32(args["qp_density"]))
        model = apply_struct_spars_v2(sparse_params, gain=args["struct_spars_factor"]) \
            if args["struct_spars_factor"] > 0 else sparse_params
        print(f"Sparsity: {get_sparsity(model)*100:.2f}%")

    bs = nnc.compress(model if not isinstance(model, nnc_core.nnr_model.ModelExecute) else model.model,
                      bitstream_path=f'{args["results"]}/d{args["model"]}_epoch{epoch}_qp_{args["diff_qp"]}_bitstream.nnc',
                      qp=args["diff_qp"],
                      nonweight_qp=args["nonweight_qp"],
                      use_dq=args["use_dq"],
                      opt_qp=args["opt_qp"],
                      param_opt=args["param_opt"],
                      bnf=args["bnf"] and not args["compress_differences"],
                      lsa=args["lsa"],
                      row_skipping=args["row_skipping"],
                      tca=args["tca"],
                      block_id_and_param_type=blkid_ptypes,
                      return_bitstream=True,
                      model=nnc_mdl,
                      model_executer=model if isinstance(model, nnc_core.nnr_model.ModelExecute) else model_executer,
                      verbose=args["verbose"],
                      approx_param_base=approx_param_base,
                      device_id=device_id,
                      int_quant_bw=args["bitdepth"],
                      compress_differences=args["compress_differences"],
                      codebook_mode=2 if args["approx_method"] == 'codebook' else 0,
                      )
    return bs
