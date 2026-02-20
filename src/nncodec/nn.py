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
import torch
import numpy as np
from nncodec import nnc
from nncodec.framework.pytorch_model import np_to_torch, torch_to_numpy
from nncodec.framework.applications.utils.sparsification import apply_struct_spars_v2, apply_unstruct_spars_v2, get_sparsity

nncargs = {'approx_method': 'uniform',
            'bitdepth': None,
            'compress_differences': False,
            'nonweight_qp': -75,
            'opt_qp': False,
            'qp': -32,
            'qp_density': 2,
            'quantize_only': False,
            'results': '.',
            'row_skipping': True,
            'sparsity': 0.0,
            'struct_spars_factor': 0.9,
            'tca': False,
            'pre_signalling': False,
            'use_dq': True,
            'param_opt': True,
            'qp_per_tensor': None,  # dict containing one qp value per parameter {Tensor1: -32, Tensor2: -40}
            'verbose': True,
            'bnf': False,
            'lsa': False,
            'wandb': False,
            'lr': 1e-3,
            'epochs': 1,
            'dataset_path': None,
            'model': 'NN',
            'max_batches': None,
            'workers': 1,
            'batch_size': 8,
           }

def encode(model, args=None, use_case_name=None, incremental=False, approx_param_base=None, epoch=0):

    if args == None:
        args = nncargs.copy()
    elif isinstance(args, dict):
        args = {**nncargs, **args}

    if incremental or isinstance(model, dict):

        if args["sparsity"] > 0 and epoch > 0:
            sparse_params = apply_unstruct_spars_v2(args["qp"], model, target_sparsity=args["sparsity"],
                                                    qp_density=np.int32(args["qp_density"]))
            model = apply_struct_spars_v2(sparse_params, gain=args["struct_spars_factor"]) \
                if args["struct_spars_factor"] > 0 else sparse_params
            print(f"Sparsity: {get_sparsity(model) * 100:.2f}%")

        bs = nnc.compress(
            model,
            bitstream_path=f'{args["results"]}/d{args["model"]}_epoch{epoch}_qp_{args["qp"]}_bitstream.nnc',
            codebook_mode=2 if args["approx_method"] == 'codebook' else 0,
            qp=args["qp"],
            nonweight_qp=args["nonweight_qp"],
            qp_per_tensor=args["qp_per_tensor"],
            use_dq=args["use_dq"],
            opt_qp=args["opt_qp"],
            param_opt=args["param_opt"],
            int_quant_bw=args["bitdepth"],
            row_skipping=args["row_skipping"],
            tca=args["tca"],
            pre_signalling=args["pre_signalling"],
            approx_param_base=approx_param_base,
            compress_differences=False,
            bnf=args["bnf"],
            lsa=args["lsa"],
            block_id_and_param_type=None,
            return_bitstream=True,
            wandb_logging=args["wandb"],
            device_id=0,
            verbose = args["verbose"],
        )

    else:

        if args["sparsity"] > 0 and epoch > 0:
            if isinstance(model, torch.nn.Module):
                sparse_params = apply_unstruct_spars_v2(args["qp"], torch_to_numpy(model.state_dict()),
                                                        target_sparsity=args["sparsity"],
                                                        qp_density=np.int32(args["qp_density"]))
                if args["struct_spars_factor"] > 0:
                    sparse_params = apply_struct_spars_v2(sparse_params, gain=args["struct_spars_factor"])
                print(f"Sparsity: {get_sparsity(sparse_params) * 100:.2f}%")
                model.load_state_dict(np_to_torch(sparse_params))
            else:
                assert 0, "sparsification of non-torch models not yet supported"

        bs, enc_mdl_info = nnc.compress_model(model,
                                              bitstream_path=f'{args["results"]}/{args["model"]}_qp_{args["qp"]}_bitstream.nnc',
                                              qp=args["qp"],
                                              nonweight_qp=args["nonweight_qp"],
                                              codebook_mode=2 if args["approx_method"] == 'codebook' else 0,
                                              lsa=args["lsa"],
                                              bnf=args["bnf"],
                                              opt_qp=args["opt_qp"],
                                              param_opt=args["param_opt"],
                                              ioq=args["ioq"],
                                              int_quant_bw=args["bitdepth"],
                                              row_skipping=args["row_skipping"],
                                              tca=args["tca"],
                                              approx_param_base=approx_param_base,
                                              use_dq=args["use_dq"],
                                              learning_rate=args["lr"],
                                              epochs=args["epochs"],
                                              use_case=use_case_name,
                                              dataset_path=args["dataset_path"],
                                              wandb_logging=args["wandb"],
                                              return_bitstream=True,
                                              return_model_data=True or args["bnf"],
                                              max_batches=args["max_batches"],
                                              num_workers=args["workers"],
                                              batch_size=args["batch_size"],
                                              verbose=args["verbose"],
                                              device_id=0,
                                              )
    return bs


def decode(bs, args=None, model=None, approx_param_base=None, return_hls=False):

    if args == None:
        args = nncargs.copy()
    elif isinstance(args, dict):
        args = {**nncargs, **args}

    update_base = approx_param_base is not None
    rec_mdl_params = nnc.decompress(bs, approx_param_base=approx_param_base, update_base_param=update_base,
                                    reconstruct_lsa=args["lsa"], reconstruct_bnf=args["bnf"], return_hls=return_hls)

    ### reconstruction
    # if args and args["bnf"]: ##TODO (incremental) BNF
    #     for n, m in model.named_modules():  # reset running statistics and trainable bn_gamma to 1
    #         if isinstance(m, torch.nn.BatchNorm2d):
    #             m.reset_running_stats()
    #             m.weight = torch.nn.Parameter(torch.ones_like(m.weight))
    #     re-name bn_beta-type params to PYT bn.bias
    #     try:
    #         rec_mdl_params = {enc_mdl_info["bnf_matching"][param] if param in enc_mdl_info["bnf_matching"]
    #                       else param: rec_mdl_params[param] for param in rec_mdl_params}
    #     except:
    #         rec_mdl_params = {bnf_matching[param] if param in bnf_matching
    #                           else param: rec_mdl_params[param] for param in rec_mdl_params}
    if return_hls:
        return rec_mdl_params[0], rec_mdl_params[1]
    else:
        return rec_mdl_params