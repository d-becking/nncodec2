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

### model determinism / reproducibility
import os

# Global seed constants
SEED_RANDOM = 909
SEED_TORCH = 808
SEED_NUMPY = 303

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # Must be before torch import
os.environ["PYTHONHASHSEED"] = str(SEED_RANDOM)
os.environ["RAY_RANDOM_SEED"] = str(SEED_RANDOM)
os.environ["RAY_DEDUP_LOGS"] = "0"

import torch
import random
import numpy as np

# PyTorch seeding
torch.manual_seed(SEED_TORCH)
torch.cuda.manual_seed_all(SEED_TORCH)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.use_deterministic_algorithms(True, warn_only=False)
if torch.backends.mps.is_available():
    torch.mps.empty_cache()
    torch.mps.manual_seed(SEED_TORCH)
    torch.mps.set_per_process_memory_fraction(0.75)

random.seed(SEED_RANDOM)
np.random.seed(SEED_NUMPY)

torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import copy
def seeded_model_copy(model, seed=SEED_TORCH):
    torch.manual_seed(seed)
    return copy.deepcopy(model)

import argparse
import shutil
import sys
import torchvision
import warnings, logging
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters

from nncodec.fl import encode, NNClient, NNCFedAvg
from nncodec import nnc
from nncodec.framework import pytorch_model
from nncodec.framework.applications.utils.transforms import split_datasets
from nncodec.framework.applications import models, datasets

parser = argparse.ArgumentParser(description='NNCodec with Flower for Federated Learning')
parser.add_argument('--qp', type=int, default=-32, help='quantization parameter for NNs (default: -32)')
parser.add_argument('--diff_qp', type=int, default=None, help='quantization parameter for dNNs. Defaults to qp if unspecified (default: None)')
parser.add_argument('--qp_density', type=int, default=2, help='quantization scale parameter (default: 2)')
parser.add_argument('--nonweight_qp', type=int, default=-75, help='qp for non-weights, e.g. BatchNorm params (default: -75)')
parser.add_argument("--opt_qp", action="store_true", help='Modifies QP layer-wise')
parser.add_argument("--use_dq", action="store_true", help='Enable dependent scalar / Trellis-coded quantization')
parser.add_argument('--bitdepth', type=int, default=None, help='Optional: integer-aligned bitdepth for limited precision (default: None); note: overwrites QPs.')
parser.add_argument("--lsa", action="store_true", help='Enable Local Scaling Adaptation')
parser.add_argument("--bnf", action="store_true", help='Enable BatchNorm Folding')
parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity rate (default: 0.0)')
parser.add_argument('--struct_spars_factor', type=float, default=0.0, help='Factor for structured sparsification (default: 0.9)')
parser.add_argument("--row_skipping", action="store_true", help='Enable Row Skipping')
parser.add_argument("--tca", action="store_true", help='Enable Temporal Context Adaptation')
parser.add_argument('--batch_size', type=int, default=4, help='Batch size for training (default=64)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--max_batches', type=int, default=None, help='Max num of batches to process (default: 0, i.e., all)')
parser.add_argument('--max_batches_test', type=int, default=None, help='Max num of batches to process in test fct (default: None, i.e., all)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 3e-4)')
parser.add_argument('--model', type=str, default='resnet56', metavar=f'any of {models.__all__} or {torchvision.models.list_models(torchvision.models)}')
parser.add_argument('--model_path', type=str, default=None, metavar='./example/ResNet56_CIF100.pt')
parser.add_argument('--model_rand_int', action="store_true", help='model randomly initialized, i.e., w/o loading pre-trained weights')
parser.add_argument('--dataset', type=str, default='cifar100', metavar=f"Any of {datasets.__all__}")
parser.add_argument('--dataset_path', type=str, default='../data')
parser.add_argument('--tokenizer_path', type=str, default='./tokenizer/telko_tokenizer.model')
parser.add_argument('--max_seq_len', type=int, default=1525, help='Custom max_seq_len for tiny Llama')
parser.add_argument('--TLM_size', type=int, default=0, help='tiny Llama size [0, 1, 2, 3]')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4), if 0 debugging mode enabled')
parser.add_argument("--wandb", action="store_true", help='Use Weights & Biases for data logging')
parser.add_argument('--wandb_key', type=str, default='', help='Authentication key for Weights & Biases API account ')
parser.add_argument('--wandb_run_name', type=str, default='NNC_WP_spars', help='Identifier for current run')
parser.add_argument('--job_id', type=str, default='', help='Identifier for job e.g. on GPU Cluster')
parser.add_argument("--pre_train_model", action="store_true", help='Training the full model prior to compression')
parser.add_argument("--verbose", action="store_true", help='Stdout process information.')
parser.add_argument('--num_clients', type=int, default=2, help='Number of clients in FL scenario (default: 2)')
parser.add_argument("--compress_upstream", action="store_true", help='Compression of clients-to-server communication')
parser.add_argument("--compress_downstream", action="store_true", help='Compression of server-to-clients communication')
parser.add_argument("--compress_differences", action="store_true", help='Compresses weight differences wrt. to base model, otherwise full base models are communicated')
parser.add_argument("--err_accumulation", action="store_true", help='Locally accumulates quantization errors (residuals)')
parser.add_argument('--weight_decay', type=float, default=1e-3)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--cuda_device', type=int, default=None)


def main():
    args = parser.parse_args()
    warnings.filterwarnings("ignore")
    logging.getLogger("flwr").setLevel(logging.CRITICAL)

    clear_results = True

    if args.diff_qp == None:
        args.diff_qp = args.qp

    if torch.cuda.is_available():
        if args.cuda_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)

    if clear_results and os.path.exists(os.path.join(args.results)):#, f'd{args.model}_epoch0_qp_{args.diff_qp}_bitstream.nnc')):
        shutil.rmtree(f"{args.results}")

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.wandb:
        import wandb
        if isinstance(args.wandb_key, str) and len(args.wandb_key) == 40:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        else:
            assert 0, "incompatible W&B authentication key"

    if args.model in models.__all__:
        if "tinyllama" in args.model:
            model, tokenizer = models.init_model(args.model, parser_args=args)
        else:
            model = models.init_model(args.model, num_classes=100)
    elif args.model in torchvision.models.list_models(torchvision.models):
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    elif args.model in torchvision.models.segmentation.deeplabv3.__all__:
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    else:
        assert 0, f"Model not specified in /framework/applications/models and not available in torchvision model zoo" \
                  f"{torchvision.models.list_models(torchvision.models)})"

    if not args.model_rand_int and args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path))

    if args.wandb:
        wandb.init(
            config=args,
            project=f"{args.model}_{args.dataset}_{args.wandb_run_name}",
            name=f"{args.job_id}{args.num_clients}_{args.model}{'_'+str(args.TLM_size) if args.model == 'tinyllama' else ''}"
                 f"_lr_{args.lr}_bs_{args.batch_size}_qp_{args.qp}"
                 f"{'_diff' if args.compress_differences else '_base'}{'_resids' if args.err_accumulation else ''}"
                 f"{'_optqp' if args.opt_qp else ''}{'_dq' if args.use_dq else ''}{'_bnf' if args.bnf else ''}"
                 f"{'_lsa' if args.lsa else ''}{'_spars_'+str(args.sparsity) if args.sparsity > 0 else ''}"
                 f"{'_struct_'+str(args.struct_spars_factor) if args.struct_spars_factor > 0 else ''}"
                 f"{'_rs' if args.row_skipping else ''}{'_tca' if args.tca else ''}",
            entity="edl-group",
            save_code=True,
            dir=f"{args.results}"
        )

    UCS = {'cifar100': 'NNR_PYT_CIF100', 'cifar10': 'NNR_PYT_CIF10', 'imagenet200': 'NNR_PYT_IN200',
           'voc': 'NNR_PYT_VOC', 'V2X': 'NNR_PYT_Telko'}
    use_case_name = UCS[args.dataset] if args.dataset in UCS else 'NNR_PYT'

    nnc_mdl, nnc_mdl_executer, mdl_params = pytorch_model.create_NNC_model_instance_from_object(
        model,
        dataset_path=args.dataset_path if not args.dataset in ['V2X'] else None,
        lr=args.lr,
        batch_size=args.batch_size,
        num_workers=args.workers,
        lsa=args.lsa,
        epochs=args.epochs,
        use_case=use_case_name
    )

    mdl_info = {"parameter_index": {k: i for i, k in enumerate(nnc_mdl_executer.model.state_dict().keys())
                                            if nnc_mdl_executer.model.state_dict()[k].shape != torch.Size([])}}

    if args.bnf:
        p_types = nnc_mdl.guess_block_id_and_param_type(nnc_mdl_executer.model.state_dict(),
                                                        bn_info=nnc_mdl.get_torch_bn_info(nnc_mdl_executer.model))
        mdl_info = nnc.compress(mdl_params, model=nnc_mdl, bnf_mapping=True, block_id_and_param_type=p_types)

    # split datasets across clients
    if not args.dataset in ['V2X']:
        trainloaders, valloaders, testloader = split_datasets(nnc_mdl_executer.test_set,
                                                              nnc_mdl_executer.train_loader.dataset,
                                                              num_partitions=args.num_clients,
                                                              batch_size=args.batch_size,
                                                              num_workers=args.workers,
                                                              val_ratio=0.0, #if > 0 creates validation split at each client
                                                              )
    else:
        trainloaders, valloaders, testloader = datasets.V2X(args)

    # function for global evaluation on the server
    def get_evaluate_fn(s_model, testloader_server):
        """Define ."""
        def evaluate_fn(server_round: int, parameters):
            s_model.load_state_dict(parameters, strict=True)
            test_res_dict = nnc_mdl_executer.handle.evaluate(s_model, criterion=nnc_mdl_executer.handle.criterion,
                                                    testloader=testloader_server, device=nnc_mdl_executer.device,
                                                    verbose=False, max_batches=args.max_batches_test)
            print(f"Server round {server_round} global {[str(k) + ': ' + str(v) for k, v in test_res_dict.items()]}")
            return test_res_dict
        return evaluate_fn

    def accumulated_bitstream_sizes(metrics):
        bs_sizes = [m["bs_size"] for _, m in metrics]
        return {"accumulated_bs_sizes": sum(bs_sizes)}

    # function to generate client instances
    def generate_client_fn(trainloaders, valloaders, criterion, device, c_model):
        def client_fn(context: Context):
            """Returns a FlowerClient containing the cid-th data partition"""
            cid = int(context.node_config['partition-id']) + 1
            print(f"CID: {cid}")

            torch.manual_seed(808 + cid)
            np.random.seed(303 + cid)
            random.seed(909 + cid)

            return NNClient(context,
                            trainloader=trainloaders[cid-1], valloader=valloaders[cid-1], id=cid,
                            criterion=criterion, device=device, c_model=c_model, args=args, mdl_info=mdl_info,
                            encode_fn=encode, decode_fn=nnc.decompress, train_fn=nnc_mdl_executer.handle.train
                            ).to_client()
        return client_fn

    client_fn_callback = generate_client_fn(trainloaders=trainloaders,
                                            valloaders=valloaders,
                                            criterion=nnc_mdl_executer.handle.criterion,
                                            device=nnc_mdl_executer.device,
                                            c_model=nnc_mdl_executer.model)

    def torch_mdl_to_flwr_params(mdl):
        param_dict = {k: np.float32(v.cpu().detach().numpy()) for k, v in mdl.state_dict().items()
                      if v.shape != torch.Size([])}
        params = [v for _, v in param_dict.items() if v.shape != ()]
        return ndarrays_to_parameters(params)

    # Federated learning strategy with NNCodec compression
    strategy = NNCFedAvg(
        # fraction_fit=0.0,
        min_fit_clients=args.num_clients,  # number of clients to sample for fit()
        min_available_clients=args.num_clients,  # total clients in the simulation
        evaluate_fn=get_evaluate_fn(seeded_model_copy(nnc_mdl_executer.model), testloader),
        accept_failures=False,
        fit_metrics_aggregation_fn=accumulated_bitstream_sizes,
        initial_parameters=torch_mdl_to_flwr_params(seeded_model_copy(nnc_mdl_executer.model)),
        model_arch=seeded_model_copy(nnc_mdl_executer.model),
        mdl_info=mdl_info,
        args=args,
        encode_fn=encode,
        decode_fn=nnc.decompress
    )

    # Launch the simulation
    hist = fl.simulation.start_simulation(
        client_fn=client_fn_callback,  # A function to run a _virtual_ client when required
        num_clients=args.num_clients,  # Total number of clients available
        config=fl.server.ServerConfig(num_rounds=args.epochs),  # Specify number of FL rounds
        strategy=strategy,  # A Flower strategy
        client_resources={"num_cpus": 1, "num_gpus": 1 if not nnc_mdl_executer.device == torch.device("mps") else 0},
        ray_init_args={"local_mode": True if sys.gettrace() is not None else False} # code is running in debug mode
    )
    print(hist)

    if args.wandb:
        wandb.finish()
        print(f"zipping W&B files to {args.results}/wandb_zipped")
        shutil.make_archive(f"{args.results}/wandb_zipped", 'zip', f"{args.results}/wandb")
        print("remove W&B dir")
        shutil.rmtree(f"{args.results}/wandb")

if __name__ == '__main__':
    main()
