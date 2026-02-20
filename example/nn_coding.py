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
from ptflops import get_model_complexity_info
import torchvision
import warnings

from nncodec.nn import encode, decode
from nncodec.framework.use_case_init import use_cases
from nncodec.framework.pytorch_model import __initialize_data_functions, np_to_torch, torch_to_numpy, model_diff, model_add
from nncodec.framework.applications.utils.train import train_classification_model
from nncodec.framework.applications.utils.evaluation import evaluate_classification_model
from nncodec.framework.applications import models, datasets

parser = argparse.ArgumentParser(description='NNCodec for entire Neural Networks')
parser.add_argument('--uc', type=int, default=0, help='Use cases (default: 0) [0: non-incremental, 1: incremental, 2: incremental differences]')
parser.add_argument('--qp', type=int, default=-32, help='quantization parameter for NNs (default: -32)')
parser.add_argument('--qp_density', type=int, default=2, help='quantization scale parameter (default: 2)')
parser.add_argument('--nonweight_qp', type=int, default=-75, help='qp for non-weights, e.g., 1D or BatchNorm params (default: -75)')
parser.add_argument("--opt_qp", action="store_true", help='Modifies QP layer-wise based on relative layer size within NN')
parser.add_argument("--use_dq", action="store_true", help='Enable dependent scalar / Trellis-coded quantization')
parser.add_argument('--approx_method', type=str, default='uniform',  help='Approximation method [uniform or codebook]')
parser.add_argument('--bitdepth', type=int, default=None, help='Optional: integer-aligned bitdepth for limited precision (default: None); note: overwrites QPs.')
parser.add_argument("--lsa", action="store_true", help='Enable Local Scaling Adaptation')
parser.add_argument("--ioq", action="store_true", help='Enable inference-based QP optimization')
parser.add_argument("--bnf", action="store_true", help='Enable BatchNorm Folding')
parser.add_argument('--sparsity', type=float, default=0.0, help='Sparsity rate (default: 0.0)')
parser.add_argument('--struct_spars_factor', type=float, default=0.0, help='Factor for structured sparsification (default: 0.9)')
parser.add_argument('--row_skipping', action='store_true', help='Enable Row Skipping')
parser.add_argument('--tca', action='store_true', help='Enable Temporal Context Adaptation')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default=64)')
parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train (default: 10)')
parser.add_argument('--max_batches', type=int, default=None, help='Max num of batches to process (default: 0, i.e., all)')
parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate (default: 1e-3)')
parser.add_argument('--model', type=str, default=None, metavar=f'any of {models.__all__} or {torchvision.models.list_models(torchvision.models)}')
parser.add_argument('--model_path', type=str, default=None, metavar='./example/ResNet56_CIF100.pt')
parser.add_argument('--model_rand_int', action="store_true", help='model randomly initialized, i.e., w/o loading pre-trained weights')
parser.add_argument('--dataset', type=str, default=None, metavar=f"Any of {datasets.__all__}")
parser.add_argument('--dataset_path', type=str, default=None, metavar='../data')
parser.add_argument('--results', type=str, default='./results')
parser.add_argument('--workers', type=int, default=4, help='Number of data loading workers (default: 4)')
parser.add_argument("--wandb", action="store_true", help='Use Weights & Biases for data logging')
parser.add_argument('--wandb_key', type=str, default='', help='Authentication key for Weights & Biases API account ')
parser.add_argument('--wandb_run_name', type=str, default='NNC_WP_spars', help='Identifier for current run')
parser.add_argument("--pre_train_model", action="store_true", help='Training the full model prior to compression')
parser.add_argument("--print_comp_complexity", action="store_true", help='Print model computational complexity')
parser.add_argument("--verbose", action="store_true", help='Stdout process information.')
parser.add_argument('--cuda_device', type=int, default=0)


def main():
    warnings.filterwarnings("ignore", category=UserWarning)
    args = parser.parse_args()

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        if torch.cuda.is_available():
            if args.cuda_device is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = str(args.cuda_device)
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

    if not os.path.exists(args.results):
        os.makedirs(args.results)

    if args.wandb:
        import wandb
        if isinstance(args.wandb_key, str) and len(args.wandb_key) == 40:
            os.environ["WANDB_API_KEY"] = args.wandb_key
        else:
            assert 0, "incompatible W&B authentication key"

    if args.model in models.__all__:
        model = models.init_model(args.model, num_classes=100)
    elif args.model in torchvision.models.list_models(torchvision.models):
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    elif args.model in torchvision.models.segmentation.deeplabv3.__all__:
        model = torchvision.models.get_model(args.model, weights="DEFAULT" if not args.model_rand_int else None)
    else:
        print(f"Model not specified in /framework/applications/models and not available in torchvision model zoo" \
                  f"{torchvision.models.list_models(torchvision.models)})")
        model = None

    if model and not args.model_rand_int and args.model_path and os.path.exists(args.model_path):
        model.load_state_dict(torch.load(args.model_path, map_location=device))

    if args.wandb:
        wandb.init(
            config=args,
            project=f"{args.model}_{args.dataset}_{args.wandb_run_name}",
            name=f"UC_{args.uc}_lr_{args.lr}_qp_{args.qp}_opt_{args.opt_qp}_dq_{args.use_dq}_bnf_{args.bnf}_lsa_{args.lsa}",
            entity="edl-group",
            save_code=True,
            dir=f"{args.results}"
        )
        # wandb.log({f"orig_{n}": v for n, v in model.state_dict().items() if not "num_batches_tracked" in n})

    criterion = torch.nn.CrossEntropyLoss()
    UCS = {'cifar100': 'NNR_PYT_CIF100', 'cifar10': 'NNR_PYT_CIF10', 'imagenet200': 'NNR_PYT_IN200', 'voc': 'NNR_PYT_VOC'}
    use_case_name = UCS[args.dataset] if args.dataset and args.dataset in UCS else 'NNR_PYT'

    if args.dataset:
        test_set, test_loader, val_set, val_loader, train_loader = __initialize_data_functions(handler=use_cases[use_case_name],
                                                                                            dataset_path=args.dataset_path,
                                                                                            batch_size=args.batch_size,
                                                                                            num_workers=args.workers)
        if not args.model_rand_int:
            test_perf = evaluate_classification_model(model, criterion, test_loader, test_set, device=device,
                                                      verbose=args.verbose, max_batches=args.max_batches)
            print(f"Initial test performance: {[str(k) + ': ' + str(v) for k, v in test_perf.items()]}")

    if args.print_comp_complexity:
        for idx, (i, l) in enumerate(test_loader):
            if idx >= 1:
                break
            input_shape = tuple(i.shape[1:])
        macs, params = get_model_complexity_info(model, input_shape, as_strings=False,
                                                 print_per_layer_stat=True, verbose=args.verbose,
                                                 ignore_modules=[torch.nn.MultiheadAttention])

        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))


    ### optionally pre-train the neural network model for args.epochs
    if args.pre_train_model:
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.wandb:
            wandb.watch(model, log="all", log_graph=True)
        best_acc = 0
        for e in range(args.epochs):
            print(f"Epoch {e}")
            train_perf, model = train_classification_model(model, optimizer, criterion, train_loader, device=device,
                                                          verbose=args.verbose, return_model=True, max_batches=args.max_batches)
            print(f"Reconstructed train performance: {[str(k) + ': ' + str(v) for k, v in train_perf.items()]}")

            test_perf = evaluate_classification_model(model, criterion, test_loader, test_set, device=device, verbose=args.verbose,
                                                      max_batches=args.max_batches)
            print(f"Reconstructed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf.items()]}")
            if args.wandb:
                wandb.log(test_perf)
            test_acc = list(test_perf.values())[0]
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save(model.state_dict(), f"{args.results}/{model}_retrained.pt")

    if args.uc == 0: # enc/dec examples of entire neural networks

        if model is None:
            num_layers = 5
            print(f"Example coding dict of random numpy arrays representing for instance a {num_layers}-layer NN:")
            model = {f"parameter_{i}": np.random.randn(np.random.randint(1, 36),
                                                       np.random.randint(1, 303)).astype(np.float32) for i in range(num_layers)}
            bitstream = encode(model, vars(args))
            rec_mdl_params = decode(bitstream, vars(args))

        elif args.model:
            bitstream = encode(model, vars(args), use_case_name)
            rec_mdl_params, hls_dec = decode(bitstream, vars(args), return_hls=True)

            ### evaluation of decoded and reconstructed model
            model.load_state_dict(np_to_torch(rec_mdl_params), strict=False if args.bnf else True)

            torch.save(model.state_dict(), f"{args.results}/{args.model}_dict_dec_rec.pt")

            if args.dataset:
                test_perf = evaluate_classification_model(model, criterion, test_loader, test_set, device=device, verbose=args.verbose, max_batches=args.max_batches)
                print(f"Reconstructed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf.items()]}")

        if args.wandb:
            wandb.log(test_perf)

    elif args.uc == 1: # enc/dec example of incremental full neural networks

        bs_size_acc = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.wandb:
            wandb.watch(model, log="all", log_graph=True)
        approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} if args.tca else None

        for e in range(args.epochs):

            print(f"Epoch {e}")
            train_perf, model = train_classification_model(model, optimizer, criterion, train_loader, device=device,
                                                          verbose=args.verbose, return_model=True, max_batches=args.max_batches)
            updated_mdl = copy.deepcopy(model)
            print(f"Train performance: {[str(k) + ': ' + str(v) for k, v in train_perf.items()]}")

            test_perf_uc = evaluate_classification_model(updated_mdl, criterion, test_loader, test_set, device=device,
                                                         verbose=args.verbose, max_batches=args.max_batches)
            print(f"Uncompressed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf_uc.items()]}")

            bitstream = encode(model, vars(args), use_case_name, epoch=e, approx_param_base=approx_param_base)
            rec_mdl_params = decode(bitstream, vars(args), approx_param_base=approx_param_base)

            bs_size_acc += len(bitstream)

            ### evaluation of decoded and reconstructed model
            rec_mdl = copy.deepcopy(model)
            rec_mdl.load_state_dict(np_to_torch(rec_mdl_params), strict=False if args.bnf else True)
            test_perf = evaluate_classification_model(rec_mdl, criterion, test_loader, test_set,
                                                                       device=device,
                                                                       verbose=args.verbose,
                                                                       max_batches=args.max_batches)

            print(f"Reconstructed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf.items()]}")

            if args.wandb:
                wandb.log({"train_perf": train_perf,
                           "test_perf_uc": test_perf_uc,
                           "rec_test_perf": test_perf,
                           "bs_size": len(bitstream),
                           "accumulated_bs_size": bs_size_acc})

    elif args.uc == 2: # enc/dec example of incremental differential neural networks

        bs_size_acc = 0
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        if args.wandb:
            wandb.watch(model, log="all", log_graph=True)
        approx_param_base = {"parameters": {}, "put_node_depth": {}, "device_id": 0, "parameter_id": {}} if args.tca else None

        for e in range(args.epochs):

            print(f"Epoch {e}")
            prev_mdl = copy.deepcopy(model)
            train_perf, model = train_classification_model(model, optimizer, criterion, train_loader, device=device,
                                                          verbose=args.verbose, return_model=True, max_batches=args.max_batches)
            updated_mdl = copy.deepcopy(model)
            print(f"Train performance: {[str(k) + ': ' + str(v) for k, v in train_perf.items()]}")
            test_perf_uc = evaluate_classification_model(updated_mdl, criterion, test_loader, test_set, device=device,
                                                         verbose=args.verbose, max_batches=args.max_batches)
            print(f"Uncompressed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf_uc.items()]}")

            if e == 0:
                dNN = torch_to_numpy(updated_mdl.state_dict())
            else:
                dNN = model_diff(torch_to_numpy(updated_mdl.state_dict()), torch_to_numpy(prev_mdl.state_dict()))

            # compression
            bitstream = encode(dNN, vars(args), use_case_name, incremental=True, epoch=e,
                               approx_param_base=approx_param_base)
            rec_mdl_params = decode(bitstream, vars(args), approx_param_base=approx_param_base)

            bs_size_acc += len(bitstream)

            ### evaluation of decoded and reconstructed model
            if e == 0:
                rec_mdl = copy.deepcopy(model)
            else:
                rec_mdl_params = model_add(torch_to_numpy(rec_mdl.state_dict()), rec_mdl_params)

            rec_mdl.load_state_dict(np_to_torch(rec_mdl_params), strict=False if args.bnf else True)
            test_perf = evaluate_classification_model(rec_mdl, criterion, test_loader, test_set,
                                                                       device=device,
                                                                       verbose=args.verbose,
                                                                       max_batches=args.max_batches)

            print(f"Reconstructed test performance: {[str(k) + ': ' + str(v) for k, v in test_perf.items()]}")

            if args.wandb:
                wandb.log({"train_perf": train_perf,
                           "test_perf_uc": test_perf_uc,
                           "rec_test_perf": test_perf,
                           "bs_size": len(bitstream),
                           "accumulated_bs_size": bs_size_acc})

    if args.wandb:
        wandb.finish()
        print(f"zipping W&B files to {args.results}/wandb_zipped")
        shutil.make_archive(f"{args.results}/wandb_zipped", 'zip', f"{args.results}/wandb")
        print("remove W&B dir")
        shutil.rmtree(f"{args.results}/wandb")

if __name__ == '__main__':
    main()
