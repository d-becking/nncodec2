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
# import tensorflow as tf
from sklearn.metrics import classification_report
from nncodec.framework.applications.utils.metrics import get_topk_accuracy_per_batch
# from nncodec.framework.applications.models.tokenizer import Tokenizer
from nncodec.nnc_core import nnr_model
from contextlib import nullcontext
import re

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def evaluate_classification_model(model, criterion=None, testloader=None, testset=None,  min_sample_size=1000, max_batches=None,
                                  early_stopping_threshold=None, device=DEVICE, print_classification_report=False,
                                  return_predictions=False, verbose=False, rec_mdl=False):
    """
    Helper function to evaluate model on test dataset.

    Parameters
    ----------
    model: torch.nn.Module
        Neural network model.
    criterion: torch.nn.Criterion
        Criterion for loss calculation.
    testloader: torch.utils.data.DataLoader
        DataLoader that loaded testset.
    testset: torch.utils.data.dataset.Dataset
        Test dataset
    min_sample_size: int
        Minimum sample size used for early_stopping calculation. Default: 1000
    max_batches: int
        Maximum batches evaluated, by default evaluates the complete testset. Default: None
    early_stopping_threshold: int
        A value between 0-100 corresponding to the accuracy. If it drops under a given threshold
        the evaluation is stopped.
    device: str
        Device on which the model is evaluated: cpu or cuda.
    print_classification_report: bool
        If True print the complete confusion matrix for all the classes.
    return_predictions: bool
        If True return all the predictions for all samples, otherwise return the accuracy.
    verbose: bool
        If True print the progress bar of the evaluation.

    Return
    ------
    output: float | nd.array
        Accuracy or all predictions, depending on the given return_predictions parameter.
    """
    if isinstance(model, nnr_model.ModelExecute):
        criterion = model.handle.criterion
        testloader = model.test_loader
        testset = model.test_set
        device = model.device
        if rec_mdl:
            model = model.rec_model
        else:
            model = model.model

    if criterion is None:
        criterion = torch.nn.CrossEntropyLoss()

    model = model.to(device)
    model.eval()
    test_loss = []
    all_predictions = []
    all_labels = []
    correct = 0
    total = 0
    top5_acc = 0

    # set (verbose) iterator
    total_iterations = max_batches or len(testloader)
    # iterator = tqdm(enumerate(testloader), total=total_iterations, position=0, leave=True) if verbose else enumerate(testloader)
    iterator = enumerate(testloader)

    DeepLab_condition = model.__class__.__name__ == "DeepLabV3" if not (isinstance(model, torch.nn.DataParallel) or isinstance(model, torch.nn.parallel.DistributedDataParallel)) \
                            else model.module.__class__.__name__ == "DeepLabV3"

    if DeepLab_condition:
        from torchmetrics import JaccardIndex
        num_of_classes = model.classifier[len(model.classifier) - 1].weight.shape[0]
        jaccard = JaccardIndex(task="multiclass", num_classes=num_of_classes, average="macro").to(device)

    with torch.no_grad():
        for batch_idx, (inputs, targets) in iterator:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            if DeepLab_condition:
                outputs = outputs['out']
                targets = targets * (targets != 1) * 255
                targets = targets.squeeze(1).long()

            loss = criterion(outputs, targets)

            if outputs.size(1) > 5 and not DeepLab_condition:
                c1, c5 = get_topk_accuracy_per_batch(outputs, targets, topk=(1, 5))
                top5_acc += c5 * targets.size(0)

            test_loss.append(loss.item())
            _, predicted = outputs.max(1)
            total += targets.size(0) if not DeepLab_condition else targets.numel()
            correct += predicted.eq(targets).sum().item()
            all_predictions.append(np.array(predicted.cpu()))
            all_labels.append(np.array(targets.cpu()))

            acc = 100. * correct / total

            if DeepLab_condition:
                jaccard.update(predicted, targets)
                if verbose:
                    print('Running Test/Val mIOU (batch {}/{}) over all {} classes: {}'.format(batch_idx,
                                                                                               total_iterations,
                                                                                               num_of_classes,
                                                                                               jaccard.compute() * 100))

            if batch_idx == max_batches:
                break
            elif len(all_predictions) > min_sample_size and early_stopping_threshold is not None and \
                    acc < early_stopping_threshold:
                break

        acc = 100. * correct / total
        if top5_acc != 0:
            top5_acc = top5_acc / total

        if print_classification_report:
            print(classification_report(np.concatenate(all_labels), np.concatenate(all_predictions),
                                        target_names=list(testset.mapping.keys()),
                                        labels=list(testset.mapping.values())))

        if return_predictions:
            return np.concatenate(all_predictions)
        else:
            mean_test_loss = np.mean(test_loss)
            if DeepLab_condition:
                m_IoU = jaccard.compute() * 100
                return {'acc': acc, 'm_IoU': m_IoU, 'mean_test_loss': mean_test_loss}
            else:
                return {'acc': acc, 'top5_acc': float(top5_acc), 'mean_test_loss': mean_test_loss}

def evaluate_classification_model_TEF(model, test_loader, test_set, num_workers=8, verbose=0):

    _ , val_labels = zip(*test_set.imgs)

    y_pred = model.predict(test_loader, verbose=verbose, callbacks=None, max_queue_size=10, workers=num_workers,
                           use_multiprocessing=True)

    top5 = tf.keras.metrics.sparse_top_k_categorical_accuracy(val_labels, y_pred, k=5)
    top1 = tf.keras.metrics.sparse_categorical_accuracy(val_labels, y_pred)
    loss = tf.keras.metrics.sparse_categorical_crossentropy(val_labels, y_pred)

    acc = []
    acc.append((tf.keras.backend.sum(top1) / len(top1)).numpy() * 100)
    acc.append((tf.keras.backend.sum(top5) / len(top5)).numpy() * 100)
    acc.append((tf.keras.backend.mean(loss)).numpy())

    return acc

@torch.no_grad()
def evaluate_language_model(model, testloader, device='mps', max_batches=3, verbose=False, criterion=None, detokenize=False, args=None):

    if detokenize:
        import textwrap
        import json

        def is_number(s):
            try:
                float(s)  # float() can handle both integers and floats
                return True
            except ValueError:
                return False

        enc = Tokenizer(tokenizer_model=f"{args.tokenizer_path}")
        vocab_size = enc.sp_model.get_piece_size()
        vocabulary = [enc.sp_model.id_to_piece(i) for i in range(vocab_size)]

        global_predictions, global_gt, global_mse, global_rel_diff = {}, {}, {}, {}

    model = model.to(device)

    ptdtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}["float16"]

    ctx = (
        nullcontext()
        if device == "cpu"
        else torch.amp.autocast(device_type=device.type, dtype=ptdtype)
    )

    if max_batches == None:
        try:
            max_batches = len(testloader)
        except:
            max_batches = testloader.dataset.num_samples

    model.eval()
    test_loss = []
    test_acc = []

    for k, batch in enumerate(testloader):

        if k == max_batches:
            break

        if detokenize:
            X, Y = batch, None
        else:
            X, Y = batch

        X = X.to(device, non_blocking=True)
        if Y is not None:
            Y = Y.to(device, non_blocking=True)
        with ctx:
            logits = model(X,Y)
            loss = model.last_loss

        if not detokenize:
            test_loss.append(loss.item())
            predictions = torch.argmax(logits, dim=-1)  # Shape: (batch_size, seq_length)
            correct = (predictions == Y).float()
            test_acc.append(correct.mean().item() * 100)

        if detokenize:
            output_sample = None
            text_width = 150

            exclude_strings = {'=', '-->', ''}

            def merge_consecutive_digits(token_list):  ## merging consecutiive digits to one number
                merged_digits = []
                i = 0
                while i < len(token_list):
                    if token_list[i] in {'', '='}:
                        merged_digits.append(token_list[i])
                        i += 1
                    elif re.match(r'^-?\d*\.?\d*$', token_list[i]) and token_list[i] != '':
                        num = token_list[i]
                        i += 1
                        while i < len(token_list) and re.match(r'^\d*\.?\d*$', token_list[i]):# and token_list[i] != '':
                            num += token_list[i]
                            i += 1
                        merged_digits.append(num)
                    else:
                        merged_digits.append(token_list[i])
                        i += 1
                return merged_digits

            inp_token = X[0,:]

            wrapped_text = [enc.decode(inp_token[i].int().tolist()) for i in range(X.shape[1])]

            arrow_indices = [i for i, s in enumerate(wrapped_text) if s == "-->"]
            if arrow_indices:
                input_sample = wrapped_text[:arrow_indices[-1] + 1]
                input_sample = merge_consecutive_digits(input_sample)
                print(f"Input:\n{textwrap.fill(' '.join(input_sample), width=text_width)} \n")

                output_sample = wrapped_text[arrow_indices[-1]:]
                model_input = inp_token[:arrow_indices[-1] + 1].unsqueeze(0)
                concat_results = []
                current_line_length = 0
                last_char = ""
                print(f"Predicting [...]\n")

                for i in range(max(args.max_seq_len, len(output_sample) + 1)):
                    if model_input.shape[1] > args.max_seq_len:
                        break
                    logits = model(model_input)

                    arg_max = torch.argmax(logits.squeeze(0), dim=1).item()
                    char = vocabulary[arg_max]

                    if char == 'time':
                        break
                    elif char == "<unk>":
                        break
                    elif char != '▁':
                        concat_results.append(char)
                        print(f'{char if ((is_number(last_char) or last_char in [".", "-"]) and (is_number(char) or char == ".")) else f" {char}"}', end='', flush=True)
                        current_line_length += len(char)
                        last_char = char
                    else:
                        if last_char and not (is_number(last_char) or last_char in [".", "-"]):
                            print(" ", end='', flush=True)
                        current_line_length += 1

                    model_input = torch.cat((model_input, torch.tensor([[arg_max]], device=device)), dim=1)

                    if current_line_length >= text_width:
                        print()
                        current_line_length = 0

            if output_sample:
                print("\n")
                print(f"Ground Truth:\n")
                output_sample = merge_consecutive_digits(output_sample)
                concat_results = merge_consecutive_digits(concat_results)
                print(f"{textwrap.fill(' '.join(output_sample), width=text_width)} \n")

                def extract_values(string_list):
                    values, i = {}, 0
                    while i < len(string_list) - 1:
                        key = string_list[i].strip()
                        if key not in exclude_strings:
                            j = i + 1
                            while j < len(string_list) and string_list[j].strip() in exclude_strings:
                                j += 1
                            if j < len(string_list):
                                try:
                                    value = float(string_list[j].strip())
                                    values[key] = value
                                except ValueError:
                                    pass
                        i += 1
                    return values

                predicted_values = extract_values(concat_results)
                ground_truth_values = extract_values(output_sample)

                rel_differences, diff = {}, {}
                for key in predicted_values:
                    if key in ground_truth_values:
                        diff[key] = predicted_values[key] - ground_truth_values[key]
                        if key in global_predictions:
                            global_predictions[key] += [predicted_values[key]]
                        else:
                            global_predictions[key] = [predicted_values[key]]
                        if key in global_gt:
                            global_gt[key] += [ground_truth_values[key]]
                        else:
                            global_gt[key] = [ground_truth_values[key]]

                for key in diff:
                    rel_differences[key] = (diff[key] / (ground_truth_values[key] + 1e-10) ) * 100
                    if key in global_rel_diff:
                        global_rel_diff[key] += [rel_differences[key]]
                    else:
                        global_rel_diff[key] = [rel_differences[key]]

                print("---------------------------------------------------------------------------------------\n")
                print(f"Relative differences wrt. ground truth: ")
                [print(f"{rel_diff}: {rel_differences[rel_diff]:.1f}%") for rel_diff in rel_differences]
                print("---------------------------------------------------------------------------------------\n")
                print(f"Running absolute mean relative differences wrt. ground truth (global):")
                [print(f"{grel_diff}: {np.mean(np.abs(np.array(global_rel_diff[grel_diff]))):.1f}%") for grel_diff in global_rel_diff]
                print("---------------------------------------------------------------------------------------\n")


                with open(f'{args.results}/test_predictions.json', 'w') as f:
                    json.dump(global_predictions, f)
                with open(f'{args.results}/test_ground_truth.json', 'w') as f:
                    json.dump(global_gt, f)
                with open(f'{args.results}/test_rel_differences.json', 'w') as f:
                    json.dump(global_rel_diff, f)

    loss = torch.mean(torch.tensor(test_loss)).item()
    acc = torch.mean(torch.tensor(test_acc)).item()
    ppl = torch.exp(torch.tensor(loss)).item()
    model.train()
    print(f'top1-acc: {acc:.3f}%, ppl: {ppl:.3f}, loss: {loss:.3f}')
    return {'top1_acc': acc, 'ppl': ppl, 'loss': loss}
