""""""

import os
import functools as func
from six import int2byte

import torch
from torch import nn
from terrace import learn
from terrace.evaluate import evaluate
from terrace.network import Variable, Module, HParameters
from terrace.tensor_utilities import shift_right

import data


BASE_HPARAMS = HParameters(
    training_chars=50000000,
    batch_size=64,
    hidden_size=256,
    num_hidden_layers=2,
    dropout=0.0,
    input_dropout=0.0, 
    sparse_embed_gradients=True,
    loss="nll",
    loss_ignore_index=0,
    optimizer="adagrad",
    optimizer_lr=0.01,
    sample_temperature=1.0,
    scheduled_sampling_prob=0.0,
    max_seq_length=400,
)


class LSTMLanguageModel(Module):

    def __init__(self, hparams):
        super().__init__()
        self.hparams = hparams
        self.embedding = nn.Embedding(hparams.vocab_size, hparams.hidden_size, 
                                      sparse=hparams.sparse_embed_gradients)
        self.input_dropout = nn.Dropout(p=hparams.input_dropout)
        self.lstm = nn.LSTM(hparams.hidden_size, hparams.hidden_size, 
                            hparams.num_hidden_layers, batch_first=True, 
                            dropout=hparams.dropout)
        self.output = nn.Linear(hparams.hidden_size, hparams.vocab_size)
        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, input_var, states=None, batch_size=None, autoregressive=False):
        use_cuda = next(self.parameters()).is_cuda
        sample_prob = self.hparams.scheduled_sampling_prob
        temperature = self.hparams.sample_temperature
        batch_size = self.hparams.batch_size if batch_size is None else batch_size
        no_output_feeding = ((self.training and not sample_prob) 
                             or (not self.training and not autoregressive))
        if input_var is None:
            volatile = not self.training
            input_var = Variable(torch.zeros(batch_size).long(), 
                                 use_cuda=use_cuda, volatile=volatile)
        if input_var.dim() == 1:
            input_var = input_var.unsqueeze(1)
        if no_output_feeding:
            time_steps = 1
        elif not self.training:
            time_steps = self.hparams.max_seq_length
        else:
            time_steps = input_var.shape[1]
        output = Variable(torch.zeros(batch_size, time_steps, 
            self.hparams.vocab_size if not autoregressive else 1), use_cuda=use_cuda)
        for i in range(time_steps):
            if no_output_feeding:
                step_input = input_var
            elif self.training and i > 0:
                sample = torch.bernoulli(torch.Tensor([sample_prob]))[0]
                step_input = (torch.max(step_output, 1)[1] if sample 
                              else input_var[:, i].unsqueeze(1))
            elif self.training:
                step_input = input_var[:, i].unsqueeze(1)
            else:
                step_input = (step_output.long() if i >= input_var.shape[1] 
                              else input_var[:, i].unsqueeze(1))
            embedded_input = self.input_dropout(self.embedding(step_input))
            lstm_output, states = self.lstm(embedded_input, states)
            step_output = self.logsoftmax(self.output(lstm_output))
            if not self.training and autoregressive:
                if temperature == 0.0:
                    step_output = torch.max(step_output, 1)[1].unsqueeze(-1)
                else:
                    step_output = torch.multinomial(
                        step_output.squeeze().exp()/temperature, 1).unsqueeze(-1)
            if time_steps == 1:
                output = step_output
            else:
                output[:, i, :] = step_output
            if (not self.training and autoregressive 
                    and step_output.ne(data.EOS_ID).max().data[0] == 0):
                break

        return output, states

    def init_states(self, batch_size=None, use_cuda=True):
        if batch_size is None:
            batch_size = self.hparams.batch_size
        hidden_state = Variable(torch.zeros(self.hparams.num_hidden_layers, 
                                            batch_size, 
                                            self.hparams.hidden_size),
                                use_cuda=use_cuda)
        cell_state = Variable(torch.zeros(self.hparams.num_hidden_layers, 
                                          batch_size, 
                                          self.hparams.hidden_size),
                              use_cuda=use_cuda)
        return hidden_state, cell_state


def perform_train_step(model, hparams, data_source, loss_function, 
                       optimizer, training_log, use_cuda=True):
    optimizer.zero_grad()
    ((input_data, target_data), _), exhausted = data_source.get_next_batch(
        hparams.batch_size, use_cuda=use_cuda)
    batch_size = target_data.shape[0]
    states = model.init_states(batch_size, use_cuda)
    if exhausted:
        data_source.shuffle()
    input_data = Variable(target_data.data, use_cuda=use_cuda)
    output, _ = model(shift_right(input_data, variable=True), states, batch_size)
    loss = loss_function(output.view(-1, output.size(-1)), target_data.view(-1))
    loss.backward()
    optimizer.step()
    return {"loss": loss.data[0]}


def infer_with_eval(model, hparams, batch_data, metrics, use_cuda=True):
    (input_data, target_data), _ = batch_data
    batch_size = target_data.shape[0]
    states = model.init_states(batch_size, use_cuda)
    input_data = Variable(target_data.data, use_cuda=use_cuda, volatile=True)
    output, _ = model(shift_right(input_data, variable=True), states, batch_size)
    return {metric: metric(output, target_data) for metric in metrics}


def train_with_eval(model, hparams, train_data_source, eval_data_source, 
                    logger, training_dir, steps, use_cuda=True):
    log_file_name = os.path.join(training_dir, "training.log")
    log_callback = learn.TrainingLogCallback(100, log_file_name=log_file_name)
    metrics = ["accuracy", "accuracy_top5", "sequence_accuracy", "neg_log_perplexity"]
    eval_step_fn = func.partial(infer_with_eval, use_cuda=use_cuda)
    eval_callback = learn.EvaluationCallback(metrics, 2000, 
        eval_function=func.partial(
            evaluate, eval_step_fn=eval_step_fn, use_cuda=use_cuda), 
        full_final_eval=False, generate_function=generate, log_examples=True)
    saver_callback = learn.SaverCallback(4000, max_stored=10)
    callbacks = [log_callback, eval_callback, saver_callback]
    device = "gpu" if use_cuda else "cpu"
    trainer = learn.Trainer(model, hparams, train_data_source, perform_train_step,
                            callbacks, eval_data_source=eval_data_source, 
                            logger=logger, training_dir=training_dir, device=device)
    trainer.run(steps, use_cuda=use_cuda)


class CharTextEncoder:

    def encode(self, string):
        reserved = len(data.RESERVED_TOKENS)
        return [byte + reserved for byte in string.encode("utf-8", "replace")]

    def decode(self, ids):
        reserved = len(data.RESERVED_TOKENS)
        decoded_ids = []
        for id_ in ids:
            if 0 <= id_ < reserved:
                decoded_ids.append(bytes(data.RESERVED_TOKENS[id_], "ascii"))
            else:
                decoded_ids.append(int2byte(id_ - reserved))
        return b"".join(decoded_ids).decode("utf-8", "replace")


def run_experiment(root_dir, hparams, training_steps, use_cuda=True):
    # Get data
    data_dir = os.path.join(root_dir, "data")
    if not os.path.exists(data_dir):
        os.mkdir(data_dir)
    data.download_corpus(data_dir)
    encoder = CharTextEncoder()
    train_dataset = data.get_1billion_dataset(
        data_dir, "training", hparams.training_chars, vocab_encoder=encoder.encode)
    eval_dataset = data.get_1billion_dataset(
        data_dir, "validation", vocab_encoder=encoder.encode)

    training_dir = os.path.join(
        root_dir, "model-" + "-".join(
            [str(hparams.batch_size), str(hparams.hidden_size), 
             str(hparams.num_hidden_layers), str(hparams.dropout)]))
    if not os.path.exists(training_dir):
        os.mkdir(training_dir)
    logger = learn.setup_logging(training_dir)
    hparams.update({"vocab_size": 2**8 + len(data.RESERVED_TOKENS)})
    
    train_data_source = data.convert_to_datasource(
        train_dataset, hparams.vocab_size)
    eval_data_source = data.convert_to_datasource(
        eval_dataset, hparams.vocab_size)

    # Create model and run training
    model = LSTMLanguageModel(hparams)
    train_with_eval(model, hparams, train_data_source, eval_data_source, 
                    logger, training_dir, training_steps, use_cuda)
    return model


def generate(model, hparams, data_source, input_string=None, num_examples=1, 
             file_path=None, use_cuda=True):
    model.eval()
    encoder = CharTextEncoder()
    if not input_string:
        input_data = None
    else:
        input_data = Variable(
            torch.LongTensor(encoder.encode(input_string)), use_cuda=use_cuda)
        input_data = input_data.unsqueeze(0)

    examples = []
    for i in range(num_examples):
        states = model.init_states(1, use_cuda)
        output, _ = model(input_data, states, 1, autoregressive=True)
        output_string = encoder.decode(
            [id_ for id_ in list(output.data.long().squeeze()) if id_])
        examples.append((None, output_string))

    return examples

