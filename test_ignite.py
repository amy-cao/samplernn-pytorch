from model import SampleRNN
from train import *
from trainer import create_supervised_trainer, create_supervised_evaluator

from ignite.engine import Events
from ignite.metrics.loss import Loss
# TODO: accuracy can be used if Loss is working 


def main(exp, frame_sizes, dataset, **params):
    print('main called')

    params = dict(
        default_params,
        # exp='TEST', frame_sizes=[16,4], dataset='piano',
        exp=exp, frame_sizes=frame_sizes, dataset=dataset,
        **params
    )

    results_path = setup_results_dir(params)
    tee_stdout(os.path.join(results_path, 'log'))

    model = SampleRNN(
        frame_sizes=params['frame_sizes'],
        n_rnn=params['n_rnn'],
        dim=params['dim'],
        learn_h0=params['learn_h0'],
        q_levels=params['q_levels'],
        weight_norm=params['weight_norm']
    )
    predictor = Predictor(model)
    if params['cuda']:
        model = model.cuda()
        predictor = predictor.cuda()

    optimizer = torch.optim.Adam(predictor.parameters())
    # optimizer = gradient_clipping(torch.optim.Adam(predictor.parameters()))
    loss = sequence_nll_loss_bits

    test_split = 1 - params['test_frac']
    val_split = test_split - params['val_frac']
    data_loader = make_data_loader(model.lookback, params)

    train_loader = data_loader(0, val_split, eval=False)
    val_loader = data_loader(val_split, test_split, eval=True)
    test_loader = data_loader(test_split, 1, eval=True)

    trainer = create_supervised_trainer(
        predictor, optimizer, loss, params['cuda'])
    evaluator = create_supervised_evaluator(predictor,
                                            metrics={
                                                'nll': Loss(loss)
                                            })

    # @trainer.on(Events.ITERATION_COMPLETED)
    # def log_training_loss(trainer):
    #     print("Epoch[{}] Loss: {:.2f}".format(
    #         trainer.state.epoch, trainer.state.output))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer):
        evaluator.run(train_loader)
        metrics = evaluator.state.metrics
        print("Training Results - Epoch: {}  Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['nll']))

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer):
        evaluator.run(val_loader)
        metrics = evaluator.state.metrics
        print("Validation Results - Epoch: {}  Avg loss: {:.2f}"
              .format(trainer.state.epoch, metrics['nll']))



    trainer.run(train_loader, max_epochs=2)
    print('train complete!')






parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    argument_default=argparse.SUPPRESS
    )

def parse_bool(arg):
    arg = arg.lower()
    if 'true'.startswith(arg):
        return True
    elif 'false'.startswith(arg):
        return False
    else:
        raise ValueError()

parser.add_argument('--exp', required=True, help='experiment name')
parser.add_argument(
    '--frame_sizes', nargs='+', type=int, required=True,
    help='frame sizes in terms of the number of lower tier frames, \
            starting from the lowest RNN tier'
)
parser.add_argument(
    '--dataset', required=True,
    help='dataset name - name of a directory in the datasets path \
            (settable by --datasets_path)'
)
parser.add_argument(
    '--n_rnn', type=int, help='number of RNN layers in each tier'
)
parser.add_argument(
    '--dim', type=int, help='number of neurons in every RNN and MLP layer'
)
parser.add_argument(
    '--learn_h0', type=parse_bool,
    help='whether to learn the initial states of RNNs'
)
parser.add_argument(
    '--q_levels', type=int,
    help='number of bins in quantization of audio samples'
)
parser.add_argument(
    '--seq_len', type=int,
    help='how many samples to include in each truncated BPTT pass'
)
parser.add_argument(
    '--weight_norm', type=parse_bool,
    help='whether to use weight normalization'
)
parser.add_argument('--batch_size', type=int, help='batch size')
parser.add_argument(
    '--val_frac', type=float,
    help='fraction of data to go into the validation set'
)
parser.add_argument(
    '--test_frac', type=float,
    help='fraction of data to go into the test set'
)
parser.add_argument(
    '--keep_old_checkpoints', type=parse_bool,
    help='whether to keep checkpoints from past epochs'
)
parser.add_argument(
    '--datasets_path', help='path to the directory containing datasets'
)
parser.add_argument(
    '--results_path', help='path to the directory to save the results to'
)
parser.add_argument('--epoch_limit', help='how many epochs to run')
parser.add_argument(
    '--resume', type=parse_bool, default=True,
    help='whether to resume training from the last checkpoint'
)
parser.add_argument(
    '--sample_rate', type=int,
    help='sample rate of the training data and generated sound'
)
parser.add_argument(
    '--n_samples', type=int,
    help='number of samples to generate in each epoch'
)
parser.add_argument(
    '--sample_length', type=int,
    help='length of each generated sample (in samples)'
)
parser.add_argument(
    '--cuda', type=parse_bool,
    help='whether to use CUDA'
)

parser.set_defaults(**default_params)

main(**vars(parser.parse_args()))


