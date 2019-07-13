import torch
from ignite.engine.engine import Engine
from ignite.utils import convert_tensor


def wrap(input, cuda):
    if torch.is_tensor(input):
        input = Variable(input)
        if cuda:
            input = input.cuda()
    return input

# Based on ignite.engine.__init__ code
# (cuda argument is added)
# (output_transform: get value of actual loss)
def create_supervised_trainer(model, optimizer, loss_fn, cuda=True,
                              device=None, non_blocking=False,
                              output_transform=lambda x, y, y_pred, loss: loss.item()):
    if device:
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        data = batch
        batch_inputs = data[: -1]
        batch_target = data[-1]

        batch_inputs = list(
            map(wrap, batch_inputs, [cuda for _ in range(len(batch_inputs))]))
        batch_target = Variable(batch_target)

        if cuda:
            batch_target = batch_target.cuda()

        batch_output = model(*batch_inputs)
        loss = loss_fn(batch_output, batch_target)
        loss.backward()  # same as the one in closure() defined in trainer

        optimizer.step()
        return output_transform(batch_inputs, batch_target, batch_output, loss)

    return Engine(_update)


def create_supervised_evaluator(model, metrics={}, cuda=True,
                                device=None, non_blocking=False,
                                output_transform=lambda x, y, y_pred: (y_pred, y,)):

    if device:
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            batch_inputs = batch[: -1]
            batch_target = batch[-1]

            batch_inputs = list(
                map(wrap, batch_inputs, [cuda for _ in range(len(batch_inputs))]))
            
            if cuda:
                batch_target = batch_target.cuda()
            batch_output = model(*batch_inputs)

            return output_transform(batch_inputs, batch_target, batch_output)

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine













import torch
from torch.autograd import Variable

import heapq


# Based on torch.utils.trainer.Trainer code.
# Allows multiple inputs to the model, not all need to be Tensors.
class Trainer(object):

    def __init__(self, model, criterion, optimizer, dataset, cuda=False):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.dataset = dataset
        self.cuda = cuda
        self.iterations = 0
        self.epochs = 0
        self.stats = {}
        self.plugin_queues = {
            'iteration': [],
            'epoch': [],
            'batch': [],
            'update': [],
        }

    def register_plugin(self, plugin):
        plugin.register(self)

        intervals = plugin.trigger_interval
        if not isinstance(intervals, list):
            intervals = [intervals]
        for (duration, unit) in intervals:
            queue = self.plugin_queues[unit]
            queue.append((duration, len(queue), plugin))

    def call_plugins(self, queue_name, time, *args):
        args = (time,) + args
        queue = self.plugin_queues[queue_name]
        if len(queue) == 0:
            return
        while queue[0][0] <= time:
            plugin = queue[0][2]
            getattr(plugin, queue_name)(*args)
            for trigger in plugin.trigger_interval:
                if trigger[1] == queue_name:
                    interval = trigger[0]
            new_item = (time + interval, queue[0][1], plugin)
            heapq.heappushpop(queue, new_item)

    def run(self, epochs=1):
        for q in self.plugin_queues.values():
            heapq.heapify(q)

        for self.epochs in range(self.epochs + 1, self.epochs + epochs + 1):
            self.train()
            self.call_plugins('epoch', self.epochs)

    def train(self):
        for (self.iterations, data) in \
                enumerate(self.dataset, self.iterations + 1):
            batch_inputs = data[: -1]
            batch_target = data[-1]
            self.call_plugins(
                'batch', self.iterations, batch_inputs, batch_target
            )

            def wrap(input):
                if torch.is_tensor(input):
                    input = Variable(input)
                    if self.cuda:
                        input = input.cuda()
                return input
            batch_inputs = list(map(wrap, batch_inputs))

            batch_target = Variable(batch_target)
            if self.cuda:
                batch_target = batch_target.cuda()

            plugin_data = [None, None]

            def closure():
                batch_output = self.model(*batch_inputs)

                loss = self.criterion(batch_output, batch_target)
                loss.backward()

                if plugin_data[0] is None:
                    plugin_data[0] = batch_output.data
                    plugin_data[1] = loss.data

                return loss

            self.optimizer.zero_grad()
            self.optimizer.step(closure)
            self.call_plugins(
                'iteration', self.iterations, batch_inputs, batch_target,
                *plugin_data
            )
            self.call_plugins('update', self.iterations, self.model)
