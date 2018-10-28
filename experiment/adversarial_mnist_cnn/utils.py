from tensorpack.callbacks import TensorPrinter
from tensorpack.utils import logger

class TensorPrinterEpoch(TensorPrinter):
    """ Prints the value of some tensors in each epoch.
    """

    def __init__(self, names):
        """
        Args:
            names(list): list of string, the names of the tensors to print.
        """
        super(TensorPrinterEpoch, self).__init__(names)
        self.args = None

    def _after_run(self, _, vals):
        self.args = vals.results

    def _trigger_epoch(self):
        assert len(self.args) == len(self._names), len(self.args)
        for n, v in zip(self._names, self.args):
            logger.info("{}: {}".format(n, v))

