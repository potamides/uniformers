from transformers.training_args import TrainingArguments
from transformers.utils import logging

logger = logging.get_logger("transformers")

class GlobalBatchTrainingArguments(TrainingArguments):
    """
    TrainingArguments class which evenly distributes batch_size on available
    GPUs under distributed training (DistributedDataParallel). Normal
    TrainingArguments use same batch_size on each GPU. (see
    https://discuss.pytorch.org/t/should-we-split-batch-size-according-to-ngpu-per-node-when-distributeddataparallel/72769/15)
    This should also work for DataParallel which does splitting on its own (see
    https://discuss.pytorch.org/t/a-question-concerning-batchsize-and-multiple-gpus-in-pytorch/33767).
    Additionally, batch_size is scaled according to gradient accumulation
    steps.
    """

    def __init__(
        self,
        global_train_batch_size=8,
        global_eval_batch_size=8,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.global_train_batch_size = global_train_batch_size
        self.global_eval_batch_size = global_eval_batch_size
        self.per_device_train_batch_size = self._scale_batch_size(
            global_train_batch_size
        )
        self.per_device_eval_batch_size = self._scale_batch_size(global_eval_batch_size)
        if self.world_size > 1:
            logger.info(f"Dividing batches equally on {self.world_size} processes.")

    def _scale_batch_size(self, batch_size) -> int:
        scaled_batch_size, remainder = divmod(
            batch_size,
            self.world_size * self.gradient_accumulation_steps,
        )
        if remainder != 0:
            raise ValueError(
                "`batch_size` must be divisible by number of processes times gradient accumulation steps."
            )
        return scaled_batch_size
