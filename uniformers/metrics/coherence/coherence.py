from itertools import pairwise
from statistics import mean

from datasets import Features, Value
from datasets.info import MetricInfo
import datasets.metric
from torch import device
from torch.cuda import is_available as has_cuda
from transformers.models.bert.modeling_bert import BertForNextSentencePrediction
from transformers.models.bert.tokenization_bert import BertTokenizer
from transformers.pipelines.text_classification import TextClassificationPipeline


class Coherence(datasets.metric.Metric):
    """Score coherence in a quatrain using BERT NSP"""

    def __init__(
        self,
        model_name="bert-base-multilingual-cased",
        batch_size=1,
        device=device("cuda:0" if has_cuda() else "cpu"),
        **kwargs,
    ):
        kwargs["config_name"] = model_name
        super().__init__(**kwargs)
        self.pipeline = TextClassificationPipeline(
            model=BertForNextSentencePrediction.from_pretrained(model_name),
            tokenizer=BertTokenizer.from_pretrained(model_name),
            batch_size=batch_size,
            device=device,
            top_k=None,
            truncation=True,
            padding=True,
        )

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features({"quatrains": Value("string")}),
        )

    def _compute(
        self,
        quatrains,
    ):
        results = self.pipeline(
            [
                {"text": pair[0], "text_pair": pair[1]}
                for quatrain in quatrains
                for pair in pairwise(quatrain.split("\n"))
            ]
        )

        scores = [logit['score'] for result in results for logit in result if logit['label'] == 'LABEL_0']
        return {
            "coherence": mean(scores)
        }
