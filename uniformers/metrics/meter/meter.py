import datasets.metric
from datasets import Features, Value
from datasets.info import MetricInfo
from uniformers.pipelines import MeterClassificationPipeline
from statistics import mean, harmonic_mean
from uniformers.utils.poetry import METERS

class Meter(datasets.metric.Metric):
    """Score meter in a quatrain"""

    def __init__(
        self, language, model_name="nllg/clf-canine-m", batch_size=1, **kwargs
    ):
        kwargs['config_name'] = model_name
        super().__init__(**kwargs)
        self.pipeline = MeterClassificationPipeline(
            lang=language,
            batch_size=batch_size,
            model_name=model_name,
            top_k=None,
        )

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "quatrains": Value("string"),
                    "meters": Value("string"),
                }
            ),
        )

    def _preprocess(self, quatrains, meters):
        filtered_quatrains, filtered_meters = list(), list()
        for quatrain, meter in zip(quatrains, meters):
            assert meter in METERS
            if len(splitted := quatrain.split("\n")) == 4:
                filtered_quatrains.append(splitted)
                filtered_meters.append(meter)

        return filtered_quatrains, filtered_meters, [[0.0]] * (len(meters) - len(filtered_meters))

    def _compute(
        self,
        quatrains,
        meters,
    ):

        quatrains, meters, all_scores = self._preprocess(quatrains, meters)
        results = self.pipeline([verse for quatrain in quatrains for verse in quatrain])

        for i in range(0, len(results), 4):
            quatrain_results = results[i:i+4]
            meter = meters[i//4]
            scores = list()
            for verse_results in quatrain_results:
                for result in verse_results:
                    if result['label'] == meter:
                        scores.append(result['score'])
                        break
            assert len(scores) == 4
            all_scores.append(scores)

        output_dict = {
            "meter_score": mean(mean(scores) for scores in all_scores),
            "harmonic_meter_score": mean(harmonic_mean(scores) for scores in all_scores)
        }
        return output_dict
