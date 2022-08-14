import datasets.metric
from datasets import Features, Value
from datasets.info import MetricInfo
from uniformers.pipelines import RhymeClassificationPipeline
from itertools import combinations
from statistics import mean, harmonic_mean
from uniformers.utils.poetry import QUATRAIN_RHYME_SCHEMES

class Rhyme(datasets.metric.Metric):
    """Score rhymescheme in a quatrain"""

    def __init__(
        self, language, model_name="nllg/clf-canine-r", batch_size=1, **kwargs
    ):
        kwargs['config_name'] = model_name
        super().__init__(**kwargs)
        self.pipeline = RhymeClassificationPipeline(
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
                    "schemes": Value("string"),
                }
            ),
        )

    def _preprocess(self, quatrains, schemes):
        filtered_quatrains, filtered_schemes = list(), list()
        for quatrain, scheme in zip(quatrains, schemes):
            assert scheme in QUATRAIN_RHYME_SCHEMES
            if len(splitted := quatrain.split("\n")) == 4:
                filtered_quatrains.append(splitted)
                filtered_schemes.append(scheme)

        return filtered_quatrains, filtered_schemes, [[0.0]] * (len(schemes) - len(filtered_schemes))

    def _compute(
        self,
        quatrains,
        schemes,
    ):
        quatrains, schemes, all_scores = self._preprocess(quatrains, schemes)
        results = self.pipeline([comb for text in quatrains for comb in combinations(text, r=2)])

        for i in range(0, len(results), 6):
            quatrain_results = results[i:i+6]
            scheme = schemes[i//6]
            scores = list()
            for j, (a, b) in enumerate(combinations(scheme, r=2)):
                for result in quatrain_results[j]:
                    # corresponds to this metric: https://stats.stackexchange.com/a/517971
                    if result['label'] == ("rhyme" if a == b else "dissonance"):
                        scores.append(result['score'])
                        break
            assert len(scores) == 6
            all_scores.append(scores)

        output_dict = {
            "rhyme_score": mean(mean(scores) for scores in all_scores),
            "harmonic_rhyme_score": mean(harmonic_mean(scores) for scores in all_scores)
        }
        return output_dict
