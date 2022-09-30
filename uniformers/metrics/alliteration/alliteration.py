import datasets.metric
from datasets import Features, Value
from datasets.info import MetricInfo
from statistics import mean
from uniformers.utils.phonemes import Phonemizer, alliteration_score
from uniformers.utils.poetry import ALLITERATION_LEVELS

class Alliteration(datasets.metric.Metric):
    """Score alliterations in a quatrain"""

    def __init__(
        self, language, medium, high, batch_size=1, **kwargs
    ):
        super().__init__(**kwargs)
        self.medium = medium
        self.high = high
        self.phonemizer = Phonemizer(lang=language, batch_size=batch_size)

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "quatrains": Value("string"),
                    "levels": Value("string"),
                }
            ),
        )

    def _preprocess(self, quatrains, levels):
        filtered_quatrains, filtered_levels = list(), list()
        for quatrain, level in zip(quatrains, levels):
            assert level in ALLITERATION_LEVELS
            if len(splitted := quatrain.split("\n")) == 4:
                filtered_quatrains.append(splitted)
                filtered_levels.append(level)

        return filtered_quatrains, filtered_levels, [0.0] * (len(levels) - len(filtered_levels))

    def _compute(
        self,
        quatrains,
        levels,
    ):
        assert ("high", "medium", "low") == ALLITERATION_LEVELS
        quatrains, levels, all_scores = self._preprocess(quatrains, levels)
        phonemes = self.phonemizer([verse for quatrain in quatrains for verse in quatrain])
        scores = [alliteration_score(verse) for verse in phonemes]

        for i in range(0, len(scores), 4):
            score = mean(scores[i:i+4])
            level = levels[i//4]

            if score < self.medium:
                all_scores.append(float(level == "low"))
            elif score < self.high:
                all_scores.append(float(level == "medium"))
            else:
                all_scores.append(float(level == "high"))


        output_dict = {
            "alliteration_score": mean(all_scores),
        }
        return output_dict
