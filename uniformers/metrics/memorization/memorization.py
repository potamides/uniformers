from difflib import get_close_matches

from datasets import Features, Value
from datasets.info import MetricInfo
import datasets.metric

class Memorization(datasets.metric.Metric):
    """Quantify memorization in a quatrain (based on https://arxiv.org/abs/2202.07646)"""

    def __init__(self, train_data, cutoff=0.8, **kwargs):
        super().__init__(**kwargs)

        self.train_data = train_data
        self.cutoff = cutoff

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "quatrains": Value("string"),
                    "schemes": Value("string"),
                    "meters": Value("string"),
                    "levels": Value("string"),
                }
            ),
        )

    def _compute(self, quatrains, schemes, meters, levels):
        styles, num_copied = ["rhyme", "meter", "alliteration"], 0

        for quatrain, scheme, meter, level in zip(quatrains, schemes, meters, levels):
            spl = dict(zip(styles, [scheme, meter, level]))
            candidates = self.train_data.filter(lambda ex: all(ex[s] == spl[s] for s in styles))
            assert len(candidates) > 0
            if get_close_matches(quatrain, ["\n".join(q) for q in candidates["text"]], n=1, cutoff=self.cutoff):
                num_copied += 1

        output_dict = {
            "extractive_memorization_rate": num_copied / len(quatrains)
        }
        return output_dict
