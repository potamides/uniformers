from difflib import get_close_matches

from datasets import Features, Value
from datasets.info import MetricInfo
import datasets.metric
from datasets.utils.logging import get_verbosity, set_verbosity, set_verbosity_error

class Memorization(datasets.metric.Metric):
    """Quantify memorization in a quatrain (based on https://arxiv.org/abs/2202.07646)"""

    def __init__(self, train_data, cutoff=0.7, **kwargs):
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

    def _silent(self, func, *args, **kwargs):
        old_verbosity = get_verbosity()
        set_verbosity_error()
        res = func(*args, **kwargs)
        set_verbosity(old_verbosity)
        return res

    def _compute(self, quatrains, schemes, meters, levels):
        styles, num_copied = ["rhyme", "meter", "alliteration"], 0

        for quatrain, scheme, meter, level in zip(quatrains, schemes, meters, levels):
            spl = dict(zip(styles, [scheme, meter, level]))
            # quickfix: this would produce a ton of log messages in this loop, so silence them temporarily
            candidates = self._silent(self.train_data.filter, lambda ex: all(ex[s] == spl[s] for s in styles))
            assert len(candidates) > 0
            if get_close_matches(quatrain, ["\n".join(q) for q in candidates["text"]], n=1, cutoff=self.cutoff):
                num_copied += 1

        output_dict = {
            "extractive_memorization_rate": num_copied / len(quatrains)
        }
        return output_dict
