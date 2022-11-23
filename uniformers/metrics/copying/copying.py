from datasets import Features, Sequence, Value
from datasets.info import MetricInfo
import datasets.metric
from numpy import asarray
from sentence_transformers import SentenceTransformer

from uniformers.vendor.copying import C_T

class DataCopying(datasets.metric.Metric):
    """Score emotion in a quatrain"""

    def __init__(self, train_data, test_data=None, sentence_embedder="distiluse-base-multilingual-cased-v1", **kwargs):
        kwargs['config_name'] = sentence_embedder
        super().__init__(**kwargs)

        styles = ["rhyme", "meter", "alliteration"]
        train_data = train_data.map(lambda ex: ex | {"style": "-".join(ex[s] for s in styles)}, remove_columns=styles)

        # If we do not have held out data, sample from training data. Not ideal
        # but with a small sample this should only have minimal impact.
        if not test_data:
            train_data, test_data = train_data.train_test_split(test_size=0.05, stratify_by_column="style").values()
        else:
            test_data = test_data.map(lambda ex: ex | {"style": "-".join(ex[s] for s in styles)}, remove_columns=styles)

        self.train_data = train_data
        self.test_data = test_data
        self.model = SentenceTransformer(sentence_embedder)

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "quatrains": Value("string"),
                    "schemes": Sequence(Value("string")),
                    "meters": Sequence(Value("string")),
                    "levels": Sequence(Value("string")),
                }
            ),
        )

    def _compute(self, quatrains, schemes, meters, levels):
        styles = ["-".join(styles) for styles in zip(schemes, meters, levels)]
        sorted_styles = sorted(unique_styles := set(styles))
        # retain only the styles we generated
        filtered_train = self.train_data.filter(lambda ex: ex["style"] in unique_styles)
        filtered_test = self.test_data.filter(lambda ex: ex["style"] in unique_styles)

        T = self.model.encode(["\n".join(quatrain) for quatrain in filtered_train["text"]])
        T_cells = asarray([sorted_styles.index(style) for style in filtered_train['style']])
        Pn = self.model.encode(["\n".join(quatrain) for quatrain in filtered_test["text"]])
        Pn_cells = asarray([sorted_styles.index(style) for style in filtered_test['style']])
        Qm = self.model.encode(quatrains)
        Qm_cells = asarray([sorted_styles.index(style) for style in styles])

        output_dict = {
            "data_copying_score": C_T(Pn, Pn_cells, Qm, Qm_cells, T, T_cells),
        }
        return output_dict
