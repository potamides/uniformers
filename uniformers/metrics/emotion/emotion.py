from collections import defaultdict
from statistics import harmonic_mean, mean

from datasets import Features, Sequence, Value
from datasets.info import MetricInfo
import datasets.metric

from uniformers.pipelines import EmotionClassificationPipeline
from uniformers.utils.poetry import EMOTIONS

class Emotion(datasets.metric.Metric):
    """Score emotion in a quatrain"""

    def __init__(
        self, language="de", model_name="nllg/clf-bert-e", batch_size=1, **kwargs
    ):
        kwargs['config_name'] = model_name
        super().__init__(**kwargs)
        self.pipeline = EmotionClassificationPipeline(
            lang=language,
            batch_size=batch_size,
            model_name=model_name,
        )

    def _info(self):
        return MetricInfo(
            description=str(self.__doc__),
            citation="",
            features=Features(
                {
                    "quatrains": Value("string"),
                    "emotions": Sequence(Value("string")),
                }
            ),
        )

    def _preprocess(self, quatrains, emotions):
        filtered_quatrains, filtered_emotions = list(), list()
        for quatrain, emotion in zip(quatrains, emotions):
            assert set(emotion).issubset(EMOTIONS)
            if len(splitted := quatrain.split("\n")) == 4:
                filtered_quatrains.append(splitted)
                filtered_emotions.append(emotion)

        return filtered_quatrains, filtered_emotions, [[0.0]] * (len(emotions) - len(filtered_emotions))

    def _compute(
        self,
        quatrains,
        emotions,
    ):
        quatrains, emotions, all_scores = self._preprocess(quatrains, emotions)
        results = self.pipeline([verse for quatrain in quatrains for verse in quatrain])

        for i in range(0, len(results), 4):
            quatrain_results = results[i:i+4]
            emotion = emotions[i//4]
            scores = defaultdict(int)
            for verse_results in quatrain_results:
                for result in verse_results:
                    if (label := result['label']) in emotion:
                        scores[label] = max(result['score'], scores[label])
            assert len(scores) == len(emotion)
            all_scores.append(list(scores.values()))

        output_dict = {
            "emotion_score": mean(mean(scores) for scores in all_scores),
            "harmonic_emotion_score": mean(harmonic_mean(scores) for scores in all_scores)
        }
        return output_dict
