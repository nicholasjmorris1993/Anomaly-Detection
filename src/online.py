import pandas as pd
from river import stream
from river import anomaly


def streaming(df, test_frac):
    model = Streaming()
    model.train(df, test_frac)
    model.predict()

    return model


class Streaming:
    def train(self, df, test_frac):
        self.data = df.copy()
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        self.model = anomaly.QuantileFilter(
            anomaly.HalfSpaceTrees(seed=42),
            q=0.95,
        )

        for x, _ in stream.iter_pandas(train):
            self.model = self.model.learn_one(x)

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))

        self.predictions = pd.DataFrame()

        for x, _ in stream.iter_pandas(test):
            score = self.model.score_one(x)
            is_anomaly = self.model.classify(score)
            self.model = self.model.learn_one(x)

            pred = pd.DataFrame({
                "Score": [score],
                "Anomaly": [is_anomaly],
            })
            self.predictions = pd.concat([
                self.predictions, 
                pred,
            ], axis="index").reset_index(drop=True)


