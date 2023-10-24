import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def batching(df, test_frac):
    model = Batching()
    model.train(df, test_frac)
    model.predict()

    return model


class Batching:
    def train(self, df, test_frac):
        self.data = df.copy()
        self.test_frac = test_frac
        train = self.data.copy().head(int(len(self.data)*(1 - self.test_frac)))

        # train a model to detect outliers
        self.model = IsolationForest(n_jobs=-1, random_state=0)
        self.model.fit(train)

    def predict(self):
        test = self.data.copy().tail(int(len(self.data)*self.test_frac))

        # find the outliers
        score = self.model.predict(test)
        is_anomaly = score == -1

        self.predictions = pd.DataFrame({
            "Score": score,
            "Anomaly": is_anomaly,
        })
