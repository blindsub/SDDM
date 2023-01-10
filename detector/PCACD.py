from menelaus.data_drift import PCACD
import numpy as np
import pandas as pd


class PcaCD:
    def __init__(self, window_size, ev_threshold=0.99, delta=0.1, divergence_metric="kl", sample_period=0.05):
        self.window_size = window_size
        self.ev_threshold = ev_threshold
        self.delta = delta
        self.divergence_metric = divergence_metric
        self.sample_period = sample_period
        self.current_window = []
        self.detector = PCACD(window_size=self.window_size, ev_threshold=self.ev_threshold, delta=self.delta,
                              divergence_metric=divergence_metric, sample_period=self.sample_period)
        self.drift_detected = False

    def add_batch(self, x):
        self.drift_detected = False
        # self.current_window.append(x)

        # if len(self.current_window) * self.current_window[0].shape[0] > self.window_size:
        #     detected_window = np.concatenate([x for x in self.current_window], axis=0)
        #     self.current_window = []
        #
        #     x = pd.DataFrame(detected_window)
        # print(pd.DataFrame(x))

        self.detector.update(pd.DataFrame(x))
        if self.detector.drift_state == 'drift':
            self.drift_detected = True
            # print("drift detected")

    def detected_change(self):
        return self.drift_detected
