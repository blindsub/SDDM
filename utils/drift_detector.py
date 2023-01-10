import time

import numpy as np


class DriftDetectorSupervised():
    def __init__(self, clf, drift_detector, training_buffer_size):
        self.clf = clf
        self.drift_detector = drift_detector
        self.training_buffer_size = training_buffer_size
        self.X_training_buffer = []
        self.Y_training_buffer = []
        self.changes_detected = []

    def apply_to_stream(self, X_stream, Y_stream):
        self.changes_detected = []

        collect_samples = False
        T = len(X_stream)
        since = time.time()
        for t in range(T):
            x, y = X_stream[t, :], Y_stream[t, :]

            if collect_samples == False:
                self.drift_detector.add_element(self.clf.score(x, y))

                if self.drift_detector.detected_change():
                    self.changes_detected.append(t)

                    collect_samples = True
                    self.X_training_buffer = []
                    self.Y_training_buffer = []
            else:
                self.X_training_buffer.append(x)
                self.Y_training_buffer.append(y)

                if len(self.X_training_buffer) >= self.training_buffer_size:
                    collect_samples = False
                    self.clf.fit(np.array(self.X_training_buffer), np.array(self.Y_training_buffer))
        time_elapsed = time.time() - since
        return self.changes_detected, time_elapsed


class DriftDetectorUnsupervised():
    def __init__(self, drift_detector, batch_size):
        self.drift_detector = drift_detector
        self.batch_size = batch_size
        self.changes_detected = []
        self.detector_name = self.drift_detector.__class__.__name__

    def apply_to_stream(self, data_stream):
        since = time.time()
        self.changes_detected = []

        n_data_stream_samples = len(data_stream)

        t = 0
        while t < n_data_stream_samples:
            end_idx = t + self.batch_size
            if end_idx >= n_data_stream_samples:
                end_idx = n_data_stream_samples

            batch = data_stream[t:end_idx, :]
            self.drift_detector.add_batch(batch)

            if self.drift_detector.detected_change():
                self.changes_detected.append(t)
                print(self.detector_name + " detects concept drift")

            t += self.batch_size
        print(self.detector_name + " detects concept drift at " + str(self.changes_detected))
        time_elapsed = time.time() - since
        return self.changes_detected, time_elapsed
