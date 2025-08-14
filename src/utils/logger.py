# src/utils/logger.py

import csv
import os


class CSVLogger:
    def __init__(self, filepath, header=["epoch", "loss", "test_accuracy"]):
        self.filepath = filepath
        self.header = header

        # 파일이 없다면 헤더 작성
        if not os.path.exists(self.filepath):
            with open(self.filepath, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(self.header)

    def log_epoch(self, epoch, loss):
        with open(self.filepath, mode="a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, round(loss, 4), ""])

    def log_test_accuracy(self, accuracy):
        # 마지막 줄 test_accuracy만 갱신
        with open(self.filepath, mode="r") as f:
            lines = list(csv.reader(f))

        # 마지막 epoch 줄 찾아 수정
        for i in range(len(lines) - 1, 0, -1):
            if lines[i][0].isdigit():
                lines[i][2] = str(round(accuracy, 2))
                break

        with open(self.filepath, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(lines)
