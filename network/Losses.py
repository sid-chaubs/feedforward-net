import math

class Losses:

    @staticmethod
    def cross_entropy(predicted: list, targets: list) -> float:
        loss = 0.0

        for i in range(0, len(predicted)):
            loss += float(targets[i][0]) * float(math.log(predicted[i][0]))

        return  -1.0 * loss
