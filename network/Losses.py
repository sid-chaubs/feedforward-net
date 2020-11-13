import math

class Losses:

    @staticmethod
    def cross_entropy(predicted: float, targets: list) -> float:
        loss = 0.0
        for i in range(0, len(predicted)):
            loss += float(targets[i]) * float(math.log(predicted[i]))

        return  -1.0 * loss
