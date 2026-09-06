import random

from micrograd.nn import MLP


def main():
    model = MLP(3, [4, 4, 1])

    # XOR-like toy dataset
    dataset = [
        ([2.0, 3.0, -1.0], 1.0),
        ([3.0, -1.0, 0.5], -1.0),
        ([0.5, 1.0, 1.0], -1.0),
        ([1.0, 1.0, -1.0], 1.0),
    ]

    learning_rate = 0.05

    for epoch in range(100):
        predictions = [
            model(inputs)
            for inputs, _ in dataset
        ]

        # Mean squared error
        loss = sum(
            (prediction - target) ** 2
            for prediction, target in zip(
                predictions,
                [target for _, target in dataset],
            )
        )

        model.zero_grad()
        loss.backward()

        for parameter in model.parameters():
            parameter.data += -learning_rate * parameter.grad

        if epoch % 10 == 0:
            print(
                f"epoch={epoch:03d} "
                f"loss={loss.data:.4f}"
            )


if __name__ == "__main__":
    main()
