from micrograd import Value
from micrograd.visualization import draw_dot


def main():
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")

    z = x * y
    z.label = "z"

    z.backward()

    graph = draw_dot(z)

    graph.render(
        "artifacts/computation_graph",
        cleanup=True,
    )


if __name__ == "__main__":
    main()
