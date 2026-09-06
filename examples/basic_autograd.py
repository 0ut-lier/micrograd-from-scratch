from micrograd import Value


def main():
    x = Value(2.0, label="x")
    y = Value(3.0, label="y")

    z = x * y + x ** 2

    z.label = "z"

    z.backward()

    print(f"x = {x}")
    print(f"y = {y}")
    print(f"z = {z}")

    print(f"dz/dx = {x.grad}")
    print(f"dz/dy = {y.grad}")


if __name__ == "__main__":
    main()
