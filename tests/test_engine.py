import math

from micrograd import Value


def test_addition():
    x = Value(2)
    y = Value(3)

    z = x + y
    z.backward()

    assert z.data == 5
    assert x.grad == 1
    assert y.grad == 1


def test_multiplication():
    x = Value(2)
    y = Value(3)

    z = x * y
    z.backward()

    assert x.grad == 3
    assert y.grad == 2


def test_power():
    x = Value(3)

    y = x ** 2
    y.backward()

    assert math.isclose(x.grad, 6)


def test_tanh():
    x = Value(0)

    y = x.tanh()
    y.backward()

    assert math.isclose(x.grad, 1)


def test_chain_rule():
    x = Value(2)
    y = Value(3)

    z = x * y + x ** 2
    z.backward()

    assert math.isclose(x.grad, 7)
    assert math.isclose(y.grad, 2)


def test_shared_node():
    x = Value(3)

    y = x * x
    z = y * x

    z.backward()

    assert math.isclose(x.grad, 27)
