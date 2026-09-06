import math


class Value:
    """
    A scalar value that supports automatic differentiation.

    Each Value stores:
        - data: the scalar numerical value
        - grad: derivative of the final output with respect to this value
        - _prev: parent nodes in the computation graph
        - _op: operation that produced this value
        - _backward: function that propagates gradients to its parents
    """

    def __init__(self, data, _children=(), _op="", label=""):
        self.data = float(data)
        self.grad = 0.0

        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # ------------------------------------------------------------------
    # Arithmetic operations
    # ------------------------------------------------------------------

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data + other.data,
            (self, other),
            "+"
        )

        def _backward():
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward

        return out

    def __radd__(self, other):
        return self + other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)

        out = Value(
            self.data * other.data,
            (self, other),
            "*"
        )

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward

        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        if not isinstance(power, (int, float)):
            raise TypeError("Power must be an int or float.")

        out = Value(
            self.data ** power,
            (self,),
            f"**{power}"
        )

        def _backward():
            self.grad += (
                power
                * self.data ** (power - 1)
                * out.grad
            )

        out._backward = _backward

        return out

    def __truediv__(self, other):
        return self * (other ** -1)

    def __rtruediv__(self, other):
        return other * (self ** -1)

    # ------------------------------------------------------------------
    # Mathematical functions
    # ------------------------------------------------------------------

    def exp(self):
        out = Value(
            math.exp(self.data),
            (self,),
            "exp"
        )

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward

        return out

    def tanh(self):
        t = math.tanh(self.data)

        out = Value(
            t,
            (self,),
            "tanh"
        )

        def _backward():
            self.grad += (1 - t ** 2) * out.grad

        out._backward = _backward

        return out

    # ------------------------------------------------------------------
    # Autograd
    # ------------------------------------------------------------------

    def backward(self):
        """
        Compute gradients of this value with respect to all
        values that contributed to it.
        """

        topo = []
        visited = set()

        def build_topological_order(node):
            if node not in visited:
                visited.add(node)

                for child in node._prev:
                    build_topological_order(child)

                topo.append(node)

        build_topological_order(self)

        # d(output) / d(output) = 1
        self.grad = 1.0

        for node in reversed(topo):
            node._backward()

    def zero_grad(self):
        """Reset this Value's gradient to zero."""
        self.grad = 0.0
