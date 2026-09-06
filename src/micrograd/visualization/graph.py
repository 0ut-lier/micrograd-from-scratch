from graphviz import Digraph


def trace(root):
    """
    Traverse the computation graph starting from root.

    Returns:
        nodes: all Value nodes
        edges: parent-child relationships
    """

    nodes = set()
    edges = set()

    def build(node):
        if node in nodes:
            return

        nodes.add(node)

        for child in node._prev:
            edges.add((child, node))
            build(child)

    build(root)

    return nodes, edges


def draw_dot(root):
    """
    Create a Graphviz representation of the computation graph.
    """

    dot = Digraph(
        format="svg",
        graph_attr={"rankdir": "LR"}
    )

    nodes, edges = trace(root)

    for node in nodes:
        node_id = str(id(node))

        dot.node(
            name=node_id,
            label=(
                "{ %s | data %.4f | grad %.4f }"
                % (
                    node.label,
                    node.data,
                    node.grad,
                )
            ),
            shape="record",
        )

        if node._op:
            operation_id = node_id + node._op

            dot.node(
                name=operation_id,
                label=node._op,
            )

            dot.edge(
                operation_id,
                node_id,
            )

    for source, target in edges:
        dot.edge(
            str(id(source)),
            str(id(target)) + target._op,
        )

    return dot
