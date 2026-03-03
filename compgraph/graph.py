import typing as tp

from . import operations as ops
from .external_sort import ExternalSort


class Graph:
    """Computational graph implementation"""

    def __init__(self, operation: ops.Operation, previous_nodes: list['Graph']) -> None:
        self.operation = operation
        self.previous_nodes = previous_nodes

    @staticmethod
    def graph_from_iter(name: str) -> 'Graph':
        """Construct new graph which reads data from row iterator (in form of sequence of Rows
        from 'kwargs' passed to 'run' method) into graph data-flow
        Use ops.ReadIterFactory
        :param name: name of kwarg to use as data source
        """
        return Graph(ops.ReadIterFactory(name), [])

    @staticmethod
    def graph_from_file(filename: str, parser: tp.Callable[[str], ops.TRow]) -> 'Graph':
        """Construct new graph extended with operation for reading rows from file
        Use ops.Read
        :param filename: filename to read from
        :param parser: parser from string to Row
        """
        return Graph(ops.Read(filename, parser), [])

    # If you would like to implement __init__ and/or @classmethods instead of methods above,
    #  feel free to do so. However, the __init__ method should not accept any arguments.

    def map(self, mapper: ops.Mapper) -> 'Graph':
        """Construct new graph extended with map operation with particular mapper
        :param mapper: mapper to use
        """
        return Graph(ops.Map(mapper), [self])

    def reduce(self, reducer: ops.Reducer, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with reduce operation with particular reducer
        :param reducer: reducer to use
        :param keys: keys for grouping
        """
        return Graph(ops.Reduce(reducer, keys), [self])

    def sort(self, keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with sort operation
        :param keys: sorting keys (typical is tuple of strings)
        """
        return Graph(ExternalSort(keys), [self])

    def join(self, joiner: ops.Joiner, join_graph: 'Graph', keys: tp.Sequence[str]) -> 'Graph':
        """Construct new graph extended with join operation with another graph
        :param joiner: join strategy to use
        :param join_graph: other graph to join with
        :param keys: keys for grouping
        """
        return Graph(ops.Join(joiner, keys), [self, join_graph])

    def run(self, **kwargs: tp.Any) -> ops.TRowsIterable:
        """Single method to start execution; data sources passed as kwargs"""
        if len(self.previous_nodes) == 0:
            yield from self.operation(**kwargs)
        else:
            yield from self.operation(*[node.run(**kwargs) for node in self.previous_nodes])
