from abc import abstractmethod, ABC
import typing as tp
import string
from itertools import groupby
import re
from collections import defaultdict
import heapq
import math
from datetime import datetime

TRow = dict[str, tp.Any]
TRowsIterable = tp.Iterable[TRow]
TRowsGenerator = tp.Generator[TRow, None, None]

# Either rename me to operations.py, or, better yet, create a folder `operations` with `__init__`
#  and split this large file into different files (e. g. one file for mappers, one for reducers, etc.)
#  Check Modules lecture for inspiration, ther similar approach is done in `basic_module` task

class Peekable:
    def __init__(self, it):
        self.it = iter(it)
        self.buffer = []

    def peek(self):
        if not self.buffer:
            try:
                self.buffer.append(next(self.it))
            except StopIteration:
                return None
        return self.buffer[0]

    def __next__(self):
        if self.buffer:
            return self.buffer.pop(0)
        return next(self.it)

    def __iter__(self):
        return self


class Operation(ABC):
    @abstractmethod
    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        pass


class Read(Operation):
    def __init__(self, filename: str, parser: tp.Callable[[str], TRow]) -> None:
        self._filename = filename
        self._parser = parser

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        with open(self._filename) as f:
            for line in f:
                yield self._parser(line)


class ReadIterFactory(Operation):
    def __init__(self, name: str) -> None:
        self._name = name

    def __call__(self, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in kwargs[self._name]():
            yield row


# Operations


class Mapper(ABC):
    """Base class for mappers"""
    @abstractmethod
    def __call__(self, row: TRow) -> TRowsGenerator:
        """
        :param row: one table row
        """
        pass


class Map(Operation):
    def __init__(self, mapper: Mapper) -> None:
        self._mapper = mapper

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for row in rows:
            yield from self._mapper(row)


class Reducer(ABC):
    """Base class for reducers"""
    @abstractmethod
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        """
        :param rows: table rows
        """
        pass


class Reduce(Operation):
    def __init__(self, reducer: Reducer, keys: tp.Sequence[str]) -> None:
        self._reducer = reducer
        self._keys = keys

    def __call__(self, rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        for group_key, group in groupby(rows, key = lambda x: tuple(x[col] for col in self._keys)):
            yield from self._reducer(tuple(self._keys), group)


class Joiner(ABC):
    """Base class for joiners"""
    def __init__(self, suffix_a: str = '_1', suffix_b: str = '_2') -> None:
        self._a_suffix = suffix_a
        self._b_suffix = suffix_b

    @abstractmethod
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        """
        :param keys: join keys
        :param rows_a: left table rows
        :param rows_b: right table rows
        """
        pass


class Join(Operation):
    def __init__(self, joiner: Joiner, keys: tp.Sequence[str]):
        self._keys = keys
        self._joiner = joiner

    def __call__(self, left_rows: TRowsIterable, *args: tp.Any, **kwargs: tp.Any) -> TRowsGenerator:
        left_iter = groupby(left_rows, key = lambda x: tuple(x[col] for col in self._keys))
        right_iter = groupby(args[0], key = lambda x: tuple(x[col] for col in self._keys))

        left_key, left_group = next(left_iter, (None, []))
        right_key, right_group = next(right_iter, (None, []))

        while left_key is not None or right_key is not None:
            if right_key is None or (left_key is not None and left_key < right_key):
                yield from self._joiner(self._keys, left_group, [])
                left_key, left_group = next(left_iter, (None, []))
            elif left_key is None or (right_key is not None and left_key > right_key):
                yield from self._joiner(self._keys, [], right_group)
                right_key, right_group = next(right_iter, (None, []))
            else:
                yield from self._joiner(self._keys, left_group, right_group)
                left_key, left_group = next(left_iter, (None, []))
                right_key, right_group = next(right_iter, (None, []))

# Dummy operators


class DummyMapper(Mapper):
    """Yield exactly the row passed"""
    def __call__(self, row: TRow) -> TRowsGenerator:
        yield row


class FirstReducer(Reducer):
    """Yield only first row from passed ones"""
    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        for row in rows:
            yield row
            break


# Mappers


class FilterPunctuation(Mapper):
    """Left only non-punctuation symbols"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self._column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._column] = row[self._column].translate(str.maketrans('', '', string.punctuation))
        yield row


class LowerCase(Mapper):
    """Replace column value with value in lower case"""
    def __init__(self, column: str):
        """
        :param column: name of column to process
        """
        self._column = column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._column] = row[self._column].lower()
        yield row


class Split(Mapper):
    """Split row on multiple rows by separator"""
    def __init__(self, column: str, separator: str | None = None) -> None:
        """
        :param column: name of column to split
        :param separator: string to separate by
        """
        self._column = column
        self._separator = separator or r"\s+"

    def __call__(self, row: TRow) -> TRowsGenerator:
        last = 0
        text = row[self._column]
        for match in re.finditer(self._separator, text):
            result = row.copy()
            result[self._column] = text[last: match.start()]
            yield result
            last = match.end()
        result = row.copy()
        result[self._column] = text[last:]
        yield result

class Product(Mapper):
    """Calculates product of multiple columns"""
    def __init__(self, columns: tp.Sequence[str], result_column: str = 'product') -> None:
        """
        :param columns: column names to product
        :param result_column: column name to save product in
        """
        self._columns = columns
        self._result_column = result_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        result = 1
        for col in self._columns:
            result *= row[col]
        row[self._result_column] = result
        yield row


class Filter(Mapper):
    """Remove records that don't satisfy some condition"""
    def __init__(self, condition: tp.Callable[[TRow], bool]) -> None:
        """
        :param condition: if condition is not true - remove record
        """
        self._condition = condition

    def __call__(self, row: TRow) -> TRowsGenerator:
        if self._condition(row):
            yield row


class Project(Mapper):
    """Leave only mentioned columns"""
    def __init__(self, columns: tp.Sequence[str]) -> None:
        """
        :param columns: names of columns
        """
        self._columns = columns

    def __call__(self, row: TRow) -> TRowsGenerator:
        yield {key : row[key] for key in self._columns}

class Idf(Mapper):
    def __init__(self, column: str, total_count_column: str, words_count_column: str) -> None:
        """
        :param column: result column name
        :param total_count_column: total count column name
        :param words_count_column: words count column name
        """
        self._column = column
        self._total_count_column = total_count_column
        self._words_count_column = words_count_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._column] = math.log(row[self._total_count_column] / row[self._words_count_column])
        yield row

class Haversine(Mapper):
    def __init__(self, start_point_column, end_point_column, result_column = 'length') -> None:
        self._result_column = result_column
        self._start_point_column = start_point_column
        self._end_point_column = end_point_column

    def __call__(self, row: TRow) -> TRowsGenerator:
        R = 6373.0

        lon1, lat1 = row[self._start_point_column]
        lon2, lat2 = row[self._end_point_column]

        phi1 = math.radians(lat1)
        phi2 = math.radians(lat2)
        d_phi = math.radians(lat2 - lat1)
        d_lambda = math.radians(lon2 - lon1)

        a = math.sin(d_phi / 2) ** 2 + \
            math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2

        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        row[self._result_column] = R * c

        yield row

class WeekdayAndHour(Mapper):
    def __init__(self, enter_time_col, leave_time_col, weekday_col, hour_col, time_col):
        self._enter_time_col = enter_time_col
        self._leave_time_col = leave_time_col
        self._weekday_col = weekday_col
        self._hour_col = hour_col
        self._time_col = time_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        dt1 = datetime.strptime(row[self._enter_time_col], "%Y%m%dT%H%M%S.%f")
        dt2 = datetime.strptime(row[self._leave_time_col], "%Y%m%dT%H%M%S.%f")

        row[self._weekday_col] = dt1.strftime("%a")
        row[self._hour_col] = dt1.hour
        row[self._time_col] = (dt2 - dt1).total_seconds() / 3600.0

        if row[self._time_col] >= 0:
            yield row

class Divide(Mapper):
    def __init__(self, first_col, second_col, result_col) -> None:
        self._first_col = first_col
        self._second_col = second_col
        self._result_col = result_col

    def __call__(self, row: TRow) -> TRowsGenerator:
        row[self._result_col] = row[self._first_col] / row[self._second_col]
        yield row


# Reducers


class TopN(Reducer):
    """Calculate top N by value"""
    def __init__(self, column: str, n: int) -> None:
        """
        :param column: column name to get top by
        :param n: number of top values to extract
        """
        self._column_max = column
        self._n = n

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        heap = []

        counter = 0
        for row in rows:
            value = row[self._column_max]
            counter += 1
            if len(heap) < self._n:
                heapq.heappush(heap, (value, counter, row))
            else:
                if value > heap[0][0]:
                    heapq.heapreplace(heap, (value, counter, row))

        for value, _, row in sorted(heap, key=lambda x: x[0], reverse=True):
            yield row


class TermFrequency(Reducer):
    """Calculate frequency of values in column"""
    def __init__(self, words_column: str, result_column: str = 'tf') -> None:
        """
        :param words_column: name for column with words
        :param result_column: name for result column
        """
        self._words_column = words_column
        self._result_column = result_column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count_map = defaultdict(int)
        total_count = 0
        example_row = {}
        for row in rows:
            example_row = row
            count_map[row[self._words_column]] += 1
            total_count += 1
        example_row = {key: example_row[key] for key in group_key}
        for word, count in count_map.items():
            result = example_row.copy()
            result[self._words_column] = word
            result[self._result_column] = count / total_count
            yield result


class Count(Reducer):
    """
    Count records by key
    Example for group_key=('a',) and column='d'
        {'a': 1, 'b': 5, 'c': 2}
        {'a': 1, 'b': 6, 'c': 1}
        =>
        {'a': 1, 'd': 2}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for result column
        """
        self._column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        count = 0
        example_row = {}
        for row in rows:
            example_row = row
            count += 1
        result = {key : example_row[key] for key in group_key}
        result[self._column] = count
        yield result


class Sum(Reducer):
    """
    Sum values aggregated by key
    Example for key=('a',) and column='b'
        {'a': 1, 'b': 2, 'c': 4}
        {'a': 1, 'b': 3, 'c': 5}
        =>
        {'a': 1, 'b': 5}
    """
    def __init__(self, column: str) -> None:
        """
        :param column: name for sum column
        """
        self._column = column

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        sum = 0
        example_row = {}
        for row in rows:
            example_row = row
            sum += row[self._column]
        result = {key : example_row[key] for key in group_key}
        result[self._column] = sum
        yield result

class Average(Reducer):
    def __init__(self, columns: tuple[str]) -> None:
        """
        :param column: name for sum column
        """
        self._columns = columns

    def __call__(self, group_key: tuple[str, ...], rows: TRowsIterable) -> TRowsGenerator:
        sums = defaultdict(float)
        example_row = {}
        count = 0
        for row in rows:
            example_row = row
            count += 1
            for col in self._columns:
                sums[col] += row[col]
        result = {key : example_row[key] for key in group_key}
        for col in self._columns:
            result[col] = sums[col]/count
        yield result



# Joiners


class InnerJoiner(Joiner):
    def __init__(self, suffix_a = '_a', suffix_b = '_b'):
        self.suffix_a = suffix_a
        self.suffix_b = suffix_b

    """Join with inner strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        keys_set = set(keys)
        list_b = list(rows_b)
        for row_a in rows_a:
            for row_b in list_b:
                result = {}

                for k in keys_set:
                    if k in row_a:
                        result[k] = row_a[k]
                    else:
                        result[k] = row_b.get(k)

                keys_a = set(row_a.keys()) - keys_set
                keys_b = set(row_b.keys()) - keys_set

                overlap = keys_a & keys_b

                for k in keys_a - overlap:
                    result[k] = row_a[k]

                for k in keys_b - overlap:
                    result[k] = row_b[k]

                for k in overlap:
                    result[k + self.suffix_a] = row_a[k]
                    result[k + self.suffix_b] = row_b[k]

                yield result


class OuterJoiner(Joiner):
    """Join with outer strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_b = list(rows_b)
        rows_a = Peekable(rows_a)
        a_empty = rows_a.peek() is None
        b_empty = len(list_b) == 0
        if a_empty:
            for row_b in list_b:
                yield row_b
        if b_empty:
            for row_a in rows_a:
                yield row_a
        for row_a in rows_a:
            for row_b in list_b:
                result = row_a | row_b
                yield result

class LeftJoiner(Joiner):
    """Join with left strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_b = list(rows_b)

        if not list_b:
            for row_a in rows_a:
                yield row_a
            return

        for row_a in rows_a:
            for row_b in list_b:
                yield row_a | row_b


class RightJoiner(Joiner):
    """Join with right strategy"""
    def __call__(self, keys: tp.Sequence[str], rows_a: TRowsIterable, rows_b: TRowsIterable) -> TRowsGenerator:
        list_a = list(rows_a)

        if not list_a:
            for row_b in rows_b:
                yield row_b
            return

        for row_b in rows_b:
            for row_a in list_a:
                yield row_a | row_b
