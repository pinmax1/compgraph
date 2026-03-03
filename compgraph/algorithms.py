from . import Graph, operations


def word_count_graph(input_stream_name: str, text_column: str = 'text', count_column: str = 'count') -> Graph:
    """Constructs graph which counts words in text_column of all rows passed"""
    return Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .sort([text_column]) \
        .reduce(operations.Count(count_column), [text_column]) \
        .sort([count_column, text_column])


def inverted_index_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
                         result_column: str = 'tf_idf') -> Graph:
    """Constructs graph which calculates td-idf for every word/document pair"""
    split_graph = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column))

    count_docs_graph = Graph.graph_from_iter(input_stream_name) \
        .reduce(operations.Count('total_count'), tuple())

    idf_graph = split_graph.sort((doc_column, text_column)) \
        .reduce(operations.FirstReducer(), (doc_column, text_column)).sort((text_column,)) \
        .reduce(operations.Count('count'), (text_column,)) \
        .join(operations.InnerJoiner(), count_docs_graph, tuple()) \
        .map(operations.Idf('idf', 'total_count', 'count'))

    tf_graph = split_graph.sort((doc_column,)) \
        .reduce(operations.TermFrequency(text_column), (doc_column,)) \
        .sort((text_column,))

    return tf_graph.join(operations.InnerJoiner(), idf_graph, (text_column,)) \
        .map(operations.Product(('idf', 'tf'), 'tf_idf')) \
        .sort((text_column,)) \
        .reduce(operations.TopN('tf_idf', 3), (text_column,)) \
        .map(operations.Project((doc_column, text_column, 'tf_idf')))


def pmi_graph(input_stream_name: str, doc_column: str = 'doc_id', text_column: str = 'text',
              result_column: str = 'pmi') -> Graph:
    """Constructs graph which gives for every document the top 10 words ranked by pointwise mutual information"""
    split_graph = Graph.graph_from_iter(input_stream_name) \
        .map(operations.FilterPunctuation(text_column)) \
        .map(operations.LowerCase(text_column)) \
        .map(operations.Split(text_column)) \
        .map(operations.Filter(lambda row: len(row[text_column]) > 4)) \
        .sort((doc_column, text_column))

    filter_graph = split_graph.reduce(operations.Count('count'), (doc_column, text_column)) \
        .join(operations.InnerJoiner(), split_graph, (doc_column, text_column)) \
        .map(operations.Filter(lambda row: row['count'] >= 2)) \
        .sort((doc_column, text_column)) \
        .map(operations.Project((doc_column, text_column)))

    word_freq_in_total_graph = filter_graph.sort((text_column,)) \
        .reduce(operations.TermFrequency(text_column, 'total_frequency'), tuple())

    word_freq_in_doc_graph = filter_graph.reduce(operations.TermFrequency(text_column, 'doc_freq'), (doc_column, ))

    result = word_freq_in_doc_graph.sort((text_column, )) \
        .join(operations.InnerJoiner(), word_freq_in_total_graph, (text_column,)) \
        .map(operations.Idf('pmi', 'doc_freq', 'total_frequency')) \
        .sort((doc_column, text_column)) \
        .reduce(operations.TopN('pmi', 10), (doc_column, )) \
        .map(operations.Project((doc_column, text_column, 'pmi')))

    return result


def yandex_maps_graph(input_stream_name_time: str, input_stream_name_length: str,
                      enter_time_column: str = 'enter_time', leave_time_column: str = 'leave_time',
                      edge_id_column: str = 'edge_id', start_coord_column: str = 'start', end_coord_column: str = 'end',
                      weekday_result_column: str = 'weekday', hour_result_column: str = 'hour',
                      speed_result_column: str = 'speed') -> Graph:
    """Constructs graph which measures average speed in km/h depending on the weekday and hour"""

    date_graph = Graph.graph_from_iter(input_stream_name_time) \
        .map(operations.WeekdayAndHour(enter_time_column,
                                       leave_time_column, weekday_result_column, hour_result_column, 'diff')) \
        .sort((edge_id_column,))

    length_graph = Graph.graph_from_iter(input_stream_name_length) \
        .map(operations.Haversine(start_coord_column, end_coord_column)) \
        .sort((edge_id_column,))

    result_graph = date_graph.join(operations.InnerJoiner(), length_graph, (edge_id_column, )) \
        .sort((weekday_result_column, hour_result_column)) \
        .reduce(operations.Average(('diff', 'length')), (weekday_result_column, hour_result_column)) \
        .map(operations.Divide('length', 'diff', speed_result_column)) \
        .map(operations.Project((weekday_result_column, hour_result_column, speed_result_column)))

    return result_graph
