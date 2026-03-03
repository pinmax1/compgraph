import click
import json

from compgraph.algorithms import yandex_maps_graph

def json_stream(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

@click.command()
@click.option('--input-time', required=True)
@click.option('--input-length', required=True)
@click.option('--output', required=True)
def main(input_time: str, input_length: str, output: str) -> None:
    graph = yandex_maps_graph(input_stream_name_time="input_time", input_stream_name_length="input_length")

    result = graph.run(input_time=lambda: json_stream(input_time), input_length=lambda: json_stream(input_length))
    # # pyrefly: ignore  # no-matching-overload
    with open(output, "w", encoding="utf-8") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
