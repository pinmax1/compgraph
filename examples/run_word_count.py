import click
import json

from compgraph.algorithms import word_count_graph

def json_stream(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

@click.command()
@click.option('--input', required=True)
@click.option('--output', required=True)
def main(input: str, output: str) -> None:
    graph = word_count_graph(input_stream_name="input", text_column='text', count_column='count')

    result = graph.run(input=lambda: json_stream(input))
    # # pyrefly: ignore  # no-matching-overload
    with open(output, "w", encoding="utf-8") as out:
        for row in result:
            print(row, file=out)


if __name__ == "__main__":
    main()
