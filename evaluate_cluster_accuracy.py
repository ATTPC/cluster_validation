import polars as pl
import click
from pathlib import Path


def evaluate_accuracy(df_path: Path, truth_label: int):
    results = pl.read_parquet(df_path).to_dict()

    total = len(results["event"])
    sum_matched = 0
    sum_total_acc = 0.0
    sum_inc_acc = 0.0
    sum_ex_acc = 0.0

    for idx, matches in enumerate(results["matches"]):
        if truth_label not in results["truth_labels"][idx]:
            total -= 1
            continue
        for midx, match in enumerate(matches):
            if match[1] == truth_label:
                sum_matched += 1
                sum_total_acc += results["total_acc"][idx][midx]
                sum_inc_acc += results["inclusive_acc"][idx][midx]
                sum_ex_acc += results["exclusive_acc"][idx][midx]
                break

    print(f"Evaluating accuracy of clustering truth label: {truth_label}")
    print(
        f"Total number of events which contain truth label {truth_label}: {total} out of {len(results["event"])}"
    )
    print(
        f"Total events with match: {sum_matched} Percent matched: {sum_matched/total}"
    )
    print(f"Avg. total accuracy of matches: {sum_total_acc/sum_matched}")
    print(f"Avg. inclusive accuracy of matches: {sum_inc_acc/sum_matched}")
    print(f"Avg. exclusive accuracy of matches: {sum_ex_acc/sum_matched}")


@click.command()
@click.argument("path", type=click.types.Path(exists=True))
@click.argument("label", type=click.INT)
def main(path: str, label: int):
    evaluate_accuracy(Path(path), label)


if __name__ == "__main__":
    main()
