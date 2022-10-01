from typing import List

import numpy as np
import pandas as pd
import wandb

api = wandb.Api()


def grab_run_data(run_querys: List[str]) -> pd.DataFrame:
    """
    Grabs complete run data given a query and returns formatted dataframe

    Example::
    >>> run_df = grab_run_data(["/bfattori/LJX/runs/wzltcjjb","/bfattori/LJX/runs/hgndkjh"])
    >>> run_df.head(5)
            Tokens Seen (B)  Validation LM Loss  Train LM Loss
        0         0.000000           10.922203      10.919440
        1         0.262144            5.809898       5.829369
        2         0.524288            5.228324       5.257582
        3         0.786432            4.927660       4.928319
        4         1.048576            4.728181       4.750701

    """
    full_runs_df = pd.DataFrame()

    for run_query in run_querys:

        run = api.run(run_query)
        history = run.scan_history(
            keys=[
                "_step",
                "Tokens Seen (B)",
                "Train Step Time",
                "_runtime",
                "Train LM PPL",
                "Train LM Loss",
                "Train Sequence Length",
                "_timestamp",
                "Validation LM Loss",
            ]
        )

        run_df = pd.DataFrame(
            {
                "Tokens Seen (B)": [row["Tokens Seen (B)"] for row in history],
                "Validation LM Loss": [row["Validation LM Loss"] for row in history],
                "Train LM Loss": [row["Train LM Loss"] for row in history],
            }
        )

        full_runs_df = full_runs_df.append(run_df)

    return full_runs_df.sort_values(by="Tokens Seen (B)", ascending=True)


if __name__ == "__main__":
    run_df_full = grab_run_data(
        ["/bfattori/LJX/runs/1b3f90mz", "/bfattori/LJX/runs/799j3xyb"]
    )
    run_df_full.to_csv("analysis/processed/staged_multi_epoch.csv")
