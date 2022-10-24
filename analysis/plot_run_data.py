import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import wandb

api = wandb.Api()


def grab_run_data(run_query: str) -> pd.DataFrame:
    """
    Grabs complete run data given a query and returns formatted dataframe

    Example::
    >>> run_df = grab_run_data("/bfattori/LJX/runs/wzltcjjb")
    >>> run_df.head(5)
            Tokens Seen (B)  Validation LM Loss  Train LM Loss
        0         0.000000           10.922203      10.919440
        1         0.262144            5.809898       5.829369
        2         0.524288            5.228324       5.257582
        3         0.786432            4.927660       4.928319
        4         1.048576            4.728181       4.750701

    """
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

    return run_df


if __name__ == "__main__":

    #124 M
    # run_df_full_ctx = pd.read_csv("analysis/processed/full_context.csv")

    # run_staged_1_epoch = pd.read_csv("analysis/processed/staged_single_epoch.csv")

    # run_staged_multi_epoch = pd.read_csv("analysis/processed/staged_multi_epoch.csv")

    # run_df_full_ctx = run_df_full_ctx.loc[run_df_full_ctx["Tokens Seen (B)"] <= 12.6]
    # run_staged_1_epoch = run_staged_1_epoch.loc[
    #     run_staged_1_epoch["Tokens Seen (B)"] <= 12.6
    # ]
    # run_staged_multi_epoch = run_staged_multi_epoch.loc[
    #     run_staged_multi_epoch["Tokens Seen (B)"] <= 12.6
    # ]

    # run_df_full_ctx["Validation LM PPL"] = np.exp(run_df_full_ctx["Validation LM Loss"])
    # run_staged_1_epoch["Validation LM PPL"] = np.exp(
    #     run_staged_1_epoch["Validation LM Loss"]
    # )
    # run_staged_multi_epoch["Validation LM PPL"] = np.exp(
    #     run_staged_multi_epoch["Validation LM Loss"]
    # )

    # sns.set_theme()
    # sns.lineplot(
    #     x="Tokens Seen (B)",
    #     y="Validation LM PPL",
    #     data=run_df_full_ctx,
    #     label="1 Epoch - Full Context",
    # )
    # sns.lineplot(
    #     x="Tokens Seen (B)",
    #     y="Validation LM PPL",
    #     data=run_staged_1_epoch,
    #     label="1 Epoch - Context Warmup",
    # )
    # sns.lineplot(
    #     x="Tokens Seen (B)",
    #     y="Validation LM PPL",
    #     data=run_staged_multi_epoch,
    #     label="Multi Epoch - Context Warmup",
    # )
    # plt.title("Tokens Seen vs. Validation PPL (124M Param Transformer)")
    # plt.ylim((20, 40))
    # plt.ylabel("Validation PPL")
    # plt.legend()
    # plt.show()

    # 354M
    run_df_full_ctx = pd.read_csv("analysis/processed/354_no_warmup.csv")

    run_staged_1_epoch = pd.read_csv("analysis/processed/354_ctx_warmup.csv")

    run_df_full_ctx = run_df_full_ctx.loc[run_df_full_ctx["Tokens Seen (B)"] <= 12.6]
    run_staged_1_epoch = run_staged_1_epoch.loc[
        run_staged_1_epoch["Tokens Seen (B)"] <= 12.6
    ]

    run_df_full_ctx["Validation LM PPL"] = np.exp(run_df_full_ctx["Validation LM Loss"])
    run_staged_1_epoch["Validation LM PPL"] = np.exp(
        run_staged_1_epoch["Validation LM Loss"]
    )

    sns.set_theme()
    sns.lineplot(
        x="Tokens Seen (B)",
        y="Validation LM PPL",
        data=run_df_full_ctx,
        label="1 Epoch - Full Context",
    )
    sns.lineplot(
        x="Tokens Seen (B)",
        y="Validation LM PPL",
        data=run_staged_1_epoch,
        label="1 Epoch - Context Warmup",
    )

    plt.title("Tokens Seen vs. Validation PPL (354M Param Transformer)")
    plt.ylim((15, 40))
    plt.ylabel("Validation PPL")
    plt.legend()
    plt.show()
