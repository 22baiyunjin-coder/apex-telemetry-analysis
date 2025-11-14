def load_and_prepare_apex_history(
    filename: str = "Apex_Game_History_Season15S1.csv",
) -> pd.DataFrame:
    """
    Load the real Apex game history dataset and map it to the
    columns used in the rest of the pipeline.

    - my_duration -> time + survival_time
    - my_damage   -> damage
    - my_kills / my_assists / my_knocks -> combat_events
    """

    raw_dir = Path("data/raw")
    path = raw_dir / filename

    df = pd.read_csv(path)

    # 基础清洗：去掉全空的行，重置索引
    df = df.dropna(how="all").reset_index(drop=True)

    # 把原始列映射成我们 pipeline 能用的列名
    if "my_duration" in df.columns:
        df["time"] = df["my_duration"]
        df["survival_time"] = df["my_duration"]

    if "my_damage" in df.columns:
        df["damage"] = df["my_damage"]

    # 用 击杀+助攻+击倒 的总和近似表示“战斗事件数”
    for col in ["my_kills", "my_assists", "my_knocks"]:
        if col not in df.columns:
            df[col] = 0  # 如果某一列不存在就补 0，避免报错

    df["combat_events"] = (
        df["my_kills"].fillna(0)
        + df["my_assists"].fillna(0)
        + df["my_knocks"].fillna(0)
    )

    return df
