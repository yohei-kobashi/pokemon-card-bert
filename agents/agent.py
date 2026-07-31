import os
import random

# デッキ選択フェーズで返すため、自分の60枚を読み込んでおく。
# ローカルは decks/deck.csv、Kaggle提出時は agent と同じ階層の deck.csv を読む。
_DECK_CANDIDATES = [
    # decks/deck.csv was renamed to mega_abomasnow_sample.csv (it IS the competition's
    # sample deck: byte-identical to data/sample_submission/deck.csv). "deck.csv" stays
    # in the list because a Kaggle bundle always names the deck file deck.csv.
    os.path.join("decks", "mega_abomasnow_sample.csv"),
    "deck.csv",
    "/kaggle_simulations/agent/deck.csv",
]
for _p in _DECK_CANDIDATES:
    if os.path.exists(_p):
        with open(_p) as f:
            DECK = [int(line) for line in f if line.strip()]
        break
else:
    raise FileNotFoundError("deck.csv not found in: " + ", ".join(_DECK_CANDIDATES))

def agent(obs_dict: dict) -> list[int]:
    select = obs_dict.get("select")

    # 初期のデッキ選択フェーズ: select も current も None。
    # ここでは合法手の選択ではなく、自分のデッキ(60枚のカードID)を返す。
    if select is None:
        return DECK

    # 通常フェーズ: 提示された option から maxCount 個を選ぶ
    options = select["option"]
    n = select["maxCount"]
    return random.sample(range(len(options)), n)