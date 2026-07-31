import json
from kaggle_environments import make
from battle_log import save_battle, deck_name, deck_path, load_agent
from library import load_config, resolve_deck

# 使用するデッキ名(decks/<名前>.csv)とエージェント名(agents/<名前>.py)は
# config.json から読む。管理画面(manage_server.py)で選択を変更できる。
# デッキが「ランダム」(__random__)指定なら、この実行(=1バトル)ごとに実デッキを抽選する。
_run = load_config()["run"]
AGENT0_NAME, DECK0_NAME = _run["player0"]["agent"], resolve_deck(_run["player0"]["deck"])
AGENT1_NAME, DECK1_NAME = _run["player1"]["agent"], resolve_deck(_run["player1"]["deck"])

with open(deck_path(DECK0_NAME)) as f:
    deck0 = [int(line) for line in f if line.strip()]
with open(deck_path(DECK1_NAME)) as f:
    deck1 = [int(line) for line in f if line.strip()]

agent0 = load_agent(AGENT0_NAME)
agent1 = load_agent(AGENT1_NAME)

env = make("cabt", configuration={"decks": [deck0, deck1]})
env.run([agent0, agent1])

# 描画データは env.steps[0][0]["visualize"] に入っている（vis.json と同じ形式）。
visualize = env.steps[0][0]["visualize"]

# 従来どおり vis.json も残す（visualizer.html での再生用）。
with open("vis.json", "w") as f:
    json.dump(visualize, f)

# logs/ に保存。ファイル名で AI同士であること・各エージェント名・デッキ名が分かる。
path = save_battle(visualize, [
    {"kind": "ai", "agent": AGENT0_NAME, "deck": deck_name(DECK0_NAME)},
    {"kind": "ai", "agent": AGENT1_NAME, "deck": deck_name(DECK1_NAME)},
])

print("Simulation finished. -> open visualizer.html in a browser and select vis.json")
print("rewards:", [s.get("reward") for s in env.steps[-1]])
print("saved log:", path)