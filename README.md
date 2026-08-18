# ポケモンTCG AI エージェント開発

## 学習済みモデル一覧（HuggingFace）

コンペ終了時点（2026-08-17、最終レーティング 398.2 = `dusk_v3`）に公開した3リポジトリ。
すべて `yoheikobashi/` 配下・public。合計 約16.5 GB。

| リポジトリ | 役割 | 中身 | サイズ |
|---|---|---|---|
| [ptcg-dusknoir-deberta-reranker](https://huggingface.co/yoheikobashi/ptcg-dusknoir-deberta-reranker) | **提出パイロット本体** | DeBERTa-v3-base クロスエンコーダ `fld_r49b`（SFT → mirror-RL/field DPO 67ラウンド）。`model.safetensors`(fp32) と、Kaggle 提出用の語彙刈り込み＋weight-only INT8 の `onnx/model_wonly_int8.onnx` + `onnx/vocab_remap.npy` | 0.93 GB |
| [ptcg-qwen3-4b-cardfirst-v40](https://huggingface.co/yoheikobashi/ptcg-qwen3-4b-cardfirst-v40) | **4B系の共有ベース** | Qwen3-4B に v40 プロンプト形式で SFT した LoRA（11デッキ自己対戦 約1.5M 決定）。ドメイントークン約3kの学習済み埋め込み `domain_embeddings.pt`、`cardfirst_vocab.json`、`checkpoint-6000` / `checkpoint-6178`（optimizer 状態込み＝再開可能） | 11.69 GB |
| [ptcg-qwen3-4b-dpo-loras](https://huggingface.co/yoheikobashi/ptcg-qwen3-4b-dpo-loras) | **デッキ別 DPO LoRA 群（22本）** | 上記ベースの上に、プレイアウト裁定ペアで DPO した LoRA。各ディレクトリに `adapter_model.safetensors` + `adapter_config.json` + `domain_embeddings.pt` + tokenizer 一式 | 3.87 GB |

`ptcg-qwen3-4b-dpo-loras` の内訳（ディレクトリ名 = アダプタ名）:

- デッキ別（`_r1` / `_r2` はラウンド）: `lora_alakazam_nz_r1`, `lora_archaludon_r1`,
  `lora_crustle_geco_r1`, `lora_dragapult_r1`, `lora_dragapult_r2`,
  `lora_dudunsparce_box_r1`, `lora_dudunsparce_box_r2`, `lora_ethan_hooh_r1`,
  `lora_marnie_grimmsnarl_r1`, `lora_mega_abomasnow_sample_r1`, `lora_mega_lucario_tr_r1`,
  `lora_ogerpon_mono_r1`, `lora_slowking_r1`, `lora_slowking_r2`
- 対面特化: `lora_dusk_vs_ogerpon_mono_r1`（最後まで攻略できなかった ogerpon_mono 専用）
- 全デッキ混合 / 実験系: `dpo_r8`, `lora_night4b_base`, `lora_night4b_filt`,
  `lora_night5_a`, `lora_night5_b`, `lora_night6_a`, `lora_night6_b`

使い分け: **提出物として実際に走ったのは DeBERTa リランカーのみ**（4B は Kaggle の
197.66 MiB 制限に載らず、スコア源・教師・実験用に留まる）。4B の LoRA を読むには
cardfirst-v40 のベース LoRA と `domain_embeddings.pt` を先に適用する必要がある
（ドメイントークンを含む埋め込みがベース側にあるため）。

学習インフラの実体は `tools/instance/`（両Vastインスタンスの全スクリプトのアーカイブ）、
人間対局の一次データは `human_games/` を参照。

## 自由研究キット: 人と対戦して集めたデータでモデルを追加学習する

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/yohei-kobashi/pokemon-card-bert/blob/main/notebooks/ptcg_kenkyu_colab.ipynb)

**研究テーマ: play_server でヒューリスティックエンジンとたくさん対戦して記録をため、Google Colab で
`ptcg-dusknoir-deberta-reranker` に追加学習して、ogerpon_mono（オーガポン）への勝率をどこまで上げられるか。**

ogerpon_mono はこのプロジェクトが最後まで攻略できなかった相手です。この環境で実際に測ると、
`dragapult_dusknoir` を**ヒューリスティックエンジンだけで**動かしたときの勝率は 400 試合で **3.0%**
（12勝、95% CI 1.7 - 5.2）しかありません。つまり**伸びしろが大きく、上がったかどうかが数字ではっきり出る**
題材です（同じ相手・同じ測り方で `crustle_stall` は 100 試合 92.0% なので、測り方の問題ではなく
このデッキとこの相手の相性です）。

```
 家のPC                        Google ドライブ                Colab (無料GPUでOK)
 ─────────────                 ────────────────               ──────────────────
 play_server で対戦   ──────▶  logs/  (自動でコピー)  ──────▶  集計 → 学習データ → 追加学習
        ▲                      models/ ◀──────────────────────  学習ずみモデルを保存
        │                          │
        └──── モデルをダウンロード ─┘ → 人と対戦 / エンジンと対戦して勝率を比較
```

| 使うもの | 何をするか |
|---|---|
| `tools/kenkyu/setup_local.py` | Windows / macOS / Linux で対戦環境をつくる（＋ドライブ連携） |
| `play_server.py` | ブラウザで人 vs AI の対戦。終わるたびにドライブへ自動保存 |
| `tools/kenkyu/sync_logs.py` | 過去の記録をまとめてドライブへ／zip にして手動アップ |
| `tools/kenkyu/log_stats.py` | **学習前の集計**（試合数・相手別勝率・日付別勝率） |
| `notebooks/ptcg_kenkyu_colab.ipynb` | Colab で追加学習（期間・相手・学習率・GPU を指定）。上のバッジから直接開けます |
| `agents/lm_dusknoir.py` | 学習したモデルを **人間の対戦相手**として動かす |
| `tools/kenkyu/battle_eval.py` | エンジンと対戦させて勝率を測り、**モデル同士を比較** |

---

### 0. 用意するもの

- **Python 3.9 以上**（3.11 か 3.12 がおすすめ）
- **Kaggle アカウント** … 対戦エンジン（`cg`）はコンペの配布物で、このリポジトリには入っていません。
  [コンペのページ](https://www.kaggle.com/competitions/pokemon-tcg-ai-battle)でルールに同意し、
  アカウントページの「Create New Token」で `kaggle.json` を保存して、
  `~/.kaggle/kaggle.json`（Windows は `%USERPROFILE%\.kaggle\kaggle.json`）に置きます。
- **Google アカウント**（ドライブ ＋ Colab）
- Google ドライブのパソコン用アプリ（Windows / Mac）。Linux には公式アプリが無いので、
  その場合は zip にして手動アップロードします（後述）。

### 1. セットアップ（家のパソコン、1回だけ）

```bash
git clone https://github.com/yohei-kobashi/pokemon-card-bert.git
cd pokemon-card-bert

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install kaggle huggingface_hub transformers torch

python tools/kenkyu/setup_local.py --drive "<ドライブのフォルダ>"
```

`<ドライブのフォルダ>` の例:

| OS | 例 |
|---|---|
| Windows | `"G:\My Drive\PTCG"` （ドライブレターは Google ドライブアプリの設定で確認） |
| macOS | `"~/Library/CloudStorage/GoogleDrive-<メールアドレス>/My Drive/PTCG"` |
| Linux | `"~/PTCG"`（同期アプリが無いので普通のフォルダ。あとで zip で上げます） |

このコマンドがやること:

1. コンペから対戦エンジンをダウンロードし、**この機械に合うバイナリ**を置く
   （Windows は `cg.dll`、Linux は `libcg.so`、Apple Silicon Mac は `libcg.dylib`。
   Intel Mac だけは配布バイナリが無いので、公開されている C++ ソースからコンパイルします
   ＝ 20〜60秒、`xcode-select --install` が必要）
2. ドライブに `logs/` `models/` を作り、**対戦が終わるたびに記録がドライブへコピーされる**ように
   `config.json` に書き込む。あわせて Colab 用に `cg/`（エンジン一式）を置く
   — **これがあれば Colab 側に Kaggle の鍵は要りません**（`repo.zip` も一緒に置かれますが、
   これは GitHub につながらないときの予備で、ふだんは使いません）
3. **実際に2試合プレイして動作確認**（ファイルが揃っただけでは成功と言わない）

うまくいくと最後に `OK. Start playing:` と出ます。あとから確認だけしたいときは
`python tools/kenkyu/setup_local.py --check`。

### 2. 対戦してデータを集める

```bash
python play_server.py
# ブラウザで http://localhost:8000/  （デッキ選択は http://localhost:8000/manage ）
```

`/manage` で次のように選びます。

| 項目 | 値 |
|---|---|
| human_deck | `dragapult_dusknoir`（自分が使うデッキ） |
| ai_agent | `ogerpon_mono` |
| ai_deck | `ogerpon_mono` |

1試合終わるたびに `logs/` に保存され、同時にドライブへ `.json.gz` でコピーされます。
Google ドライブアプリが次の試合の間にアップロードしてくれるので、**あとから何かする必要はありません**。

- 設定前に遊んだ分をまとめて送る: `python tools/kenkyu/sync_logs.py`
- 遊んでいる間ずっと同期する: `python tools/kenkyu/sync_logs.py --watch 60`
- **Linux など同期アプリが無い場合**: `python tools/kenkyu/sync_logs.py --zip battles.zip` で 1 つに
  まとめ、ブラウザでドライブの `PTCG` フォルダにアップロード。ノートブックに zip を展開するセルが
  あるので、そのままで大丈夫です

**どのくらい集めればいい?** 1試合 ≒ 40〜50 の判断 ≒ 40 行の学習データ。
最低 15 試合（約 500 行）、できれば 40 試合以上あると学習らしくなります。

> 相手を変えて遊ぶと、その相手との記録になります。学習データを作るときは
> `human-<自分のデッキ名>` のファイルだけが選ばれるので、**自分が違うデッキを使った試合**
> （例: 学習ずみモデルと対戦したとき）は混ざりません。

### 3. 学習前に集計する

```bash
python tools/kenkyu/log_stats.py                          # 全部
python tools/kenkyu/log_stats.py --since 2026-08-16       # 期間を切る
python tools/kenkyu/log_stats.py --opp ogerpon_mono       # 相手を切る
python tools/kenkyu/log_stats.py --logs "<ドライブ>/logs" --json stats.json
```

```
=== 対戦履歴の集計 ===
games      : 37  (2026-08-16 .. 2026-08-17)
human wins : 28 / 37 = 75.7%
decisions  : 1803 (学習データの上限, 平均 48.7/game)
turns      : 平均 10.6

--- 相手デッキ別 ---
  ogerpon_mono            17 games  10勝  7敗   58.8%    990 decisions
  mega_abomasnow_sample    8 games   8勝  0敗  100.0%    308 decisions
...
--- 日付別 ---
  2026-08-16   10 games   5勝  5敗   50.0%
  2026-08-17   27 games  23勝  4敗   85.2%
```

日付別の勝率は「**自分が上手くなった**」ことを示す大事なデータです。
上手くなってからの試合だけで学習する（`--since` で切る）とどうなるかは、そのまま研究のテーマになります。

### 4. Colab で追加学習する

[**このリンクでノートブックを Colab で開きます**](https://colab.research.google.com/github/yohei-kobashi/pokemon-card-bert/blob/main/notebooks/ptcg_kenkyu_colab.ipynb)
（`notebooks/ptcg_kenkyu_colab.ipynb`）。リポジトリもモデルも公開なので、ログインやトークンは要りません
— Colab が自分でクローンし、モデルは HuggingFace から取ってきます。

**先に「ランタイム → ランタイムのタイプを変更 → GPU」**。無料版では T4 が割り当てられます。

一番上の「設定」セルで指定できるもの:

| 設定 | 意味 |
|---|---|
| `DRIVE_DIR` | ドライブのフォルダ（手順1で作ったもの） |
| `SINCE` / `UNTIL` | **学習に使う対戦の期間**。空なら全部。集計もこの範囲で出ます |
| `OPPONENT` | 学習に使う相手デッキ。空なら全部、`ogerpon_mono` ならオーガポン戦だけ |
| `LEARNING_RATE` | **学習率**。標準 5e-6。大きくすると速く変わるが、元の強さを壊しやすい |
| `EPOCHS` | 同じデータを何周するか（2 くらい） |
| `L2SP` | 元のモデルからどれだけ離れてよいか。大きいほど元のまま（標準 1e-3） |
| `GPU_PROFILE` | `auto` / `T4` / `L4/A100` / `CPU` |
| `MODEL_NAME` | 保存する名前（ドライブの `models/<名前>`） |

GPU の切り替えは中身が違います:

| GPU | 精度 | なぜ |
|---|---|---|
| T4（無料版） | fp16 ＋ ロススケーリング | T4（Turing）に bf16 の演算器が無く、bf16 を選ぶとエミュレーションで遅くなるだけ。fp16 は勾配（1e-4 くらい）がそのままだと 0 に潰れるので、ロススケーリングを自動で入れます |
| L4 / A100 | bf16 | 数値の範囲が広く、ロススケーリング無しで安定 |
| CPU | fp32 | GPU が無いときの最後の手段（とても遅い） |

どのモードでも**重みは fp32 のまま**です（bf16 のまま学習すると、学習率 5e-6 の更新が
まるめられて消えてしまい、モデルが動かなくなります — このプロジェクトが実際に踏んだ罠です）。

`--max-minutes` を付けてあるので、無料版のセッションが切れそうな時間で打ち切っても
**そこまでの結果は必ず保存**されます。

学習が終わると `plan-conformance before / after` が出ます。これは
「その場面で人間と同じ手を選べた割合」で、上がっていれば人間のプレイを覚えたということ。
**ただし勝率が上がったという意味ではありません**（次で確かめます）。

### 5. 学習したモデルで対戦する

ドライブの `models/<名前>` をパソコンにダウンロードしてから:

**(a) 人間と対戦する**

```bash
# macOS / Linux
PTCG_MODEL=~/Downloads/kenkyu_r1 python play_server.py
# Windows (PowerShell)
$env:PTCG_MODEL="C:\Users\me\Downloads\kenkyu_r1"; python play_server.py
```

`/manage` で **ai_agent = `lm_dusknoir`** / **ai_deck = `dragapult_dusknoir`** を選び、
human_deck は好きなデッキ（`ogerpon_mono` にすれば「自分が育てたAI」と戦えます）。
`PTCG_MODEL` を指定しなければ公開モデルが自動でダウンロードされます。
モデルが読めなかったときは、そのままヒューリスティックエンジンとして対戦できます（不戦敗しません）。

**(b) エンジンと対戦させて勝率を測る／モデル同士を比べる**

```bash
python tools/kenkyu/battle_eval.py --model engine --games 200 --tag heuristic   # モデル無し
python tools/kenkyu/battle_eval.py --model base   --games 40  --tag base        # 学習前（初回は約750MBのダウンロード）
python tools/kenkyu/battle_eval.py --model ~/Downloads/kenkyu_r1 --games 40 --tag r1
python tools/kenkyu/battle_eval.py --compare --baseline base
```

出力はこの形です（数字は**書き方の例**で、実際の値はあなたの測定結果に入れ替わります）:

```
=== モデル性能比較 (同じ相手・同じデッキ・先攻後攻を交互) ===
tag              opponent            games   wins     win% 95% CI           runs
r1               ogerpon_mono           40     14    35.0% 21.8 - 50.9      1
base             ogerpon_mono           40     10    25.0% 14.2 - 40.2      1
heuristic        ogerpon_mono          400     12     3.0%  1.7 -  5.2      2

--- base (40 games, 25.0%) との差 ---
  r1                +10.0pt   p=0.313   差は誤差の範囲
```

（`heuristic` の行だけは、この環境で実際に測った値です。`--tag` が同じ測定は合算されるので、
`runs` が 2 になっています。）

- 結果は 1 つのファイル（`evaluations/kenkyu_results.json`、Colab では `--results` でドライブ）に
  たまっていくので、**何個モデルを作っても後から比べられます**。同じ `--tag` で追加すると合算されます。
- 速さの実測（16コアのデスクトップ CPU）: モデルが判断するのは 1試合あたり 10〜60回で、
  1判断 0.7 秒（全コア）〜2.6 秒（`--threads 8`）、1試合およそ 1〜3 分。
  ノートパソコンではもっとかかるので、**試合数を多く回すときは Colab の GPU** が楽です
  （ノートブックの手順7がそれをやります）。エンジン同士（`--model engine`）は 1試合 0.2 秒なので、
  ベースラインは 200〜400 試合まわして構いません。
  同時に2つ評価を走らせるときは `--threads 4` のようにコアを分けてください（分けないと
  奪い合って両方遅くなります）。

### 6. 結果の読み方（ここが研究の本体）

- **40試合では ±15ポイントもぶれます。** 表の「95% CI」が重なっている2つは、まだ差があるとは言えません。
  ちゃんと差を見たいなら 200 試合以上。「差がなかった」も立派な結果です。
- **ベースラインを3つ**とるのが大事です:
  ①モデル無し（ヒューリスティック）②学習まえの公開モデル ③学習したモデル。
  ①より②が低いこともあります（このプロジェクトの本番でも実際にありました）。
  参考: 本番のコンペ期間中、この公開モデルの ogerpon_mono に対する勝率は **17〜21%** でした
  （別マシン・別プロトコルでの測定なので、あなたの環境の数字は違って当然です。だから②は
  自分で測り直します）。
- **人間のマネが上手くなること ≠ 強くなること。**
  このプロジェクトでは、人が書いた作戦への一致率を 61.5% → 93.6% まで上げたとき、
  勝率は 18.9% → 5.2% に**下がりました**。だから一致率だけでなく必ず勝率を測ります。
- 学習して弱くなったら、学習率を下げる（1e-6）・`L2SP` を上げる（1e-2）・データを増やす、
  の順で試すと原因を切り分けられます。

### 7. 困ったとき

| 症状 | 対処 |
|---|---|
| `ModuleNotFoundError: No module named 'cg'` | `python tools/kenkyu/setup_local.py` を実行（エンジン未設置） |
| kaggle が 401 / 403 | コンペのページでルールに同意したか、`kaggle.json` の場所を確認 |
| Intel Mac でビルドに失敗 | `xcode-select --install` を実行してから `--build` |
| ドライブにログが増えない | `config.json` の `play.log_mirror` を確認。`tools/kenkyu/sync_logs.py` で手動同期 |
| Colab で `torch.cuda not available` | ランタイムのタイプを GPU に変更してから、上から実行し直す |
| T4 で out of memory | 設定セルの `GPU_PROFILE` を `T4` に固定（`--maxlen 448`）。1行だけ大きすぎる場合は自動でスキップされます |
| 対戦が遅い | 判断の数だけモデルを動かすので CPU では時間がかかります。評価は Colab の GPU で |
| Colab でリポジトリを取得できない（GitHub 側の障害など） | 手順1でドライブに置いた `repo.zip` から自動で展開されます。その zip は手順1の時点のスナップショットなので、家のPCで `git pull` したら `setup_local.py --drive ...` を実行し直してください |

---

Kaggle「pokemon-tcg-ai-battle」向けの対戦エージェント開発リポジトリ。ルールベースのヒューリスティックエンジン（`agents/engine_v2.py`）と、その自己対戦データで学習する軽量な学習エージェント（提出用）を開発している。デッキ一覧・引用元は本ファイル末尾を参照。

## ヒューリスティックエンジン（`agents/engine_v2.py`）

ルールベースの対戦AI。本リポジトリの中核で、次の**3つの役割**を持つ:

1. **提出時のデフォルトパイロット**（学習エージェント無しでも単体で対戦可能）
2. **学習エージェント（LM/reranker）のフォールバック**（スコアリングが失敗・エラー時に engを呼び、常に合法手を返す＝不戦敗しない）
3. **学習データ源**（自己対戦で意思決定ログを生成 → `tools/build_rerank.py` / `build_sft.py` が学習データ化）

**意思決定のしくみ（雑な説明）**: 合法手を種類（`OptionType`）で分類し、**優先度ラダー**で上から順に決める — `lethal`（勝ち筋KO）→ `ability` → `evolve` → `bench` → `trainer` → `attach` → `attack` → `retreat` → `end`。カード重要度は `agents/tuning.json` の **`card_roles`（win / engine / line / fuel / tech の5段階）** でサーチ・エネ付け・昇格先を制御。デッキ固有のコンボは **62デッキ分の個別ポリシー（L2）** で処理する。設計・アルゴリズムは `docs/`（`engine_v2_spec.md`, `engine_v2_algorithms.md`, `deck_archetypes.md`, `card_roles_guide.md`, `l2_*.md`）に記録。

**作成方法**: **Opus 4.8 との反復的な壁打ち**で作成。以下の**3つの信号源を突き合わせ**ながら、デッキごとに構築とプレイングを詰めた:

- **自己対戦**: 全デッキ総当たりの勝率計測（信頼下限 **150ゲーム/対戦**、`tools/_sample.py`）
- **ライブ対戦履歴**: Kaggleの対戦リプレイから相手の60枚を復元（`tools/scout_decks.py`）、勝敗・敗因の分析（`tools/_analyze_matches.py`）、ライブ→ローカルログ変換（`tools/export_live_logs.py`）、メタ分布の集計（`tools/leaderboard_distribution.py`）
- **人間の大会レポート**: LimitlessTCG の入賞リスト・デッキガイド（NAIC 2026 / 日本シティリーグ）で構築とプレイング（例: 加速エンジン、壁・スプレッド運用）を補正

**壁打ちで得た主な教訓（補足）**:

- **自己対戦は一部デッキを過大評価する**（例: `alakazam` は自己対戦とライブで最大 ~48pt 乖離）。そのため**ライブKaggleスコアを最終的な真の指標**とし、人間レポートは自己対戦では測れない構築・プレイングの補正に使う。
- **少数ゲームは分散が大きい**ため 150ゲーム下限を設け、ノイズによる誤結論を防ぐ（過去に同一コードで 160ゲームでも勝率が 8pt 動いた実績）。
- デッキの「弱さ」には **(a) 構築の問題（エンジン不足・エネ基盤）** と **(b) プレイングの問題** の2種があり、前者はカードの入替、後者はポリシーで対応。自己対戦で「勝ち筋が場に出て攻撃もしているのに勝てない」場合は構築側を疑う。

## エージェント性能比較

**評価プロトコル**（"fair protocol" ＝ 旧 Qwen3.5 SFT の `sft_v34_bal` と同一構成に固定。直接比較のため）:

- **操作デッキ（pilots, 学習エージェントが操作）**: `mega_lucario` / `alakazam_nz_fez` / `crustle_stall`
- **相手（opponents, ヒューリスティック `engine_v2` が操作）**: `alakazam` / `crustle` / `dragapult`
- **試合数**: 9組（pilot×opponent） × 30ゲーム（先後入替）＝ 計 **270ゲーム**
- **指標**: 学習エージェントの対 `engine_v2` 勝率（%）
- **注意**: (1) 30ゲーム/セルは分散大（1セル ±~14pt、全体 ±~3pt。150ゲームの信頼下限未満＝方向性の目安）。(2) 相手はローカルのヒューリスティックで、**ライブKaggleレーティングとは別物**。

**全体勝率**:

| モデル | 方式 | サイズ | 全体勝率 | 提出可否 |
|---|---|---|---|---|
| **gte-reranker-modernbert**（＋multi-pick学習） | エンコーダ（候補選択） | **149M** | **56.7%** | ✓（INT8 ~150MB）|
| Qwen3.5-0.8B | デコーダ（生成） | 0.8B | 56.7% | ✗（語彙25万→523MB, 197MB上限超過）|
| gte-reranker-modernbert（single-pickのみ） | エンコーダ（候補選択） | 149M | 51.9% | ✓（INT8 ~150MB）|
| LFM2-350M | デコーダ（生成） | 350M | 35.6% | ✓（Q3_K_M ~187MB）|

**セル別勝率（%, pilot vs opponent）**:

| pilot vs opponent | LFM2-350M | Reranker-149M<br>(single) | Reranker-149M<br>(＋multi-pick) | Qwen3.5-0.8B |
|---|---|---|---|---|
| mega_lucario vs alakazam | 3.3 | 13.3 | 23.3 | 23.3 |
| mega_lucario vs crustle | 36.7 | 60.0 | 43.3 | 60.0 |
| mega_lucario vs dragapult | 53.3 | 66.7 | 63.3 | 66.7 |
| alakazam_nz_fez vs alakazam | 10.0 | 30.0 | 63.3 | 33.3 |
| alakazam_nz_fez vs crustle | 10.0 | 10.0 | 26.7 | 13.3 |
| alakazam_nz_fez vs dragapult | 30.0 | 56.7 | 76.7 | 86.7 |
| crustle_stall vs alakazam | 50.0 | 66.7 | 56.7 | 63.3 |
| crustle_stall vs crustle | 60.0 | 63.3 | 73.3 | 76.7 |
| crustle_stall vs dragapult | 66.7 | 100.0 | 83.3 | 86.7 |
| **全体** | **35.6** | **51.9** | **56.7** | **56.7** |

- **Reranker-149M（＋multi-pick）**: 現在の本命。**提出不可の Qwen3.5-0.8B（0.8B）と同じ 56.7% を 149M で達成**。single-pick のみの版から +4.8pt。
- **Qwen3.5-0.8B**: **提出不可**（語彙25万で提出上限197MB超）。上限の参考値。
- **Reranker**（`Alibaba-NLP/gte-reranker-modernbert-base`）: 自己対戦の勝者の各意思決定を listwise で学習した cross-encoder。各合法候補を (状態, 候補) でスコアリングし argmax で選択。
- **LFM2-350M**: 提出可能なデコーダだが、このサイズ帯では能力不足。

## 開発履歴

- **2026-07-24**: 提出用 LFM2-350M を現行エンジンの自己対戦データで SFT → fair **35.6%**（mirror 20.6%）。Qwen3.5(56.7%)比で明確に能力不足と判明。
- **2026-07-24〜25**: 「意思決定＝候補からの選択」である点に着目し、デコーダ生成から**エンコーダ・リランカー**（`gte-reranker-modernbert-base` 149M）へ方針転換。listwise softmax-CE で SFT（`tools/build_rerank.py` / `train_rerank.py` / `eval_rerank.py`）。
- **2026-07-25**: reranker（single-pick）**fair 51.9%** を記録。149M で LFM2-350M を +16pt 上回り Qwen3.5 に肉薄。
- **2026-07-25**: **multi-pick 決定の学習追加**。従来は single-pick（1択）決定のみ学習し、複数枚選択（デッキサーチの選択、ベンチ選択、コスト捨て札など）は未学習だった。推論と同一の逐次分解（`multipick_substate` で部分状態を作り、残り候補＋`stop` 候補から1つずつ選ぶ）で教師データを展開し、single 118万件＋multi-pick 35万件＝**152万件**で single-pick モデルから warm-start 再学習。
  - 学習指標: 混合 eval top1 **61.7%**（single専用モデルの single-only 57.6% を、より難しい混合条件で上回る＝multi-pick 習得と single-pick 維持を両立）
  - **fair 51.9% → 56.7%（+4.8pt）で成功**（劣化なし基準を満たし、さらに向上）。提出不可の Qwen3.5-0.8B と同値に到達。
  - 内訳では `alakazam_nz_fez` が大きく改善（vs alakazam 30→63.3、vs crustle 10→26.7、vs dragapult 56.7→76.7）。この構築はエネルギー/サーチの複数枚選択が多く、未学習箇所が直接ボトルネックだったと解釈できる。逆に `mega_lucario vs crustle`（60→43.3）など下がったセルもあるが、30ゲーム/セルの分散（±~14pt）内。

---

# デッキ一覧と引用元

`decks/` に収録している構築デッキの引用元をまとめたものです。

- **対象レギュレーション**: ポケモンカードゲーム「スタンダード」（2026年4月ローテ後・メガシンカ環境）。本コンペのカードプール（`data/JP_Card_Data.csv`）に一致します。
- **収集日**: 2026-07-05
- **主な出典**:
  - **LimitlessTCG NAIC 2026**（North America International Championships、2026年6月10日開催）
  - **LimitlessTCG 日本シティリーグ**（2026年5月開催が中心）
- 各リストは実在の入賞デッキです。ただし本コンペのプールに無いカード（主に CRI セットのカードや `Special Red Card` 等）は、**プール内の同等カードに差し替え**て収録しています（詳細は末尾「収録上の注意」）。
- **現在のデッキ数は 62**（2026-07-05 の初期収集分に加え、以降リーダーボード・スカウトやチューニングで追加）。下記の「NAIC 2026」「日本シティリーグ」が初期収集分、「後日追加したデッキ」が以降の追加分です。

## NAIC 2026（2026-06-10）

| デッキ (ファイル) | アーキタイプ | 順位 | 引用元 |
|---|---|---|---|
| `lillies_clefairy` | リーリエのピッピex | 1位 | https://limitlesstcg.com/decks/list/28249 |
| `dragapult_dusknoir` | ドラパルトex＋ヨノワール | 2位 | https://limitlesstcg.com/decks/list/28236 |
| `dragapult` | ドラパルトex（純正） | 3位 | https://limitlesstcg.com/decks/list/28250 |
| `slowking` | ヤドキング | 4位 | https://limitlesstcg.com/decks/list/28251 |
| `crustle` | イワパレス | 5位 | https://limitlesstcg.com/decks/list/28252 |
| `dragapult_blaziken` | ドラパルトex＋バシャーモex | 6位 | https://limitlesstcg.com/decks/list/28253 |
| `rockets_mewtwo` | ロケット団のミュウツーex | 7位 | https://limitlesstcg.com/decks/list/28254 |
| `ogerpon_box` | オーガポンバレット | 18位 | https://limitlesstcg.com/decks/list/28262 |
| `hydrapple` | カミッチュex | 25位 | https://limitlesstcg.com/decks/list/28266 |
| `raging_bolt` | タケルライコex＋オーガポン | — | https://limitlesstcg.com/decks/list/27922 |

## 日本シティリーグ（2026年）

| デッキ (ファイル) | アーキタイプ | 順位・会場・日付 | 引用元 |
|---|---|---|---|
| `mega_lucario` | メガルカリオex | 1位・滋賀・05/06 | https://limitlesstcg.com/decks/list/jp/71992 |
| `mega_starmie` | メガスターミーex | 3位・滋賀・05/06 | https://limitlesstcg.com/decks/list/jp/71993 |
| `mega_gardevoir` | メガサーナイトex | 4位・東京・05/05 | https://limitlesstcg.com/decks/list/jp/71711 |
| `metagross` | ダイゴのメタグロスex | 11位・新潟・05/06 | https://limitlesstcg.com/decks/list/jp/71971 |
| `manectric` | メガライボルトex＋メガジュカイン | 14位・北海道・05/06 | https://limitlesstcg.com/decks/list/jp/71849 |
| `mega_absol` | メガアブソルex＋エレザード | 3位・北海道・05/06 | https://limitlesstcg.com/decks/list/jp/71840 |
| `mega_gengar` | メガゲンガーex＋マイオ | 16位・岡山・05/06 | https://limitlesstcg.com/decks/list/jp/71928 |
| `mega_froslass` | メガユキメノコex＋ヨノワール | 3位・愛知・05/06 | https://limitlesstcg.com/decks/list/jp/72009 |
| `mega_latias` | メガラティアスex＋バシャーモ | 10位・静岡・03/22 | https://limitlesstcg.com/decks/list/jp/66138 |
| `mega_dragonite` | メガカイリューex＋ハピナス | 13位・神奈川・04/28 | https://limitlesstcg.com/decks/list/jp/70221 |
| `mega_diancie` | メガディアンシーex＋ヨノワール | 10位・東京・05/04 | https://limitlesstcg.com/decks/list/jp/72090 |
| `mega_feraligatr` | メガオーダイルex | 11位・福岡・05/03 | https://limitlesstcg.com/decks/list/jp/71239 |
| `ethan_hooh` | イーサンのホウオウex | 6位・北海道・05/02 | https://limitlesstcg.com/decks/list/jp/71421 |
| `flareon` | ブースターex＋ヨルノズク | 12位・愛知・05/06 | https://limitlesstcg.com/decks/list/jp/72018 |
| `archaludon` | ジュラルドンex | 13位・福井・05/06 | https://limitlesstcg.com/decks/list/jp/71961 |
| `cynthia_garchomp` | シロナのガブリアスex＋ロズレイド | 5位・岐阜・05/06 | https://limitlesstcg.com/decks/list/jp/72074 |
| `mamoswine` | マンムーex＋フーディン | 14位・大阪・04/14 | https://limitlesstcg.com/decks/list/jp/68513 |
| `iono_bellibolt` | ナンジャモのビリリダマex | 16位・石川・05/02 | https://limitlesstcg.com/decks/list/jp/70895 |
| `hop_zacian` | ホップのザシアンex＋トレビ | 3位・岡山・05/06 | https://limitlesstcg.com/decks/list/jp/71916 |
| `hydreigon` | ヒドライドンex＋メガサメハダーex（悪） | 7位・愛媛・04/19 | https://limitlesstcg.com/decks/list/jp/69451 |
| `omatsuri` | おまつりおんど（カミッチュ連打コンボ・草／Tier3） | 2位・岐阜・05/06 | https://limitlesstcg.com/decks/list/jp/72071 |
| `alakazam` | フーディン＋ノココッチ（超／Tier3） | 2位・北海道・05/06 | https://limitlesstcg.com/decks/list/jp/71839 |
| `ceruledge` | セグレイブex＋ソルロック | 9位・大阪・05/06 | https://limitlesstcg.com/decks/list/jp/71985 |
| `trevenant_control` | ホップのオーロット（グッズロック）＋ブラッドムーン | 3位・香川・05/06 | https://limitlesstcg.com/decks/list/jp/72061 |
| `volcanion_box` | ピカチュウex＋オーガポン等トールボックス | 4位・北海道・05/06 | https://limitlesstcg.com/decks/list/jp/71841 |

## 後日追加したデッキ（2026-07-07 以降）

初期収集後に、公開リーダーボードのスカウト（`tools/scout_decks.py` で対戦リプレイから相手の60枚を復元）や、既存デッキのチューニング派生として追加したもの。**出典が単一の入賞URLでないもの（LBスカウト・当方の派生版）はその旨を明記**しています。

### リーダーボード・スカウト由来（実在の上位プレイヤーの構築）

| デッキ (ファイル) | アーキタイプ | 出典（LB順位・プレイヤー等） |
|---|---|---|
| `rockets_spidops` | ロケット団のスパイダース＋ミュウツーex | 公開LB #3 THIRD PTCG Club |
| `crustle_geco` | イワパレス＋メガガルーラ（gecogeco型・壁グラインド） | 公開LB #12 gecogeco |
| `crustle_stall` | イワパレス壁＋コーンストーンオーガポンex | 公開LB #7 LiamKirwin |
| `marnie_grimmsnarl` | マリィのオーロンゲex（悪スプレッド） | 公開LB top-12（#5 Yushin Ito / #12 渡邊征央） |
| `okidogi_box` | オコリザル箱（闘トールボックス） | 公開LB top-100 #1 Majkel1337 |
| `comfey_yveltal` | キュワワー＋イベルタル（悪/超ディスラプト） | 公開LB top-100 #7 koga_poke |
| `mega_venusaur` | メガフシギバナex（草進化） | 公開LB top-100 #29 tw_shin |
| `cubchoo_control` | ツンベアー（エネ拒否ソフトロック） | 公開LB top-100 #88 XP3RiX |
| `chandelure` | シャンデラ（Mind Ruler）＋キュワワー | 公開LB Comfey系の Chandelure 型を分離収録 |
| `rockets_honchkrow` | ロケット団のドンカラスex（Rocket Feathers グラインド） | https://limitlesstcg.com/decks/list/26267 （Vanoverschelde型） |

### 人間 Tier-1 だがライブでは低評価（AIピロット検証用）

| デッキ (ファイル) | アーキタイプ | 出典・位置づけ |
|---|---|---|
| `mega_lopunny` | メガミミロップex（Gale Thrust） | 2026-06 スタンダード Tier-1。LBでは低勝率（例: lmaffei #15 ~33%） |
| `ns_zoroark` | Nのゾロアークex | 人間 Tier-1 だがライブほぼ死に（LB shg195 #473） |

### 既存デッキのチューニング派生（当方作成、単一出典なし）

| デッキ (ファイル) | ベース | 内容 |
|---|---|---|
| `alakazam_nz` | `alakazam`（jp/71839） | Night Stretcher 採用の LB コンセンサス版 |
| `alakazam_nz_fez` | `alakazam` | ↑＋フェザンディ tech |
| `alakazam_xero` | `alakazam` | ↑＋ゼイユ/Xerosic tech |
| `mega_lucario_hg` | `mega_lucario` | 提出版の一つ（ライブ 566.7） |
| `mega_lucario_hilda` | `mega_lucario` | 提出版の一つ（ライブ 542.9） |
| `mega_lucario_tr` | `mega_lucario` | ロケット団混成版 |
| `mega_lucario_ctrl` | `mega_lucario` | コントロール寄り版 |
| `mega_lucario_lb2` | `mega_lucario` | LB上位構築への追随版 |

### 現スタンダードの基礎的な単体ex系デッキ

デッキ内容を確認したところ、いずれも**現スタンダードの実在アーキタイプ**（有力な ex を主軸にした基礎的な構築）。特定の入賞リストのコピーではなく、有力 ex ＋汎用コンサルの「基礎テンプレ」に近い。特に `black_kyurem` / `zangoose` / `mega_heracross` は **「単体 ex ＋ Dunsparce×3 ＋ Fezandipiti ex×1」という同一の基礎コンサル骨格**を共有する。

| デッキ (ファイル) | 主軸ポケモン | 備考 |
|---|---|---|
| `mega_zygarde` | メガジガルデex＋Binacle/Barbaracle（エネ加速） | 2026スタンダードの有力デッキ（闘ボックス）※web確認済 |
| `mega_heracross` | メガヘラクロスex（基本） | 2026スタンダードの競技デッキ ※web確認済。基礎コンサル型 |
| `black_kyurem` | ブラックキュレムex（基本） | 単体アタッカー＋基礎コンサル |
| `zangoose` | ザングースex（基本） | 単体アタッカー＋基礎コンサル |
| `doublade` | ギルガルド（ヒトツキ→ニダンギル→ギルガルド, Honedge→Doublade→Aegislash）＋ゲノセクトex | 鋼・進化ライン |
| `staryu` | メガスターミーex（Staryu 進化）＋Cinderace | `mega_starmie` の別構築 |
| `garchomp_lucario` | シロナのガブリアスex＋メガルカリオex | `cynthia_garchomp` の派生（ガブリアス＋ルカリオ混成） |

> いずれも `library.save_deck` の検証（60枚・ACE≤1）を通過。特定入賞URLはメモリ未記録だが、上記の通り実在の現スタンダード・アーキタイプ。

> 注: 上記の出典・位置づけは開発メモリ（`.claude/.../memory/`）由来。順位はスカウト時点のもので変動する。

## サンプルデッキ（引用元なし）

| デッキ (ファイル) | 内容 |
|---|---|
| `deck` | リポジトリ同梱のサンプルデッキ（引用元なし） |

> 注: 以前あった `deck_ai`（サンプル）は削除済みです。

## 収録上の注意

- **カードの対応付け**: 各リストのカード名を `data/*_Card_Data.csv` のカードIDへ変換して収録しています（`decks/<名前>.csv` は1行1カードID・60行）。
- **プール外カードの差し替え**: 本コンペのプールに存在しないカードは、以下のようにプール内の同等カードへ置換しています。
  - `Special Red Card`（CRIセット）→ プライムキャッチャー（ACE SPEC重複時は別のドロー・サーチ札）
  - その他 CRI/未収録カード（`Prism Tower`, `Deoxys`, `Pokémon Center Lady` 等）→ プール内の近い役割のカード
  - 中核カードがプール外のデッキ（例: デルフォックス型チャリザード、シンボラー型エンボアー、ゲッコウガ系）は**構築不能のため未収録**。
- **重複除外**: 収録済みデッキと内容が酷似するリスト（idf重み付きコサイン類似度 ≥ 0.60）は多様性確保のため不採用としています。
- すべてのデッキは60枚ちょうど・基本エネルギー以外は最大4枚、というルールを満たすことを検証済みです。
