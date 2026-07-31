# 先読みを reasoning として使う小型 LM エージェント — 先行研究と提案手法

*対象: pokemon-tcg-ai-battle (cabt) 用の Qwen3.5-0.8B エージェント / 最終更新 2026-07-13*

本ドキュメントは「未来の手(先読み)を reasoning とみなして小型 LM に蒸留する」という本手法を、
関連する先行研究の中に位置づけ、設計判断と実測結果を一箇所にまとめたものである。

---

## 1. 背景と問題設定

- **タスク**: 不完全情報の対戦カードゲーム(52 デッキ、1326 マッチアップ)を、
  **CPU・オフライン・10 分/ゲーム**の制約下でプレイするエージェント。
- **土台**: 3 層ヒューリスティックエンジン v2.4(全 52 デッキで legacy 同等以上)を教師とし、
  自己対戦から SFT データを生成(26,520 ゲーム → **201 万模倣サンプル**)。
- **目標モデル**: `Qwen3.5-0.8B`(0.752B)。GGUF Q4 + llama.cpp、KV 再利用で **~5ms/手**。
- **中心アイデア**:「今後の展開を **reasoning** して手を決める」を実現したい。ただし
  - reasoning は**自然言語である必要はなく**、ドメイン記号(手の系列)でよい、
  - コンペの時間制約上、**推論時に先読みを生成することは不可**(後述モード b)。

この 2 点が、以下の先行研究の合成として自然に解ける。

---

## 2. 先行研究

「探索/先読みを reasoning に使う」研究は 4 つの系統に整理できる。

### 2.1 探索トレースを reasoning トークンとして系列化する
- **Searchformer / "Beyond A\*"** (Lehnert et al., Meta 2024). transformer に **A\* 探索のダイナミクス**
  (探索木への状態の追加/削除)をトークン列として模倣させ、その後 **expert iteration** でより短い
  トレースへブートストラップ。倉庫番を 93.7% 最適解、A\* より最大 26.8% 少ない探索ステップ。
- **Stream of Search (SoS)** (Gandhi et al., Stanford/MIT 2024). 探索軌跡(探索・バックトラック・枝刈り)
  を文字列に系列化し、LM に**言語内で探索**を学習させる(ゲーム Countdown)。
  **知見: 失敗を含む "劣った" 探索軌跡で学習した方が、最適解のみより強い**(+25%、
  方策改善で未解決の 36% を解決)。

→ 「未来ロールアウトを reasoning ターゲットにする」本手法の最も直接的な前例。

### 2.2 探索を方策改善としてネットワークに蒸留する(AlphaZero 型)
- **AlphaZero / Expert Iteration** (Silver et al. 2017; Anthony et al. 2017). 探索(MCTS)を
  **方策改善オペレータ**として使い、探索で改善した方策へネットワークを学習。推論時はネットワーク単体で強い。
  Searchformer の bootstrapping はこの探索トレース版。

→ 本手法の **act ラベル改善**(探索が教師 v2 の手を上書き)に対応。

### 2.3 LLM を世界モデル+先読みに使う「reasoning = planning」
- **RAP: Reasoning via Planning** (Hao et al., EMNLP 2023). LLM を**世界モデルと行動主体の両方**に流用し、
  推論中に未来状態を **MCTS で先読み**。「reasoning とは世界モデルによる planning」。
  LLaMA-33B+RAP が plan generation で GPT-4 の CoT を上回る。

→ 「先を読んで手を決める」を主流の**先読み型 reasoning** として位置づける枠組み。

### 2.4 探索を学習し推論時は生成しない(内在化 / 暗黙 reasoning)
- **Implicit / Latent CoT, Stepwise Internalization** (Deng et al. 2024), **COCONUT**, **KaVa / CODI** ほか.
  **reasoning つきで学習し、推論時は中間を出さず答えを直接生成**。低遅延で効果の多くを保持。

→ 本手法の **モード b**(reason/compare で学習、推論時は行動を直接デコード)そのもの。

### 2.5 小型モデルで「探索を蒸留し推論時は直接行動」した実証
- **Grandmaster-Level Chess Without Search** (Ruoss et al., DeepMind 2024). **2.7 億パラメータ**の
  transformer を Stockfish の action-value で教師あり蒸留し、**推論時に探索なし**で
  Lichess ブリッツ Elo 2895(GM 級)。「大きめの transformer は探索ベースの価値推定を
  feed-forward に蒸留できる」。**0.8B より小さいモデルでの直接前例**(完全情報の類似課題)。

---

## 3. 提案手法

**探索蒸留による対照 SFT データ生成 + マルチタスク学習 + 推論時は行動直接デコード。**
上記 4+1 系統の合成を、不完全情報カードゲームで実装する。

### 3.1 全体像

```
自己対戦 (v2×v2)                 各 MAIN 決定点
  │                                  │
  ├─ v2 の手 / legacy の手 ──── ケース判定
  │      不一致(48%) → 対照ペア = (v2手, legacy手)          [case2]
  │      一致        → 対照ペア = (合意手, ランダム他手) 低率  [case1]
  │
  └─ 各ペアを裁定:
        同一 determinized ルートから A/B を fork          … cg ネイティブ探索 API
        v2 で両プレイヤーを 3 ターン先までロールアウト     … 先読み = reasoning
        ルールベース評価器で結果状態を採点                … 勝利優先 override
        3→5 回投票, ≥4/5 有意で採用・同点破棄            … determinization 分散を平均化
        └─→ 出力
             [ACT]     : prompt → 探索検証済みの手     (方策改善ラベル)
             [COMPARE] : prompt → "A:<枝> | B:<枝> => CHOOSE X ACT 手"  (対照 reasoning)
```

### 3.2 構成要素

1. **候補生成(v2 ⊕ legacy)**: 2 つの強ヒューリスティックの**不一致点**(実測 **47.9%**)を
   「難しい決定点」の検出器兼候補生成器として使う(active-learning 的な絞り込み)。
2. **反事実ロールアウト(先読み = reasoning)**: cg エンジンの**ネイティブ探索 API**
   (`search_begin/step/release`)で、同一決定点から A/B を独立に fork。隠れ情報は
   **両デッキ既知**なので determinization(既知構成 − 可視カード)で供給。
   ロールアウトは v2 が両プレイヤーを駆動し **3 ターン先**まで(§4 で最適と確認)。
3. **ルールベース評価器**: prize 差を辞書順支配、**勝利状態を ±無限大で絶対最優先**
   (終局 100% 正答)。弱点 ×2 補正の KO 脅威・進化段階テンポ・蓄積ダメージ・prize 価値を加味。
4. **分散低減と裁定**: A/B は同一 determinization を共有(共通乱数法)。投票は 3 回、
   割れたら +2 回、**同一側が ≥4/5 で有意(|Δ|>margin)** でなければ「差なし」で破棄。
   → 引きに依らず**一貫して優れた手のみ**が残る(低歩留まり・高品質)。
   コイン運は本プールにコイン技が無いため無視可(manual_coin 不要)。
5. **二重出力**:
   - `[ACT]` = 探索検証済みの手(**方策改善ラベル**、2.2/2.5 に対応)。
   - `[COMPARE]` = A/B 2 枝の未来ロールアウト + 裁定(**対照 reasoning**、2.1 に対応)。
6. **推論(モード b, 2.4 に対応)**: 学習は act/reason/compare のマルチタスク、
   **推論時は `[ACT]` を直接デコード**(reasoning を生成しない)→ 反復ループ回避・低遅延。
7. **既存模倣データとの統合**: build_sft の 201 万サンプルと**混合**。形式の衝突を避けるため
   3 つの一貫タスクに分離:`[ACT]`(手)/ `[REASON]`(単一実現未来, build_sft)/
   `[COMPARE]`(A/B 対照, 本手法)。

### 3.3 トークナイザ(補足)
形式言語のため scratch BPE で **~9×** 圧縮可(unk 0・可逆)。ただし Qwen 事前学習を活かすため、
scratch 語彙を **Qwen に追加**しても **~8×** を維持でき、新トークンはサブピース平均で温初期化する
経路が有力(別途検討)。

---

## 4. 実測結果

| 項目 | 結果 |
|---|---|
| v2/legacy 不一致率 | 47.9%(勝者手 54.9%) — 対照ペア豊富 |
| リプレイ再現性 | 100%(測定パイプライン健全) |
| 評価器 終局正答 | 100%(勝利優先 override) |
| 評価器 中盤/終盤判別 | 50%→64%, 75%→77%, 最終→90%(序盤は prize 未変動で本質的に未確定) |
| 局所トレンド(使用場面相当) | 78.9% |
| **地平の最適** | h2: 1.6% / **h3: 30%** / h5: 21% 採用 → **3 ターン**(有用データ/秒が最良) |
| 採用時の方策改善 | **採用の 75% で探索が v2 手を上書き**(= 実効的な policy improvement) |
| 歩留まり | 2.72 採用決定/ゲーム、5.44 サンプル/ゲーム、5.95 秒/ゲーム(単核) |

> **SoS の知見との整合**: 地平 2→3 で採用率が跳ね、5 で低下。不完全情報では地平を延ばすと
> determinization 分散が増大し符号反転が増える。「短すぎず深すぎず(3)」が最適という結果は、
> 「劣った/現実的な探索軌跡が有効」という SoS の観察と整合的。

---

## 5. 本手法の位置づけ:借用と新規性

| 系統 | 借用 | 本手法での役割 |
|---|---|---|
| Searchformer / SoS (2.1) | 探索/未来を reasoning トークンに系列化 | `[COMPARE]` の A/B ロールアウト |
| AlphaZero / ExIt (2.2) | 探索 = 方策改善の蒸留 | 探索検証済み `[ACT]` ラベル |
| RAP (2.3) | reasoning = 世界モデルによる planning | 「未来手で手を決める」枠組み |
| Implicit/Latent CoT (2.4) | 学習あり・推論なしの内在化 | モード b(行動直接デコード) |
| Searchless Chess (2.5) | 小モデルへ探索を蒸留し推論時直接 | 0.8B 実現可能性の直接前例 |

**新規性**: これらの多くは**完全情報**課題。本手法は **(1) 探索トレース=reasoning + (2) 探索による
方策改善の蒸留 + (3) 推論時は生成しない内在化** を、**不完全情報カードゲーム**で
**determinization ベースの反事実ロールアウト**として合成した点にある。

**明示的トレードオフ**: Searchformer/SoS/RAP は探索を**推論時に生成**して利得を得るが、本手法は
遅延のためこれを**捨て**、内在化(2.4)で代替する。明示 reasoning 利得の一部を失う意図的選択。

---

## 6. リスクと評価計画

- **0.8B の容量**: 生成的 NL CoT は小モデルで困難(2025-26 文献)。ただし本手法は
  **形式領域の探索蒸留+推論時直接**であり、searchless chess(270M)が直接の反例。まず 0.8B、
  容量律速なら 1.5B へ。
- **不完全情報の分散**: 地平を延ばすと引き運が支配。3 ターン+共通乱数+投票破棄で対処済み。
- **reasoning 補助の上乗せ効果は不確実**: `[COMPARE]`/`[REASON]` が act ヘッドを助けるかは経験的。
  → **事前登録アブレーション**で決着:
  **(A) act のみ / (B) act+reason / (C) act+compare** を小規模 SFT で比較し、
  アリーナ勝率(と遅延)で判定。
- **フォールバックの堅牢性**: 仮に reasoning 補助がゼロ効果でも、**`[ACT]` ラベル(探索改善済み方策、
  75% が v2 上書き)は単体で有効**(= searchless chess のレシピ)。生成作業は無駄にならない。

---

## 参考文献
- Ruoss et al. *Grandmaster-Level Chess Without Search.* 2024. arXiv:2402.04494
- Lehnert et al. *Beyond A\*: Better Planning with Transformers via Search Dynamics Bootstrapping (Searchformer).* Meta, 2024. arXiv:2402.14083
- Gandhi et al. *Stream of Search (SoS): Learning to Search in Language.* Stanford/MIT, 2024. arXiv:2404.03683
- Hao et al. *Reasoning with Language Model is Planning with World Model (RAP).* EMNLP 2023.
- Deng et al. *From Explicit CoT to Implicit CoT: Stepwise Internalization.* 2024. / *Latent CoT Reasoning: A Survey.* 2025. arXiv:2505.16782
- Silver et al. *Mastering the game of Go without human knowledge (AlphaZero).* 2017. / Anthony et al. *Thinking Fast and Slow with Deep Learning and Tree Search (ExIt).* 2017.
