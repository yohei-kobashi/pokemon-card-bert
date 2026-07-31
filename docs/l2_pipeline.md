# L2 Discovery Pipeline — デッキ改善のシステム化（提案 2026-07-11）

## 0. 目的と制約
- **目的**: v1(legacy) では人手・場当たり的だった「デッキ固有の条件発見 → bespoke 方策化」を、
  **再現可能なパイプライン**にする。汎用の動きは L0/L1 に託し、L2 を legacy 水準へ。
- **制約**: **live 対戦履歴を使わない**。入力は (a) デッキ60枚＋カードDB（静的）、(b) セルフプレイ（動的）のみ。
- **方法論上の核心（今回の教訓）**: WR という最終指標だけでは穴の位置が分からない
  （alakazam で Boss-nuke に注力し、真の欠陥＝エネ誤ルートを見逃した）。legacy は
  「Dawn 0/2963 plays」のような**中間挙動指標**から穴を特定していた。→ **計測を挙動レベルに下げ、
  「症状 → 定型修正」のカタログで規則化する**。

## 1. パイプライン（デッキ1つあたり5フェーズ）

```
P0 静的解析 → P1 計測付きセルフプレイ → P2 ギャップ診断（自動レポート）
   → P3 規則合成（パターン適用）→ P4 受け入れ評価 →（不合格なら P1 へ戻る）
```

### P0. 静的解析（カードDBのみ・自動）
デッキ60枚から**仮説リスト**を機械生成:
- スケール技の実打点式（`for each ...` テキスト → 資源Xに比例）→ **共有知覚 `_expected_dmg` 候補**
- ルール相互作用の棚卸し: エネは進化で持ち上がる／ダメカン配置は軽減無視／サポ1枚制／自己ドロー→山切れ
- サイド算術: 主砲HP vs 環境の典型打点（~200）→ **毎ターンKOされる前提か**（→チェイン仮説）
- 各カードの「想定される使い所」宣言（後で実プレイ率と突き合わせる）

### P1. 計測付きセルフプレイ（テレメトリ・自動）
多様パネル相手（matchup評価・プロセス分離）に L0/L1 パイロットで N 戦し、**挙動テレメトリ**を記録:
| 指標 | 定義 | 何を暴くか |
|---|---|---|
| **カード別プレイ率** | plays / (copies × games) | **死にカード**＝使い方を知らないカード（Dawn 0/2963 方式） |
| **トリガ発火率** | 各 L0/L1 規則の発火回数/試合 | このデッキの状態分布で**発火しない汎用閾値**（手札12-18枚 vs 手札≤6ゲート） |
| **エネ配分先分布** | 手貼り先ポケモン別回数 | **誤ルート**（Dudunsparce 独占 等）を即座に可視化 |
| **状態分布** | 手札枚数・盤面エネ・ベンチ幅のターン別ヒストグラム | 閾値較正の根拠 |
| **非攻撃ターン率** | 攻撃可能だが撃たず/撃てないターン | 消極性・装填不全 |
| **KO後即応率** | 自 active KO → 次の自ターンに攻撃できたか | **チェイン欠如**（140HP主砲問題） |
| **敗因分類** | deckout / サイド負け / 盤面全滅 / タイムアウト | 修正の優先度（山戻し・再建・テンポ） |
| **サイドテンポ** | ターン別 取得サイド曲線（自 vs 相手） | レースの遅れの発生点 |

### P2. ギャップ診断（自動レポート）
P1 の計測を P0 の仮説と突き合わせ、**症状リスト**を出力:
- 「Dawn: プレイ率 0.00 — 想定=ラインのチュートル → **発火条件が死んでいる**」
- 「エネ手貼りの 92% が Dudunsparce — payoff は Alakazam → **誤ルート**」
- 「自 active KO 後の即応率 18% → **チェイン欠如**」
- 「敗因の 31% が deckout → **山戻し規則欠如**」

### P3. 規則合成（パターンライブラリ適用・半自動）
症状ごとに**定型修正**を当てる。カタログは legacy の発見をテンプレート化したもの（§2）。
実装は L2 override（数行〜数十行）。**1パターン=1コミット=1計測**で効果を分離。

### P4. 受け入れゲート
- **挙動ゲート**: 対象の症状指標が改善（死にカード解消・即応率↑・敗因シェア↓）
- **成績ゲート**: matchup/field 評価（プロセス分離・高n）で
  - legacy が存在するデッキ: **≥ legacy**（同条件比較）
  - legacy が無いデッキ: ≥ L1 かつ症状ゲート通過
- 不合格 → P1 の計測に戻り、次の症状へ（WR ではなく**症状の解消**を積む）

## 2. パターンライブラリ（症状 → 定型修正）— legacy の発見のテンプレ化
| # | 症状（P2 が検出） | 定型修正（L2 テンプレ） | legacy での実例 |
|---|---|---|---|
| 1 | スケール技が nominal 評価 | 実打点式を**共有知覚** `_expected_dmg` に昇格（gust/promote/エネ配分が参照） | `_my_active_dmg` の 743=20×hand |
| 2 | チュートル/ドローのプレイ率~0 | 手札閾値 → **所持ベース発火**（「keyカードが盤面/手札に無い」） | Dawn/Hilda |
| 3 | エネ誤ルート（payoff 以外に集中） | **ライン先貼り**（進化で持ち上がる）＋ payoff 優先の attach 表 | Abra 先貼り |
| 4 | KO後即応率が低い | **チェイン**: ベンチの2体目 payoff を予備装填（上限k） | Alakazam×2 装填 |
| 5 | 敗因に deckout | 山戻しカードの条件発火（deck≤N かつ 捨て札に資源≥M） | Sacred Ash |
| 6 | keyカードがトラッシュに落ちて勝ち筋喪失 | **再建ループ**（回収カードの所持ベース発火） | Stretcher/Lana |
| 7 | 手札抱えすぎ/攻撃遅延 | 「手札≤k のみドロー、以外は即攻撃」均衡則 | hand≤6 |
| 8 | gust が価値を逃す | サイド期待値優先（ex=2枚を KO 可能なら優先） | Boss ex優先 |
| 9 | 妨害カードの浪費/腐り | 「相手が撃つ寸前」等の**相手盤面条件**に限定 | Hammer |
| 10 | 無差別ベンチ | ベンチ許可リスト＋上限 | Abra/Dudun/Fez のみ |
（新デッキで新種の症状が出たら、修正をテンプレ化してカタログに追記＝**システムが学習を蓄積**）

## 3. 実装物
1. `tools/telemetry.py` — 任意の policy をラップして P1 指標を JSONL 収集・集計（arena 互換）
2. `tools/l2_report.py` — P0 静的解析 + P1 集計 → P2 症状レポート（Markdown 出力）
3. `docs/l2_patterns.md` — §2 のカタログ（テンプレコード断片つき）
4. デッキ毎の作業記録 `docs/l2_<deck>.md`（症状 → 適用パターン → 計測結果）

## 4. 検証計画（メソッド自体の受け入れ試験）
1. **キャリブレーション（正解既知）**: alakazam に盲検適用し、パイプラインが legacy の発見
   （Dawn死に・エネ誤ルート・チェイン欠如・deckout）を**自動レポートで全て検出できるか**を確認。
   検出できれば方法は妥当。→ パターン適用で legacy 水準（field で legacy±ノイズ）に到達するか実測。
2. **本番（正解未知）**: legacy bespoke が無い/弱いデッキ（52中 bespoke は少数）へ展開し、
   L1 比で有意改善が出るかで**新規価値**を実証。
3. 以降、新デッキ追加時の標準工程として運用（P0→P4 で1デッキ数時間）。

## 5. この設計が v2 の意図に応える理由
- **システム化**: 「人が気づく」を「計測が検出する」に置換。legacy の商才（0/2963 を見た）を常設化。
- **live 非依存**: 全指標がセルフプレイ＋カードDBから計算可能。
- **L0/L1 との整合**: 汎用はそのまま、L2 は症状ベースの最小 override。パターンは全デッキで再利用。
- **知識の蓄積**: 発見はカタログに残り、次のデッキで自動候補になる（場当たりの反対）。

---

# Pipeline v2 (2026-07): lines, bundles, forced probes, honest stopping

Validated end-to-end on ns_zoroark: old pipeline plateaued at Δ−6.2 vs legacy
(convergence-by-noise); v2 produced **Δ+8.7, CI [+3.7,+13.7], n=764 → ACCEPT**.

Changes over v1 (fixes the six failure modes found in the ns_zoroark post-mortem):

- **P0′ (a,c)**: p0_<deck>.json adds `lines` (setup/loop/finisher playbooks),
  `invariants` (measurable board properties with deadlines), and `combo_edges`
  with `individually_negative` flags — cross-card loops are declared as BUNDLES.
- **P0.5 (e)**: mechanism claims backing L2 rules MUST carry forced-choice probe
  evidence (inject choices at the decision point; never infer mechanics from the
  default pilot's behavior — that's how the copy-menu misread happened).
- **P1′ (d)**: tools/p1_telemetry.py measures `line_conformance` (invariant
  satisfaction by deadline, generic evaluators driven by the p0 json) and a
  `loss_postmortem` block — causal drill-down, not just symptoms.
- **P2′ (c,d)**: discrimination rule — conformance↑ but WR→ ⇒ revise the LINE
  (P0′); conformance→ ⇒ fix the implementation (P3′). Legacy-policy reading is a
  sanctioned oracle step (own code, not live data).
- **P3′ (b)**: engine_v2 BasePolicy gains `ladder` (rule-method names evaluated
  before the generic steps — legacy's priority-ladder shape as a primitive) and
  `game_state(ctx)` (per-game scratch: lock flags, designations).
- **P4′ (c,f)**: tools/p4_accept.py — the accept/revert unit is a LINE BUNDLE
  (never a single rule; individually-negative edges would be greedily rejected),
  adaptive n (extend when |Δ|<2SE), 95% CI, and a PROCESS stopping criterion:
  all lines implemented AND conformance ≥ target AND CI-upper ≥ 0 — never
  "WR stopped moving" (that is noise, not convergence).

## v2.1 additions (from the ns_zoroark / ethan_hooh / manectric rollouts —
## generalized, deck-agnostic)

**P1′ decision-point histogram** (`decision_histogram` in p1_telemetry output):
per own turn, did an attack OPTION exist and was one taken — split by the active
body. Two DIFFERENT failure families:
  - `wall_share` / `wall_by_active` (no option at any MAIN visit): arming or
    pivot failure — energy never reaches the front, or an armed body waits on
    the bench behind a dead active (ethan r6: first_attack median 14.5).
  - `declined_share` / `declined_by_active` (option present, not taken): an
    attack-choice GATE is wrong (ethan r4: lethal-only Lava gate + display-0 →
    a loaded Magcargo passed its turn; attacks_flow was insensitive to three
    energy-policy revisions because the bug was not in energy policy).
Note: turn-weighted (long walling games dominate), vs attacks_flow which is
game-weighted; read both.

**P2′ oracle conformance benchmarking** (`p1_telemetry <deck> --pilot legacy`):
run the SAME invariants/metrics on a reference pilot. Interpretation rules:
  - An invariant BOTH pilots fail at similar rates is deck-intrinsic, NOT the
    differentiator (ethan: attacks_flow ~0.2 for both — passivity was intrinsic).
  - The differentiator is the invariant the reference satisfies and you don't
    (ethan: grinder_armed 62.5% vs 47.5% → flip the accel priority).
  - No reference exists (new deck)? Substitute: bundle-A/B the plausible
    variants (e.g. both accel-priority orderings) — same discrimination, paid
    in sims instead of oracle reads.

**P2′ escalation rule**: a symptom metric that is INSENSITIVE to N successive
revisions means the lever is wrong, not the effort — stop revising, get the
decision-point histogram (or the oracle benchmark) before the next P3 round.

**Line-design priors** (recurring across decks; check in P0′ before coding):
  - display-0 bodies whose ONLY attack displays 0 need an explicit fire rule
    (threshold-based, e.g. legacy's "Lava at >=3R"), or they wall silently.
  - a body the accel ability cannot reach (bench-only accel + it sits active)
    starves; the line must say who tanks while the accel target charges.
  - free-pivot resources (rc0 bodies, cost-0-when-empty, Skyliner-class
    abilities, retreat-cost stadiums) usually pay for PROACTIVE pivots; the
    generic engine never pivots a "loaded" active, so lines must.

## v2.2: config-driven L2 (the pipeline WRITES the policy)

- **L0 dynamic damage estimator**: `_opt_atk_dmg`/`_my_active_dmg` resolve
  "for each ..." scalers against the LIVE board (benched both/yours, hand
  sizes, damage counters, attached energy); static text estimate remains the
  card-level fallback. Fleet effect: omatsuri +6.9 (Do the Wave live-valued),
  zygarde/kyurem to parity with no deck code.
- **L0 energy-search supporters** (Crispin class): recognized by text
  ("search your deck for ... basic energy") and played when the hand holds no
  basic energy — thin-energy decks' energy engine was NEVER fired before
  (lillies wall_share 95% root cause).
- **P0.5 `tools/p05_deckconfig.py`**: derives a line config for ANY deck from
  the card DB alone — focus ranking by deck-context value-per-energy (bench-
  widening stadiums re-value benched-scalers; type-payability filter),
  stadiums, bench_target, discard-fuel edges ("attach from discard" trainers).
  `--write` puts it in tuning.json as `l2: "config"`.
- **`ConfigL2`**: ONE generic class consuming that config (FocusL2 doctrine +
  rule_stadium + wide bench + fuel-discard edge). No hand-written per-deck L2.
- Loop: p05 --write → P1' conformance/histogram → P4' bundle accept. Validated:
  lillies_clefairy −8 → **+9.4 ACCEPT** (config + Crispin rule),
  mega_lopunny untouched → **+8.3 ACCEPT in one automated shot**.

## v2.3: exact-counters combo vocabulary

- **L0**: conditional-KO estimator pattern ("exactly N damage counters ...
  Knocked Out" → live value = opp active HP when the condition holds, static
  rank value 250); `decide_count(ctx)` hook (COUNT selects were hardcoded max —
  base unchanged, combo policies override).
- **p05**: derives `combo: {type: exact_counters, n, finisher, finisher_attack}`
  from attack text and `support_energy` (counter-mover abilities, "move up to K
  damage counters ... to 1 of your opponent's") — mega_absol's Terminal-6 line
  found from cards alone.
- **ConfigL2**: N-precision count selection (never overshoot the window),
  counter-move targeting (active while short of N, else stock a fresh bench
  body), mover-first energy/ability-attach semantics.
- Validation: execution metrics 4x (Adrena 14→53/18g, at-exactly-6 3→16,
  Terminal 1→4) — but mega_absol WR stayed ≈−8 vs legacy: the line EXECUTES but
  doesn't CONVERT (finisher must be armed+front exactly when the window opens —
  cross-turn planning beyond config vocabulary; the documented LM-layer
  boundary). absol remains the fleet's single borderline deck.

### v2.3 rev-3: conversion rules (the "cross-turn planning" that was NOT an
### LM-boundary after all — user pushback #3, correct again)

- **L0**: self-switch UTILITY abilities (Subjugating-Chains class, text
  "switch 1 of your Benched ... with your Active") never fire via the generic
  ability step — they randomize the front; ladder rules invoke them
  deliberately.
- **ConfigL2 rule_combo_pivot**: when the exact-N window is OPEN or REACHABLE
  this turn (deficit <= armed-movers' capacity AND our shippable counter
  stock), put the armed finisher in front NOW (free switch ability > Switch
  card > retreat); movers then set N before the attack step.
- **ConfigL2 window discipline**: while building toward N on a body we cannot
  KO outright, never chip its counters past N — exact setter (damage ==
  10*(n-c)) > bench snipe > chip-as-last-resort (anti-passivity preserved).
- Result: mega_absol 21.9% -> 30.9%, Δ−8 -> **−4.2 CI[−9.0,+0.6] ACCEPT
  (parity)**; guards hydreigon +4.2 / marnie +1.6 held. Lesson (recurring):
  before declaring a residual "beyond heuristics", write the DELIBERATE
  sequencing rule — pivot-when-window, don't-break-the-window — they are
  ordinary ladder rules.

## v2.4: conditional damage + support economy at L1

- **L0 estimator**: 5 damage-state conditional clause classes evaluated live
  (opp-active-damaged bonus / base-override (Huge Bite 260->30) / does-nothing
  vs undamaged / self-damaged bonus / self-undamaged bonus) — 16 attacks fleet-wide.
- **Support economy demoted to BasePolicy** (`_read_line`/`_support_attach`
  via step_attach): mover/support lines now work under ANY archetype pilot —
  `p05 --support-only --write` leaves l2 unchanged (for decks where the focus
  doctrine is measured harmful).
- **predamage enabler**: p05 emits `predamage: true` when a payable attack has
  an opp-damaged bonus; generic spread-targeting then chips a CLEAN opponent
  active first (unlocks Mortal-Crunch-class 400s with Munkidori).
- **`support_gate`**: "strict" (default: never feed movers while any attacker
  is one energy short) vs "lenient" (ready-attacker only) — starmie needs
  strict (+2.3 vs −8.2), mover-core decks need lenient (blaziken +7.4 vs +1.6;
  absol −0.2 vs −4.9). A pipeline-tunable config knob, chosen by keep-better.
- Results: mega_feraligatr −3.6→+0.7 (config adopted; the Munkidori-chip →
  Mortal Crunch 400 line converts), crustle_stall +2.0→+7.2 (≥legacy; Adrena
  as the wall-heal engine), hydreigon +2.6→+6.5 (≥legacy), mega_absol → −0.2
  (best ever). metagross reverted to hand (multi-focus toolbox depth).
