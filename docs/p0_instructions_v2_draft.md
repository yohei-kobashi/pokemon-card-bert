# P0 指示書 改訂案 v2（ドラフト・2026-07-11）
根拠: alakazam/marnie 盲検テスト、live 自滅事故（sub 54573105）、P1-P4 収束ループ3周の実測。
現行 `p0_instructions.md` への **7つの修正**。各修正に「由来した実例」を付す。

---

## 改訂1: 仮説に severity（重大度）を必須化 ★最重要
**由来**: 盲検P0は Run Away Draw の危険（H6）を正しく出力していたが、実装者が**無断でスキップ**し
live で自滅負けした。仮説が並列リストだと取捨選択で安全性が落ちる。

- JSON `hypotheses[]` に `severity: "catastrophic" | "major" | "minor"` を必須追加。
  - **catastrophic** = 単独でゲームを失う（自己除去・自滅・山切れ・盤面全滅系）
  - **major** = 勝率に有意に効く（エネ誤ルート・死にエンジン）
  - **minor** = 効率改善
- **契約**: P3 は全仮説を「実装/延期(理由)/棄却(理由)」で**必ず disposition** し、
  catastrophic の未実装は P4 通過不可。P0 出力に disposition 台帳の雛形を含める。

## 改訂2: 「自己害の棚卸し」を必須セクションに昇格（新§2b）
**由来**: 自滅は L0 の「特性は無条件で使う」×自己シャッフル特性で発生。alakazam H6・marnie H6
（Lillie's Determination の手札流し）と、**両盲検が独立に同型の危険を発見**した＝体系化可能。

新設 §2b: 60枚を対象に以下を**明示的に走査**し、該当カードは severity=catastrophic の仮説にする:
- 自身/自軍を場から除去する特性・技（"shuffle/put this Pokémon into your deck/hand"）
- 手札を山に戻す/捨てるサポート（コンボ核を抱えた手で撃つと自壊）
- 自傷ダメージ・自軍への配置（Froslass型）、強制ベンチ・強制入替
- 自己ミル（山を掘るエンジン → デッキアウト地平とセット）
- 各項目に「**発火してよい盤面条件**」（bodies≥N 等）を状態変数で宣言

## 改訂3: expect は「方向＋暫定閾値」の2層に（事前分布として扱う）
**由来**: パリティ到達後も postKO 0.63（期待0.7）・H8 0.58（期待0.7）が「違反」のまま残った。
盲検の数値 expect は推測であり、**後半ループでは修正でなく期待値の再較正が正解**だった。

- `expect` を分割: `direction`（必ず成立すべき向き。例「エネは主にラインへ」）と
  `threshold`（暫定値・`provisional: true`）。
- P2 の判定は direction 違反＝症状、threshold 違反＝「較正 or 修正」の2択として報告。
- 可能なら「L0/legacy の同指標を先に測って基準化せよ」（絶対値の当て推量を避ける）。

## 改訂4: ゲート条件は調整可能パラメータとして書く
**由来**: H6 の「手札≤4」ゲートは実測で **68→58% の退行**（RAD の+3枚＝核の燃料を止めた）。
「盤面≤2 or エネ搭載」への緩和で最良に。**P0 の条件値は仮説であり、強度は P4 で較正される**。

- card_intents / 仮説の数値条件は `param` 表記で範囲を宣言:
  例 `fire_if: bodies >= B (B∈{2,3}), hand <= H (H∈{4, ∞})` — P4 が sweep する前提。
- 「条件を強くしすぎる方向のリスク」（何の燃料を止めるか）を1行で必ず書く。

## 改訂5: 軸間の対立（tension pairs）を必須宣言（§5 拡張）
**由来**: deckout は3周回しても残った——**山を掘るエンジン自体が核の燃料**という構造的
トレードオフで、進化優先の修正（チェイン改善）が deckout を**増やした**。個別軸の採点だけでは
whack-a-mole になる。

- §5 に `tensions: [{axes: [resource_economy, power_curve], balance: "deck>10 の間は掘る", ...}]`
  を必須化。各 tension に「意図するバランス点」を状態変数で宣言。
- P3 への指示: tension に属する指標を修正するときは**対になる指標を同時に監視**。

## 改訂6: 複数ドクトリンの併記と判別実験（§1 拡張）
**由来**: marnie 盲検は legacy と異なる有効プラン（壁＋スワップ vs **その場進化**）を独立に提案した。
1つに決め打ちさせると、より強い代替案が闇に消える。

- §1 勝ち筋: 実行可能なゲームプランが複数あるときは**両方を仕様として書き**、
  `doctrines: [{name, plan, discriminating_metric}]` で**判別する実験（A/B）**まで指定。
- 判別は P4 のメニューに乗る（今回なら swap 型 vs in-place 型の field A/B）。

## 改訂7: catastrophic 仮説には再現シナリオを添付（P4 regression 用）
**由来**: live 自滅の修正検証は「実盤面スナップショット→regression テスト」が決定打だった
（注: visualize リプレイは enum が文字列なので変換が必要、というのも資産化済み）。

- severity=catastrophic の各仮説に `scenario:`（危険が発火する盤面の構成手順:
  「active=Dudunsparce 単騎・ベンチ0・特性オプション提示」）を必須添付。
- P4 はこれをプローブで構築し、修正前=発火/修正後=回避 の2値テストにする。

---

## JSON スキーマ差分（まとめ）
```json
{
 "hypotheses": [{
   "id": "H6", "severity": "catastrophic",
   "claim": "...", "metric": "...",
   "direction": "RADは盤面条件つきでのみ発火",
   "threshold": {"value": "fires<=0 when bodies<=B", "provisional": true,
                  "params": {"B": [2,3]}},
   "risk_of_overtightening": "RADの+3枚=PH+60打点の燃料を止める",
   "scenario": "active=Dudunsparce単騎/bench空/ability提示",
   "violation_symptom": "...", "pattern": "..."
 }],
 "tensions": [{"axes": ["resource_economy","power_curve"],
                "balance": "deck>10 の間は掘る、以下で停止"}],
 "doctrines": [{"name": "wall+swap", "plan": "...",
                 "discriminating_metric": "field WR A/B"}],
 "disposition_ledger": {"H1": null, "H2": null}
}
```

## 校正スイートへの追記（§4）
- **実例(c) alakazam live 事故**を教材化: 「P0 が出した catastrophic 仮説のスキップは
  そのまま live の敗因になった（H6→自滅）。全仮説 disposition が契約である」。
- **実例(d) 収束の3類型**: 綺麗に消える症状／構造的トレードオフで平衡する症状／
  expect 較正が必要なだけの症状——P0 時点で tension 宣言（改訂5）があれば2つ目を予告できる。

## 変更しない点（実証済みで有効）
- 盲検で機能した観点リスト（実打点式・型 vs 供給・進化持ち上がり・状態分布による閾値検査・
  60枚全カードの使用宣言・プローブによる verified/unverified）
- 校正例の anchoring 禁止則、陰性対照（crustle_stall）
- 仮説最低8本・「破られたら適用パターンが決まっている仮説が良い仮説」の定義
