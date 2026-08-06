# RL カリキュラム改訂 — A/B/C（3段）→ 1/2（2段）

2026-08-06 ユーザー決定。`docs/rl_design.md` の A/B/C を置き換える。実装は
`tools/rl_config.py:stage()`、データ生成は `docs/rl_mirror_design.md`。

    旧  A  P=全65, O=一様65      → B  P=全65, O=LIVE_META  → C  P=1デッキ, O=LIVE_META
    新  1  P=11,   O=11（自己対戦）                        → 2  P=dragapult×2, O=11

---

## なぜ A/B を消すのか

**証拠1: 幅広ステージは2回とも平坦だった。** `rl-plateau-five-refutations`（rlDL2、12ラウンド
平坦）と `rl-stage-a-plateau-diagnosis`（平坦であって強弱のトレードオフではない、ゲーム単位の
一様クレジットが原因、ラウンド3倍は検証済みで無効）。Stage A の設計そのものが否定されている。

**証拠2: 幅は RL が供給していない。base が供給している。** 実測（i2_r7 の mix）:

    258,511 行 | base 77.4% | dagger 13.4% | valued 9.2%

base は全65デッキを含む。RL/DAgger 分は 22.6% しかない。これを65デッキに散らすと
1デッキ 0.35%、11デッキに集めると 2.05% で **6倍の濃度**。65デッキに広げることの意味は
「幅を持たせる」ことではなく「効く量を下回らせる」ことだった。**破滅的忘却は構造的に
起きない** — 忘れさせない仕事は base の 77.4% がやっている。

**証拠3: 提出先が11に確定した。** `stage-c-eleven-decks`。65デッキを鍛える理由が消えた。

Stage B（O を LIVE_META に寄せる）は独立ステージとして残す価値がない。新 Stage 1 が
最初から live 重みで対戦相手を引くため、B の中身は 1 に吸収される。

---

## 出発点（decoder i2_r6、mirror screen 40ゲーム）

    marnie_grimmsnarl  55.0   w .358      cynthia_garchomp   47.5   w .038
    alakazam_nz        57.5   w .134      mega_lucario_tr    55.0   w .028
    crustle_geco       57.5   w .057      dudunsparce_box    37.5   w .024
    ogerpon_mono       47.5   w .057      dragapult_dusknoir 57.5   w .000
    crustle            65.0   w .047
    alakazam           57.5   w .040      11デッキ 非加重平均 51.6 / live加重 53.7
    dragapult          30.0   w .040      参考: 全65デッキ平均 53.5

**dragapult が 30.0% で 11 中の最下位**、しかも Stage 2 の対象。2段構成の根拠がここにある —
1 で11デッキを底上げし、2 で最も遅れていてかつ提出する1つを仕上げる。

（DeBERTa 側の同じ screen は 30.6%。Stage 1 開始時点は loop8 の到達点で読み直すこと。）

---

## Stage 1 — 11デッキの自己対戦

**P = 11デッキ**（`STAGE_C_TARGETS`）、**O = 11デッキ**。11×11 のグリッドを回す。

    対角（11マス）    ミラー。両席同一デッキ＋同一シャッフル順（--mirror --seed）
    非対角（110マス）  ladder が実際に起きている形。両席とも11に属するので
                      1ゲームから両側の決定が学習に使える

**重み。** 行（pilot）= ヘッドルーム加重、列（opponent）= LIVE_META を11に制限して再正規化。
対角の比率は **0.25** に固定する（放っておくと重みの積で対角が薄くなりすぎる）。

**ヘッドルームの出所を間違えないこと。** 既存の `_stageA_pilot_weights` は
`rl_ratings`（= engine_v2 の総当たりでのデッキ強度）を読む。これは**デッキの強さ**であって
**LM の操縦の下手さ**ではなく、そのまま使うと配分が反転する:

    デッキ                rl_ratings  LM実測   旧w     正しくは
    dragapult_dusknoir       34.8     57.5%   0.303   最小（LMは2番目に上手い）
    dragapult                46.9     30.0%   0.157   最大（LMの最下位）
    dudunsparce_box           --      37.5%   0.060   2番目（LMの下から2番目）

弱いデッキと下手に操縦されているデッキは別物で（`fleet-roundrobin-and-weak-decks`、
`weak-decks-pilot-vs-structural`）、RL が動かせるのは後者だけ。新規に
`_stage12_pilot_weights` を追加し、**LM 自身の mirror screen**
（`evaluations/lm_mirror_screen.json`）から測る。修正後:

    dragapult 0.267 / dudunsparce_box 0.200 / ogerpon_mono 0.111 / cynthia_garchomp 0.111
    残り7デッキは床の 0.044

screen が無い場合は一様に落ちるが、**stderr に警告を出す** — ここの無言フォールバックは
正常動作と見分けがつかない。

**なぜ対角を残すか。** ミラーは *評価の計器* であって目的ではない。Q ラベルは playout から
来るのでミラーである必要はない（`rl_branch.py` の determinization 共有で足りる）。それでも
0.25 残すのは、(1) ゲートがミラー paired screen であり分布を合わせたい、(2) null が厳密に 0
になる唯一の設定だから（`mirror-shuffle-mode`）。

**なぜ非対角を主にするか。** 目的は ladder であり、ladder は非対角。対角だけで鍛えると
`live-weighted-eval-protocol` の教訓どおり判定が反転しうる。

**学習信号は測定 Q。GRPO ではない。** `docs/rl_mirror_design.md` の Phase 0-3 をそのまま使う。
GRPO の一様クレジットは平坦の診断済み原因なので、同じ賭けを3度目はしない。

**どの決定に予算を使うかは価格付けで決まる（2026-08-06 実測、反映済み）。** pilot 重みが
「どのデッキで対戦するか」を決めるのに対し、`QLABEL_TARGETS`（= `evaluations/
lm_targets_priced.json`）が「どの (deck, kind) の決定を分岐するか」を決める。観測 gap だけで
配分すると外す:

    end     艦隊最大の観測効果（プール z -10.90）だが、最初に測った側では**打ち手がない** —
            LM が「切り上げない」と決めた場面では切り上げないのが正しい（dQ -0.10〜-0.38）。
            反対側には打ち手がある: 切り上げた場面では切り上げるべきでなかった
            （4デッキプール -0.0696、z -4.28）。**早仕舞いであって、粘りすぎではない。**
    evolve  mega_lucario_tr が decline 側で誤りと出た唯一のセル（+0.079, z+2.88 / take 側も
            +0.082 で整合）。観測 gap は「勝者は進化が 29.9pp 少ない」で**逆を向いている**。

生成の詳細（符号×側、kind プール、1セル 25% 上限、dQ_max を使わない理由）は
`docs/rl_mirror_design.md` の Phase 2 に置く。`rl_config.QLABEL_MAX_CELL_SHARE = 0.25` と
`VALUED_MAX_FRAC = 0.10` は、どちらも過去に一度払った失敗（1デッキ集中で艦隊 -2.75pt /
valued 比率が届いたバッチ数で無言に決まる）に対する上限。

---

## Stage 2 — dragapult 2種の特化

**P = {dragapult, dragapult_dusknoir}**、**O = 11デッキ**（live 重み）。20マス
（dragapult_dusknoir は leaderboard 使用ゼロなので相手側には出ない → 2 pilot × 10 opponent）。

Stage 1 との違いは pilot の絞り込みだけで、機構（測定 Q ラベル、Phase 0-3）は同一。

**推奨する逸脱が1つ。** O を11に限ると ladder の 21.4% が視界から消える
（mega_lopunny .028 / mega_lucario .028 / archaludon .024 / omatsuri .018 …）。
**O = 11（LM操縦、live重み）×0.85 + 残りの LIVE_META テール（engine_v2 操縦）×0.15**
を推奨する。テールは engine_v2 が操縦するので追加学習コストはゼロ、11のみとの差分も
ゲートで分離報告できる。ユーザー指定は「11との対戦」なので、これは提案であって既定ではない。

---

## ステージ境界

Stage 1 → 2 は、次のどちらか早い方。ラウンド上限 6。

    (a) 11デッキ paired が 2ラウンド連続で ≤ +1pt（下記 SE 1.14pt に対し有意でない）
    (b) dragapult が 50% を超える（最下位が並んだ = 底上げの役目が終わった）

Stage 2 の終了は live Kaggle レーティング（`deck-status-and-live-scores`: live が唯一の
本物の信号）。ラウンド上限 6、live 提出は2ラウンドごと。

---

## ゲートの標本設計（ここが一番の落とし穴）

**63デッキ → 11デッキにゲートを狭めると、同じゲーム数では精度が 2.4倍悪化する。**

実測から逆算する。i2 r7 の paired は 63デッキ×40ゲームで `+0.0077 ± 0.0114`、
つまり per-deck デルタの sd = 1.14 × √63 = **9.05pt**。40ゲームでの独立2画面の予測は
√(2×0.25/40) = 11.2pt なので、観測 9.05 < 予測 11.2 → **デルタは標本ノイズ支配**で、
デッキ間の真の分散はほぼ無い（seed pairing が効いている分だけ下回っている）。

標本ノイズ支配なら sd は 1/√G で落ちるので:

    現行  63デッキ × 40ゲーム = 2,520ゲーム   sd 9.05pt  SE 1.14pt
    新    11デッキ × 229ゲーム = 2,519ゲーム   sd 3.78pt  SE 1.14pt

**総ゲーム数を保ったまま11デッキに再配分すれば SE は変わらない。** 壁時計も同じ。
おまけに 229 games/deck は 150 games/matchup の床（`evolution-spare-ex-guard`）を余裕で
超える。40ゲームのまま11デッキに狭めると SE 2.9pt になり、±1pt の判定は不可能になる。

前提の再確認: 「デッキ間の真の分散 ≈ 0」は観測 9.05 < 予測 11.2 から来ている。
Stage 1 が進んで一部のデッキだけ伸びると真の分散が立ち上がるので、**2ラウンドごとに
sd を再測定**し、11.2pt を超えたらゲーム数ではなくデッキ数が効き始めた合図。

Stage 2 は 20マス × 150ゲーム = 3,000ゲーム → sd 4.67pt / SE 1.04pt。
ゲート自体は Stage 1 と同じ 11デッキ × 229ゲームを使う（提出可否は11全体で読む）。

---

## タイムボックス（2026-08-06 決定）

締切は **2026-08-16 23:59 UTC = 2026-08-17 08:59 JST**（月曜の朝）。実質日曜いっぱい。

ユーザー指示: **4B の Stage 1+2 は「3〜4日でどこまで行けるか」**。到達点を目標にせず、
期限で切る。

### ラウンド長は実測値を使う（見積もりではない）

instance2 の r6→r7 の実績（一度きりのベースライン再screen を除いた定常値）:

    screen 2.13h + collect/mix 1.75h + train 7.22h = 11.1h / ラウンド

Phase 2（分岐 + playouts、3h）は instance1 の CPU なので instance2 の
クリティカルパスに乗らない。**11.1h/ラウンドがそのまま予算単位。**

### カレンダー

    08-06 06:30   Phase 0 Pass A 開始（i1 CPU、1.5h）  ← 他と並行、GO/NO-GO
    08-06 08:00   Phase 0 結果。NO-GO ならここで終了
    08-06 09:50   loop8 round 1 完了 → valued watcher が round 2 で再起動
    08-06 10:10   i2 r7 完了 → instance2 が空く
    08-06 11:00   Stage 1 round 1 開始
    ...
    08-10 00:00   ★4B の RL 打ち切り（ハードストップ、89.5h = 3.7日 = 8ラウンド分）
    08-12 00:00   DeBERTa 最終学習 + デプロイ検証（語彙再スイープ / ONNX / INT8 / 速度）
    08-12 12:00   ★最初の LM 提出（T-4.5日）
    08-12 〜 16    live レーティング測定、余裕があれば反復
    08-16 23:59   締切

### ラウンド配分

8ラウンド入るが、遅延分を見て **7 で組む**。

    Stage 1   4ラウンド   11デッキの底上げ
    Stage 2   3ラウンド   dragapult 2種の特化
    予備      1ラウンド   どちらかの遅延吸収

**Stage 1 → 2 の境界にカレンダー条件を追加する。** 既存の (a) plateau / (b) dragapult ≥50%
に加え、**(c) 08-08 12:00 UTC を過ぎたら無条件で Stage 2 へ**。plateau 判定を待って
Stage 2 が1ラウンドも回らない、という失敗を防ぐ。

### 期限を守るために捨てたもの

- Stage 1/2 の max_rounds 6 → 4/3。上限まで回す前提を放棄した
- 「plateau するまで回す」→ 期限で切る。**到達点ではなく経過時間が停止条件**

### DeBERTa は待たない（これが最大のリスク低減）

DeBERTa の学習を「4B が終わってから」にすると、失敗したときに作り直す時間が無い。
**loop8 は既に走っており、round 2 から v41_attach を VALUED で食べる。** Stage 1 の各
ラウンドが吐く Q データも同じ経路（`VALUED=`）でそのまま loop8 に入れられるので、
**DeBERTa は 4B と並行して継続的に強くなる**。08-10 の打ち切り後に始まるのは
「最終学習 + デプロイ検証」だけで、ゼロからではない。

---

## 見積もり

1ラウンドあたり（`docs/rl_mirror_design.md` の Phase 構成）:

    Phase 1  on-policy 収集   GPU  ~20min   11デッキ × 150ゲーム
    Phase 2  分岐 + 16 playouts  CPU  ~3h   instance1 のみ（61.4 実効コア）
    Phase 3  学習              GPU  5-7h
    Phase 4  ゲート screen      GPU  ~1h    11 × 229ゲーム

    ≈ 8-10h/ラウンド。Stage 1 が最大6、Stage 2 が最大6 → 4-5 GPU日（上限まで回した場合）

**配置。** CPU 重工程は instance1（instance2 は 13.44 実効コアで不可 — `vast-cpu-quotas`）。
instance1 の GPU は loop8（DeBERTa DAgger）が使うので、**instance1 = データ工場 + DeBERTa、
instance2 = RL 学習**。生成物は既存の `ship_pool_v41.sh` の経路で流れるので新規配管は不要。

---

## 4B の役割（ユーザー確定 2026-08-06）

4B は提出できない（197.66 MiB 上限）。それを承知の上で、**2つの役割**を担わせる。

    (a) 天井の測定器   4B でどこまで行けるかを先に測る
    (b) データ工場     4B が回した RL のデータで DeBERTa を学習させる

**これは殺した蒸留分岐とは別物。** `teacher-9b-adds-nothing`（9B教師、60倍のパラメータ、
149M生徒に対して手の順位付けの改善ゼロ）で死んだのは「大きいモデルの**意見**をラベルにする」
経路。本設計のラベルは **playout 測定 Q** で、engine_v2 が16回続きを打った結果であり、
**誰が状態を集めたかに依存しない**。

前例が我々のコードにある。`tools/dagger_loop7.sh` のヘッダ:

> The valued-attach files are NOT DAgger and are NOT dropped: they are playout-measured
> labels that do not depend on which pilot collected them, so they stay in every round.

`attach_label.py` も同じ立場を明示している（「LM が到達する状態に**あえて限定しない** —
これは決定にシグナルがあるかを問うのであって、特定のパイロットがそこに踏み込むかではない」）。
そして実際、その valued データは decoder の mix に 9.2% 入って効いている。

### (a) 天井として何が決まるか

4B は 65デッキ平均 53.5%、DeBERTa は 30.6%。**4B が測定 Q で頭打ちになったら、DeBERTa が
それを超えることはまず無い。** `rl-plateau-five-refutations` は decoder について既に
「representation limit が支配的」と結論しており、`teacher-9b-adds-nothing` は容量が
律速でないことを示している。つまり **4B の天井はモデルの天井ではなく手法の天井**として
読める。両方走らせるより先に片方で測る方が安い、という判断は妥当。

判定: Stage 1 を2ラウンド回して 11デッキ paired が動かなければ、**DeBERTa 側は着手しない**
（Phase 0 の GO/NO-GO とは別の、上位の中止判断）。

### (b) データ工場として、残る本物の注意点

ラベルは pilot 非依存だが、**2つだけ 4B の形が残る**:

    状態分布      DeBERTa が到達する局面ではなく 4B が到達する局面を学ぶ。
                  4B 53.5% / DeBERTa 30.6% は別の方策で、DeBERTa が崩れる局面
                  （30%側の負け方）が過小にしか出てこない
    分岐点の選別   margin が小さい決定を分岐するが、その margin は 4B のもの。
                  DeBERTa が迷う場所とは一致しない

どちらも安い対策がある。**Phase 1 の on-policy 収集を2本立てにする** — 4B で 0.7、
DeBERTa（loop8 の最新チェックポイント）で 0.3。Phase 2 以降は共通。CPU 工程はどのみち
instance1 で走り、DeBERTa も instance1 に居るので追加コストはほぼゼロ。

比率は測ってから決める。**ラウンド1で 4B 状態と DeBERTa 状態の両方を集め、
「分岐点として選ばれた決定」の重なりを見る**。重なりが高ければ 4B 単独で十分（0.3 を 0 に
落とす）、低ければ DeBERTa 側を厚くする。1ラウンド分の追加収集で決まる。

### 保存形式は reranker スキーマが正

    生成      rerank スキーマ（state / candidates / chosen / qvals）で書く
    → 4B     tools/rerank_to_sft.py（imitation行）、tools/valued_to_sft.py（Q付き行）
    → DeBERTa そのまま（VALUED= に渡す）

変換は rerank → decoder の一方向しか存在せず、rerank 側が候補リストと qvals を持つ
上位互換。**decoder スキーマで作ると DeBERTa に戻せない**ので、生成側は必ず rerank
スキーマで書くこと。

---

## 決定済み（2026-08-06、残る自由度なし）

ユーザー確定「Stage 2 に関しては既定通り」— 提案した2つの逸脱は**どちらも採らない**。

    テール 15%   採用しない。O = 11デッキのみ。`RL_STAGE2_TAIL` は 0.0 のまま
    GRPO         採用しない。両ステージとも測定 Q ラベル一本

テールを入れないので、**ladder の 21.4% は Stage 2 の学習中は一度も現れない**
（mega_lopunny .028 / mega_lucario .028 / archaludon .024 / omatsuri .018 …）。
これは意図された取引で、代わりに得るのは「11 の相手は全て LM が競技レベルで操縦している」
という一貫性である。テール側は engine_v2 操縦になるため、混ぜると
`live-alakazam-beats-us`（自己対戦は alakazam の強さを 48pt 読み違えた）と同じ質の
汚染を持ち込む。**ゲート側は影響を受けない** — Phase 4 の live-weighted 評価は
LIVE_META 全25エントリで読むので、テールでの弱さは学習に入らなくても計測はされる。

`RL_STAGE2_TAIL` の実装は残す。もし Phase 4 で「11 には強いがテールに弱い」が観測されたら、
そのときだけ 0.15 にして1ラウンド回せばよい。既定を戻すのではなく、証拠が出てから。

---

## 廃止の扱い

`stage("A"/"B"/"C")` はコードから消さず deprecated として残す（過去ログの再現用）。
`tools/rl_loop.sh` は STAGE_C_TARGETS 全件で Stage C を回す作りなので、そのままでは
11回走る。新ドライバは Stage 1 を1回、Stage 2 を dragapult 2種のみで回す。
