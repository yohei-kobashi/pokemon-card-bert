# ミラー自己対戦 RL — 測定 Q ラベル版設計

2026-08-06。ユーザー提案（mirror 自己対戦 → 負け側の僅差手を1手差し替え → 勝敗が変わったら
bad/good ペア → DPO）の骨格を維持し、n=1 勝敗ラベルを playout 測定 Q に差し替えた設計。

## 元提案のどこを何に差し替えたか

| 元提案 | 問題（実測根拠） | 差し替え |
|---|---|---|
| 1ゲームの勝敗変化をラベルにする | ±6pp の効果で正解率56%、value-neutral 7割込みで**52%** ≈ `dagger-label-is-a-coin-flip` の51.1% | 16 playouts × split-sample argmax × permutation null（`attach_label.py` の既存機構） |
| top1-top2 の確率差が小さい手を選ぶ | モデルの迷い ∝ どちらでもいい手。value-neutral の7割を優先的に拾う | **margin 小 ∩ 実測 Q spread 大** の積集合。margin は分岐候補の絞り込みにのみ使う |
| 負けた側の選択から探す | 既に詰んでいる局面を優先サンプリング。mirror では必ず片側が負けるのでデータ半減 | 両側の全決定から分岐し、Q に判定させる |
| DPO を新規導入 | reranker には `train_rerank --margin-weight`、decoder には `valued_to_sft.py` が既にある | 既存損失を使う。decoder に選好項を足す場合のみ DPO 形の追加（小差分） |
| （暗黙）ゲーム単位のクレジット | Stage A plateau の原因そのもの（`rl-stage-a-plateau-diagnosis`） | ラベルが決定単位の測定 Q なのでクレジット割当自体が消える。`shaping-potential-refuted`: playout Q は代替案の17倍のシグナル |

維持するもの: mirror shuffle（`mirror-shuffle-mode`、対戦ごとに並びは変わり両席が同一）、
同一カード候補の除外（`indistinguishable-board-slots`、reranker は menu_dedup 済み）、
対象デッキ = dragapult 系 2 + leaderboard 上位（下記11デッキ）。

## 対象デッキ（11 = Stage-C 候補と同一）

ユーザー確定 2026-08-06。**この11デッキは RL のパイロット対象であると同時に提出候補**で、
`tools/rl_config.py:STAGE_C_TARGETS` と一致させてある。以後どちらかだけを変えないこと。

出典 docs/meta/leaderboard_top500_2026-08-05.md。括弧は 500 チーム中の使用数と最高順位。

    dragapult (20, #21)             Stage-C 継続（final RL round = dragapult 集中）
    dragapult_dusknoir (0, --)      leaderboard 不在。提出候補として意図的に残す
    marnie_grimmsnarl (177, #1)     field の 35.4%
    alakazam_nz (66, #12)           Alakazam 族の主流
    alakazam (20, #20)
    crustle_geco (28, #6)           Crustle 族の主流
    crustle (23, #5)
    ogerpon_mono (27, #7)           新規追加の meta デッキ
    dudunsparce_box (13, #8)
    cynthia_garchomp (19, #42)      上昇中（stageB-meta-refresh-task で要注意指定）
    mega_lucario_tr (14, #2)        後述

    カバー率  top-500 393/500 = 78.6% | top-100 79/100 | top-20 16/20

**選定基準は2つある。使用数だけで切ると間違える。** `mega_lucario_tr` は使用数14で
`mega_lopunny` と同数だが、最高順位は #2 と #3（lb 1186.6 / 1160.7）で、片や #30。
使用数で足切りした最初の10デッキ案はこれを落としていた。**到達高度は使用数と別の信号。**

しかも `decks/mega_lucario_tr.csv` は #2/#3 の構築と **match 1.00** で一致する。構築作業は
ゼロで、ギャップは全て操縦。`mega-lucario-live-matchup-profile`（提出した mega_lucario は
live 47%、構造的な弱さ）と `mega-lucario-real-list-tech`（答えはプール内にある、差し替え
ではなく作り直し）が、この型で既に手元にあったということ。

**入れなかった top-20 の4つ**（再検討時はここを読むこと）:

    slowking      #4 だが top-500 に1チームのみ。プレイヤー効果とデッキの強さを分離できない
    hydrapple     #7、match 0.73。live は Meganium 型で我々のリストと差がある
    mega_starmie  #10、match 0.63。live は Mega Froslass ex x4 との混成で別物
    raging_bolt   #11、match 0.84 だが top-100 に1、かつシグネチャに Raging Bolt が無く
                  分類ラベル自体が怪しい（Mega Kangaskhan ex / Meowth ex 主体）

いずれも「RL の前に構築の話」か「標本1」。`mega_lopunny` (14, #30) と `omatsuri` (9, #47) は
使用数はあるが到達高度が無い。

mirror は同一デッキ同士でしか成立しない。同デッキ mirror だけで鍛えると
`live-weighted-eval-protocol` の教訓どおり field への転移が保証されないので、
状態収集は **(a) 各対象デッキの mirror + (b) 対象デッキ × LIVE_META 上位のシード付き通常戦**
の両方から行う。(a) は分散削減と mirror 評価用、(b) が転移の担保。

## Phase 0 — ヘッドルーム監査（GO/NO-GO、~1.5h、instance1 CPU）

`attach_value.py` を attach 以外の決定種（play / retreat / evolve / promote / sub-select）に
一般化した `decision_value.py` で、対象11デッキについて決定種ごとに測る:

    非 neutral 率      permutation null を超える分岐の割合（attach 実測 25-31%）
    capture_engine     [Q(engine の手) − mean Q(他)] / [Q(best) − mean Q(他)]   attach 実測 48%
    capture_LM         同じものを LM の手について                              未測定
    spread             Q(best) − mean Q(他)                                    attach 実測 0.1221
    頻度               全決定に占める割合                                       attach 実測 8.5%

### 判定は比率ではなく残り価値の絶対量で行う

capture 80% でも spread が大きければ、capture 50% で spread が小さい決定より残りは大きい。
比率で足切りすると換算ができないので、**attach を 1.0 とした残り価値**で並べる:

    残り価値_k = 頻度_k × 非neutral_k × (1 − capture_k) × spread_k
    attach     = 0.085 × 0.30      × 0.52            × 0.1221  = 0.00162  ≡ 1.0

attach は「LM の attach だけ engine に委譲すると +11.4pt」（`attach-decisions-at-chance`）
という実プレイでの校正点を持つ唯一の決定種なので、これが物差しになる。

### capture は engine と LM の両方で測る。片方では打ち手が決まらない

伸びしろは `1 − capture_LM`。そのうち**模倣で届くのは capture_engine まで**で、
**Q ラベルが要るのはその先だけ**。

                         capture_engine 低          capture_engine 高
    capture_LM 低    Q ラベルが効く              模倣で足りる → DAgger（安い）
    capture_LM 高    Q ラベルが効く              その決定種は完了。触らない

### Q^engine ≠ Q* — バイアスの向きを知った上で使う

分岐先を評価するとき**その後を打つのは engine_v2**。engine_v2 は自分の型の外に出た手の
フォローが下手なので、型外の手は δ だけ不当に下がる（MCTS の rollout policy bias）。

    engine が選んだ手 a    自分の型どおりに続く → 減点なし
    それ以外の手 b        型外に押し出される  → δ 下がる
    capture = [Q(a) − 平均Q(b)] / [Q(最善) − 平均Q(b)]
              分子が δ 水増し、分母はほぼ不変 → capture 過大 → ヘッドルーム過小

**したがって Pass A は保守側に外れる。GO は信頼できるが NO-GO は信頼できない。**
Pass A 単独の NO-GO で止めてはならない。

標本数では消えない限界も残る: 「一手変えれば効く手」は検出できるが、**「複数ターンの
一貫した計画が要る手」は構造的に過小評価される** — engine_v2 がその計画を実行しないため。
これは Q ラベル方式全体の天井。根拠は成功例1件（attach で capture 48% を出し、下流予測
「attach だけ engine に委譲で +11.4pt」「valued が decoder の mix で効く」が的中）であって
体系的検証ではない。

なお **16 playouts のバラつき源は engine の気まぐれではない**（`rl_branch.py:182-187`）。
1シナリオ = 隠れプールの1回の配り直しで、シナリオ内では全候補が同じ配りを共有し、16
シナリオ間で隠し状態を積分する。engine が決定的でも 16回はバラける（attach 実測の
per-decision sd は ±1 スケールで 0.29）。permutation null が退化する心配はない。

### capture_LM が本命。capture_engine は参考

**4B は engine_v2 に既に勝ち越している**（全65デッキ平均 53.5%）。Q ラベル方式の前提は
「教師が見えていないものを playout が見る」だが、多くのデッキで LM は既に教師を超えている。
engine の capture が答えるのは「教師の見落とし」であって「LM の見落とし」ではない。

**Pass B は全量不要。ゲート判定には 2,000決定で足りる。**

    attach_value 実測   2,519決定 → engine edge ±0.0058（per-decision sd 0.29）
    2,000決定           SE = 0.29/√2000 = ±0.0065   ほぼ同等
    GPU コスト          2,000 × 199ms（3系統競合下の実測値）= 約 6.6分

7分なので削る理由がない。**ゲートの前に必ず走らせる。**

### GO / NO-GO

    残り価値は capture_LM で計算する（capture_engine は参考値）

    GO           残り価値の合計 ≥ 1.0（attach 相当）
                 → 該当する決定種にだけ分岐予算を寄せる（全種でなくてよい）
    DAgger に振替 capture_LM が capture_engine より 20pt 以上低い決定種
                 → 教師が知っていることをまだ学べていない。RL より安い
    STOP         capture_LM / capture_engine の両方が高く、かつ spread も小さいとき「だけ」
                 → 結論は「LM に伸びしろが無い」ではなく「Q ラベルはその梃子ではない」。
                   DAgger 側の判断に回す

    Pass A 単独の NO-GO では止めない。バイアスの向きが分かっており、誤検出の側だから。

0.3 は判断であって測定値ではない。根拠は、ゲートの解像度が SE 1.14pt で、1ラウンド 8-10h ×
最大6ラウンドを賭ける対象として 3-4pt は薄すぎる、というだけ。ただし比率で切るのと違い
換算の筋道は示せる。

**「全決定種が高ければ中止」は強すぎるので採らない。** attach だけ 48% で他が 90% なら
attach 系に絞って回せばよく、実際 `attach_label.py` は既にセル重みでそうしている。

### 事前予想

**retreat が最有力。** `systematic-divergence-diagnostic`（DAgger は play/attach を閉じたが
retreat は一度も閉じていない、ns_zoroark 0-40 は retreat の失敗）に加え、
`prompt-lies-about-retreat-cost` により **v41 まで retreat コストの表示自体が嘘だった**
（N's Castle / Latias ex が 0 にするのに印刷値を出していた、24/63デッキ該当）。つまり
retreat は読める情報が無い状態で測られていたので、v41 で初めてまともに測れる。

逆に **evolve / promote は capture が高い予想** — engine_v2 の per-deck ルールが最も濃い
領域で、`card-roles-authored-all-60` の tier 規則が直接効いている。

## Phase 1 — on-policy 状態収集（~15min GPU）

**学習対象モデル自身**で (a)(b) を打つ（engine 収集の状態では自分の到達分布を直せない）。
決定ごとに blob 付き obs と候補、モデルの score margin を記録。11デッキ × 150 ゲーム規模。

## Phase 2 — 分岐と測定（~3-4h、instance1 CPU 40 workers）

margin 下位の決定から `rl_branch.py` で K 候補に分岐（determinization 共有）、
16 playouts/候補、両側 engine_v2 継続。split-sample argmax + permutation null を通った
決定だけを `qvals` 付きで書く。同一カード候補は分岐前に統合。

既知バイアスを明示: Q は「engine_v2 が続きを打った場合」の値で、モデル継続の値ではない。
attach_label と同じ妥協で、同ラベルは v40 で decoder に +（i2 mix の valued 9%）の実績あり。

### 予算配分は観測 gap ではなく **価格付け済み** の gap で行う（2026-08-06 実測）

`tools/diag_lm_losses.py --targets` が出す `share` は**観測**の大きさであって因果ではない。
`tools/price_targets.py --rollout engine --points 100` を上位6セルに掛けた結果:

    deck                 kind     観測gap        dQ       z    読み
    ogerpon_mono         end       -9.8pp   -0.3825   -6.18   LM は既に正しい
    cynthia_garchomp     end       -8.1pp   -0.1625   -5.14   同上
    mega_lucario_tr      end       -3.7pp   -0.2988   -6.97   同上
    marnie_grimmsnarl    end       -6.6pp   -0.0975   -3.68   同上
    mega_lucario_tr      evolve   -29.9pp   +0.0788   +2.88   ★ 唯一の実弾
    marnie_grimmsnarl    ability  -11.9pp   +0.0437   +1.65   有意でない

**艦隊全体で z −10.90 だった最強の観測所見（`end`）が、反実仮想では梃子ではなかった。**
`setup-execution-audit-and-budew-overattack`（over-attack は症状）と同型で、価格付けを
挟まずにルール化していれば外していた。

**ただし初版の価格付けには片手落ちがあった。** 観測 gap は全セルで**負**（勝者は取る割合が
低い）なので、問うべき仮説は「取りすぎ」である。しかし初版の `price_targets.py` は
**LM が取らなかった決定**しか分岐しておらず、取りすぎ側を見ることができなかった。
`--side take` を追加して再測定した結果、**`end` の判定は逆転した**:

    deck                 kind     dQ(take)      z    読み
    ogerpon_mono         end       -0.1346   -4.14   取りすぎ（有意）
    cynthia_garchomp     end       -0.0608   -1.68   同方向、単独では有意でない
    mega_lucario_tr      end       -0.0523   -1.51   同上
    marnie_grimmsnarl    end       -0.0371   -1.31   同上
    mega_lucario_tr      evolve    +0.0821   +2.44   進化した判断は正しい
    marnie_grimmsnarl    ability   +0.0204   +0.71   —

    分散逆数プール（end / take、4デッキ）  -0.0696  z -4.28

**両側を合わせた読み: LM はターンを切り上げるのが早すぎる。** 切り上げないと決めた場面の
判断（decline 側 dQ 大きく負）は正しい。`mega_lucario_tr/evolve` は decline +0.079(z+2.88)、
take +0.082(z+2.44) で両側整合 — 進化は**過小**利用であり、観測 gap（勝者は進化が 29.9pp
少ない）とは逆を向く。相関と反実仮想が食い違った実例。

**`vs best alt`（dQ_max）は判定に使わない。** k=3 の最大値は推定ノイズを ~0.85sd 拾うので、
真値に関わらず下方バイアスが乗る。実際 6セル全部が -0.13〜-0.21 / z -3.8〜-6.1 で負になり、
これには他の全測定が正と言う `mega_lucario_tr/evolve` も含まれる。自分の選択バイアスを
測っている統計なので診断値どまり。

配分は `tools/retarget_cells.py` が観測 share に倍率を掛けて作る。**ヘッドルームは符号では
なく符号×側で決まる**:

    decline 側 dQ > 0  誤り  x3.0   見送ったが取るべきだった
    take    側 dQ < 0  誤り  x3.0   取ったが見送るべきだった
    上記の逆（有意）    x0.35  **測った側では**正しい。qlabel_gen はメニューに出れば両側を
                              分岐するので削除ではなく降格
    |z| < 2 / 未測定    x1.0   観測を prior として据え置く

符号だけで判定すると `ogerpon_mono/end`（decline -0.38 / take -0.13）が降格され、最大の
当て先を3分の1に削っていた。

**kind 単位でプールする。** 100分岐点はセル単体で dQ 0.06 程度しか解像しないので、
`end` の4デッキは1つだけが |z|≥2 を超える。しかし DECK[]/ID ME で条件付けられた**1つの
モデル**であって11個のモデルではないから、全デッキで同方向に倒れる kind はモデルの性質
1つであり、証拠の単位はプール推定である。プールが有意なら、その kind の全セルに床
`x1.5` を与える（個別に通ったセルの x3.0 は下げない）。セル単位判定だけだと
`ogerpon_mono/end` が 15.7%、他の `end` が 1.2〜1.5% になり、これは n=100 でどのセルが
運良く有意になったかの記述であってポリシーの記述ではない。

**1セルの上限は 25%**。無制限だと `mega_lucario_tr/evolve` が単独で 43.9% を取り、これは
`narrow-dagger-overfits` が実測した形そのもの（1デッキ集中で対象 +11.9pt / 艦隊 −2.75pt）。
価格付けは「どこに価値があるか」を答えるが「そこに全額賭けてよいか」は答えない。

### 席（先攻/後攻）を固定できるセルがある

`diag_lm_losses.py` の section 6 が Alakazam 系の**席崩壊**を検出した（先攻 34-6 / 32-8）。
これは take-rate gap とは別種の失敗で、**勝っている席で集めたラベルでは直らない**。
`qlabel_gen.py --seat second`（またはセルの `seat` フィールド）で、負ける席の決定だけを
ラベルする。席は開幕 obs の最初の `yourIndex` で判定できるのでエンジン内部情報は要らない。

## Phase 3 — 学習

    reranker (instance1)   既存ループの VALUED 経路そのもの。生成物を VALUED= に渡し
                           VALUED_FRAC=0.05-0.10。追加コード無し
    decoder  (instance2)   valued_to_sft.py で重み付き SFT 継続。選好項を試すなら
                           (argmax-Q, モデル選択) ペアの DPO を LoRA 上で — ただし
                           `rl-plateau-five-refutations`（clean Q への supervised fit 失敗）
                           が decoder 側の先例なので、1ラウンドで paired が動かなければ撤退

base 混合は必須（85%+）。valued 単独学習は過去に破滅している（分布が attach に偏る）。

## Phase 4 — 評価ゲート

    mirror paired screen   固定シード、前ラウンドと同一 shuffle_fp、63/65 デッキ
    live-weighted          LIVE_META 重み、≥300 games/cell（4分で済む）
    per-deck               対象11デッキは 150 games/matchup 床で個別判定

判定は screen のみ。held-out top1 / eval loss は使わない（`teacher-9b-adds-nothing`、
DeBERTa ベンチで3度目の実証: top1 で選ぶと逆を選ぶ）。

## スケジューリング

CPU 重（Phase 0/2）は **instance1 のみ**（61.4 実効コア。instance2 は 13.44 で不可 —
`vast-cpu-quotas`）。DeBERTa ループの5時間学習ウィンドウ中に load-gate 付きで走らせ、
生成物は既存 shipper の経路で instance2 へ。instance2 は GPU 学習のみ。

## 中止基準

- Phase 0 の残り価値の合計が attach の 0.3 未満 → 開始しない（判定式は Phase 0 の節）
- 2ラウンド連続 paired ≤ +1pt（SE ~1.2）→ 停止。plateau 先例2件と同型
- 対象デッキ +でも live-weighted −（mirror 過適応）→ (b) の比率を上げて1回だけ再試行

## 未決定（ユーザー判断）

instance2 の 4B は 197.66 MiB 制限で提出不能、蒸留経路は死んでいる
（`teacher-9b-adds-nothing`）。本設計は instance2/decoder で実行可能だが、同じ生成物が
instance1/reranker（提出可能）にそのまま入るので、**主消費者を reranker、decoder は
副消費者**とするのを推奨。
