# P0 静的解析: `omatsuri`（Festival Grounds / Dipplin 「Do the Wave」ベンチスケール）

BLIND zero-shot 解析。接地は (a) 実60枚のカードテキスト/数値、(b) シミュレータ・プローブ計測のみ。
プローブは全て omatsuri ミラー（両者 non-ex）で実施。verified フラグは実測に紐づく。

---

## 0. デッキ同定（全60枚 / 30 種）

**ポケモン 22枚**
| cid | 名前 | 枚 | 段階/型 | HP | 技/特性(cost=energies) |
|---|---|---|---|---|---|
| 89 | Grookey | 4 | たね/草 | 70 | Smash Kick10[G], Branch Poke30[GG] → Thwackey へ進化元 |
| 90 | Thwackey | 3 | 1進/草 | 100 | Beat50[GG]; **特性 Boom Boom Groove**: 自ターン1回、Activeが Festival Lead 持ちならデッキから1枚サーチ |
| 92 | Applin | 4 | たね/草 | 40 | Tumbling10[G]+コイン表で+20 → Dipplin へ進化元 |
| 42 | Applin | 1 | たね/**竜** | 40 | **Find a Friend**0[.] ポケサーチ; Rolling Tackle30[G+**Fire**] → Dipplin へ進化元 |
| 93 | **Dipplin** | 4 | 1進/草 | 80 | **Do the Wave 表示0[G]=20×自ベンチ数**; **特性 Festival Lead** |
| 100 | Goldeen | 1 | たね/水 | 50 | Whirlpool10[CC]+コインで相手エネ破壊; 特性 Festival Lead |
| 240 | Seaking | 1 | 1進/水 | 110 | Rapid Draw60[C]+2ドロー; 特性 Festival Lead |
| 73 | Rellor | 1 | たね/草 | 50 | Slight Intrusion30[C] **自分に10** → Rabsca 進化元 |
| 74 | Rabsca | 1 | 1進/草 | 70 | Psychic10[G]=+30×相手Activeエネ; **特性 Spherical Shield**（自ベンチ全ダメ/効果無効） |
| 858 | Psyduck | 1 | たね/水 | 70 | Ram20[CC]; **特性 Damp**（自壊系特性を無効化） |
| 343 | Shaymin | 1 | たね/草 | 80 | Smash Kick30[CC]; **特性 Flower Curtain**（ルールボックス無し自ベンチをダメ無効） |

**トレーナー/エネルギー 38枚**
| cid | 名前 | 枚 | 種 | 効果 |
|---|---|---|---|---|
| 1227 | Lillie's Determination | 4 | サポ | 手札を山に戻し6ドロー（サイド6枚残なら8） |
| 1086 | Buddy-Buddy Poffin | 4 | グッズ | 山からHP70以下たね2体をベンチへ |
| 1152 | Poké Pad | 4 | グッズ | 山から非ルールボックスポケ1枚を手札 |
| 1094 | Bug Catching Set | 4 | グッズ | 山上7枚から草ポケ/基本草エネを最大2枚 |
| 1245 | **Festival Grounds** | 4 | スタジアム | エネ付き全ポケが特殊状態無効。**＋ Festival Lead の発動条件** |
| 1 | 基本草エネ | 4 | エネ | G |
| 1182 | Boss's Orders | 3 | サポ | 相手ベンチをActiveに（ガスト） |
| 1175 | Brave Bangle | 2 | ツール | 非ルールボックスの技 **+30（相手Active {ex} 限定・実測で non-ex には無効）** |
| 1184 | Lana's Aid | 1 | サポ | トラッシュから非ルールボックスポケ/基本エネを最大3枚回収 |
| 1191 | Kieran | 1 | サポ | 入替 or 当ターン **+30（相手 {ex}/{V} 限定）** |
| 1211 | Black Belt's Training | 1 | サポ | 当ターン **+40（相手 {ex} 限定）** |
| 1213 | Judge | 1 | サポ | 両者手札を山に戻し4ドロー |
| 1183 | Perrin | 1 | サポ | 手札のポケ最大2枚を山へ→同数サーチ |
| 1129 | Sacred Ash | 1 | グッズ | トラッシュのポケ最大5枚を山に戻す |
| 1080 | Unfair Stamp | 1 | グッズ | 自ポケが前ターンKOされた時のみ：両者手札を山、自5枚/相手2枚 |
| 1174 | Air Balloon | 1 | ツール | 逃げ-2 |
| 1261 | Forest of Vitality | 1 | スタジアム | 草ポケが出したターンに草へ進化可（初手番除く） |
| 18 | Grow Grass Energy | 1 | 特殊エネ | G供給、付けた草ポケ **+20HP** |

---

## 1. 勝ち筋の仕様（win condition spec）

**主砲は Dipplin (93) の「Do the Wave」で、表示は 0 だが実打点はベンチ数に比例する。**
これは汎用エンジンが「見えない」典型で、テレメトリの ~10%WR とエネ誤配線の根源。

### 実打点式（★最重要・共有知覚 `_expected_dmg` の実装候補）
```
per_attack   = 20 * bench_count              (＋ 30/attack だけ相手 Active が {ex} かつ Brave Bangle 装着時)
per_turn     = per_attack * (2 if Festival Grounds が場に出ている else 1)   ← Festival Lead で2回攻撃
```
- **verified:true**（プローブ probe2/probe5）: bench 0–5 で per_attack = 20×bench が線形一致（0,20,40,60,80,100）。
- **verified:true**（probe2）: Festival Grounds 在場時、1ターンに Do the Wave の攻撃選択が **2回** 発生（62 ターン中 44 が2回、18 が1回）。→ per_turn = 40×bench。
- **verified:true**（probe5）: Brave Bangle 装着でも non-ex 相手には dmg=20×bench で変化なし → **+30 は {ex} 限定**。Kieran/Black Belt's Training も同様（テキスト {ex}/{V}）。
- コスト [G] 1個のみ。Grow Grass 付与で Dipplin は 80→**100HP**。

### 育てるべき資源（式の変数）
1. **bench_count（最重要）** — 打点そのもの。目標5。
2. **Festival Grounds 在場** — 打点×2の乗数。落とされたら半減。
3. **Dipplin の草エネ1個** — オン/オフのスイッチ（0か200か）。

### 主勝ち筋（手順）
- T1–T3: Poffin/Pad/Bug Set でベンチに たね（Grookey/Applin/…）を並べ、Applin→Dipplin へ進化、Festival Grounds を貼る。
- T3 以降: Dipplin Active＋草1で毎ターン Do the Wave×2 = 40×bench。bench5 で **200/ターン**。
- Festival Lead の「初撃KOなら相手が新Active選択後もう一度攻撃」で **1ターン複数サイド** を狙う。全ポケ単サイドなので 6KO レース。

### 副勝ち筋 / プランB
- Dipplin が引けない/落ちた時: **Thwackey Beat50[GG]** or **Seaking Rapid Draw60[C]** で場繋ぎ（いずれも表示打点があり L0 が見える点は皮肉にも利点）。
- **Rabsca Psychic**（10+30×相手エネ）は重エネ相手への対角火力。
- ただし副砲は全て 60 未満〜60 で、主砲不在時は明確に格落ち。プランBは「延命して Dipplin ラインを再建」が本質。

---

## 2. ルール相互作用の棚卸し

- **エネは進化で持ち上がる**（**verified:true**, probe4: エネ付き個体の EVOLVE を115件観測、serial 保存でエネ継続）。→ **Applin に先貼りしても Dipplin へ乗る**。ゆえに「進化前への手貼り」自体は無駄ではない。**ただし Grookey→Thwackey ラインに乗ると Thwackey（非攻撃エンジン）で塩漬け**になる（誤配線の本体）。
- **ダメージ vs ダメカン**: Do the Wave は「ダメージ」（弱点/軽減が乗る通常ダメージ）。配置ではない。
- **技コストの型 vs 供給型**: 供給は **基本草4 + Grow Grass1 = 全て草**。全主要アタッカー（Dipplin[G], Thwackey[GG], Applin[G], Goldeen[CC], Seaking[C], Rabsca[G], Rellor[C]）は草/無色で支払い可。**唯一 Applin(42) の Rolling Tackle[G+Fire] のみ型死**（Fire 不在）だが同カードは実質 Find a Friend サーチ用なので影響軽微。→ **型死は問題でない。問題は量（5枚）と配線先**。
- **1ターン1回制**: サポート1・手貼り1・スタジアム1・逃げ1。Boom Boom Groove（特性）は毎ターン別枠。理想シーケンス（進化＋手貼り＋スタジアム＋Boom Boom Groove サーチ）は1ターンに収まる。
- **山掘り/デッキアウト地平**: Poffin4/Pad4/BugSet4/Lillie4＋Boom Boom Groove（毎ターン1サーチ）＋Perrin＝掘削が非常に厚い。単サイド主砲ゆえ試合が長引きやすく、**デッキアウトが現実的な敗因**。戻し札は Sacred Ash1 / Lana's Aid1 のみ（薄い）。→ §5 tension。
- **スタジアム競合**: Festival Grounds（×2乗数の生命線）と **Forest of Vitality（草の即進化）が同じ1枠を奪い合う**。Forest を貼ると Festival Lead が消え、打点が半減。→ tension/doctrine。
- **特殊エネ**: Grow Grass のみ（G＋20HP）。ACE SPEC は無し。

### §2b 自己害の棚卸し（60枚走査）★
「自軍を場から除去する」効果は **なし**（盤面即死型の自己害カードは0）。該当する自己害は以下3類：
- **手札を山に戻す系**: Lillie's Determination(4) / Judge(1) / Unfair Stamp(1) / Perrin(1)。組み上げた combo（Dipplin進化体＋Festival Grounds＋草）を抱えた手で発火すると **山へ散り、薄い山では再集約に失敗**。→ **発火してよい盤面条件**: `hand に "今使う予定の勝ち筋パーツ" が無い` かつ（Lillie/Judge）`hand_size <= 4` の時のみ。Judge(引4)は Lillie(引6) より危険。
- **自傷**: Rellor Slight Intrusion（自10）。**発火条件**: Rellor を攻撃に使う場面自体が稀（副砲以下）。`main 攻撃札が他に無く、かつ Rellor HP>10` の時のみ。severity=minor。
- **強制入替（自軍）**: Kieran 選択1（自 Active⇄ベンチ）。**発火条件**: `Dipplin を前に出す時のみ`。誤って Dipplin を引っ込めない。

→ §2b 該当ありのため **severity=catastrophic の仮説を1本（H5, 手札リセットで combo 消失）** を立てる（scenario 付き）。

---

## 3. サイド算術（prize arithmetic）

- 全ポケが **単サイド（ex 無し）** → 双方 6KO レース。
- 主砲 Dipplin: HP 80（Grow Grass で100）。環境典型打点 ~200 → **主砲は毎ターン取られる前提**。
- ただし Dipplin は **草1で即再装填**でき4枚あり、Do the Wave×2=200 で **相手1体を確実KO＋Festival Lead 継続で2体目KO** を狙える。トレードは 1-for-1〜2-for-1 で有利。
- チェイン要件: 毎ターン「Dipplin1体を Active＋草1＋bench≥4」を維持。**2体目 Dipplin の予備装填（進化元 Applin をベンチに温存）** が継戦の鍵。
- 敗北条件: (a) ベンチを削られ打点が 40×低bench に萎む、(b) Festival Grounds を割られ×2が消える、(c) 山切れ、(d) Dipplin を引けず副砲60で押し負け。

---

## 4. フェーズプラン（計測可能）

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| 序盤 | `turn<=3` かつ `Dipplin not in play` | Poffin/Pad/BugSet でベンチに たね≥3、手札に Applin＋Dipplin＋Festival Grounds＋草を集約 | 進化元/スタジアム欠→ Pad/BugSet/Perrin で不足パーツ指名サーチ |
| 中盤 | `Dipplin in play & Festival Grounds in play & bench>=3` | Dipplin Active、草1装填、Do the Wave×2 を毎ターン実行（≥120/ターン、bench4以上で≥160） | エネ誤配線→ 逃げ/入替で Dipplin を前に。Festival Grounds 割られ→ 予備を即再設置 |
| 終盤 | `自サイド<=3` または `deck<=10` | KO チェーン継続で詰め。掘り停止し Sacred Ash/Lana's Aid で山と Dipplin ラインを再充填、山切れ回避 | 山薄→サーチ/ドロー札を打たず温存。Dipplin 全滅→ Lana's Aid で回収し再建 |

各フェーズの「使う/使わない」:
- 序盤に使う: Poffin, Pad, Bug Set, Applin(42) Find a Friend, Perrin, Lillie（手札薄時）。使わない: Boss's Orders（対象価値薄）, 副砲攻撃, Judge。
- 中盤に使う: Do the Wave, Boom Boom Groove（毎ターン）, Boss's Orders（KO を運ぶ時）, Grow Grass（Dipplin へ）。使わない: Forest of Vitality（Festival Grounds を消すため原則封印）。
- 終盤に使う: Sacred Ash, Lana's Aid, Unfair Stamp（KO 返し）。使わない: 追加のサーチ掘り（デッキアウト回避）。

---

## 5. 多面的評価（scorecard）＋ 対立軸

| 軸 | 点(1-5) | このデッキ固有の定義と検証指標 |
|---|---|---|
| 速度 | 3 | 勝ち筋 online = Dipplin＋Festival Grounds＋bench≥4＋草1。概ね T3–4。指標 `first_attack_turn` |
| 火力曲線 | 4 | per_turn=40×bench。bench5＋FG で 200。ただし bench 依存で分散大。指標 `bench_size_dist@attack` |
| 安定性 | 3 | 掘削は厚いが「Dipplin＋FG＋草＋広ベンチ」の4条件同時要求でブリック余地。指標 `play_rate[93]` |
| 継戦力 | 3 | Dipplin4＋Applin5＋Sacred Ash/Lana's Aid。再装填は草1で安いが山戻し札が薄い。指標 `post_ko_attack_rate` |
| 対応力 | 2 | 壁/スプレッド/手札破壊への回答が乏しい。ベンチ削り（スプレッド）に打点が直結で崩れる。Shaymin/Rabsca の盤面保護が唯一。指標 `bench_size_dist` |
| 資源経済 | 2 | エネ僅か5枚（草1が命）で誤配線に極端に脆い。掘るほど山が減る。指標 `energy_attach_share[93]`, `loss_share[deckout]` |
| 妨害耐性 | 2 | スタジアム破壊で×2消失、ベンチガストで打点源露出、エネ破壊で唯一の草1を失う＝急所が多い。指標 `loss_share[board]` |
| サイドレース | 3 | 単サイド同士、200打点＋Festival Lead 連撃で 2-for-1 を取れれば有利。指標 `post_ko_attack_rate` |
| **★発明軸: 打点可視性** | 1 | 主砲の表示が **0**。L0 の全判断（手貼り先/進化先/ガスト/昇格/リーサル）が主砲を無視。**このデッキ最大の欠陥軸**。指標 `energy_attach_share[93]` |

### 対立軸（tensions）★
- **T1 打点(ベンチ幅) × デッキアウト**: bench を広げる Poffin/Pad/BugSet は同時に山を削る。単サイド長期戦で山切れ。**バランス点**: `bench<4 または 手札に勝ち筋パーツ欠 の間だけ掘る。bench>=4 かつ combo 完成後は掘削停止。deck<=8 で Sacred Ash 起動`。
- **T2 ×2乗数(Festival Grounds) × 即進化(Forest of Vitality)**: 単一スタジアム枠の奪い合い。**バランス点**: `原則 Festival Grounds 固定。Forest は "その1枚で Dipplin が即出て、かつ翌ターン Festival Grounds を再設置できる" 場合のみ`。
- **T3 主砲装填 × 副砲装填**: 草5枚しかなく分割すると Dipplin が死ぬ。**バランス点**: `attackable な Dipplin が場に出るまで全草を Dipplin ライン優先。副砲へは Dipplin 入手不能が確定した時のみ`。
- P3 含意: T1 を締める（掘り停止）と安定性が落ちる。**deckout 指標と play_rate[93] を同時監視**しないと whack-a-mole。

---

## 6. カード別「使用宣言」（全60枚）＋ L0 既定監査

### L0 既定の監査（各項目がこのデッキで誤作動するか）
1. **特性は無条件使用** → 該当する能動特性は Boom Boom Groove のみで、これは**使ってほしい**（Active が Festival Lead 持ちの時だけ提示される）。Damp/Spherical Shield/Flower Curtain は受動。**自己害の無条件発火は無し（§2b）**。→ この項目は概ね安全。
2. **手貼りは表示ダメ最大へ** → **★致命的誤作動**。Dipplin 表示0 < Thwackey50 < Seaking60。草が Thwackey/Seaking/Grookey へ流れ Dipplin が撃てない（probe4 で Thwackey/Rabsca にエネが乗り塩漬けを観測）。→ H1。
3. **ドローサポは手札≤5で発火** → 本デッキは掘削過多で手札が膨れがち＝Lillie が必要時に出ない、または combo 抱え時に発火し散らす。→ H5。
4. **サーチ/ボールは無条件毎ターン** → 山を過剰に削り **デッキアウト**（戻し札薄い）。→ H6。
5. **進化は進化後表示最大優先** → Thwackey(50) > Dipplin(0)。**Grookey→Thwackey を優先し Applin→Dipplin が後回し**＝勝ち筋ライン遅延。→ H2。
6. **逃げは装填済(エネ数)なら逃がさない** → 草が誤って Thwackey/Grookey(Active)に乗ると「装填済」判定で逃がさず、Dipplin を前に出せない。→ H8。
7. **gust は自Active表示≥60時のみ** → Dipplin 表示0 → **Dipplin Active 時に Boss's Orders が永遠に出ない**（実打点200を見ない）。→ H3。
8. **KO後昇格は表示最大** → Dipplin 落ち後、次のActiveに Thwackey/Seaking を昇格し Dipplin ラインを晒さない/撃たない。→ H7系。

### 使用宣言（`意図｜発火条件(状態変数)｜期待プレイ率/試合｜腐る条件`）
- **93 Dipplin ×4** — 主砲。｜`bench>=1 & Dipplin Active & 草>=1`｜~5+回（毎ターン攻撃）｜bench0 で0打点／FG無で半減。`fire_if bench>=A, A∈{2,3,4}`（締めすぎ＝低bench早撃ちの初動を止める）
- **92 Applin ×4** — Dipplin 進化元＆ベンチ胴｜`Dipplin 未着 or 予備が要る`｜~3｜Dipplin 4枚が既に場/手にあると余剰
- **42 Applin ×1** — Find a Friend で不足ポケ指名サーチ＆進化元｜`序盤 & 欲しいポケが山`｜~0.6｜Rolling Tackle は型死で腐る（Fire無）
- **89 Grookey ×4** — ベンチ胴＆Thwackey 進化元｜`序盤ベンチ展開`｜~2.5｜Thwackey 不要な手では単なる胴（打点源としては可）
- **90 Thwackey ×3** — **Boom Boom Groove エンジン**＆bench 胴＆副砲｜`Active が Festival Lead 持ち（=Dipplin等）`｜~2｜Active が非FL だと特性死。エネを吸うと誤配線源（H1）
- **1245 Festival Grounds ×4** — ×2乗数の生命線｜`場にFG無し`｜~2｜既にFG在場なら温存（割り返し用に手札保持）
- **1086 Buddy-Buddy Poffin ×4** — ベンチ即2体｜`bench<4 & HP70以下たねが山`｜~3｜Shaymin(80)は対象外。bench満杯で腐る
- **1152 Poké Pad ×4** — 非ルールボックスポケ指名サーチ｜`鍵ポケ(Dipplin/Applin)欠`｜~3｜必要ポケ全て手/場で腐る
- **1094 Bug Catching Set ×4** — 草ポケ/草エネ回収｜`草エネ or 草ポケ欠 & 山上に有`｜~2.5｜非草しか無い時ハズレ
- **1 基本草エネ ×4** — Dipplin の点火剤｜`Dipplin ライン未装填`｜~3（貼り）｜**Thwackey/副砲へ流れると死に札化（H1）**
- **18 Grow Grass Energy ×1** — 草供給＋Dipplin +20HP｜`Dipplin へ`｜~0.5｜非草へ付くと HP ボーナス無
- **1182 Boss's Orders ×3** — 詰め/厄介ベンチ除去のガスト｜`実打点で相手をKO/価値ある対象が居る`｜~1｜**表示0ゲートで出ない懸念(H3)**。対象価値薄で腐る
- **1227 Lillie's Determination ×4** — 手札更新｜`hand<=4 & combo を抱えていない`｜~1.5｜combo 抱え時に打つと自壊(H5)
- **1175 Brave Bangle ×2** — 技+30｜`相手Active が {ex}`｜~0.3｜**non-ex 相手には完全に無効（verified）**
- **1191 Kieran ×1** — 入替 or +30(ex/V)｜`Dipplin を前に出す or 相手 ex`｜~0.4｜入替不要＆non-ex で腐る
- **1211 Black Belt's Training ×1** — +40(ex)｜`相手Active が {ex}`｜~0.1｜non-ex 相手に無効
- **1213 Judge ×1** — 手札破壊｜`相手手札多 & 自 combo 未保持`｜~0.3｜自 combo 抱え時に自壊(H5)
- **1183 Perrin ×1** — ポケ入替サーチ｜`手札の不要ポケ→必要ポケ`｜~0.3｜不要ポケ無しで腐る
- **1129 Sacred Ash ×1** — ポケを山へ戻し山切れ/再建回避｜`deck<=8 or Dipplin ライン枯渇`｜~0.5｜序盤は腐る
- **1184 Lana's Aid ×1** — トラッシュから Dipplin/エネ回収｜`Dipplin ライン or 草がトラッシュ`｜~0.5｜序盤腐る
- **1080 Unfair Stamp ×1** — KO返しドロー｜`前ターンに自ポケKO`｜~0.4｜非KOターンは使用不可
- **1174 Air Balloon ×1** — 逃げ-2（Dipplin を前に）｜`Active が誤配線した非Dipplin`｜~0.3｜Dipplin が既に前で不要
- **1261 Forest of Vitality ×1** — 草即進化（T2 で速攻 Dipplin）｜`Festival Grounds を犠牲にしても即 Dipplin が要る`｜~0.2｜**FG を消す＝原則封印**（doctrine）
- **100 Goldeen ×1 / 240 Seaking ×1** — 水 Festival Lead 副砲（Seaking Rapid Draw60+2ドロー）｜`Dipplin 不在の場繋ぎ`｜~0.3｜主砲健在なら不要な胴（bench 胴としては可）
- **73 Rellor ×1 / 74 Rabsca ×1** — Rabsca は **Spherical Shield（ベンチ保護）**＆重エネ対角火力｜`スプレッド相手 or 対 高エネ`｜~0.3｜多くの試合で腐る 1-of
- **858 Psyduck ×1** — Damp（自壊特性メタ）＆bench 胴｜`相手が自壊特性を使う`｜~0.2｜大半で腐るメタ枠
- **343 Shaymin ×1** — **Flower Curtain（非ルールボックス自ベンチ保護）**｜`スプレッド/ベンチ狙い相手`｜~0.4｜HP80で Poffin 不可、手貼り/手出し要

**注**: 1-of の水ライン・Rellor/Rabsca・Psyduck・Shaymin・ex用ブースター(Brave Bangle×2/Kieran/Black Belt)・Judge/Perrin/Forest 等、**汎用エンジンが活かせない散らし枠が計 ~14枚**。テレメトリの「14 dead cards」に整合。デッキの芯は Grookey/Applin/Dipplin ＋ Festival Grounds ＋ Poffin/Pad/Bug Set。

---

## 7. 敗因仮説と L2 規則候補（概要 / 詳細は JSON）

**予想敗因分布**: (1) テンポ/主砲不発（エネ誤配線＋進化順ミス＝表示0盲目）＝最大, (2) デッキアウト（掘削過多×単サイド長期戦）, (3) 盤面（ベンチ削り＋スタジアム破壊で打点崩壊）, (4) レース負け（Dipplin 引けず副砲60）。

**中核の一撃**: **Do the Wave の実打点式 `40×bench(FG時)` を共有知覚 `_expected_dmg` に昇格**すれば、手貼り先(H1)・進化順(H2)・ガスト(H3)・昇格・リーサル判定が一斉に主砲を認識する。単一修正でカスケードが解ける。

適用パターン: 「スケール技過小評価→実打点式を共有知覚へ」「エネ意図外→デッキ固有給餌計画」「山切れ→掘り停止＋戻し発火」「gust 価値逃し→実打点ベース対象選択」。

### L2 要否の結論: **L2 必要**
主砲表示0が L0 の手貼り/進化/ガスト/昇格/リーサルを横断的に誤らせ、実測 ~10%WR を説明する。L1（汎用床）ではこのデッキは自走不能。最小構成の L2 は「(a) Do the Wave 実打点を `_expected_dmg` に注入」「(b) 草を Dipplin ライン優先で配線」「(c) Festival Grounds 保持（Forest 封印）」の3点。以降は較正（threshold は暫定）。
