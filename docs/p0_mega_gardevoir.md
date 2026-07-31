# P0 静的解析: `mega_gardevoir`

対象: cabt シミュレータ環境の 60 枚デッキ。手法は blind zero-shot（カード DB + ルール + シミュレータ・プローブのみ）。
中核機構は**プローブで実証済み**。全主張はカードテキスト/数値 or 計測に接地。

## 60 枚ダンプ（27 種）

| cid | ×n | カード | 型 | 要点 |
|---|---|---|---|---|
| 747 | 2 | Mega Gardevoir ex (Stage2/megaEx, HP360, weak Dark×2) | — | **主砲**。Overflowing Wishes(コスト{P}, 0dmg): ベンチ各体に山から基本{P}を1枚付与。Mega Symphonia(コスト{P}, 表示0): **50×(全自軍の{P}エネ数)** |
| 746 | 2 | Kirlia (Stage1, HP100) | P | Call Sign(コスト{P}): ポケ最大3枚サーチ。Psyshot 30 |
| 745 | 3 | Ralts (Basic, HP70) | P | Collect(コスト{C}) ドロー。Rare Candy の起点 |
| 183 | 2 | Smoochum (Basic, HP30) | P | **Delightful Kiss(コスト無=無料)**: 山から基本{P}2枚をベンチ1体に付与。無料加速 |
| 184 | 2 | Latias ex (Basic, HP210, weak Dark) | P | Skyliner(特性): 自軍**基本ポケの逃げ0**。Eon Blade(P+P+C) 200、次ターン攻撃不可 |
| 272 | 2 | Lillie's Clefairy ex (Basic, HP190) | P | Fairy Zone(特性): 相手{N}の弱点を{P}に。Full Moon Rondo(P+C) 20+20/ベンチ(両者) |
| 751 | 1 | Xerneas (Basic, HP120) | P | Geo Gate(コスト{P}): 山から基本{P}ポケ3体をベンチへ(横展開)。Bright Horns 120 |
| 112 | 1 | Munkidori (Basic, HP110) | P | Adrena-Brain(特性,**要{D}**): ダメカン3移送。Mind Bend(P+C) 60+混乱 |
| 140 | 1 | Fezandipiti ex (Basic, HP210) | D | Flip the Script: KO された次に3ドロー。**Cruel Arrow(C+C+C) 相手1体に100**(表示0) |
| 1071 | 1 | Meowth ex (Basic, HP170) | C | Last-Ditch Catch: 登場時サポサーチ。**Tuck Tail(C+C+C): 自身+付与物を手札へ**(自己除去) |
| 117 | 1 | Cornerstone Mask Ogerpon ex (Basic, HP210) | F | Cornerstone Stance: **特性持ちからのダメージ全防御(壁)**。Demolish(F+C+C) 140 |
| 5 | 7 | Basic {P} Energy | P | **Symphonia 唯一の燃料(7枚のみ)** + 加速の対象 |
| 19 | 3 | Telepath Psychic Energy(特殊) | P | {P}供給(Symphonia に**数える**)。{P}ポケに手貼り時、基本{P}ポケ2体をベンチへ |
| 16 | 2 | Prism Energy(特殊) | C | {C}供給。基本ポケ上では全型を1度に1つ。**Symphonia には数えない**(型0) |
| 1079 | 2 | Rare Candy | — | Ralts→**747 直接進化(実証済)**。1ターン目/登場ターン不可 |
| 1225 | 3 | Hilda | — | 進化ポケ+エネをサーチ(Kirlia/747 + 基本{P}) |
| 1121 | 4 | Ultra Ball | — | 手札2枚捨ててポケサーチ(捨札コスト=燃料/コンボの危険) |
| 1152 | 4 | Poké Pad | — | Rule Box 無しポケをサーチ(Ralts/Kirlia/Smoochum/Xerneas/Munkidori)。捨札コスト無 |
| 1146 | 2 | Wondrous Patch | — | 捨札の基本{P}をベンチ{P}ポケへ(**天井修復**) |
| 1097 | 2 | Night Stretcher | — | 捨札のポケ/基本エネを手札へ(再建) |
| 1263 | 2 | Mystery Garden(スタジアム) | — | 手札エネ1枚捨てて手札=場の{P}ポケ数まで引く |
| 1182 | 2 | Boss's Orders | — | 相手ベンチを active に(gust) |
| 1227 | 4 | Lillie's Determination | — | **手札を山に戻して6枚ドロー**(残サイド6なら8) |
| 1219 | 1 | Team Rocket's Petrel | — | 任意トレーナーをサーチ |
| 1241 | 1 | Jacinthe | — | {P}ポケ1体を150回復 |
| 1194 | 1 | Colress's Tenacity | — | スタジアム+エネをサーチ |
| 1092 | 1 | Secret Box | — | 手札3枚捨ててItem+Tool+Sup+Stadiumサーチ |
| 1172 | 1 | Lillie's Pearl(ツール) | — | Lillie's Clefairy が KO 時、相手のサイド-1 |

---

## §1 勝ち筋の仕様（win condition spec）

**主勝ち筋（Symphonia ramp）**: 全自軍に {P} エネを積み、Mega Gardevoir ex で毎ターン
`50 × P_count` を叩き込み、多サイド KO を連取する。

手順:
1. **T1-2**: Ralts をベンチに。Smoochum の *Delightful Kiss*（**無料**）で基本{P}をベンチに貯金。
   Hilda / Ultra Ball / Poké Pad で Kirlia・747・Rare Candy を集める。横展開に Xerneas *Geo Gate*。
2. **T2-3**: **Rare Candy で Ralts→747 に直接進化**（プローブで合法確認）。または Kirlia 経由。
3. **T3-4**: *Overflowing Wishes*（表示0の加速技）でベンチ各体に基本{P}を配り `P_count` を積み上げる。
4. **T4+**: *Mega Symphonia* を毎ターン。`P_count≥5`→250、`≥6`→300… で ex/mega を OHKO。
   *Boss's Orders* で急所を引き摺り出してリーサル/サイド加速。

**実打点式（実証済）**:
`Mega Symphonia damage = 50 × P_count`、`P_count = 全自軍ポケに付いた {P} 供給エネの数`。
- **基本{P}(id5) と Telepath(id19) は数える**。**Prism(id16) は {C} 供給で数えない**（プローブで 3P+2Prism=150 を確認）。
- エネの**位置(active/bench)は打点に無関係**。「どこに貼るか」は打点でなく KO 耐性の問題（下記 tension）。
- 天井: 数える供給源は 基本{P}7 + Telepath3 = **最大10 → 理論500**。ただし全部場に出すのは非現実的で、実効は 5-8 枚 ≈ 250-400。

**副勝ち筋/プランB**:
- **Latias ex** Eon Blade 200（P+P+C、次ターン攻撃不可の自己ロック）。
- **Lillie's Clefairy ex** Full Moon Rondo `20+20×(両者ベンチ)`。序盤の横展開で伸びる。
- **Fezandipiti ex** Cruel Arrow：相手1体に**100（表示0）**、コスト C+C+C は無色で払える万能スナイプ/フィニッシュ。

**複数ドクトリン**（JSON `doctrines` に判別実験）:
- A: Symphonia ramp（主）。
- B: 二次アタッカー tempo（747 が組めない時）。
- C: エネ集中 vs 拡散（高HP体に固めて gust 耐性 vs 全ベンチに撒いて OW シンク増やす）。

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Ralts/Kirlia に先貼りした{P}は 747 に持ち上がる。ただし打点は場所非依存なので、
  先貼りの意義は「昇格後も count に残る」こと（＝どのみち count される）。Rare Candy 直進化なら Ralts の下貼りが 747 に乗る。
- **ダメージ配置 vs ダメージ**: Munkidori Adrena-Brain は「ダメカン移送」(配置系)、Symphonia は通常ダメージ(弱点/軽減適用)。
  Cruel Arrow はベンチに弱点/軽減を無視。
- **技コストの型 vs 供給型**（§型検査、全アタッカー）:
  | アタッカー | コスト | 供給可否 |
  |---|---|---|
  | 747 Overflowing/Symphonia | {P}×1 | ◎ 基本{P}/Telepath 潤沢。1枚で撃てる |
  | Kirlia Call Sign/Psyshot | {P} | ◎ |
  | Latias Eon Blade | P+P+C | ○ P×2 + Prism/基本 で C |
  | Clefairy Full Moon Rondo | P+C | ◎ |
  | Xerneas Geo/Bright | P / P+P+C | ○ |
  | Munkidori Mind Bend | P+C | ◎（ただし Adrena-Brain は**{D}要=Prism基本上のみ**） |
  | Fezandipiti Cruel Arrow | C+C+C | ◎ 無色、基本{P}/Prism で可 |
  | Meowth Tuck Tail | C+C+C | ◎ |
  | **Ogerpon Demolish** | **F+C+C** | △ **{F}は Prism(基本上,1度1型)のみ**＝実戦ではほぼ壁運用 |
  - **型死**: Ogerpon の Demolish と Munkidori の Adrena-Brain は、デッキ唯一の{F}/{D}供給が Prism-on-Basic（2枚, 1度1型）に限られ、実質フル装填困難。両者は**攻撃体でなくユーティリティ**（壁 / Mind Bend / 手札供給）として数えるべき。
- **1ターン1回制**: サポート1・手貼り1・スタジアム1。Smoochum(無料技)・Wondrous Patch(item)・Overflowing Wishes(技) は手貼り枠を消費せず加速できる＝理想シーケンスと衝突しにくい（強み）。
- **デッキアウト地平**: 掘りは重い（Determination 6/8, Hilda, Mystery Garden, Call Sign, Geo Gate）が、Determination は手札を山に戻すため純減が小さく、Night Stretcher/Wondrous Patch が回収。→ deckout は少数派（H10, minor）。
- **ACE SPEC / 特殊エネ / スタジアム**: ACE SPEC 無し。特殊エネ = Telepath(P, count する)/Prism(C, count しない)。スタジアム = Mystery Garden(自前, 競合すれば張り替え合戦)。

### §2b 自己害の棚卸し ★（60枚走査）

| カード | 自己害の種類 | 発火してよい盤面条件（状態変数） | severity |
|---|---|---|---|
| **1121 Ultra Ball / 1092 Secret Box / 1263 Mystery Garden** | 手札→捨札コスト（基本{P}/Rare Candy/747 を捨てると天井崩壊 or コンボ喪失） | 捨てるのが **非燃料**（Prism/重複/腐り札）である間のみ。基本{P}は Wondrous Patch/Night Stretcher で回収可能な時だけ | **catastrophic (H1)** |
| **1227 Lillie's Determination** | 手札を山に戻す（組み上がったコンボ/リーサルを流す） | 手札に **Rare Candy+747 や当ターンのリーサル手が無い**時のみ | **catastrophic (H2)** |
| **1071 Meowth Tuck Tail** | 自身+付与物を手札へ（唯一体なら即負け、count エネ喪失） | **ベンチ≥1 かつ Meowth に必要{P}が無い**時のみ | major (H6) |
| 184 Eon Blade / 751 Bright Horns | 次ターン自己攻撃ロック | 二の矢/別アタッカーが立っている時のみ | minor |
| 112 Adrena-Brain | 自ポケのダメカンを相手へ移送（自軍を回復する側=**利益**）| 常時可（害でない） | — |
| その他（自傷ダメージ/自ミル/強制自入替）| **該当なし** | — | — |

→ §2b 該当あり。catastrophic 仮説 H1・H2 を scenario 付きで宣言（下記）。

## §3 サイド算術

- **747 = megaEx = 3サイド**、HP360。単発 OHKO は 360 必要（環境典型200-330では**2-3発耐える**）。
  **例外: 弱点 Dark×2**。Dark アタッカーが 180 出せば ×2=360 で **OHKO=3サイド献上**（メタに Dark 有→ H8 重大）。
- ex 多数（Latias/Clefairy/Fezandipiti/Meowth/Ogerpon=各2サイド）。**サイド献上の重いデッキ**。Lillie's Pearl で Clefairy の献上を 2→1 に軽減可（ニッチ）。
- **必要 KO数**: 6サイド。Symphonia は単体だが 250-400 で ex/mega を毎ターン OHKO → 2-3サイド/ターン。
- **チェイン要件**: 燃料が場に残る限り 747 は毎ターン `50×count` を撃てる（追加装填不要=攻撃コスト{P}1のみ）。
  主砲が落ちても **2枚目の 747** か二次アタッカーへ繋ぐ（最低2体目まで装填想定）。

## §4 フェーズプラン（計測可能）

| フェーズ | 遷移条件 | 「計画通り」 | 逸脱リカバリ |
|---|---|---|---|
| 序盤 | 場に 747 無し | T2 までに Ralts/Kirlia≥1、Smoochum で{P}≥2 貯金、Rare Candy か Kirlia+747 到達可 | Hilda/Petrel で欠片サーチ、Poké Pad で下位ポケ補充 |
| 中盤 | 747 在場だが P_count<5 | T3 に 747 online、OW≥1、T4 に P_count≥5 | Wondrous Patch/Night Stretcher で{P}回収、Telepath 手貼りで横展開+加速 |
| 終盤 | P_count≥5 かつ 747 active | Symphonia≥250/ターン、Boss's Orders でリーサル対象指定、2サイド/ターン | 主砲被弾なら2枚目747/Latias/Clefairy、Jacinthe で 747 延命 |

**使う/使わないカード宣言**（→ P1 プレイ率照合）:
- 序盤 使う: Poké Pad, Hilda, Ultra Ball, Ralts, Smoochum, Xerneas, 基本{P}, Telepath。使わない: Boss's Orders, Symphonia, Secret Box。
- 中盤 使う: Rare Candy, 747, Overflowing Wishes, Wondrous Patch, Telepath。使わない: Determination(コンボ保持中), Tuck Tail。
- 終盤 使う: Mega Symphonia, Boss's Orders, Fezandipiti(スナイプ), Jacinthe。使わない: Overflowing Wishes(リーサル可の時), 過剰な掘り(山薄時)。

## §5 スコアカード + 対立軸

| 軸 | 点 | このデッキ固有定義 / 検証指標 |
|---|---|---|
| 速度 | 3 | 747 online ~T3(Rare Candy)、Symphonia 実効 ~T3-4 / first_attack_turn |
| 火力曲線 | 5 | 50×count、終盤250-500 / damage per Symphonia |
| 安定性 | 3 | Stage2+燃料プール+コンボ。冗長(747×2, RareCandy×2, サーチ多) / ブリック率 |
| 継戦力 | 4 | 747×2, Night Stretcher/Wondrous Patch 再利用, Fezandipiti ドロー / post_ko_attack_rate |
| 対応力 | 4 | Boss's Orders, Cruel Arrow スナイプ, Ogerpon 壁, Munkidori 混乱/移送 |
| 資源経済 | 3 | **基本{P}7枚は薄い**、捨札コストが燃料を食う。回収あり / energy_attach_share |
| 妨害耐性 | 3 | 手札破壊は Determination リセットで回復可。エネは場に分散→hammer 効き薄。gust で燃料体を狩られる |
| サイドレース | 3 | ex/mega 多く献上重い。747 は Dark 弱点で OHKO 圏 |
| **energy_ceiling(発明)** | 3 | 場に到達する{P}上限=Symphonia キャップ。現実 5-8/理論10 / P_count@firstSymphonia |

**対立軸（tensions）**:
1. **ramp_turns ↔ tempo_damage**: OW(0打点)で積むほど大きいが遅い。均衡「P_count<A かつ非リーサルの間だけ OW、以降 Symphonia」。
2. **energy_ceiling ↔ 掘りの捨札コスト**: Ultra Ball/Secret Box/Mystery Garden が基本{P}を食い天井を下げる。均衡「基本{P}を捨てるのは回収札(Wondrous Patch/Night Stretcher)がある時のみ。Mystery Garden へは Prism を優先廃棄」。**掘るほど強く掘るほど燃料が減る**自己敗因構造がここに出る。
3. **bench_width ↔ bench_exposure**: 横に広げるほど OW シンク増だが、燃料を持つ低HP体(Smoochum30/Ralts70)を gust/スナイプで狩られると count が減る。均衡「{P}は≥100HP 体に集中」。
   → P3 含意: これらの指標を弄る際は**対の指標を同時監視**（whack-a-mole 防止）。

## §6 カード別使用宣言（全27種）

`カード ×n | 意図 | 発火条件(状態変数) | 期待プレイ率 | 腐る条件`

- **747 ×2** | 主砲 | Rare Candy/Kirlia で進化即。P_count≥5 or リーサルで Symphonia、否なら OW | 0.95 | 手札に留まり進化材料欠く
- **746 ×2** | 中継/サーチ | Rare Candy 無い時に Kirlia 経由・Call Sign | 0.7 | Rare Candy 直進化で不要
- **745 ×3** | 起点 | T1-2 ベンチ、Rare Candy 標的 | 0.95 | 複数被り
- **183 ×2** | **無料加速** | ベンチに置き Delightful Kiss で基本{P}2枚を貯金 | 0.7 | 山に基本{P}残無/HP30 で狩られる
- **184 ×2** | 逃げ0特性 + 200プランB | Skyliner 常時、Eon Blade は 747 不在時 | 0.55 | 燃料を Symphonia に回したい時
- **272 ×2** | 序盤/プランB | 横展開時 Full Moon Rondo、Fairy Zone | 0.5 | ベンチ狭く打点伸びず
- **751 ×1** | 横展開 | Geo Gate で基本{P}ポケ3体 | 0.35 | 既にベンチ埋まり
- **112 ×1** | 混乱/移送 | Mind Bend、Adrena-Brain は Prism{D}時のみ | 0.25 | {D}供給無く特性死
- **140 ×1** | 100スナイプ/ドロー | Cruel Arrow で仕留め、KO被弾で Flip 3ドロー | 0.4 | エネ3を主砲に回したい
- **1071 ×1** | サポサーチ | 登場時 Last-Ditch。**Tuck Tail は封印寄り** | 0.3 | 唯一体で Tuck Tail=即負け
- **117 ×1** | 壁 | 特性アタッカー相手に Cornerstone Stance | 0.2 | {F}無く Demolish 不発
- **5 ×7** | **唯一の燃料** | 手貼り/OW/Smoochum/Patch の対象、保存優先 | 0.9 | 捨札コストで浪費
- **19 ×3** | count エネ+加速 | {P}ポケに手貼り→count+基本{P}ポケ2体 | 0.75 | 場に{P}ポケ少
- **16 ×2** | 無色支払い | Eon Blade/Clefairy/Cruel Arrow の C、Mystery Garden 廃棄候補 | 0.6 | Symphonia には無貢献
- **1079 ×2** | 直進化 | Ralts→747(T2+) | 0.65 | Kirlia 既在/1ターン目
- **1225 ×3** | 進化+エネ | Kirlia/747+基本{P}サーチ | 0.7 | サポ枠を Boss's に譲る時
- **1121 ×4** | ポケサーチ | 捨2は**非燃料**のみ | 0.75 | 捨てる非燃料が無い(型死)
- **1152 ×4** | 無コストサーチ | 非ExポケをT1から | 0.7 | 欲しいのが Ex(747)の時
- **1146 ×2** | **天井修復** | 捨札に{P}あり P_count<目標時 | 0.5 | 捨札に基本{P}無/ベンチ{P}ポケ無
- **1097 ×2** | 再建 | KO/捨札コスト後にポケ or 基本{P}回収 | 0.55 | 捨札に欲しい札無
- **1263 ×2** | ドローエンジン | エネ(Prism優先)捨てて手札=場{P}ポケ数 | 0.4 | 手札既に多い/基本{P}しか無い
- **1182 ×2** | gust | 実 Symphonia がベンチ標的 HP≥ でリーサル/2-1 | 0.5 | 主砲打点未詳で撃たない(=H3 バグ)
- **1227 ×4** | リセット+6/8ドロー | 手札に組上りコンボ/リーサル**無**時 | 0.55 | コンボ保持中に撃つと自壊(H2)
- **1219 ×1** | 万能チューター | 欠けた欲しいトレーナー | 0.25 | 既に揃っている
- **1241 ×1** | 延命 | 747 が2発圏で被弾 | 0.2 | 回復対象が満タン
- **1194 ×1** | スタジアム+エネ | Mystery Garden+燃料 | 0.2 | スタジアム既張り
- **1092 ×1** | 爆発サーチ | 捨3が**非燃料**で確保できる時 | 0.2 | 捨てる非燃料無(天井破壊)
- **1172 ×1** | サイド軽減 | Clefairy 前線時に装着 | 0.15 | Clefairy 不使用

### L0 既定の監査（1項目ずつ）
1. **特性無条件使用**: 危険特性は無し（Skyliner/Fairy Zone/Cornerstone は受動、Adrena-Brain は{D}要で不発、Flip/Last-Ditch は利益）。Meowth の Tuck Tail は**技**であり特性でないが、L0 の攻撃選択で暴発し得る（H6）。
2. **手貼り→表示最大アタッカー**: Symphonia/OW は表示0。だが**打点は場所非依存**なので誤配でも count はされる。真の害は「Smoochum/手貼り/Ultra Ball が基本{P}プールを枯らし OW が0付与になる」こと（H4, プローブで OW=0 多発を確認）。
3. **ドローサポ 手札≤5 発火**: **Lillie's Determination がコンボ保持手で発火→山戻し（H2 catastrophic）**。Hilda は手札多いと不発で組成遅延。
4. **サーチ/ボール無条件毎ターン**: Ultra Ball/Secret Box/Mystery Garden が基本{P}/コンボを捨てる（H1 catastrophic）。
5. **進化=進化後表示最大**: 747 は表示0 → Rare Candy/進化が後回しになり win-con 遅延（H7）。競合進化ラインは無いので誤進化自体は起きにくい。
6. **逃げ=装填済みなら逃さない**: Latias Skyliner で基本の逃げ0。型死壁の暴発は限定的。
7. **gust=自 active 表示≥60**: Symphonia 表示0 → **Boss's Orders がほぼ発火せずリーサル/サイドを取り逃す（H3）**。
8. **KO後昇格=表示最大**: 未装填の Smoochum/Ralts を晒す恐れ。装填済みor 2枚目747を昇格すべき。

## §7 敗因仮説と L2 規則候補

**敗因分布(予想)**: (1) prize race 負け=最多（747 Dark OHKO で3枚献上、ex 多く献上重い）、
(2) tempo/組成遅延（Stage2+燃料で立ち上がり遅い、H4/H7）、(3) gust 不発でリーサル取り逃し（H3）、
(4) 自壊（H1/H2 の燃料/コンボ喪失）、(5) deckout=少数（H10）。

**適用パターン（優先順）**:
1. スケール技過小評価 → **実打点式(50×count)を共有知覚へ**（gust/昇格/エネ配分/リーサル判定が参照）＝ H3。
2. 自己害無条件発火 → **盤面ゲート**（H1/H2/H6 の §2b 宣言を実装）。
3. エネが意図しないプール → **給餌計画**（基本{P}保存、Wondrous Patch/Night Stretcher 回収）＝ H4。
4. 攻撃せず抱え込み/早撃ち → **攻撃優先＋fire閾値の均衡**＝ H5。
5. 主砲KO後に殴れない → **2体目装填チェイン**＝ H8。
6. 山切れ → 山薄時の掘り停止＝ H10。

**L2 要否**: このデッキは L0 既定と**構造的にズレる箇所が多い**（表示0スケール技×2、§2b の catastrophic 2件、7枚しかない燃料プール、Dark 弱点の主砲）。特に H1/H2(catastrophic)・H3・H4 は汎用エンジンの既定で**確実に誤作動**する見込みが高く、単なる閾値較正(L1)では吸収しきれない。→ **L2 作成を推奨**（最優先: 実打点式の共有知覚化 と §2b ゲート）。ただし H5/H7/H10 は L1 較正で足りる可能性があり、P1 テレメトリで H1-H4 が支持されるかを先に確認してから L2 実装範囲を絞ること。

---
### プローブ実証ログ（verified の根拠）
- **Mega Symphonia = 50×P_count**: 自己対戦で P_count 3→150 / 4→200 / 5→250 を観測。`[5,5,5,0,0]`(3P+2Prism)=150 で **Prism 非算入**を確認。Telepath は `energies=[5]` で算入。→ `verified:true`
- **Overflowing Wishes**: 付与数 0-3、ベンチ5でも山の基本{P}枯渇時は**0付与**を多発観測（燃料律速を実証）。→ `verified:true`
- **Rare Candy Ralts→747**: EVOLVE ログ `cardId=747, cardIdTarget=745`（Kirlia を飛ばし Ralts から直接）を観測。→ `verified:true`
- 未実証（テキスト接地, `verified:false`）: Full Moon Rondo 式、Cruel Arrow 100、megaEx=3サイド、Prism-on-Basic の{P}/型供給。
