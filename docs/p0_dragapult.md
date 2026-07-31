# P0 静的解析 — `dragapult`

対象: cabt シミュレータ環境の 60 枚デッキ `decks/dragapult.csv`。
本書はカードDB（`agents._engine._CARDS/_ATTACKS`, `cg.api`）とルール知識、および
セルフプレイ・プローブ（`cg.game.battle_start`）のみから導出。全機構主張に `verified` を付す。

エネルギー型コード: 0=無色C, 1=草G, 2=炎R, 5=超P, 6=闘F, 7=悪D, 9=竜Dragon。

## 全60枚ダンプ（枚数 / カード / 主要テキスト）

**Pokémon**
- **Dreepy ×4** (119, Basic, 竜, HP70, 弱点なし, 逃1) — Petty Grudge[P]10 / Bite[R,P]40。進化元。
- **Drakloak ×4** (120, Stage1←Dreepy, 竜, HP90, 逃1) — 特性 *Recon Directive*: 自ターン1回、山上2枚見て1枚手札・1枚山底。Dragon Headbutt[R,P]70。
- **Dragapult ex ×3** (121, Stage2←Drakloak, 竜, **HP320**, 弱点なし, 逃1) — Jet Headbutt[C]70 / **Phantom Dive[R,P]200 + 相手ベンチに6ダメカンを好きに配置**。主砲。
- **Munkidori ×2** (112, Basic, 超, HP110, 弱点D, 逃1) — 特性 *Adrena-Brain*: 悪エネ付帯時、自ポケ1体→相手ポケ1体へダメカン最大3個移動。Mind Bend[P,C]60+こんらん。
- **Dunsparce ×1** (305, Basic, 無, HP70, 逃1) — Trading Places[C]0(自ベンチと入替) / Ram[C,C]20。
- **Dudunsparce ×1** (66, Stage1←Dunsparce, 無, HP140, 逃3) — 特性 *Run Away Draw*: 自ターン1回3ドロー、引いたら**自身と付帯を山に戻す**。Land Crush[C,C,C]90。
- **Budew ×1** (235, Basic, 草, **HP30**, 逃0) — Itchy Pollen[無コスト]10 + 相手次ターン アイテム使用不可。
- **Fezandipiti ex ×1** (140, Basic, 悪, HP210, 弱点F) — 特性 *Flip the Script*: 前相手ターンに自ポケがKOされていれば3ドロー。Cruel Arrow[C,C,C]**表示0→相手ポケ1体に100**(ベンチは弱点無視)。
- **Meowth ex ×1** (1071, Basic, 無, HP170, 弱点F, 逃1) — 特性 *Last-Ditch Catch*: ベンチに出した時サポート1枚サーチ。Tuck Tail[C,C,C]60 + **自身と付帯を手札に戻す**。

**Trainer/Supporter/Item/Stadium/Energy**
- **Lillie's Determination ×4** (1227, Sup) — 手札を山に混ぜ6ドロー（残サイド6なら8ドロー）。
- **Boss's Orders ×4** (1182, Sup) — 相手ベンチ1体をアクティブに。
- **Crispin ×2** (1198, Sup) — 山から異なる型の基本エネ2枚、1枚手札・1枚を自ポケに付帯。
- **Rosa's Encouragement ×1** (1240, Sup) — **自分のサイドが多い時のみ**、トラッシュの基本エネ最大2枚をStage2に付帯。
- **Buddy-Buddy Poffin ×4** (1086, Item) — 山からHP70以下の基本ポケ最大2体をベンチに。
- **Poké Pad ×4** (1152, Item) — ルールボックス無しポケ1枚サーチ（ex/V不可）。
- **Ultra Ball ×4** (1121, Item) — **手札2枚トラッシュ**して任意ポケ1枚サーチ。
- **Night Stretcher ×3** (1097, Item) — トラッシュのポケ or 基本エネ1枚を手札に。
- **Rare Candy ×2** (1079, Item) — Basic→そのStage2へ（Stage1飛ばし）。初ターン/出したてのBasic不可。
- **Crushing Hammer ×2** (1120, Item) — コイン表で相手エネ1個トラッシュ。
- **Unfair Stamp ×1** (1080, **ACE SPEC** Item) — 自ポケが前相手ターンにKOされた時のみ。両者手札を山へ、自分5・相手2ドロー。
- **Risky Ruins ×2** (1260, Stadium) — いずれかのプレイヤーが**基本の非悪ポケをベンチに出すたび**そのポケにダメカン2個。
- **Basic {P} Energy ×4** (5), **Basic {R} Energy ×3** (2), **Basic {D} Energy ×2** (7) — 計9枚。

---

## §1 勝ち筋の仕様

**主砲アイデンティティ**: Dragapult ex は **HP320・弱点なし**。環境典型打点(~200-330)で**1発では落ちない**タンク型スプレッダー。
毎ターン Phantom Dive を撃ち続け、**アクティブ200＋ベンチ60スプレッド**で盤面を並行して削り、
Boss's Orders(×4) で削れたベンチを引きずり出して連続KOする「スプレッド＋ガスト」型。

**主勝ち筋（手順）**:
1. T1: Dreepy をアクティブ/ベンチに（Buddy-Buddy Poffin/Poké Pad/Ultra Ball でサーチ）。ベンチにDreepyを複数。
2. T2-3: Dreepy→Dragapult ex（**Rare Candy 直接** or Drakloak 経由）。同時に **炎R+超P** を Dragapult に集める。
3. T3-4以降: 毎ターン **Phantom Dive**。アクティブ200＝大半のアタッカーを1-2発でKO、ベンチに60を「1-2体に集中」して置く。
4. 削れたベンチ threat を **Boss's Orders** でアクティブへ→次のPhantom Diveでとどめ、または **Fezandipiti Cruel Arrow(100)** / **Munkidori Adrena-Brain(最大30移動)** でベンチのままKO。
5. exを2-3枚取らせる前提でも、320タンクでトレードに勝ちながら6サイド。

**実打点式（表示と乖離する技のみ）**:
- **Phantom Dive (121)**: `dmg = 200(アクティブ, 弱点適用) + 60(相手ベンチ, 10刻み6カウンタを任意配分, 弱点/抵抗/効果耐性を無視=配置)`。エネは消費しない（1度組めば毎ターン撃てる）。**育てる資源 = 「スプレッド先の相手ベンチ数」と「既存ダメージとのコンボ」**。`verified: true`（プローブ: 6→1のダメカン配置select確認、アクティブ40/50→0を確認）。
- **Cruel Arrow (140)**: `dmg = 100 を相手ポケ1体（ベンチは弱点無視）`。表示0。`verified: true`（プローブ: 相手盤面総HP減少 max 100-110 を観測、残HP<100の対象はcap）。
- 参考: **Munkidori Adrena-Brain** = 疑似打点 `最大30` を「自ポケの既存ダメージ」から相手へ移送（=自軍回復も兼ねる）。`verified: true`（悪エネ付帯時のみ提示: 3/3, 非付帯: 0/293）。
- Jet Headbutt70 / Dragon Headbutt70 / Land Crush90 / Mind Bend60 / Ram20 は表示＝実効。

**副勝ち筋 / プランB**:
- 主砲が組めない/落ちた時: **2枚目Dragapult** を装填（Night Stretcher で回収可）。
- Dragapultが炎不足で撃てない時: 暫定 **Jet Headbutt(C 70)** で殴りつつ炎を待つ。
- **Fezandipiti Cruel Arrow(100)** 単体でベンチスナイプ（[C,C,C] 3エネ＝重い、フィニッシャ限定）。
- **Dudunsparce/Drakloak** のドローで山を回し、Munkidori/Fezで盤面のダメージを export。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Dreepy/Drakloak に炎・超を先貼りしてから進化すれば Dragapult に載る → **下位ラインへの先貼りは有効かつ推奨**（1手貼り/ターン制約を回避して主砲を早く online に）。`verified: 進化でダメージ/付帯が持ち上がる挙動を観測`（Risky Ruins 20ダメが進化後Dudunsparce 120/140に持ち上がるのを確認）。
- **ダメカンは「配置」**: Phantom Dive スプレッド、Risky Ruins、Munkidori移送はすべて**配置**＝弱点/抵抗/効果耐性を無視。Munkidori Adrena-Brain も配置。
- **技コストの型 vs 供給型（型死検査）**: 供給=P4/R3/D2。
  - Dragapult **Phantom Dive[R,P]** = **炎1＋超1が必須**。`verified: 超2・炎0では Phantom Dive が一切提示されない（504/504）` → **炎が絶対条件**。デッキの炎はわずか3枚 → **炎が最大のボトルネック資源**。
  - Jet Headbutt[C] / Cruel Arrow[C,C,C] / Land Crush[C,C,C] / Tuck Tail[C,C,C] / Ram[C,C] = 無色 → 任意型でOK。
  - Mind Bend[P,C] = 超1＋無色1。Munkidori の主役は特性であり、Adrena-Brain は **悪エネ付帯が条件**（D2枚が唯一の供給）。
  - **型死ポケなし**（全アタッカーの要求型は供給内）だが、**炎の希少性で「実質Jet Headbuttしか撃てないDragapult」に陥るリスク**が中核。
- **1ターン1回制**: サポート1・手貼り1・スタジアム1・特性は各ポケ1回。理想手 = 手貼り(炎or超)＋サポート(Crispin/Boss's/Lillie's)＋Poffin/Ball複数（アイテムは無制限）＋Drakloak Recon＋Phantom Dive。手貼り1回制約が炎+超の同時装填を2ターンに割る → Crispin/Rosa's が加速弁。
- **自己ドロー/山掘り → デッキアウト地平**: Lillie's は**手札を山に戻す**ため実質デッキ減が緩い。Recon(実質±0)、Dudunsparce(自身を山に戻し実質圧縮弱)、Poffin/Ball/Poké Pad で掘る。概算 **山切れは~14-16ターン以降**で、通常決着より遅い → **deckout は minor**。ただし Ultra Ball の手札2捨ては資源とエネ(炎)を削り得る。
- **ACE SPEC**: Unfair Stamp（1枚, 正しく単一）。KO被弾後の手札リセット＋相手を2枚に絞る妨害。
- **スタジアム競合**: Risky Ruins ×2（自前）。相手スタジアムと張替え合戦。**下記§2bの自己害と直結**。
- **特殊エネなし**（全て基本エネ）→ 支払いは素直。Crispin/Rosa's/Night Stretcher で基本エネを回す。

---

## §2b 自己害の棚卸し（60枚走査） ★

該当**あり**。発火してよい盤面条件を状態変数で宣言:

| カード | 自己害 | severity | 発火してよい条件（状態変数） |
|---|---|---|---|
| **Risky Ruins ×2** | 自軍の**基本・非悪ポケをベンチに出すたび-20**（配置）。`verified: 自軍 Dreepy50/70, Munkidori90/110, Budew10/30, Meowth150/170, Dunsparce50/70 を観測。Fezandipiti(悪)は免除` | **major** | `場に出したいのが悪ポケ or 高HP、かつ 相手が非悪基本を多く展開 or Munkidoriでexport可能` の時のみ維持。Budew(30)や次ターンPoffinで低HP基本を並べる直前は**張らない/相手スタジアムで上書きさせる** |
| **Dudunsparce Run Away Draw** | 使うと**自身を山に戻す**（自軍除去） | **catastrophic** | `場のポケ総数 ≥ 2`（Dudunsparceが最後の1体なら使用＝場0体で即敗北）。かつドローが必要な時 |
| **Meowth ex Tuck Tail** | 攻撃で**自身＋付帯を手札に戻す**（アクティブ除去） | **catastrophic** | `ベンチに昇格可能ポケが1体以上`（ベンチ空で撃つと場のアクティブ消滅→昇格不可で敗北）。基本はリセット/回避目的 |
| **Ultra Ball ×4** | 手札2枚**トラッシュ** | **major** | `手札に余剰2枚があり、かつ 捨札に「最後の炎エネ/最後のDragapult/Rare Candy」を含めない`。薄い山を無条件に掘り続けない |
| **Lillie's Determination ×4** | **手札を山に戻す**（勝ち筋パーツを抱えた手で撃つと霧散） | **major** | `手札が低品質（実行可能なPhantom Dive装填コンボを保持していない）`。理想はT1(6サイド→8ドロー) |
| **Unfair Stamp** | 自手札を山に戻す（条件付き, KO被弾後のみ） | minor | `直前にKO被弾`（カード側で強制ゲート済み） |

Boss's Orders / Crispin / Rosa's / Night Stretcher / Poké Pad / Buddy-Buddy / Crushing Hammer / Budew / 各エネ は自己害なし
（ただし Buddy-Buddy は Risky Ruins の自己ダメを**間接誘発**）。

---

## §3 サイド算術

- Dragapult ex: **HP320 / 2サイド**。環境典型200-330では**1発耐える**のが基本 → 「主砲が毎ターン取られる」前提**ではない**（tank）。相手は2ターンかけて落とす必要 → その間にこちらは連続Phantom Dive。
- Fezandipiti ex: HP210/2サイド。Meowth ex: HP170/2サイド。Munkidori:110/1。Dudunsparce:140/1。Dreepy70/Drakloak90/Budew30/Dunsparce70=1。
- **こちらのサイド献上**: ex主体（Dragapult3, Fez1, Meowth1）→ 落とされると2枚ずつ。ただしDragapult自体が落ちにくい設計で相殺。
- **必要KO数**: exを狩れば6サイド=3KO。Phantom Diveアクティブ200＋スプレッド60で「アクティブKO＋ベンチ弱体化」を並行 → Boss's/Cruel Arrow/Munkidoriで**ベンチKOを差し込み1ターン2枚取り**が可能。**チェイン要件 = Dragapult 2体目までの装填**（炎2枚目の確保が鍵）。

---

## §4 フェーズプラン（計測可能）

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| **序盤 early** | `turn ≤ 3 かつ Dragapult ex 未装填` | Dreepyが場に≥1（理想2）、ベンチ構築、Dragapultへ炎or超を1個は載せ始め、Rare Candy/Drakloak を確保 | Dreepy無し=Poffin/Poké Pad/Ultra Ball連打・Lillie'sで引き直し |
| **中盤 mid** | `Dragapult ex が [R,P] 装填完了` | **first_attack_turn(Phantom Dive) ≤ 4**。毎ターンPhantom Dive、スプレッドを**1-2体に集中**、Boss'sで削れたbenchを処理 | 炎不足=Jet Headbutt(70)で繋ぎ＋Crispin/Night Stretcherで炎供給 |
| **終盤 late** | `自サイド ≤ 3` | `post_ko_attack_rate ≥ 0.8`、2体目Dragapult装填済、Cruel Arrow/Munkidoriでベンチのスプレッド対象を刈り取り6サイド | 主砲全滅=Night Stretcherで回収し再装填、Fezで殴る |

**使う/使わないカード宣言**: 序盤=Poffin/Poké Pad/Ball/Lillie's/Crispin/Rare Candy を使う、Boss's/Crushing Hammer/Rosa's は温存。中盤=Boss's/Phantom Dive主体、Poffin/Poké Padは役目薄。終盤=Boss's/Cruel Arrow/Munkidori/Unfair Stamp、序盤サーチは腐る。

---

## §5 スコアカード＋対立軸

| 軸 | 点(1-5) | このデッキ固有の定義 / 検証指標 |
|---|---|---|
| 速度 | 3 | Phantom Dive online = Rare Candy成立で T2-3、通常T3-4。`first_attack_turn(154)` |
| 火力曲線 | 4 | mid以降 200+60/ターン＋スナイプ。表示過小評価に注意（実260盤面ダメージ） |
| 安定性 | 3 | Stage2依存＋炎3枚ボトルネック＝ブリック要因。冗長性: Rare Candy2/Drakloak4/サーチ多数。`nonattacking_turn_rate` |
| 継戦力 | 4 | 320タンク＋Night Stretcher3で再装填容易。Fez Flip the ScriptでKO被弾を燃料化 |
| 対応力 | 3 | 壁=スプレッド＋Boss'sで貫通、エネ破壊=Crushing Hammer、手札破壊=Unfair Stamp。ただし炎割られると停止 |
| 資源経済 | 3 | エネ9枚(炎3が急所)。掘りは強いが Ultra Ball手札2捨てが逆風。山切れ~14T=遠い |
| 妨害耐性 | 2 | **炎エネへのCrushing Hammer/gustでの主砲剥がし** に脆い。炎の実物が3枚しかない |
| サイドレース | 4 | 320タンクで有利トレード。ex被KOの2サイドは主砲の硬さで相殺 |
| **(発明)炎充足度** | 2 | `Dragapultの攻撃のうちPhantom Diveが占める率`＝炎routingの健全性。式変数「炎1必須」に直結 |

**対立軸（tensions）**:
1. **炎希少性 × エネ供給先の広さ**: 炎はわずか3枚、Phantom Dive毎に炎1が要る。炎を非Dragapult(Munkidori/Fez)に貼ると主砲が死ぬ。バランス: `炎は「超を既に持つ/持つ予定のアクティブDragapult」にのみ`。
2. **Risky Ruins の攻(相手基本を削る) × 守(自軍基本を削る)**: バランス: `RR維持は 自ベンチが悪/高HP or Munkidori online の時のみ。Budew等低HP基本を並べる直前は張らない`。
3. **掘り深度 × 山切れ/手札リセット**: Ultra Ballは2枚捨て、Lillie'sは成立コンボを混ぜ得る。バランス: `Phantom Diveが今ターン実行可能になった時点で掘りを止める`。
4. **スプレッド即時 × ガスト後追い**: スプレッド60は後で回収(Boss's/Cruel Arrow/2発目)しないと死に札。バランス: `スプレッドは1-2ターン内にKO可能な相手(現/次アタッカー)に集中、Boss'sを1枚温存`。

P3含意: tension内の指標(例: 炎routing)を締める時は対になる指標(非攻撃ターン率/初撃ターン)を同時監視。

---

## §6 カード別「使用宣言」（全60枚）

`名前 ×n | 意図 | 発火条件(状態変数) | 期待プレイ率/試合 | 腐る条件`

- **Dreepy ×4** | 主砲進化元 | 常に場に≥1、`Dragapult系が場に<2` で追加展開 | ~1.0 | ライン完成後の余剰引き
- **Drakloak ×4** | 進化橋＋Recon Directiveドロー | `Dreepy在場 かつ Rare Candy非採用ルート` | ~0.9 | Rare Candy直進ルート時
- **Dragapult ex ×3** | 主砲/勝ち筋 | `Drakloak在場 or (Rare Candy+Dreepy) かつ 炎超装填計画` | ~1.0(1体必須,2体目狙い) | 3体目・炎が無い盤面
- **Lillie's Determination ×4** | 初動掘り/手札更新 | `手札低品質 かつ 実行可能コンボ未保持`（理想T1/残6サイド=8ドロー） | ~0.6 | コンボ完成手で撃つと自壊
- **Boss's Orders ×4** | 削れたベンチ threat を処理 | `相手ベンチにスプレッドでKO可能/高価値な対象が存在` | ~0.6 | 相手ベンチが空/全て高HP
- **Buddy-Buddy Poffin ×4** | HP70以下基本を一気に展開 | `序盤 かつ ベンチにDreepy/Dunsparce/Budewを増やしたい`（RR自己ダメ注意） | ~0.8(序盤) | 中盤以降/70超の展開狙い時
- **Poké Pad ×4** | 非ルールボックス(Dreepy/Drakloak/Munki/Dun/Budew)をサーチ | `Dreepy/Drakloak/Munkidoriが手札に無い` | ~0.7 | ex主砲が欲しい時(Dragapult不可)
- **Ultra Ball ×4** | 任意ポケ(Dragapult含む)サーチ | `手札に余剰2枚 かつ 捨てるものに炎/Dragapult/Rare Candyを含めない` | ~0.7 | 手札が薄く捨てる余裕なし/薄山
- **Basic {P} Energy ×4** | Phantom Dive/Mind Bendの超 | `Dragapultに超未装備` でDragapultへ | ~1.0装着 | 手貼り1回制約下の余り
- **Night Stretcher ×3** | KO主砲/捨てた炎エネの回収 | `トラッシュにDragapult/Dreepy or 炎エネがあり必要` | ~0.5 | 序盤トラッシュ空
- **Basic {R} Energy ×3** | **Phantom Diveの炎(ボトルネック)** | `Dragapult(超保持/予定)に炎未装備` → 最優先でDragapultへ | ~1.0装着 | 非Dragapultへ貼ると死に(貼らない)
- **Munkidori ×2** | ダメカン移送(疑似30打点＋自軍回復)＋Mind Bend | `悪エネ付帯 かつ 自軍にダメージ かつ 相手にKO/前進する移送先` | ~0.4 | 悪エネ無し/移送先が高HP full
- **Crispin ×2** | 異型エネ加速(炎+超を掴む) | `Dragapultへ炎/超をrouteしたい`（山にエネ有） | ~0.5 | 山にエネ枯渇/既に装填済
- **Crushing Hammer ×2** | 相手主砲のエネ剥がし | `相手アクティブ/主砲に外したいエネが付いている` | ~0.3 | 相手エネ無し/序盤
- **Risky Ruins ×2** | 相手基本の削り/Munki燃料 | §2b条件: `自ベンチが悪or高HP かつ 相手が非悪基本多い or Munki online` | ~0.3 | 低HP基本展開直前(自壊)
- **Basic {D} Energy ×2** | Munki Adrena-Brainの起動鍵/無色払い | `Munkidoriに悪未装備でexport運用したい` | ~0.6装着 | Munki不在
- **Rare Candy ×2** | Dreepy→Dragapult直進(高速化) | `Dreepy在場(出したて不可,初T不可) かつ 手札にDragapult` | ~0.5 | Dreepy未展開/Dragapult手札に無し
- **Dunsparce ×1** | Dudunsparce進化元/序盤の穴埋め(Trading Places) | `序盤に他基本が無い時の埋め` | ~0.3 | 中盤以降
- **Dudunsparce ×1** | Run Away Drawで3ドロー圧縮 | `Dunsparce在場 かつ 場のポケ総数≥2 かつ 追加ドロー要` | ~0.3 | 場が自身1体のみ(使用禁止=catastrophic)
- **Budew ×1** | アイテムロックで相手の展開ターンを妨害 | `相手がアイテム依存(Ball/Poffin)で回す盤面` | ~0.2 | RR下で自壊(30→10)/相手非アイテム
- **Fezandipiti ex ×1** | KO被弾ドロー(Flip)＋Cruel Arrow100スナイプ(悪でRR免除) | `KO被弾後(ドロー) or ベンチに100圏内の対象＋3エネ` | ~0.3 | 3エネ届かず/序盤
- **Meowth ex ×1** | ベンチ出しでサポートサーチ(Boss's/Crispin/Lillie's) | `欲しいサポートが山にあり手札に無い、初出し` | ~0.3 | 既にサポート充足/ベンチ空でTuck Tail禁止
- **Rosa's Encouragement ×1** | 捨札から基本エネ2枚をDragapultへ一括装填 | `自サイド>相手サイド(条件) かつ 捨札にエネ かつ Dragapult装填不足` | ~0.2 | 先行/リード時は使用不可
- **Unfair Stamp ×1(ACE)** | KO被弾後の手札補充＋相手を2枚に絞る妨害 | `直前にKO被弾`(強制ゲート) | ~0.3 | KO被弾していない

**L0 既定の監査（1項目ずつ）**:
1. **特性無条件使用** → Munkidori Adrena-Brain は悪エネ付帯時のみ提示(engine側で正しくゲート済=verified)だが**移送先選択がKO/前進に繋がらない浪費**あり。Dudunsparce Run Away Drawは**場1体時に使うと即敗北**（§2b/H2）。Drakloak Recon/Fez Flipは害なし。
2. **エネ手貼り=表示最大アタッカー** → Phantom Dive表示200でDragapultが最大 → 給餌先は正しい。**だが型を見ず「超2枚」を貼るとPhantom Dive不能(炎必須, verified)**＝最大の誤作動(H1)。
3. **ドローサポは手札≤5で発火** → Lillie'sが該当。手札を山に戻すため**成立コンボ保持時の発火が自壊**(H5/§2b)。
4. **サーチ/ボール無条件毎ターン** → Ultra Ballの2枚捨てが**炎/Dragapult/Rare Candyを捨て得る**(H3)。薄山でも掘り続ける。
5. **進化=進化後表示最大優先** → Dragapult(200)が最大で主砲ライン優先＝整合。Dudunsparce進化は競合せず。Rare Candy運用の学習が要。
6. **逃げ=装填済(エネ数)なら逃さない** → 炎欠落の「超だけDragapult」を装填扱いして壁化する誤作動(型死壁, H1裏面)。
7. **gust=自アクティブ表示≥60で発火** → Phantom Dive200で発火する。**だが対象選択がスプレッドで削れた相手を狙わない**（実打点非認識, H4）。
8. **KO後昇格=表示最大** → 未装填Dragapultを昇格させ2サイド晒し得る(H6)。

---

## §7 敗因仮説と L2 規則候補（要約、詳細はJSON）

**予想敗因分布**: (1) **炎routing失敗でPhantom Dive不能→テンポ喪失/レース負け**（最有力）、
(2) §2b自己害の無条件発火（Risky Ruins自壊 / Dudunsparce・Meowthの自軍除去）、
(3) スプレッド価値の未回収（gust/リーサル判定が実打点非認識）、(4) 未装填主砲の晒し。
deckout は minor。

**適用パターン**（汎用パターン集より）:
- スケール技過小評価 → **Phantom Dive実打点式(200+60)を共有知覚に昇格**（gust/昇格/リーサルが参照）。
- エネが意図しないポケへ → **型対応の給餌計画（Dragapultに炎優先, 非Dragapultへ炎を貼らない）**。
- 型死ポケを装填扱い → **型対応の装填判定（超のみDragapult=未装填として扱う）**。
- 自己害無条件発火 → **盤面ゲート実装（§2bの宣言条件: RRの張り控え, 場1体時Run Away/Tuck Tail禁止, Ultra Ball捨て札保護）**。
- gustが価値を逃す → **実打点＋スプレッド既存ダメージベースの対象選択**。
- 主砲KO後殴れない → **2体目Dragapult予備装填（炎2枚目確保, 上限つき）**。

**L2 要否**: §2bにcatastrophic該当が実在し（H1炎型死・H2自軍除去）、かつ L0既定が
「型を見ない給餌」「特性/アイテム無条件発火」で確実に誤作動する。よって **L2 は必要**。
最優先実装は H1(型対応給餌)・H2(自軍除去ゲート)。これらが未実装なら P4 を通せない。
