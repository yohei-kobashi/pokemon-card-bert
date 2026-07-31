# P0 静的解析: `manectric`

対象: cabt シミュレータ環境の 60 枚デッキ `decks/manectric.csv`。
手法: カードDB + ルール知識 + シミュレータ・プローブ（BLIND: 他 deck の解析物・agents ソースは未参照）。

## 0. 60枚ダンプ（全カード）

### ポケモン (28枚)
| cid | 枚 | 名前 | 段階/進化元 | HP | 弱点 | 逃 | 主要技/特性（コスト/実効） |
|---|---|---|---|---|---|---|---|
| 511 | 3 | Tynamo | Basic | 40 | 闘 | 0 | Hold Still: 0dmg、自分を10回復（実質パス技） |
| 512 | 3 | Eelektrik | Stage1 ←Tynamo | 90 | 闘 | 2 | **[特]Dynamotor**: 自ターン1回、トラッシュの基本{L}を**ベンチ**の1体へ貼る。Electric Ball LLC=50 |
| 736 | 2 | Electrike | Basic | 70 | 闘 | 1 | Thunder Jolt L=30（自分に10） |
| 737 | 2 | **Mega Manectric ex** | **Stage1(Mega) ←Electrike** | 330 | 闘 | 0 | Flash Ray LL=120（次ターン、Basic の技ダメ無効）/ **Riotous Blasting LLL=200、全エネ捨てで+130=330** |
| 272 | 2 | Lillie’s Clefairy ex | Basic(ex) | 190 | 鋼 | 1 | [特]Fairy Zone: 相手の{無}ポケの弱点を{超}に。Full Moon Rondo P+無=20+20×(両者ベンチ数) |
| 868 | 1 | Mega Eelektross ex | **Stage2(Mega) ←Eelektrik** | 350 | 闘 | 2 | Split Bomb LL: 相手2体に各60 / Disaster Shock LLL=190（{L}2捨てで相手をマヒ） |
| 377 | 1 | Zeraora | Basic | 100 | 闘 | 1 | Scratch 無=20 / Thunder Raid LLL: 全エネ捨て、相手**ベンチのex**1体に210 |
| 140 | 1 | Fezandipiti ex | Basic(ex) | 210 | 闘 | 1 | [特]Flip the Script: 前ターンKOされたら3ドロー。Cruel Arrow 無無無: 相手1体に100 |
| 1071 | 1 | Meowth ex | Basic(ex) | 170 | 闘 | 1 | [特]Last-Ditch Catch: ベンチ出し時サポート1枚サーチ。**Tuck Tail 無無無=60、自身+付与を手札へ戻す** |
| 235 | 1 | Budew | Basic | 30 | 炎 | 0 | Itchy Pollen 0コスト=10、次相手ターン相手はItem使用不可 |
| 343 | 1 | Shaymin | Basic | 80 | 炎 | 1 | [特]Flower Curtain: ルールボックス無しの自ベンチへの技ダメ無効。Smash Kick 無無=30 |
| 946 | 1 | Cramorant | Basic | 110 | 雷 | 1 | Water Gun W=20（**水エネ0＝死に技**）/ Spit Shot 無無無: 全エネ捨て、相手1体に120 |

### トレーナー (22枚) / エネルギー (10枚)
| cid | 枚 | 名前 | 種別 | 効果要点 |
|---|---|---|---|---|
| 1227 | 4 | Lillie's Determination | Sup | 手札を山に戻し6ドロー（残サイド6なら8） |
| 1121 | 4 | Ultra Ball | Item | 手札2枚捨てて任意ポケ1サーチ（**Lを捨てれば discard 種蒔き**） |
| 1152 | 4 | Poké Pad | Item | ルールボックス無しポケをサーチ（**Mega/ex は取れない**） |
| 1086 | 3 | Buddy-Buddy Poffin | Item | 山からHP70以下の基本を2体までベンチへ（=Tynamo/Electrike/Budew のみ） |
| 1239 | 2 | Naveen | Sup | 手札を任意枚捨て→5枚まで引く |
| 1182 | 2 | Boss’s Orders | Sup | 相手ベンチを active に（gust） |
| 1097 | 2 | Night Stretcher | Item | トラッシュのポケ or 基本エネを手札へ（回収） |
| 1174 | 2 | Air Balloon | Tool | 逃げ-2（Eelektrik/Eelektross 用） |
| 1264 | 2 | Battle Cage | Stadium | 相手の技/特性による**ダメカン配置**をベンチに置かせない（ダメージ本体は通る） |
| 1203 | 1 | Surfer | Sup | active とベンチ入替→5枚まで引く |
| 1213 | 1 | Judge | Sup | 両者手札を戻し4ドロー |
| 1219 | 1 | Team Rocket's Petrel | Sup | トレーナー1サーチ |
| 1119 | 1 | Energy Search | Item | 基本エネ1サーチ |
| 1123 | 1 | Switch | Item | active⇔ベンチ入替 |
| 1093 | 1 | Scoop Up Cyclone | Item | **自分のポケ1体+付与を手札へ**（自バウンス） |
| 4 | 8 | Basic {L} Energy | Ene | 雷。Dynamotor/Night Stretcher で循環 |
| 5 | 2 | Basic {P} Energy | Ene | 超。**Clefairy 専用・加速不能** |

---

## §1 勝ち筋の仕様（win condition spec）

### 主勝ち筋 — 「Mega Manectric ピンポン核」（Doctrine A）
Mega Manectric ex（Stage1、Electrike から1段進化）を **330打点の核**として、Eelektrik の Dynamotor でトラッシュの {L} をベンチへ供給し、装填済みの機体を昇格 → Riotous Blasting で毎ターン330を叩く。

**重要な機構的制約（プローブで確認）**: Dynamotor は **ベンチ限定**でしか貼れない。よって active の Manectric に3枚を1ターンで供給する道は無い。手順は必然的に:
1. **T1-3**: Tynamo/Electrike を並べ（Poffin/Poké Pad）、Eelektrik を1-2体立てる。Ultra Ball で{L}をトラッシュに落として discard を種蒔き。
2. **T3-5**: ベンチの Electrike→Mega Manectric に**進化**し、Dynamotor（Eelektrik 数だけ/ターン）で**ベンチ上の Manectric に LLL を貯める**。
3. 昇格（前の空 active を逃げ costs 0 で下げる or KO 交代）→ **Riotous Blasting、全エネ捨てで 330**。捨てた3{L}は discard へ戻り、次ターン Dynamotor で2体目の Manectric を再装填 → ピンポン。

実測: Riotous Blasting の実ダメージ = **330（verified: probe3）**。ただし汎用 greedy 自己対戦40戦で Riotous の発火は **0回**（active に LLL を届けられない）。核が撃てず Flash Ray(120)/Split Bomb(60×2) の小打点でお茶を濁し、**山切れ42%**。→ ここが本デッキ最大の設計‐実装ギャップ。

### 副勝ち筋 / Plan B
- **B1 スプレッド＋スナイプ（Doctrine B）**: Mega Eelektross Split Bomb（60×2）で複数を削り、Zeraora Thunder Raid（ベンチexに210）・Cramorant Spit Shot（120）・Fezandipiti Cruel Arrow（100）で急所を抜いてサイドを刻む。核が組めない/落ちた時の勝ち筋。ただし Eelektross は Eelektrik を消費し**加速エンジンを食う**（§2 参照）。
- **B2 Clefairy 対{無}メタ**: Fairy Zone で相手の{無}ポケの弱点を{超}にし、Full Moon Rondo（超技）×2 で当てる。専用の基本{超}が2枚のみ＆加速不能なので、armできても遅い局所解。

### 実打点式（表示と実効が乖離する技・全て）
| ポケ(cid) | 技 | 実打点式 | verified |
|---|---|---|---|
| 737 | Riotous Blasting | `200`（据置） or **`330`（全エネ捨て選択時 = 200+130）** | **true (probe3: 330 観測)** |
| 272 | Full Moon Rondo | `20 + 20×(自ベンチ数 + 相手ベンチ数)`。Fairy Zone 下で対象が{無}なら **×2** | false（テキスト導出） |
| 868 | Split Bomb | 表示0 → `相手2体に各60`（ベンチはW/R無視） | false |
| 377 | Thunder Raid | 表示0 → `相手ベンチのex 1体に210`（全エネ捨て、W/R無視、ex限定） | false |
| 946 | Spit Shot | 表示0 → `相手1体に120`（全エネ捨て、W/R無視） | false |
| 140 | Cruel Arrow | 表示0 → `相手1体に100`（W/R無視） | false |
| 736 | Thunder Jolt | `30`＋**自身に10** | false |

→ `_expected_dmg` 実装候補: Riotous=330（全エネ有時）, Split Bomb=60×2, snipe系=210/120/100（表示0を昇格せねば gust/昇格/エネ配分が全て過小評価する）。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Electrike に貼った{L}は Mega Manectric へ持ち上がる → **ベンチ Electrike への先貼り**が Dynamotor と別経路の装填になる。下位ライン先貼りは有効。
- **ダメカン配置 vs ダメージ**: Battle Cage は相手の「**配置**」をベンチに無効化（ダメージ本体は通る）。自軍 Split Bomb/snipe は「ダメージ」なので相手 Battle Cage の影響を受けない。相手の配置スプレッドには Battle Cage が刺さる。
- **技コストの型 vs 供給型**: §型検査（下表）。加速は{L}のみ。{超}は Clefairy 専用で加速不能。
- **1ターン1回制**: 手貼りは1回。Dynamotor は Eelektrik ごとに1回（複数体で重ね可、機構は probe で discard→ベンチ貼り確認）。手貼り+Dynamotor×n で1ターンに複数エネ供給可能だが **active には手貼り1回分しか乗らない**。
- **自己ドロー/山掘り → デッキアウト地平**: Lillie's Determination×4（手札を山に戻すので山は減らないが、手札が薄い時に6/8引くと**山が純減**）、Naveen×2/Judge/Surfer/Ultra Ball×4/Poké Pad×4/Poffin×3。開始47枚から毎ターン大量に掘る → **概算 T12-18 で枯渇**。核が330で相手を1確しつつ盤面を膠着させるため試合が長引き、**deckout が主要敗因**（probe: greedy 42%、smart でも 24%）。
- **ACE SPEC / 特殊エネ**: 特殊エネ無し（基本{L}8・基本{P}2）。ACE SPEC 無し。Stadium は自 Battle Cage のみ（競合は相手依存）。

### 型検査（全アタッカー: コスト型 vs 供給型）
供給: 基本{L}×8（+Dynamotor/Night Stretcher で循環）, 基本{P}×2（加速不能）。
| アタッカー | コスト | 供給可否 |
|---|---|---|
| Mega Manectric (Flash/Riotous) | LL / LLL | ◎ {L}＋加速 |
| Mega Eelektross (Split/Disaster) | LL / LLL | ◎ {L}＋加速 |
| Zeraora (Thunder Raid) | LLL | ◎ {L}（撃つと全捨て） |
| Eelektrik (Electric Ball) | LLC | ◎ |
| Cramorant (Spit Shot) | 無無無 | ◯ {L}×3 で可（Water Gun は水0で**死に技**） |
| Fezandipiti (Cruel Arrow) | 無無無 | ◯ {L}×3、ただし active 手貼りのみ（加速不可） |
| Meowth (Tuck Tail) | 無無無 | ◯（自バウンス技） |
| Shaymin (Smash Kick) | 無無 | ◯ |
| **Clefairy (Full Moon Rondo)** | **P + 無** | **△ 基本{P}2枚のみ・Dynamotor不可 → 実質 arm 困難。特性Fairy Zone専用機として扱うべき** |

**型死判定**: Cramorant Water Gun（水0）＝完全死に技（Spit Shot が本体なので機体は生きる）。Clefairy は「加速線に乗らない{超}要求」で**実質エンジン外**。

---

## §2b 自己害の棚卸し（60枚走査）★

| カード(cid) | 自己害の内容 | 発火してよい盤面条件 | severity |
|---|---|---|---|
| **1071 Meowth ex** | **Tuck Tail: 自身+付与を手札へ戻す** | 「他に active になれる控えが**必ず**居る」かつ「盤面を畳みたい/エネを回収したい明確な理由がある」時のみ。**単体active時は絶対禁止（=場から消え即敗北）** | **catastrophic** |
| **1093 Scoop Up Cyclone** | 自分のポケ1体+付与を手札へ（進化Megaを戻すと進化・装填が全消滅） | 「瀕死の自機を救って再展開できる」時のみ。組み上げた Mega Manectric/Eelektross を無条件に戻すのは自壊 | **catastrophic** |
| **1121 Ultra Ball** | 手札2枚捨て | 捨てる2枚が「腐り札 or Dynamotorの燃料になる{L}」の時。**Mega/最後のBoss's/最後の{L}を捨てない** | major |
| 377 Thunder Raid | 全エネ捨て（撃つと自機丸裸） | 相手ベンチexを210で1確し、かつそのエネ損を Dynamotor で戻せる時 | major |
| 946 Spit Shot | 全エネ捨て | snipe価値がエネ損を上回る時（使い切り前提） | major |
| 737 Riotous / 868 Disaster | 「may」全エネ捨て/{L}2捨て | 330/マヒの価値がある時（=ほぼ常に押してよいが、直後の再装填計画とセット） | minor |
| 736 Thunder Jolt | 自身に10 | 70HP なので数回で自壊圏。序盤の繋ぎのみ | minor |
| 1239 Naveen / 1227 Lillie's / 1213 Judge | 手札を捨て/山に戻す | 抱えた核（Mega/Boss's）を流さない手札状態で | minor |
| 511 Tynamo Hold Still | 0打点（実質パス） | active で撃つと1ターン無為。禁止に近い | minor |

**→ catastrophic 該当あり（Meowth/Scoop Up Cyclone）。汎用エンジンは特性・技・トレーナーを無条件発火する既定を持つため、盤面ゲート必須。** §7 に catastrophic 仮説 H1/H2 を設置。

---

## §3 サイド算術（prize arithmetic）

| アタッカー | HP | 取られるサイド | 環境典型打点(~200)で | 与サイド設計 |
|---|---|---|---|---|
| Mega Manectric ex | 330 | **3** | 2発必要（壁として有効） | Riotous 330 で相手を1確 |
| Mega Eelektross ex | 350 | **3** | 2発必要 | 190/60×2 |
| Clefairy/Fezandipiti/Meowth ex | 190/210/170 | **2** | 1確され得る | — |
| Zeraora/Eelektrik/Electrike/Cramorant等 | ≤110 | **1** | 1確される | 210/50/... |

- **構造リスク**: 主砲2種が**Megaで3サイド**。核が2回落ちるとサイド6=負け。しかも全ラインが**弱点=闘**（Manectric/Eelektross/Zeraora/Eelektrik/Tynamo/Electrike/Fezandipiti/Meowth すべて闘弱点）→ 闘アタッカーは実効2倍で 330HP でも1確圏。
- **勝利までのKO数**: 相手構成次第だが、Riotous 330 が毎ターン1確を続ける**チェイン**が生命線。チェイン要件 = 「常にベンチに装填途中の2体目 Manectric（or Eelektross）」。§4 で装填上限=2体を宣言。
- **トレード方針**: こちらが3サイド機を晒す以上、**「毎ターン取り返す」以外に勝ち目が薄い**。膠着（Flash Ray 壁で殴らない）は deckout に直結し敗北。

---

## §4 フェーズプラン（計測可能）★

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| **序盤** | `turn ≤ 3` かつ `Eelektrik in play < 2` | T3までに Electrike≥1 と Eelektrik≥1 が場、discard に{L}≥1（Ultra Ball種蒔き）。使う: Poffin/Poké Pad/Ultra Ball/Tynamo/Electrike。使わない: Riotous/Zeraora/Cramorant/Clefairy攻撃/Tuck Tail | Poké Pad/Poffin で下位ライン再サーチ、Night Stretcher で回収 |
| **中盤** | `Eelektrik ≥ 2` かつ `bench Manectric のエネ < 3` | Dynamotor でベンチ Manectric を LLL に。first_attack_turn ≤ 5。初 Riotous(330) or Flash Ray で先制。使う: Dynamotor/進化/Boss's（急所を引き摺り出す） | 手貼り＋先貼り Electrike で装填、Air Balloon で逃げ確保 |
| **終盤** | `own prizes ≤ 4` or `deck ≤ 12` | ピンポンで毎ターン Riotous 1確、`post_ko_attack_rate` 高。**deck ≤ 12 で掘り(Item search)を停止**し山切れ回避。Boss's で本命を落としリーサル | 2体目未装填なら Split Bomb/ snipe で刻む、Fezandipiti でKO返しドロー |

---

## §5 多面的評価軸（scorecard）＋ 対立軸

| 軸 | 点(1-5) | 本デッキ固有の定義・検証指標 |
|---|---|---|
| 速度 | 2 | 初 Riotous はベンチ装填→昇格が要り T4-5。first_attack_turn。Flash Ray 先行は可(T3) |
| 火力曲線 | 4 | 核 330 は環境最上位級。ただし表示0/scale技が多く過小評価される（expected_dmg 昇格必須） |
| 安定性 | 2 | Tynamo→Eelektrik と Electrike→Mega の2ライン同時要求。ブリック率高。redundancy: Poffin/PokéPad/Ultra Ball 多数 |
| 継戦力 | 3 | Dynamotor 循環＋Night Stretcher 回収で {L} は戻る。ただし Eelektrik が落ちるとエンジン停止 |
| 対応力 | 3 | Boss's×2/Zeraora 狙撃/Battle Cage(配置メタ)/Budew(Itemロック)/Fairy Zone(対無メタ) と道具は多い |
| 資源経済 | **2** | **エネ10枚は薄い**。掘り過多で山切れ地平が近い（T12-18）。掘るほど強く掘るほど死ぬ |
| 妨害耐性 | 2 | 全ライン闘弱点。gust で3サイド機を晒すと痛い。エネ破壊/手札リセットで engine が止まる |
| サイドレース | **2** | 主砲が3サイド。トレード不利、1確チェインで押し切る前提 |
| **[発明] 装填効率** | 2 | Dynamotor がベンチ限定＝active に直接届かない。`energy_attach_share[737,868]` / 総L供給 |

### 対立軸（tensions）★
1. **火力曲線 × 資源経済**: Riotous は毎ターン全エネを捨てる → Dynamotor 再供給が追いつかないと枯れる。バランス点: 「**discard の{L} ≥ 3 かつ Eelektrik ≥ 2 の時のみ Riotous 全捨てモード**。それ未満は Flash Ray 温存 or 手貼り待機」。
2. **安定性(掘り) × 資源経済(山切れ)**: 掘って2ライン揃えるほど山が減る。バランス点: 「**deck > 12 の間だけ無条件サーチ。以降は手札に必要札があれば掘らない**」。
3. **継戦力(Eelektross展開) × 装填効率(Eelektrik維持)**: Mega Eelektross は Eelektrik を消費し Dynamotor ノードを減らす。バランス点: 「**Eelektrik ≥ 3 が場に居る時のみ1体を Eelektross に進化**。エンジン節点を割ってまで Eelektross 化しない」。
P3 含意: tension 指標を締める時は対の指標（掘り停止↔ブリック、全捨て↔枯渇、Eelektross↔Eelektrik数）を同時監視。

---

## §6 カード別「使用宣言」（全60枚）

（発火条件は状態変数、数値は調整可能パラメータ `∈{候補}`。「締めすぎ→止まる燃料」を併記。L0既定監査は §7 表参照。）

| カード ×n | 意図 | 発火条件 | 期待プレイ率/試合 | 腐る条件 |
|---|---|---|---|---|
| Tynamo ×3 | Eelektrik の種 | 序盤、`Eelektrik+Tynamo < 3`。攻撃(Hold Still)は撃たない | 高(0.9) | 中盤以降手札滞留 |
| Eelektrik ×3 | **加速エンジン**。Dynamotor をベンチ Manectric へ | `L in discard ≥ 1` かつ `benched attacker のエネ<3`。給餌先=**装填中の Manectric/Eelektross 優先**（Clefairy/Tynamoへ貼らない） | 高(1.0) | discard に{L}無い序盤 |
| Electrike ×2 | Mega Manectric の種、先貼り台 | ベンチで進化待機。{L}先貼りで持ち上げ | 高(0.9) | Manectric を引けない |
| Mega Manectric ex ×2 | **主砲330** | ベンチで LLL 装填後に昇格→Riotous。`discard L≥3 & Eelektrik≥2` で全捨てモード | 高(0.8) | Electrike/エネ未達 |
| Lillie's Clefairy ex ×2 | **特性Fairy Zone**（対無メタ）主。攻撃は副 | Fairy Zone は場に居れば常時。攻撃は`基本{P}が手貼りできる`時のみ | 低(0.3) | 相手に{無}ポケ不在＆{P}無し→特性だけ置物 |
| Mega Eelektross ex ×1 | スプレッド/マヒ Plan B | `Eelektrik ≥ 3` の時のみ1体進化。Split Bomb で複数削り | 低(0.3) | Eelektrik 2以下（エンジン割る） |
| Zeraora ×1 | ベンチexスナイプ(210) | 相手ベンチに脅威exが居り、`self L≥3` かつエネ損許容時 | 低(0.2) | 相手ベンチにex不在→死に技 |
| Fezandipiti ex ×1 | KO返しドロー特性＋100スナイプ | Flip the Script=前ターン被KO時。攻撃は`3エネ`揃時 | 中(0.5) | 被KO無し＆エネ無し |
| Meowth ex ×1 | **Last-Ditchでサポート確定サーチ**（出した瞬間）。Tuck Tailは非常用 | ベンチ出し時に特性。**Tuck Tail は控え有時のみ** | 中(0.5) | 単体active時に Tuck Tail=即死 |
| Budew ×1 | Itemロックで相手の展開遅延 | 相手が Item 依存（序盤）。0コスト常時 | 低(0.3) | 相手Item薄い |
| Shaymin ×1 | 特性 Flower Curtain（非RB自ベンチ保護）＝Tynamo/Electrike 守り | 場に居れば常時。攻撃はほぼ不使用 | 低(0.3) | 相手がベンチ狙撃しない |
| Cramorant ×1 | Spit Shot 120 スナイプ（使い切り） | 急所120で盤面が動く時。Water Gun は撃たない(水0) | 低(0.2) | エネ乗らない/価値なし |
| Lillie's Determination ×4 | 主ドロー・山リフレッシュ | `手札 ≤ A (A∈{3,4,5})` かつ核を抱えていない。**残サイド6で8ドロー**は序盤限定 | 中(0.6) | 手札を抱えるデッキで A=5固定だと過剰発火→核を巻き込む |
| Ultra Ball ×4 | ポケサーチ＋**discard種蒔き** | `key Pokémon 未所持` or `discard の{L}を増やしたい`。捨て2枚は腐り札/余剰{L} | 高(0.8) | Mega/最後のBoss's/最後の{L}しか捨てられない時は打たない |
| Poké Pad ×4 | 非RBポケ（下位ライン）サーチ | `Tynamo/Eelektrik/Electrike 不足`。**Megaは取れない** | 中(0.6) | 下位ライン充足後は腐る |
| Buddy-Buddy Poffin ×3 | HP70以下基本を2体展開（Tynamo/Electrike/Budew） | `bench < 3` の序盤 | 高(0.8) | 対象基本を山から出し切った後 |
| Naveen ×2 | 手札調整ドロー | `手札 ≤ 4` かつ捨てたい腐り札がある | 中(0.5) | 抱える手で無理引きは核を流す |
| Boss's Orders ×2 | gust リーサル/急所引き摺り出し | `Riotous/snipe で相手ベンチ本命を1確できる`時 | 中(0.5) | active を無為に釣ると1サポ浪費 |
| Night Stretcher ×2 | 核/エネ回収（再建ループ） | `Mega or {L} が discard/prize外で不足`時 | 中(0.5) | 回収対象がトラッシュに無い |
| Air Balloon ×2 | Eelektrik/Eelektross の逃げ確保 | 逃げ2の機体が active に取り残される時 | 中(0.4) | Manectric(逃0)には不要 |
| Battle Cage ×2 | 相手の配置スプレッド無効化 | 相手がベンチにダメカン配置してくる時 | 低(0.3) | 相手が配置系でない |
| Surfer ×1 | 入替＋ドロー | active を退避しつつ引きたい時 | 低(0.3) | 入替不要時 |
| Judge ×1 | 相手手札流し | 相手が展開札を抱えた時 | 低(0.2) | 自分も良い手札を流す危険 |
| Team Rocket's Petrel ×1 | トレーナー確定サーチ（Boss's/Battle Cage等） | 特定トレーナーが欲しい時 | 低(0.3) | 何でも良い局面では非効率 |
| Energy Search ×1 | 基本エネ1確保 | `手札/場に{L}不足`時 | 低(0.3) | エネ足りてる時 |
| Switch ×1 | 無償入替 | 逃げ払えず昇格したい時 | 低(0.2) | Air Balloon/逃0で足りる時 |
| Scoop Up Cyclone ×1 | 瀕死自機の救出・再展開 | `自機が瀕死かつ手札に再展開手段`時のみ | 低(0.15) | 進化済み核を無条件バウンス=自壊 |
| Basic {L} Energy ×8 | 全 {L} アタッカーの燃料＋Dynamotor循環 | 手貼りは**active の攻撃機 or 先貼り Electrike**へ | 高(1.0) | 非攻撃機(Tynamo/Clefairy)へ貼ると死蔵 |
| Basic {P} Energy ×2 | Clefairy 専用 | `Clefairy を攻撃に使う`時のみ手貼り | 低(0.15) | Clefairy 攻撃を使わない=死蔵 |

---

## §7 敗因仮説と L2 規則候補

### 予想敗因分布（プローブ接地）
- **山切れ (deckout)**: greedy 自己対戦42%、smart でも24%（reason=2 観測）。**最有力**。掘り過多＋膠着＋10枚エネの薄さ。
- **盤面/no-active (reason=3)**: smart で 7/25。自バウンス(Tuck Tail/Scoop)や闘弱点1確からの盤面崩壊。
- **サイドレース (reason=1)**: 3サイド機の連続被KO。
- **テンポ**: 核が組めず小打点で押し負け（Riotous 発火0の greedy が象徴）。

### L0 既定監査（各項目がこのデッキで誤作動するか）
| L0 既定 | 本デッキでの誤作動 | 対処パターン |
|---|---|---|
| 特性を無条件使用 | Dynamotor が**Clefairy/Tynamo など非装填先へ**貼る（probe: 272/511 へ流出）。Meowth Last-Ditch は無害。 | 給餌許可リスト（装填中 Manectric/Eelektross 限定） |
| 手貼り＝表示ダメ最大へ | Riotous 表示200 で Manectric に寄るのは○。だが Split Bomb/Zeraora/Cramorant/Fezandipiti は表示0 → エネ来ない。Clefairy 表示20 も過小 | expected_dmg 昇格 |
| ドローサポは手札≤5で発火 | 手札に核を抱える型。Lillie's/Naveen が核ごと流す危険 | 手札所持ベース＋抱え札除外ゲート |
| サーチ/ボールを毎ターン | 山薄でも掘り続け → deckout | `deck>12`ゲートで掘り停止 |
| 進化＝進化後表示最大 | **Eelektrik→Mega Eelektross(190)** を選び加速エンジンを割る危険 | Eelektrik≥3 ゲート |
| 逃げ＝装填済(エネ数)なら逃さない | Clefairy に{L}が乗ると「装填」誤認。Manectric は逃0で無関係 | 型対応の装填判定 |
| gust は表示60以上で | 表示0のsnipe/Split を知らず gust 判断を誤る | 実打点式参照 |
| KO後昇格＝表示最大 | 未装填 Manectric(0エネ)を晒す/ Tuck Tail 機を昇格 | 装填済み＆非自害機を昇格 |

### 汎用パターン適用（優先順）
1. スケール技過小評価 → **実打点式を共有知覚へ**（Riotous330/Split60×2/snipe210,120,100）
2. キーカード(Riotous)発火~0 → **ベンチ装填→昇格ドクトリンの明示＋発火条件を盤面所持ベースに**
3. エネが意図しない体へ → **給餌許可リスト（装填中Megaのみ）＋Clefairy除外**
4. 山切れ負け → **deck>12 掘り停止＋Lillie's の手札抱え除外**
5. 自己害無条件発火 → **Tuck Tail/Scoop Up Cyclone に盤面ゲート（控え存在・救出目的）**
6. 主砲KO後殴れない → **2体目 Manectric の予備装填（上限2）**
7. 消極性（Flash Ray 壁で膠着）→ **攻撃優先均衡（deckout回避のため殴る）**

### L2 要否
**L2 作成を推奨する**。理由: (a) §2b に catastrophic 自己害（Tuck Tail/Scoop）が実在し無条件発火を止める盤面ゲートが要る、(b) 主砲 Riotous の発火が汎用エンジンで実質0＝勝ち筋そのものが機能しない（ベンチ装填→昇格の特殊手順とエネ給餌許可リストが必須）、(c) deckout が主要敗因で掘り停止規則が要る。これらは L0/L1 の汎用閾値では表現できずデッキ固有規則を要する。仮説群は「支持される見込み」ではなく「破れる見込み」が高い。
