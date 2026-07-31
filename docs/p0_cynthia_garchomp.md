# P0 静的解析: `cynthia_garchomp`

対象: Cynthia's Garchomp ex（Fighting Stage2  nuke/sustain デッキ）。cabt シミュレータ環境、post-rotation Standard。
本書の全数値主張は (a) カードテキスト引用、(b) `tools/arena.py` セルフプレイ・ログ計測、のいずれかに接地。
**BLIND**: agents ソース・他 docs・logs/evals/submissions は不参照。カードDBは import のみ。

## 0. 全60枚ダンプ（cid / 名前 / HP / type / テキスト）

| n | cid | 名前 | 種別 | HP | energyType | 技/特性 (cost, dmg) |
|---|-----|------|------|----|-----------|----|
| 4 | 379 | Cynthia's Gible | Basic | 70 | F(6) | Rock Hurl [F] 20（抵抗力無視） |
| 4 | 380 | Cynthia's Gabite | Stage1←Gible | 100 | F | Dragonslice [F] 40 ／ 特性 Champion's Call: 1回/ターン、山からCynthia'sポケモンを手札へ |
| 3 | 381 | **Cynthia's Garchomp ex** | Stage2←Gabite | **330** | F | **Corkscrew Dive [F] 100**（手札6枚まで引く）／ **Draconic Buster [F][F] 260**（このポケモンのエネを全て捨てる） |
| 4 | 341 | Cynthia's Roselia | Basic | 70 | G(1) | Spike Sting [C] 20 |
| 3 | 342 | **Cynthia's Roserade** | Stage1←Roselia | 130 | G | Leaf Step [G][C][C] 80 ／ **特性 Cheer On to Glory: 自分のCynthia'sポケモンの技は相手アクティブに +30（W/R適用前）** |
| 1 | 387 | Cynthia's Spiritomb | Basic | 70 | D(7) | Raging Curse [C] 0（自分ベンチのCynthia'sの全ダメカン×10、弱点無視） |
| 4 | 6  | Basic {F} Energy | Energy | – | F | F供給 |
| 4 | 20 | Rock Fighting Energy | 特殊Energy | – | F | F供給＋貼られたF ポケモンは**相手の技の効果を無効**（ダメージは効果でない） |
| 1 | 10 | Neo Upper Energy | 特殊Energy | – | – | 通常C供給。**Stage2に貼ると全タイプを供給・ただし一度に2個分** |
| 4 | 1227 | Lillie's Determination | Supporter | – | – | 手札を山に混ぜ6枚引く（残サイド丁度6なら8枚） |
| 4 | 1182 | Boss's Orders | Supporter | – | – | 相手ベンチ1体をアクティブに（gust） |
| 4 | 1086 | Buddy-Buddy Poffin | Item | – | – | 山からHP70以下の基本ポケモン最大2体をベンチへ |
| 4 | 1152 | Poké Pad | Item | – | – | 山からルールボックス無しポケモン1枚を手札へ |
| 3 | 1142 | Fighting Gong | Item | – | – | 山から基本F エネor基本Fポケモン1枚を手札へ |
| 3 | 1173 | Cynthia's Power Weight | Tool | – | – | 貼ったCynthia'sポケモン **+70 HP** |
| 2 | 1225 | Hilda | Supporter | – | – | 山から進化ポケモン1＋エネ1を手札へ |
| 2 | 1141 | Premium Power Pro | Item | – | – | このターン、自分のF ポケモンの技は相手アクティブに +30 |
| 2 | 1079 | Rare Candy | Item | – | – | 基本→Stage2 スキップ進化（初ターン/出したてBasic不可） |
| 1 | 1213 | Judge | Supporter | – | – | 両者手札を山に戻し4枚引く |
| 1 | 1122 | Pokégear 3.0 | Item | – | – | 山上7枚からSupporter1枚を手札へ |
| 1 | 1123 | Switch | Item | – | – | アクティブとベンチを入替 |
| 1 | 1097 | Night Stretcher | Item | – | – | トラッシュのポケモンor基本エネ1枚を手札へ |

弱点（DB確認）: **Fighting系（Garchomp/Gible/Gabite/Spiritomb）は弱点=Grass（type1）**。Roselia/Roserade は弱点=Fire。Garchomp ex 逃げ **0**。他 Basic は逃げ1。

---

## §1 勝ち筋の仕様（win condition spec）

### 主勝ち筋（Doctrine D1: Corkscrew 継続機関）
「T2–3 までに **Garchomp ex を場に**（Gabite 経由 or Rare Candy）、Garchomp に **F エネ1**、
ベンチに **Roserade を1体以上**（Cheer On to Glory 常時+30）。以降**毎ターン Corkscrew Dive** を撃つ。
Corkscrew はエネを捨てず手札を6枚まで補充するため**装填を維持したまま無限に撃てる唯一の技**。」
Garchomp 330（Power Weight で400）HP は環境打点（~200–260）を耐える2プライズ壁として殴り続ける。

### 副勝ち筋 / 決め手（Doctrine D2: Draconic 一撃）
相手が Corkscrew(+buff) で落ちない 300+ HP の ex/mega を晒したとき、**Draconic Buster 260(+buff)** で
2プライズを一撃。ただし**全エネ discard** するため、(a) Neo Upper 1枚払い（損失1枚）、(b) Night Stretcher/銀行済み2エネで再装填可、(c) KO 確定、のいずれかを満たすときのみ。無条件連打は D1 を破壊する自傷（§2b）。

### プランB（主砲喪失時）
Garchomp が落ちたら 2体目の Garchomp（計3枚）を Gabite/Rare Candy で再構築、繋ぎに Gabite **Dragonslice 40(+buff)**。
Roserade Leaf Step は **type死**（下記）で使えないため繋ぎにならない。

### 実打点式（★ probe で検証済み）
`R` = 場の Cynthia's Roserade 数（Cheer On to Glory は**加算スタック**、probe で +30×最大4を観測）
`PPP` = このターン使った Premium Power Pro 数（各+30、最大2）。両者は**相手アクティブ限定**、W/R 適用前。

- **Corkscrew Dive 実打点 = 100 + 30·(R + PPP)** — verified（観測 100/130/160/190/220）
- **Draconic Buster 実打点 = 260 + 30·(R + PPP)**、撃った後 **自エネ全 discard** — verified（観測 260/290/320/350、discard ログ確認）
- Rock Hurl = 20 + 30·(R+PPP)、Dragonslice = 40 + 30·(R+PPP) — verified
- **Raging Curse = 10·(自ベンチCynthia'sの総ダメカン) + 30·(R+PPP)**、弱点無視 — verified（scaling+buff 観測）。**通常は自ベンチにダメカンが乗らず ≒0 → 死に技**
- 弱点適用: `(base+30·(R+PPP)) × 2`（Grass 弱点相手など、buff は弱点前）— unverified（テキスト由来）

これらの変数（R, PPP, 装填エネ数）がこのデッキの「育てる資源」であり、`_expected_dmg` 実装候補。
**共有知覚に昇格すべき鍵**: Corkscrew の実打点（表示100だが実効130–190）と Raging Curse の表示0。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Gible/Gabite に先貼りした F は Garchomp まで持ち上がる。→ Rare Candy 前提でも
  Basic Gible に F 先貼りは有効（進化後そのまま装填済み）。ただし **Neo Upper は Stage2 でのみ全タイプ2個供給**
  （probe: Draconic Buster を Neo Upper 単騎で発射＝discard 署名 `(10,)` を観測）。Gible/Gabite に貼った Neo Upper は C のみ供給で **F 技を払えない** → Neo Upper は Garchomp 専用。
- **ダメージ vs ダメカン配置**: 本デッキの技は全て通常ダメージ（配置技なし）。Raging Curse は自ベンチの**ダメカン**を参照するが自分では乗せない。
- **技コストの型要求 vs 供給型（型死検出）**:
  - Garchomp [F]/[F][F]、Gible/Gabite [F]: 供給=Basic F 4 + Rock Fighting 4（=F 8枚）+ Neo Upper(Stage2でF可)。**充足**。
  - **Roserade Leaf Step [G][C][C]: Grass エネが山に0枚 → 永遠に払えない = 型死**。Roserade は特性専用、アタッカーではない。
  - Roselia Spike Sting [C]／Spiritomb Raging Curse [C]: colorless、F で払える（が実益薄）。
- **1ターン1回制の衝突**: 手貼り1・Supporter1・逃げ1。理想シーケンス「Garchomp に手貼り＋Gabite Champion's Call（特性は回数別枠）＋Supporter でドロー」は両立可。ただし **Rare Candy(item)** と **手貼り** と **Poffin/Poké Pad(item)** は同ターン併用可（item 無制限）。
- **自己ドロー/山掘り地平**: Poké Pad4/Poffin4/Gong3/Hilda2/Pokégear1/Judge1/Lillie4 + Gabite特性 + Corkscrew(6まで) と掘りが極端に厚い。60枚中エネ9枚と薄いため、**長期戦で山切れ地平が現実的**（掘り札の無条件連打で加速）。
- **特殊エネ/ACE/スタジアム**: ACE SPEC 無し。スタジアム無し（競合なし）。特殊エネ= Rock Fighting（F供給＋効果無効の攻防両用）、Neo Upper（Stage2 2個供給、Garchomp 専用）。

## §2b 自己害の棚卸し（60枚走査）★

| カード | 自己害の種類 | 発火してよい盤面条件（状態変数） | severity |
|---|---|---|---|
| **Draconic Buster (381)** | **自エネ全 discard**（自資源喪失） | `KO確定 ∧ (Neo Upper払い ∨ 再装填手段所持 ∨ 銀行2エネ)`。素の[F][F]を捨てて次ターン無攻撃になるなら**撃つな** | **catastrophic** |
| Lillie's Determination (1227) | 手札を山へ戻す（抱えた勝ち筋を流す） | `手札に即使用できないカードが多い（good hand でない）`。Rare Candy+Garchomp を握った完成手では撃たない | major |
| Judge (1213) | 自手札も4枚に切る | `自手札 ≤ 3 かつ 相手手札を削りたい`。完成手では自傷 | minor |
| Corkscrew Dive (381) | 6枚まで**引き過ぎ**（山消費・次ドロー価値低下・deckout 寄与） | 常時撃つ（攻撃なので許容）。ただし山薄時に deckout 地平を早める | minor |

**盤面除去系の自己害（自ポケを山/手札に戻す・自傷ダメージ・自ミル）は該当なし**（Spiritomb は自傷せず、他に bounce/self-KO 無し）。
→ 最大の事故源は **Draconic Buster の無条件連打**（L0 は表示260最大を優先し、Corkscrew で足りる場面でもエネを捨てる）。catastrophic 仮説 H1 を立てる。

---

## §3 サイド算術（prize arithmetic）

- **Garchomp ex**: HP 330（Power Weight で **400**）。環境典型 200–260 では**1発で落ちない** = 耐久2プライズアタッカー。
  ただし **Grass 弱点**なら実質 ×2（例: Grass 200 → 400 で 330 素体も 400壁も貫通）→ サイド構造が反転する急所。
- 与えるサイド: Garchomp ex=**2**、Gible/Gabite/Roselia/Roserade/Spiritomb=各1。
  副アタッカー（Gabite Dragonslice）は1プライズ献上で低打点。Roserade は型死で攻撃不可・晒すと1プライズ損。
- **必要KO数**: ex 2枚KO(=4) + 1プライズ2つ、等で6。Garchomp の毎ターン KO 継続が前提。
- **チェイン要件**: 主 Garchomp が落ちても 2体目 Garchomp を即昇格・装填できる状態が理想（`post_ko_attack_rate` で計測）。
  ただし Draconic Buster で自エネを捨てた直後に KO 返されると**装填ゼロの2体目を晒す**最悪手 → H1/H8 連動。

---

## §4 フェーズプラン（計測可能）★

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| **序盤** early | `turn ≤ 2 ∧ Garchomp 未進化` | 場に Gible ライン＋Roselia1、手札/場に F≥1、Gabite の Champion's Call でパーツ確保。Poffin/Poké Pad で盤面展開 | Gong/Hilda/Pokégear でパーツ再サーチ。Gible が無ければ Poké Pad/Poffin |
| **中盤** mid | `Garchomp 場 ∧ Roserade 場 ≥1` | Garchomp に F≥1、**Corkscrew Dive を毎ターン**（実打点 ≥130）。装填維持 | Roserade 未展開なら Roselia 進化を最優先（buff online 化）。F 不足なら Gong |
| **終盤** late | `取得サイド ≥ 2 ∨ 相手主砲露出` | `nonattacking_turn_rate ≈ 0`、`post_ko_attack_rate` 高。300+HP には Draconic Buster を条件付き投入、2体目 Garchomp 準備 | 主砲KO時は Rare Candy/Gabite で再構築、繋ぎに Dragonslice |

**使う/使わないカード宣言**: 序盤=Poffin/PokéPad/Gong/Gabite特性/Rare Candy を使う、Draconic Buster/Boss は使わない。
中盤=Corkscrew/Boss/PowerWeight を使う、Draconic は条件時のみ。終盤=Draconic/Boss/PremiumPowerPro を決め手に。
Roserade Leaf Step は**全フェーズで使わない**（型死）。Raging Curse は自ベンチ被弾時のみ。

---

## §5 デッキスコアカード（1-5）＋ 対立軸

| 軸 | 点 | このデッキ固有の定義 / 検証指標 |
|---|---|---|
| 速度 | 3 | Stage2＋buff用Stage1の二系統。Rare Candy で短縮。Corkscrew 初撃 ~T2–3（`first_attack_turn`） |
| 火力曲線 | 4 | Corkscrew 130–190 持続、Draconic 320+ 瞬発（`power_curve` フェーズ別実打点） |
| 安定性 | 3 | 掘り札過多で高再現だが、二系統セットアップ＋エネ9枚でブリック/エネ枯れ余地（`bench_size_dist`, brick率） |
| 継戦力 | 3 | Garchomp3・Night Stretcher1・再進化可。ただし Draconic 後のエネ再装填が細い（`post_ko_attack_rate`） |
| 対応力 | 3 | Boss gust、Rock Fighting で相手技効果無効、Garchomp 逃げ0 でピボット。エネ破壊/手札破壊への回答は薄い |
| 資源経済 | 2 | **エネ9枚と少**、Draconic 全捨て、掘りで山消費 → deckout 地平が近い（`loss_share[deckout]`, deckCount） |
| 妨害耐性 | 3 | Rock Fighting が相手技効果を無効化（gust 効果自体は移動なので通る）。Grass 弱点と Roselia gust が急所 |
| サイドレース | 3 | 400壁の2プライズ耐久トレードは有利。ただし Grass 弱点で反転（`loss_share[prize|board]`） |
| **[発明] buff_online** | – | ≥1 Roserade の Cheer On to Glory が乗るまでのターン数。これが遅いと Corkscrew が素100で腐る |
| **[発明] reload_after_nuke** | – | Draconic Buster 後、再び攻撃可能になるまでのターン数（0が理想。Neo Upper/Night Stretcher 依存） |

### 対立軸（tensions）★
1. **火力(Draconic) × 資源経済/継戦**: Draconic の +160 瞬発はエネ全捨てを代償にし、無攻撃ターンを生む。
   バランス点: `Draconic は KO確定 ∧ (Neo Upper払い ∨ 再装填所持) のときのみ。それ以外は Corkscrew`。
2. **安定性(掘り) × 資源経済(deckout)**: サーチ/ドロー連打は再現性を上げるが山を溶かす。
   バランス点: `パーツ既充足 ∧ deckCount ≤ K のとき無条件サーチを停止`。
3. **速度(Garchomp優先) × 火力上限(Roserade online)**: 表示打点優先で Garchomp だけ育てると buff が乗らず Corkscrew が素100。
   バランス点: `Garchomp path 確保後、mid までに Roserade を1体 online`。
4. **盤面展開(Poffin で複数Basic) × 妨害耐性(gust 露出)**: ベンチの 1プライズ Basic 増は buff/予備に要るが gust の的。
   バランス点: `ベンチ Basic は必要最小（Roselia1・予備Gible1）＋Roserade`。

P3 含意: これら tension の片側指標（例 draconic_play_rate）を締めたら**対の指標**（nonattacking_turn_rate / deckout / power_curve）を同時監視。

---

## §6 カード別「使用宣言」（全60枚）

| カード ×n | 意図 | 発火条件（状態変数, params） | 期待プレイ率 | 腐る条件 |
|---|---|---|---|---|
| Cynthia's Gible ×4 (379) | Garchomp 種・繋ぎ | 常時場出し。F 先貼り可 | 1.0 | 既に Garchomp 十分 |
| Cynthia's Gabite ×4 (380) | 進化中継＋Champion's Call サーチ機関 | `Gible 場 ∧ 未進化`。特性は毎ターン | 0.8 | Rare Candy 直進化で飛ばす場面 |
| Cynthia's Garchomp ex ×3 (381) | 主砲 | `Gabite 場 ∨ Rare Candy 可`。装填後 Corkscrew 常時 | 1.0 | 全滅・エネ枯れ |
| Cynthia's Roselia ×4 (341) | Roserade 種（buff源） | 序盤ベンチへ最低1 | 0.9 | Roserade 既に十分 |
| Cynthia's Roserade ×3 (342) | **+30 buff 特性**（アタッカーにあらず） | `Roselia 場 ∧ mid まで`。**Leaf Step は撃たない** | 0.8 | 既に buff online（2枚目以降は追加+30） |
| Cynthia's Spiritomb ×1 (387) | 保険（自ベンチ被弾時のスプレッド返し） | `自ベンチCynthia'sにダメカン多 ∧ Raging Curse 実打点 ≥ 相手HP` | 0.05 | 通常（自ベンチ無傷）＝死に札 |
| Basic F Energy ×4 (6) | 主砲装填 | Garchomp/Gible line へ手貼り | 1.0 | エネ過剰 |
| Rock Fighting Energy ×4 (20) | F装填＋相手技効果無効の攻防 | Garchomp へ優先（効果無効付与） | 1.0 | – |
| Neo Upper Energy ×1 (10) | **Stage2で[F][F]単騎払い→Draconic 即撃/再装填** | `対象が Garchomp(Stage2)`。Gible/Gabite には貼らない（C のみ） | 0.5 | Stage2 不在で貼ると腐る |
| Lillie's Determination ×4 (1227) | 主ドロー | `手札に即戦力少（good hand でない）, hand ≤ A, A∈{4,5,6}` | 0.7 | 完成手を握って撃つと自壊 |
| Boss's Orders ×4 (1182) | gust で KO 対象指定 | `Corkscrew/Draconic 実打点 ≥ 相手ベンチ標的HP ∨ 1プライズ狩り ∨ 主砲隔離` | 0.6 | 相手ベンチに好標的なし |
| Buddy-Buddy Poffin ×4 (1086) | 序盤展開（Gible/Roselia/Spiritomb=70以下） | `序盤 ∨ 盤面不足`。**山薄時は抑制** | 0.7 | 中盤以降盤面充足で山消費だけ |
| Poké Pad ×4 (1152) | ルールボックス無しサーチ（種/Roselia/Gabite） | `必要パーツ不足`。パーツ充足＋山薄で停止 | 0.7 | 手札にパーツ既存 |
| Fighting Gong ×3 (1142) | F エネ/F種サーチ | `場エネ < 目標 ∨ F種不足` | 0.6 | エネ充足 |
| Cynthia's Power Weight ×3 (1173) | Garchomp +70HP（400壁化） | `Garchomp 場 ∧ 未装備`。攻撃前に付与 | 0.7 | 非Cynthia's/既装備 |
| Hilda ×2 (1225) | 進化＋エネ同時サーチ | `Garchomp/Roserade or エネ不足` | 0.5 | 両方揃済 |
| Premium Power Pro ×2 (1141) | このターン +30（リーサル/壁貫通） | `+30 で KO 圏に入る ∨ Draconic で ex 貫通` | 0.4 | 既にKO圏 |
| Rare Candy ×2 (1079) | Gible→Garchomp 直進化（検証済） | `Gible 場（出したて不可）∧ 手に Garchomp ∧ 非初ターン` | 0.6 | Gabite 経由で足る場面 |
| Judge ×1 (1213) | 相手手札干渉＋自ドロー | `自手札 ≤ 3 ∧ 相手手札多` | 0.15 | 自完成手で撃つと自傷 |
| Pokégear 3.0 ×1 (1122) | Supporter サーチ（Boss/Lillie 確保） | `手にSupporter無 ∧ 欲しいSup有` | 0.4 | 既にSupporter所持 |
| Switch ×1 (1123) | 型死/被弾ポケの退避 | `アクティブが不適（Roselia等）∧ Garchomp退避不可時`。Garchomp は逃げ0で不要な事多 | 0.2 | Garchomp が active（逃げ0で足りる） |
| Night Stretcher ×1 (1097) | Draconic 後の**F エネ/主砲回収**（再装填の核） | `トラッシュにF エネ or Garchomp/Gible ∧ 再装填/再建したい` | 0.4 | トラッシュ空 |

**L0 既定監査（1項目ずつ）**:
1. 特性無条件使用 → 本デッキの特性(Roserade buff, Gabite サーチ)は**無害**。§2b の危険特性なし。**安全**。
2. エネ手貼り=表示最大へ → Garchomp(260) が表示最大なので**正しく Garchomp へ**。ただし Neo Upper を Gible/Gabite に貼ると型死（C供給）→ H3。
3. ドローサポート hand≤5 発火 → Corkscrew が6枚まで引くため hand≥6 が常態化し **Lillie/Judge が過小発火**の恐れ（good）だが、抱え崩し方向は逆に低い。H9。
4. サーチ無条件毎ターン → Poffin/PokéPad/Gong を山薄でも連打 → **deckout 加速**。H5。
5. 進化=進化後表示最大 → Garchomp(260) > Roserade(80) で **Roserade 進化が後回し→buff online 遅延**。H2（major）。
6. 逃げ=装填済みなら逃がさない → Garchomp 逃げ0 で無害。ただし型死 Roselia/Roserade を「F装填済み」と誤認して active 固定の恐れ。H6。
7. gust=自 active 表示≥60 → Corkscrew100/Draconic260 で常時発火。対象選択が実打点/サイド価値ベースでないと価値逃し。H7。
8. KO後昇格=表示最大 → 未装填 Garchomp を晒す/Draconic 後の空 Garchomp 昇格。H8/H1連動。

---

## §7 敗因仮説の分布と L2 規則候補

**予想敗因分布**:
- **テンポ崩壊（Draconic 乱射でエネ枯れ・無攻撃ターン）**: 最有力。L0 の表示打点最大選択が Corkscrew で足りる場面でも Draconic を選ぶ。→ H1（catastrophic）。
- **火力不足（Roserade 未 online で Corkscrew 素100）**: 進化ヒューリスティックが buff 源を後回し。→ H2。
- **山切れ**: 掘り連打＋Corkscrew 引き過ぎ。長期戦で顕在化。→ H5。
- **盤面/レース**: Grass 弱点相手にサイド構造反転（デッキ構築上の弱点、L2 で救えない部分）。
- **主砲KO後不発**: Draconic 直後に返され空 Garchomp 昇格。→ H8。

**適用パターン**（症状→修正）:
| 症状 | 修正 |
|---|---|
| Draconic 乱射でエネ枯れ | 自己害効果の盤面ゲート（§2b 条件を実装）＋「Corkscrew で KO 可能なら Draconic を選ばない」実打点比較 |
| Corkscrew 表示100 過小評価 | 実打点式（100+30(R+PPP)）を共有知覚へ昇格（リーサル/gust/昇格判定が参照） |
| Roserade buff online 遅延 | Roserade 進化を「buff 源所持ベース」で優先（表示打点でなく特性価値で） |
| 山切れ | パーツ充足＋deckCount≤K でサーチ停止 |
| 主砲KO後不発 | 2体目 Garchomp 予備装填（チェイン上限つき） |
| 型死 Roselia/Roserade を装填扱い | 型対応の装填判定（Grassエネ0=Roserade は永遠に壁扱いにしない/active固定回避） |

## L2 要否の判断
**L2 は必要（作成推奨）**。理由: L0 の「表示打点最大」既定が **(a) Draconic 乱射（catastrophic 自傷）**、
**(b) Roserade buff 源の進化後回し（major）** という**このデッキ固有の2大誤作動**を確定的に踏む。
両者は役割ラベルでなく状態ゲート（実打点比較・特性価値ベース進化）でのみ回避可能で、L1 汎用床では救えない。
一方、特性は無害・エネ型は Garchomp に自然集約・gust は発火する等、L0 が既に正しい部分も多いため、
L2 の介入面は **①Draconic ゲート ②Roserade 優先 ③サーチ throttle** の3点に絞れる（過剰介入不要）。
