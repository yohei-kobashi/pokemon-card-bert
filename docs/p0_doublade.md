# P0 静的解析: `doublade`

対象: cabt シミュレータ環境の 60 枚デッキ `decks/doublade.csv`。
BLIND zero-shot 解析（agents ソース・他デッキ docs・logs 等は不参照。カードDBと library、
シミュレータ probe のみ使用）。

エネルギー型コード: **8 = Metal {M}, 7 = Darkness {D}, 0 = Colorless {C} 要求**。
cardType: 0=Pokémon, 1=Item, 3=Supporter, 5=Basic Energy, 6=Special Energy。

## 0. 全60枚ダンプ（テキスト付き）

### ポケモン（12枚）
| cid | ×n | 名前 | 種別 | HP | 型 | 進化元 | 技/特性 |
|---|---|---|---|---|---|---|---|
| 1065 | 4 | Honedge | Basic | 70 | M | — | **Cut** [1C] 10 |
| 1066 | 3 | Doublade | Stage1 | 100 | M | Honedge | **Weaponized Swords** [2C] 表示0: 「手札から Honedge/Doublade/Aegislash を好きな枚数公開し、公開1枚につき60ダメージ」 |
| 1067 | 3 | Aegislash | Stage2 | 150 | M | Doublade | **Slash** [3C] 80 / **Metal Slash** [1M+3C] 230「次の自分の番、このポケモンは技を使えない」 |
| 547 | 2 | Genesect ex | Basic | 220 | M | — | **Protect Charge** [2M+1C] 150「相手の次の番、このポケモンが受けるダメージ-30(弱抵抗適用後)」／特性 **Metallic Signal**「自分の番に1回、山から進化 {M} ポケモンを最大2枚サーチし手札に加える(公開して山切る)」 |

### エネルギー（12枚）
- **Basic {M} Energy** (8) ×8 — 型 Metal
- **Basic {D} Energy** (7) ×3 — 型 Dark（Crispin の「異なる2型」要件を満たす第2型。全技のコストでは実質 C として支払える）
- **Enriching Energy** (13) ×1 — 特殊。「{C}を供給。手札から貼ったとき4枚引く」

### トレーナー（36枚）
- **Buddy-Buddy Poffin** (1086) ×4 [Item] 山から HP70以下の基本ポケモンを最大2枚ベンチへ（**Honedge=HP70 が対象**、Genesect ex=220 は不可）
- **Ultra Ball** (1121) ×4 [Item] 手札2枚を**トラッシュ**して山から任意ポケモン1枚を手札へ
- **Rare Candy** (1079) ×4 [Item] 基本→手札のStage2に進化（Stage1飛ばし。初ターン/その番に出した基本には不可）
- **Crispin** (1198) ×4 [Supporter] 山から**異なる型の基本エネ2枚**を公開、1枚手札・1枚を自ポケモンに貼る
- **Boss's Orders** (1182) ×4 [Supporter] 相手ベンチを active に引きずり出す（gust）
- **Pokégear 3.0** (1122) ×4 [Item] 山上7枚からサポート1枚を回収
- **Poké Pad** (1152) ×4 [Item] ルールボックス無しポケモンをサーチ（Honedge/Doublade/Aegislash 対象、**Genesect ex 不可**）
- **Night Stretcher** (1097) ×3 [Item] トラッシュからポケモン or 基本エネ1枚を手札へ
- **Lillie's Determination** (1227) ×3 [Supporter] **手札を山に戻して6枚引く**（残サイド6なら8枚）
- **Hilda** (1225) ×2 [Supporter] 山から進化ポケモン1＋エネ1を手札へ
- **Enriching Energy** は上記エネ欄

合計 12 + 12 + 36 = 60。

---

## §1 勝ち筋の仕様

このデッキは **Metal 単色トゥールボックス**。攻撃者が3系統あり、いずれも状況で切替える。

### 実打点式（表示と実効が乖離する技 — 最重要）
- **Weaponized Swords (Doublade, 1066)** — 表示 **0**。
  `damage = 60 × N`、`N = 手札の {Honedge,Doublade,Aegislash} のうち公開する枚数`。
  **verified: true**（probe: 60/120 のダメージ値が実測され、-60 は Doublade が盤面に居るときに 35/37 発生。§0参照）。
  - 公開はカードを**消費しない**（reveal のみ）→ 手札にライン札を抱える限り毎ターン高打点を反復可能。
  - `N` の資源上限 = デッキのライン札総数10（Honedge4+Doublade3+Aegislash3）− 進化で盤に出した分 − active の1体。実戦 N=2〜3（120〜180）が現実解、稀に4(240)。
  - **この式が L0 に見えない**（表示0）ことが本デッキ最大の欠陥点。エネ配分・gust・リーサル・昇格の全判断が Doublade を過小評価する。
- **Metal Slash (Aegislash, 1067)** — 表示 230、**verified(値): true**（230 実測）。副作用「次の自分の番、技不可」= **自己ロック**。lockout の挙動は text 明示だが behavioral probe 未実施 → **verified: false（副作用）**。
- **Protect Charge (Genesect ex, 547)** — 150 + 被ダメ-30（次の相手番）。表示=実効、乖離なし。

### 主勝ち筋（手順）
1. **T1–2**: Poffin/Poké Pad/Ultra/Pokégear/Hilda/Metallic Signal で Honedge を盤に、ライン札とエネを手札に集める。エネを active 候補に手貼り開始。
2. **T2–3**: **Doublade に進化 → Weaponized Swords**（2C と安価）で 120〜180 を叩きつつ、手札にライン札を **温存**（=打点維持）。または Rare Candy で **Aegislash 直行**。
3. **中盤以降**: 相手の HP 帯に応じて
   - 大型(2プライズ/HP220)を **Metal Slash 230** で一撃（ただし自己ロック→翌番は別アタッカー）
   - 継続圧は **Doublade 120–180** か **Slash 80**（ロック無し）
   - **Genesect ex** は主に **Metallic Signal エンジン兼壁**（220HP, 被ダメ-30）。
4. **Boss's Orders ×4** で相手の未装填/低HPを引きずり出し盤面を制圧、6プライズを取り切る。

### 副勝ち筋 / プランB
- 主砲 Aegislash が組めない/落ちた → **Doublade スワーム**（Stage1 は安く反復容易、単プライズ）で刻み続ける。
- Doublade も細い → **Genesect ex Protect Charge 150** を壁兼アタッカーに（HP220で場持ち）。
- 攻撃者が枯れたら **Night Stretcher + Metallic Signal + Hilda** でライン札とエネを回収して再建（全て単プライズなので再建コストが低い）。

### 複数ドクトリン（決め打ち禁止 → JSON doctrines で A/B 判別）
- **Doctrine A「Aegislash ビッグスイング」**: Rare Candy で Aegislash まで上げ、230/80 を主軸。判別: `energy_attach_share[1067]` 高・`play_rate[1067]` 高・230/80 のダメージ事象が支配。
- **Doctrine B「Doublade ハンドスケール・スワーム」**: Stage1 に留まりライン札を手札に温存、120–180 を反復。判別: `play_rate[1066]` 高・手札ライン札数 ≥2 が持続・120/180 事象が支配・`energy_attach_share[1066]` 高。
- 実像は**ハイブリッド**（Doublade 早期＋安価、Aegislash が KO ティア、Genesect がエンジン/壁）。判別指標 = 「Doublade発の攻撃比率」と「攻撃時の平均手札ライン札数」。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Honedge→Doublade→Aegislash でエネ継承。Honedge/Doublade へ**先貼り**しておけば Aegislash 完成時に装填済み。全技コストが C 主体なので先貼りの取り回しが良い（Metal Slash の 1M だけ Metal を1枚確保）。
- **ダメージ vs ダメカン配置**: 全攻撃は通常**ダメージ**（probe: type16 log `putDamageCounter: false`）。弱点・抵抗・軽減が適用される（Protect Charge の -30 も「ダメージ」側）。
- **技コストの型 vs 供給型**（全アタッカー突合、下表）: **型死なし**。全技 C 主体で Metal 供給8が潤沢。M 要求は Metal Slash(1) と Protect Charge(2) のみ。Dark(3) と Enriching(1) は C として C 枠を払えるが **M 枠は払えない**。
- **1ターン1回制**: サポート1・手貼り1・逃げ1。Crispin(貼り)/Boss's/Lillie's/Hilda は互いにサポ枠を食い合う（1ターン1枚）。手貼り1 なので複数エネ要求技の立ち上がりは複数ターン要。
- **自己ドロー/山掘り**: Enriching+4、Lillie's+6/8、Ultra/Poffin/Poké Pad/Pokégear/Hilda/Metallic Signal と極めて厚い。**デッキアウト地平**: 掘りを無条件連打すると 60枚デッキでも長期戦で山切れ得る（副次的敗因）。Metallic Signal は毎ターン最大2枚山を削る。
- **ACE SPEC / スタジアム / 特殊エネ**: ACE SPEC なし。スタジアムなし。特殊エネ = Enriching Energy ×1（C供給+4ドロー、M枠は不可）。

### エネルギー 型 vs 供給 突合表
| アタッカー | 技 | コスト | 要求 | 供給で払えるか |
|---|---|---|---|---|
| Honedge 1065 | Cut | [1C] | 任意1 | ○ 何でも |
| Doublade 1066 | Weaponized Swords | [2C] | 任意2 | ○（最安の実アタッカー） |
| Aegislash 1067 | Slash | [3C] | 任意3 | ○ |
| Aegislash 1067 | Metal Slash | [1M+3C] | **Metal≥1**+任意3 | ○（Metal8潤沢。ただし M を1枚確保する装填判定が要る） |
| Genesect ex 547 | Protect Charge | [2M+1C] | **Metal≥2**+任意1 | ○（Dark/Enriching では M 枠を払えない → 型配分注意） |

---

## §2b 自己害の棚卸し（60枚走査・severity 判定）

該当 **あり**。汎用エンジンは特性・トレーナーを無条件発火するため下記が事故源。

| カード | 自己害 | 発火してよい盤面条件（状態変数） | severity |
|---|---|---|---|
| **Ultra Ball** (1121) | 手札2枚を**トラッシュ** | `hand_line_count − 2 ≥ needed_ammo(=2)` かつ トラッシュ対象に Rare Candy / 未使用進化パーツ / ライン札を含めない | **catastrophic**（Doublade の弾＝ライン札や進化パーツを捨てると打点・進化が直接崩壊） |
| **Lillie's Determination** (1227) | **手札を山に戻す**(全部) | `hand_line_count ≤ 1`（＝温存中の弾が無い）ときのみ。ライン札/Rare Candy 温存中は撃たない | **catastrophic**（温存した Weaponized Swords の弾を丸ごと山へ戻す＝勝ち筋の中核喪失） |
| **Metal Slash** (1067) | **次の番 技不可**（自己ロック） | 「これで KO 成立」or「翌番に別アタッカーが装填済み」or「相手の返しが無害」 | major（テンポ。ping-pong 化） |
| **Metallic Signal** (547 特性) | 山を最大2枚消費（自己ミル微） | 常時可。ライン札補充で**むしろ有益**。山薄(deckCount低)時のみ抑制 | minor |
| **Enriching Energy** (13) | （害なし。貼れば+4ドロー） | — | none |
| Crispin(1198) | 貼り先を誤ると死にエネ | M枠を要る攻撃者へは M を、Dark は C 枠専用に | minor |

その他 52 枚（Honedge/Doublade/Aegislash/Genesect本体, Poffin, Rare Candy, Boss's, Pokégear, Poké Pad, Night Stretcher, Hilda, 基本エネ）に**手札破棄・自軍除去・自傷・強制入替は無し**。

→ §7/JSON に Ultra Ball と Lillie's の **catastrophic** 仮説（scenario 付き）を計上。

---

## §3 サイド算術

- **HP/プライズ**: Honedge 70(1)・Doublade 100(1)・**Aegislash 150(1)**・**Genesect ex 220(2)**。
- 環境典型打点 ~200 前提だと **Aegislash(150) は毎ターン取られる**前提。ただし**単プライズ**なので取られても損は小さい（230 打点を1プライズで供給＝**トレード有利**）。
- **Genesect ex は2プライズの露出リスク**。active で KO されると一気に不利。ベンチでエンジン運用が原則。
- 必要 KO 数: 相手が単プライズ主体なら6KO、ex 主体なら3KO。
- **チェイン要件**: Metal Slash は自己ロック → **常時攻撃には2体の装填済みアタッカーが必要**（Aegislash A/B 交互、または Aegislash+Doublade+Genesect のローテ）。Doublade/Slash 単独運用なら1体でも継続可。

---

## §4 フェーズプラン（計測可能）

| フェーズ | 遷移条件(状態変数) | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| **序盤** | `turn ≤ 2` かつ Aegislash 未完成 | Honedge が盤に≥1、手札にライン札≥2 と Rare Candy/エネ、`first_attack_turn ≤ 3`（Doublade 120+ か Cut）。Metallic Signal を毎ターン発火 | Pokégear/Poké Pad/Hilda で欠片を探す。Poffin で Honedge 補充 |
| **中盤** | Aegislash≥1 か 装填 Doublade(2C) 完成、`hand_line_count ≥ 2` | 毎ターン KO(120–230)、装填済みアタッカー2体維持、手札に弾温存、Boss's で狙撃 | Night Stretcher で落ちた攻撃者/エネ回収、2体目を装填 |
| **終盤** | `prize_remaining ≤ 3` | 2プライズ標的を Metal Slash 230 で処理 or gust でリーサル。**deckout 回避で掘りを止める**。Genesect は壁 | Boss's でサイド取り切り。山薄なら Ultra/Poffin/Metallic Signal を抑制 |

**使う/使わないカード宣言**: 序盤=Poffin/Poké Pad/Pokégear/Ultra/Metallic Signal/Rare Candy を使う、Boss's は温存。中盤=Boss's/Crispin/Metal Slash 解禁。終盤=掘り系(Ultra/Poffin)を**止める**、Boss's 集中、Lillie's は手札に弾が無い時のみ。

---

## §5 スコアカード＋対立軸

| 軸 | 点(1-5) | このデッキ固有の定義 / 検証指標 |
|---|---|---|
| 1 速度 | **4** | Doublade が 2C・T2 で 120。Rare Candy で Aegislash 直行。`first_attack_turn` |
| 2 火力曲線 | **4** | 120→180→230 と伸びる。Metal Slash 220HP を一撃。`expected_dmg` 分布 |
| 3 安定性 | **4** | サーチ7種＋Rare Candy。ただし「正しい札を手札に抱える」依存でブリック中程度。`nonattacking_turn_rate` |
| 4 継戦力 | **4** | 単プライズ主体で再建安い。Night Stretcher/Metallic Signal/Hilda 回収。`post_ko_attack_rate` |
| 5 対応力 | **3** | Boss's×4 で壁を貫通。ただしエネ破壊/手札破壊への回答なし、スプレッドで HP70/100 が崩れる |
| 6 資源経済 | **3** | Metal 供給潤沢。だが Ultra 破棄・Lillie's 戻しが手札を圧迫、deckout 尾リスク。`loss_share[deckout]` |
| 7 妨害耐性 | **2** | **勝ち筋が手札のライン札温存に依存** → Iono/Marnie 系ハンドリセットが致命的。Genesect gust=2プライズ。`hand_size_dist` |
| 8 サイドレース | **4** | 単プライズ 230 攻撃者＝トレード有利。Genesect の2プライズが唯一の穴 |
| ★発明軸 **弾レジリエンス** | **2** | 「攻撃時の手札ライン札数」が打点そのもの。ハンド干渉・自己破棄(Ultra/Lillie's)で直接目減り |

### 対立軸（tensions）
- **T1 手札の弾 ↔ 盤面展開/進化**: ライン札を手札に温存すると Doublade 打点↑だが、進化やベンチ厚みに回せない。バランス点: `hand_line_count ≥ 2` を維持しつつ、超過分のみ盤へ。
- **T2 Metal Slash バースト ↔ 継続テンポ**: 230 は強いが自己ロック。バランス点: Metal Slash は「KO 成立」or「翌番に第2アタッカー装填済み」の時のみ、それ以外は Slash80/Doublade。
- **T3 掘りエンジン ↔ 自己破棄+デッキアウト**: 掘るほど欠片は揃うが Ultra 破棄・Lillie's 戻し・山薄化が進む。バランス点: 欠片が揃うまで掘り、揃ったら停止。`hand_line_count < 2` の間は破棄/戻し系を撃たない。
- **T4 Genesect エンジン/壁 ↔ 2プライズ負債**: 220HP で場持ちする一方、active で落ちると-2。バランス点: Metallic Signal はベンチから、active 送りは「壁」用途に限定。

P3 含意: T1/T3 の指標を修正する際は対の指標（盤面厚み / deckout・play_rate）を同時監視しないと whack-a-mole。

---

## §6 カード別使用宣言（全60枚）

`意図 | 発火条件(状態変数) | 期待プレイ率/試合 | 腐る条件`

- **Honedge 1065 ×4** | ライン起点＆Doublade の弾 | 盤に0体なら即出し（Poffin対象）。手札の余剰は温存 | 序盤 ~1.0 | 4枚全て手札滞留で盤が薄い時
- **Doublade 1066 ×3** | 主力・ハンドスケール砲。盤に1・手札に弾で温存 | active 化: Honedge 進化可かつ 2C 見込み。弾: 手札 ≥1 keep | ~0.9 | 全て盤に出て手札の弾0
- **Aegislash 1067 ×3** | KO ティア(230/80) | Rare Candy or 段階進化で完成、Metal≥1 装填時に Metal Slash | ~0.7 | エネ届かず/Doublade で足りる時
- **Genesect ex 547 ×2** | Metallic Signal エンジン＋壁 | ベンチに1体、毎ターン特性発火。active は壁用途のみ | ~0.8 | active で晒され2プライズ献上
- **Basic {M} Energy 8 ×8** | 主エネ・M枠 | アタッカーへ手貼り、Metal Slash/Genesect の M 枠優先確保 | 高 | —
- **Basic {D} Energy 7 ×3** | Crispin 第2型・C 枠埋め | C コストへ。**M 枠には貼らない** | 中 | M 要求技へ誤配分すると死にエネ
- **Enriching Energy 13 ×1** | +4ドローの C 供給 | 手札から貼れる時（テンポ加速）。M枠不可 | ~0.6 | M枠しか空いてない攻撃者へ貼ると無駄
- **Buddy-Buddy Poffin 1086 ×4** | Honedge 展開(HP70対象) | 盤の Honedge/基本が薄い時 | ~0.9 | ライン基本を既に十分展開後
- **Ultra Ball 1121 ×4** | ポケサーチ | 欲しいポケがある**かつ 破棄2枚に弾/Rare Candy/進化を含めない** | ~0.8 | 手札が弾のみ＝破棄で勝ち筋損傷
- **Rare Candy 1079 ×4** | Honedge→Aegislash 直行 | Aegislash 手札＋Honedge 盤（その番出しでない、初手番でない） | ~0.7 | Doublade 運用で十分な時
- **Crispin 1198 ×4** | エネ加速(型2種) | エネ不足の攻撃者がいる。Metal を M枠へ | ~0.6 | 既に装填十分でサポ枠を食う時
- **Boss's Orders 1182 ×4** | gust リーサル/壁貫通 | 相手ベンチに低HP/未装填/2プライズ標的が居て、実打点で取れる時のみ | ~0.7 | 盤面確立前に浪費
- **Pokégear 3.0 1122 ×4** | サポ回収 | 欲しいサポ(Crispin/Boss's/Hilda/Lillie's)が要る時 | ~0.7 | サポ充足時
- **Poké Pad 1152 ×4** | 非ルールボックス(ライン)サーチ | ライン札が欲しい時（Genesect不可） | ~0.7 | ライン充足時
- **Night Stretcher 1097 ×3** | 落ちた攻撃者/エネ回収 | トラッシュに攻撃者 or エネがあり再建/装填が要る時 | ~0.5 | 序盤トラッシュ空
- **Lillie's Determination 1227 ×3** | 手札総入替ドロー | **`hand_line_count ≤ 1`（弾温存中でない）**かつ手札が細い時のみ | ~0.4 | 弾を抱えた手で撃つ＝勝ち筋を山へ
- **Hilda 1225 ×2** | 進化＋エネ同時サーチ | 進化パーツとエネが欲しい立ち上がり | ~0.5 | 盤面完成後
- **Genesect ex(再掲)/エネ小計は上記** | | | |

（60枚 = ポケ12 + エネ12 + トレーナー36。上表で全 cid を網羅。）

---

## §7 敗因仮説と L2 規則候補

### 予想敗因分布
1. **Weaponized Swords 過小評価**（表示0）→ Doublade にエネが行かず・攻撃されず・リーサル/gust が式を無視 → テンポ負け。**最大要因**。
2. **自己破棄事故**（Ultra Ball が弾/進化を破棄、Lillie's が弾を山へ）→ 盤面/レース崩壊。
3. **Metal Slash 自己ロック ping-pong**（230表示を無条件連打→隔ターン攻撃）→ テンポ喪失。
4. **エネ型誤配分**（Dark/Enriching が M 枠へ→ Metal Slash/Genesect 撃てず）。
5. **Genesect 2プライズ露出** / **spread による HP70-100 崩壊** / **ハンドリセットで弾消失**。
6. 副次: 掘り連打による **deckout**。

### 適用パターン（優先順）
1. スケール技過小評価 → **実打点式 `60×hand_line_count` を共有知覚 `_expected_dmg` に昇格**（gust/昇格/エネ配分/リーサルが参照）。**最優先**。
2. 自己害無条件発火 → **盤面状態ゲート**（Ultra/Lillie's を §2b の宣言条件で gate）。
3. 型死ポケを装填扱い → **型対応装填判定**（M 枠は Metal で埋める）。
4. 主砲KO後殴れない/自己ロック → **第2アタッカー予備装填**（Metal Slash は KO or 予備装填時のみ）。
5. gust が価値を逃す → **実打点＋サイド期待値ベース**の対象選択。
6. 山切れ → 山薄時の**掘り停止**（Ultra/Poffin/Metallic Signal 抑制）。

### L2 要否の判断
**L2 必要**。理由: (a) 勝ち筋の中核が表示0のスケール技で L0 に**原理的に不可視**、(b) L0 の無条件発火既定（Ultra 破棄・Lillie's・Metal Slash 連打・型無視配分）が catastrophic 級の自己害を起こす。
最重要 L2 規則 = **`_expected_dmg[Doublade] = 60 × min(手札ライン札数, 温存可能数)` を実装**し、その上で Ultra/Lillie's を `hand_line_count` でゲート、M 枠の型配分を保証する。仮説群の多くが破られる見込みが高く、L1 では不足。

---

## L0 既定 監査リスト（本デッキでの誤作動可否）

| L0 既定 | 本デッキでの判定 |
|---|---|
| 特性は無条件に使う | Metallic Signal は**有益**（ライン補充）。山薄時のみ抑制。**概ねOK** |
| エネ手貼りは表示ダメ最大へ | **誤作動**: Doublade 表示0で無視され、Aegislash/Genesect に偏る。式昇格が必要 |
| ドローサポは手札≤5で発火 | **誤作動**: Lillie's は手札を山へ戻す→弾温存中に撃つと自壊。所持ベースゲートへ |
| サーチ/ボールを無条件毎ターン | **誤作動**: Ultra は破棄2＝弾/進化を捨てうる。山薄でも掘り続け deckout |
| 進化は進化後表示ダメ最大優先 | 概ねOK（Aegislash 230 が最大で妥当）。ただし Doublade 打点が式で見えないと Stage1 運用を過小評価 |
| 逃げは装填(エネ数)なら逃がさない | 軽微。型死壁が無いので誤作動小 |
| gust は自 active 表示60以上時 | **誤作動**: Doublade 表示0でゲートを跨げない。実打点式参照が必要 |
| KO後昇格は表示最大 | **誤作動リスク**: 未装填 Aegislash や Genesect ex(2プライズ)を晒しうる。装填済み単プライズを優先 |

---

## 検証ログ（probe）
- Weaponized Swords スケール: **verified true** — self-play(random) 150 局で type16 ダメージ値 -60/-120 を実測、-60 の 35/37 は Doublade が盤面在時。60×N を確認。公開はカード非消費。
- ダメージ vs ダメカン: type16 log `putDamageCounter:false` → 通常ダメージ（弱抵抗軽減適用）を確認。
- Metal Slash 230 / Slash 80 / Protect Charge 150 / Cut 10: ダメージ値実測で確認。**Metal Slash の自己ロック副作用は text 明示だが behavioral 未検証（verified:false）**。
- Rare Candy 直行・Metallic Signal サーチ対象は text 明示（未 behavioral 検証）。
