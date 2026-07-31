# P0 静的解析: `archaludon`

対象: cabt シミュレータ, 60枚. 解析はカードDB＋ルール知識＋シミュレータ・プローブに接地.
ブラインド実施（他デッキ P0・エンジン実装・ログ等は未参照）.

## 0. 60枚ダンプ（全カード・テキスト）

| cid | ×n | 名称 | 型 | 要点 |
|---|---|---|---|---|
| 190 | 4 | Archaludon ex | Pokémon Stage1(←Duraludon) M, HP300, weak Fire, resist Grass, retreat2 | **Metal Defender [M,M,M] 220**「相手の次の番、このポケモンは弱点を持たない」. 特性 **Assemble Alloy**「手札から進化させたとき、トラッシュの基本Mを最大2枚、自分のMポケモンに付けてよい」 |
| 169 | 4 | Duraludon | Pokémon Basic M, HP130, weak Fire, resist Grass, retreat2 | Hammer In [M] 30 / **Raging Hammer [M,M,C] 80**「このポケモンのダメカン1個ごとに+10」 |
| 666 | 4 | Cinderace | Pokémon **Stage2(←Raboot)** Fire, HP160, weak Water, retreat0 | **Turbo Flare [C] 50**「山札から基本Eを最大3枚、ベンチに好きに付ける」. 特性 **Explosiveness**「セットアップ時に手札にあれば、裏でバトル場に置いてよい」 |
| 57 | 1 | Relicanth | Pokémon Basic **Fighting**, HP100, weak Grass, retreat1 | Razor Fin [F,C] 30（**型死・後述**）. 特性 **Memory Dive**「自分の進化ポケモンは、進化前の技を全て使える（Eは別途必要）」 |
| 8 | 12 | Basic {M} Energy | 基本E | 唯一の供給型=Metal |
| 1121 | 4 | Ultra Ball | Item | 手札2枚トラッシュ→ポケモン1枚サーチ |
| 1122 | 4 | Pokégear 3.0 | Item | 上7枚見てサポート1枚回収 |
| 1185 | 4 | Explorer’s Guidance | Supporter | 上6枚見て**2枚手札・4枚トラッシュ** |
| 1227 | 4 | Lillie's Determination | Supporter | 手札を山に戻し6枚ドロー（サイド残6ちょうどなら8枚） |
| 1097 | 4 | Night Stretcher | Item | トラッシュのポケモン or 基本Eを1枚手札へ |
| 1152 | 3 | Poké Pad | Item | ルールボックス無しポケモンをサーチ（Duraludon/Cinderace/Relicanth のみ、Archaludon ex は不可） |
| 1147 | 3 | Jumbo Ice Cream | Item | E3個以上付いたバトルポケモンを80回復 |
| 1244 | 4 | Full Metal Lab | Stadium | **Mポケモン（両者）は相手の攻撃から30軽減**（弱点計算後） |
| 1182 | 2 | Boss’s Orders | Supporter | 相手ベンチを引きずり出す（gust） |
| 1197 | 2 | Xerosic’s Machinations | Supporter | 相手を手札3枚まで捨てさせる |
| 1159 | 1 | Hero’s Cape | Tool **ACE SPEC** | 付けたポケモン +100HP |

**プローブ検証済み（verified）機構**:
- **Explosiveness で Cinderace はセットアップ時バトル場に置ける**（Raboot 不在でも合法。ctx=SETUP_ACTIVE で Cinderace が単独 or 実基本と並んで提示されるのを確認）. `verified:true`
- **Cinderace はセットアップ以外では場に出せない**（12ゲーム走査で手札に居続けても PLAY 選択肢に一度も現れず。Stage2 かつ Raboot 不在＝進化経路も無い）. → **加速器は「初手にあり、かつ setup で active に置いた時だけ」1回限り**. `verified:true`
- **Turbo Flare** = 表示50 + 山から基本M最大3枚をベンチに付与（実観測: attach ×3）. Fire なので Duraludon(weak Fire)へ 50×2=100 を観測. `verified:true`
- **Assemble Alloy** = 進化時にトラッシュのM最大2枚を付与（進化直後に ATTACH Metal→Archaludon ×2 を観測）. `verified:true`
- **Memory Dive** = Relicanth がベンチに居ると Archaludon の技候補が {Metal Defender} → {Hammer In, Metal Defender, Raging Hammer} に増える（居ないと Metal Defender のみ）を観測. `verified:true`
- **Metal Defender 220 flat**（Full Metal Lab 無し時 -220、相手 M ×Full Metal Lab 下で -190＝-30軽減）を観測. `verified:true`
- **Raging Hammer のスケール式**は Memory Dive 経由の使用可否のみ verified、**+10/ダメカンの数値スケールはプローブで再現できず** `verified:false`（テキスト由来）.

## §1 勝ち筋の仕様

**主砲**: Archaludon ex（HP300, +Hero's Cape=400, +Full Metal Lab -30/被弾, +Jumbo Ice Cream 80回復）が
**Metal Defender 220 を毎ターン**撃ち続ける「不沈壁」. weak Fire は Metal Defender 自身の「次の番弱点なし」で相殺.

**手順（加速ドクトリン, 初手 Cinderace 有り≈40%）**:
1. setup: **Cinderace を Explosiveness でバトル場**、ベンチに Duraludon ×1-2 ＋ Relicanth.
2. T1: Cinderace に基本M手貼り1枚 → **Turbo Flare**（ベンチ Duraludon/Archaludon に M×3）.
3. T2: Duraludon→**Archaludon ex 進化**（Assemble Alloy でトラッシュ M×2 を追加）. さらに Turbo Flare で装填継続.
4. ベンチ Archaludon が **M≥3** に達したら **Cinderace を逃がし（retreat0=無料）Archaludon を昇格** → **Metal Defender 220**.
5. 以降: Full Metal Lab 維持、Jumbo Ice Cream で回復、Hero's Cape を Archaludon に、Boss's Orders で急所昇格. ~4発でサイド6.

**副勝ち筋 / プランB（初手 Cinderace 無し≈60%）**: 加速器が**永久に使えない**ため、Duraludon をバトル場開幕→
手貼り1/ターン＋**進化時 Assemble Alloy（トラッシュに M を貯めてから進化）** で Archaludon を装填. Explorer's Guidance で
基本Mをトラッシュに送り Assemble Alloy/Night Stretcher の弾にするのが実質の加速. 立ち上がりは1-2ターン遅い（Metal Defender online ~T4）が、
壁の耐久で間に合わせる grind プラン.

**実打点式（育てるべき資源＝式の変数）**:
- `Metal Defender = 220 × (2 if 防御側 weak Metal) − 30×[防御側が M かつ場に Full Metal Lab] − その他軽減`（flat, verified）.
- `Turbo Flare = 50 × (2 if 防御側 weak Fire) ；真の payload = min(3, 山の基本M) 枚をベンチへ`（verified）. 表示50は過小評価される→**共有知覚に「加速器」として昇格すべき**.
- `Raging Hammer(Memory Dive) = 80 + 10 × (Archaludon 自身のダメカン数)`（可否 verified、数値 unverified）. Metal Defender を超えるのは自傷140+（＝瀕死圏）のみ→ニッチな返し技.
- `Hammer In = 30`（flat, verified）— 装填1個時の繋ぎ.

## §2 ルール相互作用の棚卸し

- **エネの進化持ち上がり**: Duraludon に載せた M は Archaludon 進化後も残る（ベンチ装填→進化が有効. 実観測で Archaludon が装填状態で昇格）. → **下位 Duraludon への先貼り推奨**.
- **ダメージ vs ダメカン配置**: 本デッキの技は全て「ダメージ」（Full Metal Lab/弱点が乗る）. 相手のダメカン配置系は Full Metal Lab を無視する点に注意（軽減が効かない）.
- **技コストの型 vs 供給（Metal のみ）**:
  - Cinderace Turbo Flare [C] → M で支払可 ✓（ただし1枚の手貼りが必要）.
  - Duraludon Hammer In [M] ✓ / Raging Hammer [M,M,C] ✓.
  - Archaludon Metal Defender [M,M,M] ✓.
  - **Relicanth Razor Fin [F,C] = Fighting 要求。デッキに Fighting 無し → 恒久型死**. Relicanth は **Memory Dive 供給専用のベンチ置物**（決してアタッカーではない）.
- **1ターン1回制の衝突**: 手貼り1・サポート1. 「Cinderace を装填する手貼り」と「Archaludon を装填する手貼り」が競合→ Turbo Flare（山から3枚）が実質の追加手貼り. 逃げ(Cinderace retreat0)は手貼りと非競合.
- **自己ドロー/山掘り**: Lillie(6/8) + Explorer's Guidance(-6山) + Pokégear(-1) + Ultra Ball(-1+手2捨) + Poké Pad. 掘りが厚い＝**デッキアウト地平が近い**（壁デッキで長期戦になるほど危険。概算: 毎ターン Explorer's Guidance を撃つと ~15-18 ターンで山切れ）.
- **ACE SPEC**: Hero's Cape ×1（規定通り1枚）. Archaludon に付け HP400 の壁を作るのが最大価値.
- **スタジアム競合**: Full Metal Lab ×4. 相手スタジアムで剥がされても貼り直せる冗長性. 自 M を守る守備スタジアム.

## §2b 自己害の棚卸し（60枚走査）★

自軍除去特性・自傷技・自軍ダメカン・強制入替は**該当なし**. 該当する自己害ベクトルは**手札→トラッシュ**と**自己ミル**の2系統:

| カード | 自己害 | 発火してよい盤面条件（状態変数） |
|---|---|---|
| Ultra Ball ×4 | 手札2枚をトラッシュ | `捨てる2枚が勝ち筋核でない`（Cinderace は場外に出せば復帰不可＝捨てても機能損なし、Duraludon/Archaludon/M は Night Stretcher/Assemble Alloy で復帰可）. `deckCount > D` |
| Explorer's Guidance ×4 | **4枚をトラッシュ（自己ミル4/回）** | `deckCount > D`（薄い山で撃つと山切れ加速）. トラッシュ先が M ならむしろ Assemble Alloy の弾で**利得** |
| Lillie's Determination ×4 | 手札を山に戻す | いつでも可（復帰可能）. **死に札 Cinderace を掃く手段**として積極利用可 |

→ **catastrophic 仮説（H1）**: 汎用エンジンが Ultra Ball / Explorer's Guidance / Pokégear を無条件毎ターン使用し、壁デッキの長期戦で**自らデッキアウト**する.
自己害の出口: 山切れ側にゲート（`deckCount≤D で掘り停止`）が必須.

## §3 サイド算術

- **Archaludon ex = サイド2**. HP300（cape 400）は環境打点~200では1発で落ちない＝相手は2ターン投資して2枚. Full Metal Lab -30 と Jumbo Ice Cream +80 で更に落ちにくい → **トレード極めて有利な壁**.
- **Cinderace = サイド1**, HP160, **Fire で Full Metal Lab の保護対象外**＝無防備. 役目後は盤面に居るだけでサイド1の的（Boss's Orders で狙われる）.
- **Relicanth = サイド1**, HP100, 型死のベンチ置物. gust されて active で殴られると Memory Dive も失う（Raging Hammer 封じ、影響は小）.
- **Duraludon = サイド1**, HP130（Full Metal Lab で実 KO 閾値↑）.
- 必要 KO: 220 flat は多くの非ex を1発（サイド1）、大型ex を2発（サイド2）. **~3-4 発の Metal Defender で6枚**. チェイン要件は基本 **Archaludon 1体を延命し続ける**こと（壁なので生存前提）。予備は 2体目 Archaludon（山に Duraludon4/Archaludon4）で十分冗長.

## §4 フェーズプラン

| フェーズ | 遷移条件（状態変数） | 「計画通り」 | 逸脱時リカバリ |
|---|---|---|---|
| 序盤 | `turn ≤ 2` または `ベンチ Archaludon の M < 3` | Cinderace(有れば)が active、Duraludon≥1＋Relicanth ベンチ、Turbo Flare を≥1回、手貼りは**加速器 Cinderace 優先** | Cinderace 無し→Duraludon 開幕、Explorer's Guidance で M をトラッシュへ（Assemble Alloy 準備）. Poké Pad/Ultra Ball で Duraludon 補充 |
| 中盤 | `ベンチ or active Archaludon の M ≥ 3` | **Cinderace を逃がし Archaludon を昇格**し Metal Defender 初弾（online 目標 T3、B案 T4）、Full Metal Lab 設置、Hero's Cape を Archaludon | 主砲遅延時は Hammer In/Turbo Flare で場繋ぎ、Jumbo Ice Cream 温存 |
| 終盤 | `1発目 Metal Defender 着弾後` | 毎ターン Metal Defender（2ターンに≥1KO）、回復と cape で壁維持、Boss's Orders で急所. **`deckCount ≤ D で非必須の掘り停止**、Night Stretcher でM/主砲を再供給 | 主砲 KO 時は 2体目 Archaludon（装填済みを昇格、Relicanth/未装填 Duraludon は昇格させない） |

「使うカード/使わない」宣言: 序盤=Turbo Flare/Poké Pad/Ultra Ball/Pokégear を厚く、Boss's Orders/Xerosic は温存. 終盤=掘りアイテムを抑制、Boss's Orders/Jumbo/Cape を解禁.

## §5 スコアカード＋対立軸

| 軸 | 点(1-5) | このデッキ固有の定義 / 検証指標 |
|---|---|---|
| 速度 | 3 | Metal Defender online = 加速時T3/非加速T4。指標 `first_attack_turn`(=Metal Defender 初弾) |
| 火力曲線 | 4 | 220 flat の平坦高火力（ramp 無し）。指標 各フェーズの `expected_dmg` |
| 安定性 | 2 | mulligan≈30%、初手 Cinderace 欠け≈60%（加速喪失）。指標 `first_attack_turn` 分散, mulligan 率 |
| 継戦力 | 4 | Night Stretcher×4 + Assemble Alloy 再利用 + 2体目 Archaludon + 壁生存。指標 `post_ko_attack_rate` |
| 対応力 | 3 | Boss×2 / Xerosic×2 / Full Metal Lab / Jumbo / cape。非ダメージ妨害には薄い |
| 資源経済 | 2 | 12 M ＋ Assemble Alloy/Night Stretcher 再循環 vs Explorer's Guidance 自己ミル。指標 `loss_share[deckout]`, `deckCount` 末期 |
| 妨害耐性 | 3 | Archaludon 壁は gust 耐性高。Cinderace/Relicanth は gust の餌。手札破壊で立ち上がり阻害 |
| サイド構造 | 4 | 2プライズ不沈壁で優位トレード。ただしクロック遅い |
| **（発明）加速器脆弱性** | 2 | 加速は1回限り・初手依存・無防備。指標 `play_rate[666]`, `energy_attach_share[666]` |

**対立軸（tensions）**:
1. **加速 ⇄ 火力**: Cinderace active（Turbo Flare 50 + 加速）を続けるほど本命 Metal Defender 220 が遅れる。バランス点: `ベンチ Archaludon の M≥3 になった最初のターンに Cinderace を逃がし恒久昇格`（それ以前は Turbo Flare 優先）.
2. **掘り/安定 ⇄ 山切れ**: 掘るほど事故は減るが壁デッキは長期戦で山が尽きる。バランス点: `deckCount > D の間のみ非必須の掘り（Explorer's Guidance/Ultra Ball/Pokégear）を許可、D∈{10..18}`.
3. **壁の耐久 ⇄ クロック**: 過剰なストール（回復・防御優先で殴らない）は自デッキアウトを招く。バランス点: `online 後は 2ターンに1回以上 Metal Defender を撃つ（nonattacking_turn_rate を上限で監視）`.
P3 含意: tension の一方を締める修正は必ず対の指標（例: 掘り停止↔ブリック率 / 加速抑制↔first_attack_turn）を同時監視.

## §6 カード別「使用宣言」（全60枚）

- **Archaludon ex ×4** | 主砲・不沈壁 | `ベンチ Duraludon が M≥3 で進化 / active で Metal Defender` | ~2.5/試合 | ラインが揃わない・型が Metal で染まらない時（実質無し）
- **Duraludon ×4** | 進化元・先貼り台 | `序盤ベンチに1-2、M を先貼りして進化準備` | ~2/試合 | 4枚引き切り時の余剰、Assemble Alloy/Night Stretcher で回す
- **Cinderace ×4** | 一回限りの加速器 | `初手にあれば setup で必ず active（Explosiveness）。以後は Turbo Flare→装填後に退場` | 実プレイ ~0.4/試合（初手率）、残りは**死に札**（Ultra Ball/Explorer's Guidance/Lillie の弾） | 初手に無い＝この試合腐る（60%）
- **Relicanth ×1** | Memory Dive 供給（Raging Hammer 解禁） | `序盤ベンチに1枚置き放置。決して active/攻撃させない` | ~0.7/試合 | 型死につき攻撃は常に腐る。gust で晒される
- **Basic {M} Energy ×12** | 唯一の燃料 | `手貼り＋Turbo Flare/Assemble Alloy 経由でMポケへ集中` | 全消費 | 過剰トラッシュは Night Stretcher で回収
- **Ultra Ball ×4** | ポケモン万能サーチ | `deckCount>D かつ Duraludon/Archaludon/Relicanth が不足時。捨てる2枚に死に Cinderace/余剰Eを充てる` | ~2/試合 | 山薄時/捨て札が惜しい時は打たない（H1）
- **Poké Pad ×3** | 非ルールボックス補充 | `Duraludon/Cinderace/Relicanth を確保（Archaludon ex は不可）` | ~1.5/試合 | 序盤で用済み・Archaludon しか要らない終盤
- **Pokégear 3.0 ×4** | サポート確保 | `序盤〜中盤、Boss/Lillie/Explorer's Guidance を掘る` | ~1.5/試合 | サポート不要な終盤、山薄時（H1）
- **Explorer's Guidance ×4** | 掘り＋M をトラッシュへ（Assemble Alloy 弾込み） | `deckCount>D。手札4枚以下でなくても掘りたい時に` | ~2/試合 | **山薄時は自己ミル4で厳禁（H1）**、手札に核が多い時は捨て事故 |
- **Lillie's Determination ×4** | 手札総入替・死札掃除 | `手札に死に札(Cinderace 余剰)が滞留 or 手札枯渇。T1はサイド6で8ドロー` | ~1.5/試合 | 手札に核を抱えた状態でむやみに戻す時（H7）
- **Night Stretcher ×4** | 主砲/E の再供給 | `Duraludon/Archaludon がトラッシュ or M が枯れた時` | ~2/試合 | 序盤で回収対象が無い時
- **Jumbo Ice Cream ×3** | 壁延命（80回復, E3+） | `active Archaludon が被弾 ~80-160 で、次弾で耐えると生存する時` | ~1.5/試合 | E3未満の active（Cinderace等）には腐る、無傷時
- **Full Metal Lab ×4** | 守備スタジアム(-30/被弾) | `場に自スタジアム不在で常設。相手に剥がされたら貼り直す` | ~1.5/試合 | 相手が非Mで自M不在の極序盤（価値低）
- **Boss's Orders ×2** | gust 急所 | `相手ベンチに未進化アタッカー/低HP/回復役が居て、Metal Defender で取れる時` | ~1/試合 | 対象が薄い・active を先に処理すべき時に空撃ち（H5）
- **Xerosic's Machinations ×2** | 手札破壊 | `相手が手札を溜めた次の番の直前、または相手コンボ阻害` | ~0.8/試合 | 相手手札が既に薄い時は無駄
- **Hero's Cape ×1（ACE SPEC）** | +100HP=壁強化 | `盤面が安定した Archaludon(active or 昇格予定) に1回だけ` | ~0.7/試合 | Cinderace/Relicanth 等に誤着（単一 ACE SPEC 浪費, H9）

（定常状態の推定: 手札は掘りが厚く 5-8 枚で滞留しがち、死に Cinderace が混ざる→ Lillie の手札≤5 ゲートは**過小発火**しうる. 盤面 M は Turbo Flare/Assemble Alloy で急速に増える. → §6 の発火条件は所持ベースに寄せた.）

## §7 敗因仮説と L2 規則候補

**予想敗因分布**: ①デッキアウト（壁×自己ミルの長期戦, 最大リスク） ②テンポ（Cinderace 未装填/未昇格で 50 しか出ない立ち上がり） ③初手事故（Cinderace 欠け＝加速喪失、mulligan30%）. 盤面全滅・レース負けは壁性能ゆえ低め.

**適用パターン（優先順）**:
1. 山切れ負け → 山戻し発火＋**山薄時の掘り停止**（Explorer's Guidance/Ultra Ball/Pokégear を deckCount ゲート）.
2. スケール技過小評価 → **Turbo Flare を「加速器」として共有知覚昇格**（表示50でなく「+3E/ターン」で評価）→ 手貼り配分・昇格判断が参照.
3. 主砲KO後に殴れない/型死昇格 → 昇格を**装填済み Archaludon 限定**、Relicanth/未装填 Duraludon を昇格対象から除外.
4. キーカードがトラッシュ/山 → Night Stretcher を所持ベース発火（M/Duraludon 枯渇時）.
5. gust/妨害の浪費 → 対象存在＋実打点で取れる時に限定.
（新パターン提案: **「setup で一回限りの起動役を active に固定する」規則** — Explosiveness の Cinderace の様な post-setup 再展開不能カードは、手札にあれば setup active を最優先で確保. 症状=加速永久喪失, 条件=当該カードが opening hand かつ setup, 修正=setup active 選択で最優先.）

## L2 要否の判定

**L2 必要（作成推奨）**. 理由: 本デッキは汎用 L0 が
(a) **setup で Cinderace を active に置かず加速を永久喪失**（H2, verified: post-setup 再展開不能）、
(b) **Cinderace active で Turbo Flare 50 を撃ち続け Archaludon 220 へ昇格しない**（H3, 実プローブでも greedy は明示コードなしに昇格しなかった）、
(c) **無条件の掘りで壁デッキが自デッキアウト**（H1）
という **3つのデッキ固有誤作動**を高確率で起こす. これらは表示ダメージ／汎用閾値だけでは是正されず、Turbo Flare の共有知覚昇格・昇格ゲート・掘り停止ゲートという L2 規則が要る. 一方、火力・継戦・サイド構造は素の L0 でも機能するため、L2 は上記3点＋補助（H4/H6/H9）に絞れば足りる.
