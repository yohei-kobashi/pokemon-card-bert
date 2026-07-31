# P0 静的解析（盲検）: `alakazam`

対象: cabt シミュレータ用デッキ `decks/alakazam.csv`（60枚）。
本書はカードDB全文と、シミュレータ実証プローブのみから記述する（既存の方策・docs・ログは未参照）。

---

## 0. 60枚リスト（全文 / 攻撃・特性テキスト）

| id | 枚 | 名称 | 種別 | HP | 進化元 | 主要テキスト |
|---|---|---|---|---|---|---|
| 743 | 4 | **Alakazam** | Stage2(超) | 140 | Kadabra | **特性 Psychic Draw**: 手札から進化させて出したとき1回、3ドロー。/ **技 Powerful Hand [P] ダメ表示0**: 相手バトルポケモンに「手札の枚数ぶん ダメカン2個ずつ」置く |
| 742 | 4 | Kadabra | Stage1(超) | 80 | Abra | **特性 Psychic Draw**: 進化時1回、2ドロー。/ **技 Super Psy Bolt [P] 30** |
| 741 | 4 | Abra | たね(超) | 50 | — | **技 Teleportation Attack [P] 10**: ベンチと入れ替え |
| 66 | 3 | Dudunsparce | Stage1(無) | 140 | Dunsparce | **特性 Run Away Draw**: 自ターン1回、3ドロー。引いたら自身＋付与カードを山に戻す。/ **技 Land Crush [C][C][C] 90** |
| 305 | 3 | Dunsparce | たね(無) | 70 | — | **技 Trading Places**: ベンチと入替 / **技 Ram [C][C] 20** |
| 1079 | 4 | Rare Candy | グッズ | — | — | たね→Stage2 スキップ進化（初ターン不可・その場に出したてのポケモン不可） |
| 1152 | 4 | Poké Pad | グッズ | — | — | 山からRule Box無しポケモン1枚を手札へ |
| 1086 | 4 | Buddy-Buddy Poffin | グッズ | — | — | 山からHP70以下のたね2枚をベンチへ |
| 1081 | 4 | Enhanced Hammer | グッズ | — | — | 相手ポケモンの**特殊エネルギー**1枚をトラッシュ |
| 1097 | 3 | Night Stretcher | グッズ | — | — | トラッシュのポケモン or 基本エネ1枚を手札へ |
| 1129 | 1 | Sacred Ash | グッズ | — | — | トラッシュのポケモン最大5枚を山に戻す |
| 1231 | 4 | Dawn | サポート | — | — | 山からたね/Stage1/Stage2 各1枚を手札へ |
| 1225 | 4 | Hilda | サポート | — | — | 山から進化ポケモン1＋エネ1を手札へ |
| 1182 | 3 | Boss's Orders | サポート | — | — | 相手ベンチ1体をバトル場へ（ゲイン/gust） |
| 1184 | 1 | Lana's Aid | サポート | — | — | トラッシュから Rule Box無しポケモン/基本エネ 最大3枚を手札へ |
| 5 | 4 | Basic {P} Energy | 基本エネ | — | — | 超エネルギー |
| 19 | 4 | Telepath Psychic Energy | 特殊エネ | — | — | {P}を供給。**{P}ポケモンに手貼りしたとき、山から基本{P}たね最大2枚をベンチへ** |
| 13 | 1 | Enriching Energy | 特殊エネ | — | — | **{C}のみ**供給。手貼り時4ドロー |
| 1264 | 1 | Battle Cage | スタジアム | — | — | 相手の技・特性の効果によるベンチへのダメカン配置を無効（両者ベンチ）。技のダメージは通る |

構成比: たね系アタッカー基盤（Abra4/Dunsparce3）＋進化（Kadabra4/Alakazam4/Dudunsparce3）＋サーチ16枚（Dawn4/Hilda4/Poffin4/PokePad4）＋ドロー/回収多数。エネは実質「超1枚あれば撃てる」設計。

---

## §1 勝ち筋の仕様（win condition spec）

### 主勝ち筋: Alakazam「Powerful Hand」＝**手札スケール技**の単騎ビート
- **実打点式（実証済み）**: `ダメカン = 2 × 攻撃時の手札枚数`、すなわち `実HP換算 = 20 × 手札枚数`。
  - プローブ実測: 手札6→120HP、手札4→80HP（`serial`一致で確認）。カードテキスト「手札の枚数ぶんダメカン2個ずつ」と一致。
  - **配置（ダメカン）であってダメージではない** → 弱点・抵抗・ダメージ軽減効果を無視して相手**バトル場**に通る。
  - コストは `[P]` **1個のみ**。装填が極端に軽い＝「育てる資源は手札枚数、エネではない」。
- **手順**:
  1. 序盤: Abra をバトル場、ベンチに進化元＆Dunsparce を並べ、Alakazam線に**基本{P} or Telepath**を1枚手貼り。
  2. T2〜: Rare Candy（Abra→Alakazam）または Kadabra 経由で Alakazam を立てる。**進化は「そのターン最後の手札増加行動」にする**（Psychic Draw の3枚を攻撃時の手札に載せるため）。
  3. 以降毎ターン: 手札を減らす行動（ベンチ/手貼り/Rare Candy）を先に済ませ、**ドロー/進化を攻撃直前**に寄せ、`20×手札` が相手アクティブHP以上になったら Powerful Hand。必要なら Boss's Orders で「今KOできる/価値の高い」相手を引きずり出す。
- **サイド構造**: Alakazam は **サイド1枚**の非exながら ex 級火力（手札10で200）。単騎で殴り勝つ「1プライズ・アグロ」。

### 副勝ち筋 / プランB
- **Dudunsparce Land Crush 90 [C][C][C]**: Alakazam が組めない/落ちた時の暫定アタッカー。ただし Dudunsparce は本来ドローエンジンで、3エネは重く off-plan。過度に依存しない。
- **Kadabra Super Psy Bolt 30 [P]**: Alakazam 未達時のチップ。
- **継戦**: Alakazam は140で毎ターン取られる前提。Night Stretcher/Lana's Aid/Sacred Ash で線を回収して**Alakazam を毎ターン供給し続ける**のが実質のプランB。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる（実証済み）**: Abra に貼った{P}が Kadabra→Alakazam に持ち上がる（プローブで energies=[5] を確認）。→ **たねへの先貼りが有効**。T1に Abra へ{P}先貼り→T2 Rare Candy で即撃てる。
- **ダメカン配置 vs ダメージ**: Powerful Hand は**配置**。弱点/抵抗/軽減を貫通。逆に自分は相手のダメージ軽減に一切影響されない。
- **技コストの型要求 vs 供給型（型死検出）**: Powerful Hand は `[5]=Psychic` 必須。供給源は基本{P}4＋Telepath4＝**超8枚**。一方 **Enriching Energy は {C} のみ**（プローブで Alakazam に貼ると energies=[0]）。→ **Enriching では Powerful Hand を撃てない**。Enriching だけ載った Alakazam は「装填済み」に見えて実は型死。Enriching は Dudunsparce/Dunsparce の無色コスト用、または純粋な4ドロー要員。
- **1ターン1回制**: サポート（Dawn/Hilda/Boss/Lana は1枚のみ）・手貼り1・特性 Psychic Draw は「進化して出したとき1回」。理想シーケンス「サーチ→展開→手貼り→進化(最後)→攻撃」は1回制と衝突しにくいが、**Boss を撃つ最後のサイド詰めターンはサーチサポートを撃てない**（Boss と Dawn/Hilda は排他）点に注意。
- **自己ドロー/山掘り地平**: Alakazam(+3)/Kadabra(+2)/Dudunsparce(+3)/Enriching(+4)＋サーチ16枚。**掘りが非常に厚い**。プローブ自己対戦で **deckCount 0 到達例あり** → デッキアウトは実在の副敗因。山戻しは Sacred Ash 1枚のみ。
- **ACE SPEC / 特殊エネ供給型 / スタジアム**: ACE SPEC 無し。特殊エネは Telepath(P)・Enriching(C)。スタジアムは Battle Cage 1枚（相手のベンチspreadを無効化する防御札。自分の Powerful Hand は相手アクティブ狙いなので自傷しない）。

---

## §3 サイド算術（prize arithmetic）

- **主砲HP**: Alakazam 140。環境典型打点 ~200 に対し **毎ターン確実にOHKOされる前提**。ただし献上は**サイド1枚**。
- **こちらの打点 vs 相手HP**: `20×手札`。
  - 手札7で140（ミラーや中型たねを1発）、手札10で200（ex を1発）、手札15で300（mega 圏）。
  - 現実的な攻撃時手札は概ね **8〜10（=160〜200）**（プローブ分布 mean 8.2）。→ ex(1体2枚)を1発でKOできれば**サイドレース有利**（1献上で2取り）。
- **必要KO数とチェイン要件**: 相手が全て単プラなら6KO必要。ex 主体なら3KO。**毎ターン新しい Alakazam を装填**する必要（4 Alakazam＋Rare Candy＋回収3種で供給）。チェインが途切れる＝敗着。

---

## §4 フェーズプラン

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ | 使う/使わない札 |
|---|---|---|---|---|
| **序盤** | Alakazam 未在 or Alakazam に{P}0 | T1:Abra アクティブ＆ベンチ基盤、Alakazam線に基本{P}/Telepath 1枚、Alakazam+Rare Candy or Kadabra を手札集約、手札が増加基調 | Dawn/Hilda/Poffin/PokePad で線と基本を確保、Night Stretcher で落ちた札を回収 | 使う: サーチ全般, Poffin, 手貼り / 使わない: Boss（まだ的が無い）, Enhanced Hammer（的次第）|
| **中盤** | Alakazam アクティブ＋{P}≥1 かつ 手札 ≥ 相手HP/20 | 手札減少行動を先に→ドロー/進化を攻撃直前→`20×手札≥的HP`で Powerful Hand。必要なら Boss で始末できる的を釣る | 手札が小さくKO届かない時は**撃たずに再ドロー**（Hilda/Dawn/Dudunsparce）して次ターンKO | 使う: Powerful Hand, Boss, 進化, 手貼り / 使わない: Land Crush（原則）|
| **終盤** | 取得サイド≥4 or deckCount≤8 | 毎ターン新 Alakazam を装填し6枚目まで詰め。deckCount≤8 かつ トラッシュにポケモン≥3 で Sacred Ash | Lana's Aid で線＋基本エネを一括回収、Night Stretcher で単発補充 | 使う: 回収3種, Sacred Ash / 使わない: 無駄な Poffin |

---

## §5 多面的評価軸（deck scorecard）

1. **速度 3/5** — 根拠: Rare Candy で **最速 player-T2** に Powerful Hand 可（初T不可のため T1 は無い）。ただし十分な手札を要し実効オンラインは T2-4。検証指標: `first_attack_turn`（期待: 中央値 player-T2〜4, 手札≥7）。欠けたら: オンライン判定を「手札≥閾値 かつ KO可能」に接地。
2. **火力曲線 4/5** — 根拠: `20×手札`。中盤160〜200、上振れ300超。表示0だが実火力は高い。検証指標: 攻撃時 `hand_size_dist` × 20 の期待打点。欠けたら実打点式を共有知覚へ昇格。
3. **安定性 4/5** — 根拠: サーチ16枚＋厚いドローで線を組む確率が高い。ブリック要因は「基本{P}が全く引けない」「Stage2偏り」程度。検証指標: 初手ブリック率, `play_rate[サーチ群]`。
4. **継戦力 4/5** — 根拠: Night Stretcher×3, Lana's Aid, Sacred Ash, Alakazam×4。主砲KO後の再装填ループが厚い。検証指標: `post_ko_attack_rate`（期待: 高い）。欠けたら回収の所持ベース発火。
5. **対応力 3/5** — 根拠: Enhanced Hammer×4（特殊エネ）, Battle Cage（spread）, Boss（壁貫通=ダメカン配置で軽減無視）。エネ加速妨害への直接回答は無い。検証指標: 各対策札の条件発火率。
6. **資源経済 3/5** — 根拠: エネは1枚で撃てるため楽。だが**山経済は掘り過多で逼迫**（deckout到達例）。検証指標: `loss_share[deckout]`, min `deckCount`。
7. **妨害耐性 2/5（急所）** — 根拠: **Powerful Hand=手札そのもの**。相手の手札リセット（Marnie/Iono系）で手札が数枚に落ちると打点が瞬時に崩壊。検証指標: 手札破壊後ターンの `hand_size_dist` と打点。欠けたら「低手札時は再ドロー優先」則。
8. **サイドレース構造 4/5** — 根拠: サイド1枚の非exが ex 級を1発でKO＝交換有利。相手が ex/mega 主体ほど有利。検証指標: 相手プロファイル別勝率。

### 追加軸（このデッキ固有）
9. **手札枚数マネジメント 5/5（最重要）** — このデッキの唯一の火力変数。攻撃直前の terminal hand を最大化する「温存＋ドロー後置き」が生命線。汎用エンジンの「盤面展開で手札を吐く」挙動と真っ向衝突。検証指標: Powerful Hand ターンの `hand_size_dist`（期待 mean≥8, 手札<6での攻撃率≈0）。
10. **デッキアウト地平 2/5** — 掘り過多で山寿命が短い。Sacred Ash 1枚が命綱。検証指標: min `deckCount`, `trigger_fire[sacred_ash]`。

---

## §6 カード別「使用宣言」（全19種 / 60枚）

このデッキの**定常状態**（§1-2から推論）: 攻撃時**手札8〜10枚**、ベンチに Alakazam線＋Dunsparce/Dudunsparce、Alakazamへの必要エネは**わずか{P}1**。→ 汎用既定「手札≤5でドロー」はこのデッキの大手札分布では**ほぼ発火せず**、むしろ問題は逆（吐き過ぎ）。以下は状態変数ベースの発火条件で較正。

- **Alakazam 743 ×4** | 主砲 | 発火: Alakazam線が場、{P}装填済、手札を最後に増やした後 | 率≈1.0/試合 | 腐: 手札が小さい/{P}が Enriching しか無い時に撃つと過小。
- **Kadabra 742 ×4** | 橋渡し＆+2ドロー | 発火: Rare Candy を使わない時のみ通常進化 | 率≈0.7 | 腐: Rare Candy 直行時は場に出さず手札で腐る。
- **Abra 741 ×4** | 線の起点 | 発火: T1アクティブ/ベンチ、Teleportationは緊急スイッチのみ | 率≈1.0 | 腐: 単体では50HPの的。
- **Rare Candy 1079 ×4** | T2オンライン | 発火: 手札にAlakazam＋前ターンからの Abra、非初T | 率≈0.8 | 腐: Alakazamが手札に無い/Abra が出したてのみ。
- **Dawn 1231 ×4** | 線一括サーチ | 発火: 進化ラインが手札に不足 | 率≈0.6 | 腐: 既に線が揃っている時。
- **Hilda 1225 ×4** | 進化＋エネ、手札+2 | 発火: 進化 or エネ不足、かつ手札を増やしたい | 率≈0.6 | 腐: Bossを撃ちたいターン（サポート排他）。
- **Buddy-Buddy Poffin 1086 ×4** | ≤70HP たね2展開 | 発火: 序盤ベンチ空き（Abra/Dunsparce 供給） | 率≈0.7 | 腐: 中盤以降ベンチ埋まり。
- **Poké Pad 1152 ×4** | 任意ポケモン1サーチ＋手札+1 | 発火: 特定ポケモン欠品 | 率≈0.6 | 腐: 盤面完成後。
- **Telepath Psychic Energy 19 ×4** | {P}供給＋基本{P}たね2展開 | 発火: {P}ポケモンへ手貼り（Alakazam線） | 率≈0.7 | 腐: 山に基本{P}たねが無い時は展開効果だけ空振り（供給は有効）。
- **Basic {P} Energy 5 ×4** | Powerful Hand コスト | 発火: Alakazam線に{P}0の時1枚 | 率≈0.8 | 腐: 2枚目以降は過剰（1個で撃てる）。
- **Boss's Orders 1182 ×3** | gust | 発火: ベンチに「今KO可能(HP≤20×手札)」or 高サイド/サポの的 | 率≈0.5 | 腐: 表示0で価値算定を誤ると空撃ち。
- **Night Stretcher 1097 ×3** | 単発回収 | 発火: Alakazam/Abra or 基本{P}がトラッシュ＆再建必要 | 率≈0.5 | 腐: トラッシュに対象皆無。
- **Dudunsparce 66 ×3** | ドローエンジン/予備 | 発火 Run Away Draw: 手札≤4 かつ 自身が装填済アタッカーでない（**自身を山に戻す**）| 率≈0.6 | 腐: 手札大の時に撃つと盤面損＆無意味churn。
- **Dunsparce 305 ×3** | Dudunsparce の種＋スイッチ | 発火: ベンチ展開、Trading Placesは緊急 | 率≈0.6 | 腐: Ram20はほぼ不使用。
- **Enhanced Hammer 1081 ×4** | 特殊エネ破壊 | 発火: **相手が特殊エネ装備時のみ**（撃つ直前に剥ぐ） | 率≈0.3 | 腐: 基本エネのみの相手に0。
- **Enriching Energy 13 ×1** | 4ドロー / 無色供給 | 発火: 手貼りで即4ドロー。**Alakazamの主砲コストには不可**（{C}）| 率≈0.7 | 腐: Alakazamに貼って装填したつもりが型死。
- **Lana's Aid 1184 ×1** | 一括回収 | 発火: 終盤トラッシュに線＋基本エネ蓄積 | 率≈0.4 | 腐: 序盤トラッシュ薄い時。
- **Sacred Ash 1129 ×1** | 山戻し（deckout保険） | 発火: deckCount≤8 かつ トラッシュにポケモン≥3 | 率≈0.3 | 腐: 早撃ちは資源の無駄。
- **Battle Cage 1264 ×1** | 対spreadベンチ防御 | 発火: 相手がベンチspread/snipeを持つ時 | 率≈0.3 | 腐: 非spread相手には不活性。

---

## §7 敗因仮説と L2 規則候補

### 予想敗因分布
1. **テンポ/火力ロス（最有力）**: 汎用エンジンが Powerful Hand の表示0を見て（a）攻撃せず/弱い技を選ぶ、または（b）盤面展開で手札を吐き切ってから小手札で撃つ。→ 主砲が機能不全。
2. **妨害負け（急所）**: 相手の手札リセットで打点崩壊（§5-7）。
3. **デッキアウト**: 掘り過多で山切れ（実測到達例）。
4. **チェイン切れ**: 主砲KO後に次の Alakazam を装填できず殴打が止まる。

### L2 規則候補（優先順）
| 優先 | 症状 | 定型修正（パターン） |
|---|---|---|
| P1 | スケール技が過小評価（表示0） | **実打点式 `20×手札` を共有知覚 `_expected_dmg` に昇格**（gust/攻撃選択/エネ配分が参照）— H1,H4 |
| P1 | 攻撃直前に手札を吐く | **手札温存則**: Powerful Hand ターンは非必須プレイを停止、draw/evolve を攻撃直前に寄せる — H2,H9 |
| P2 | 型死装填（Enriching）| **型対応装填判定**: Powerful Hand は {P}(id5/19) が載る時のみ「装填済」— H3 |
| P2 | エネが Dudunsparce へ流出 | **給餌計画**: {P}は Alakazam線優先、Dudunsparce は予備のみ — H8 |
| P2 | Run Away Draw の誤発火（自身を山に戻す）| **所持/盤面ベース発火**: 手札≤4 かつ 盤面上不要時のみ — H6 |
| P3 | 妨害カードの腐り/浪費 | Enhanced Hammer=相手特殊エネ条件、Battle Cage=spread条件で限定 — H5,H11 |
| P3 | 山切れ | Sacred Ash を deck≤8 かつ discard内ポケモン≥3 で発火 — H7 |
| P3 | 手札破壊で打点崩壊 | 低手札時は撃たず再ドロー優先（型スケール技版の消極性均衡）— H10 |

---

## 付録: 実証（プローブ）状況

| 主張 | 手段 | 判定 |
|---|---|---|
| Powerful Hand = 2×手札 ダメカン配置（20×手札 HP）| 自己対戦で Alakazam に Powerful Hand を撃たせ、相手アクティブの maxHp-hp を計測（手札6→120, 手札4→80）| **実証済** |
| エネは進化で持ち上がる（Abra→Kadabra/Alakazam に{P}残存）| 進化後の active.energies/energyCards を serial 追跡 | **実証済** |
| Telepath は{P}を供給 | Telepath(19) 装填後 energies=[5] を確認 | **実証済** |
| Enriching は{C}のみ→Powerful Hand不可 | Enriching 装填後 Alakazam energies=[0] を確認（型コード [5]≠[0]）| **実証済（型コード）** |
| Telepath の「基本{P}たね2ベンチ」展開 | プローブで search select を明確捕捉できず | **未実証（テキスト由来）** |
| Powerful Hand は弱点/抵抗を無視（配置）| カードテキスト「ダメカンを置く」から演繹（配置=非ダメージ）| **未実証（テキスト演繹）** |
| デッキアウト実在 | 自己対戦で min deckCount=0 到達を観測 | **実証済（存在のみ）** |
