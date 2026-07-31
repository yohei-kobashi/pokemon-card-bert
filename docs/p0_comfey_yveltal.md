# P0 静的解析: `comfey_yveltal`

**アーキタイプ**: 妨害/エネ破壊コントロール（非ルールボックス・ロック型）。
**勝ち筋は prize race ではなく、資源枯渇＋ダメージ無効化のグラインド**。

全主張は (a) 60枚のカードテキスト/数値、(b) シミュレータプローブ、のいずれかに接地。
検証済みプローブ結論は §0 に要約し、各所で `[PROBE]` と明記する。

---

## §0 プローブで検証した機構（verified）

| # | 機構 | 結果 | verified |
|---|---|---|---|
| P1 | **Yveltal はダーク型死** | 純Psychicエネ（Telepath/BasicP）を何枚積んでも 997/998 は**一度も選択肢に出ない**。997 は Prism ≥1、998 は Prism ≥2 の時のみ出現 | true |
| P2 | **Prism はベーシックに付くと任意型（=Dark可）** | Prism 付与時 active.energies に `10`(rainbow) が入り、Yveltal のダークコストを支払う。Prism 1枚=1エネ | true |
| P3 | **Neutralization Zone の ex/V 無効化** | NZ が場にある連続 4272 フレームで、こちらの非ルールボックス active は ex 攻撃（Kangaskhan ex 120）から HP 減少 0。完全無効 | true |
| P4 | **NZ は ACE SPEC（1枚制限）** | NZ×4 のデッキは engine が拒否（obs=None）。`aceSpec=True` | true |
| P5 | **無条件サーチ乱用で自己山切れ** | 5種サーチ+Comfey215 を毎ターン撃つ挙動で、**何もしない相手に 27/40 で自滅敗北**、山が engine-turn 51〜95 で 0 に | true |
| P6 | Telepath の付与時ベンチサーチ | プローブでは TO_BENCH select を再現できず（付与フロー要因）。テキスト準拠、**unverified** | false |

数値: Comfey 70HP(弱点Metal,逃1) / Yveltal 110HP(弱点Lightning,抵抗Fighting,**逃0**) / Shaymin 80HP(弱点Fire,逃1)。全て非ルールボックス=サイド1。

---

## §1 勝ち筋の仕様

### 主砲と実打点式
- **Yveltal 998 (Oblivion Wing)**: `damage = 110`（固定、スケール無し。テキスト空）。コスト `[7,7,0]` = ダーク2+無1。
- **Yveltal 997 (Fear)**: `damage = 20`＋「相手は次ターン逃げられない」。コスト `[7]` = ダーク1。
- **Comfey 216**: `damage = 20 + 20*[heads]`（期待30、実 {20,40}）。コスト `[5]` = P1。
- **Comfey 215**: `damage = 0`、**両プレイヤーが3ドロー**（＝相手も引かせる）。コスト `[5]`。
- **Shaymin 477**: `damage = 30`。コスト `[0,0]` = 無2。

**型 vs 供給（型死検査）**: 供給は Prism×3（唯一のダーク源, ベーシック上で任意型）／Telepath×4（Pのみ）／BasicP×3（Pのみ）＝計10枚。
- Yveltal: ダークは **Prism でしか払えない**。3枚しかない Prism のうち **2枚で1体だけ 998 を成立**（残1で 997 or 予備）。Psychicエネは何枚積んでも無効 `[PROBE P1]`。→ **最重要の型ボトルネック**。
- Comfey / Shaymin: P/無で常に支払い可、型死なし。

### 主勝ち筋（Doctrine A: Yveltal control-chip）
1. **序盤(〜round3)**: Comfey を engine/壁として展開（Poffin=Comfey専用サーチ, PokePad）。**Colress で NZ + エネを直接サーチ**して NZ を即設置。Yveltal を1体ベンチへ。
2. **中盤**: Prism 2枚を1体の Yveltal に集中 → 998 online（110/ターン）。NZ で相手 ex/V の攻撃を 0 に固定 `[PROBE P3]`。Crushing Hammer + Handheld Fan でエネ破壊、Xerosic で相手手札を3に固定して再構築を封じる。
3. **終盤**: Yveltal 110 で削り切る（相手 ex を KO=**こちらがサイド2枚**取得）。Boss で KO 対象を引き摺り出す。Acerola（相手サイド≤2で解禁）で詰めの盾。

### 副勝ち筋 / プランB
- **Doctrine B（resource-lock deckout）**: 相手盤面が硬すぎてレースが遅い場合、エネ+手札を完全ロックし**相手の山切れ**で勝つ。ただし Comfey 215（相手に3ドローさせる）は封印必須、かつ**自分が先に山切れしない**こと（§2b/H1）。
- **PlanB（Prism 喪失時）**: Prism が prize/discard で失われ 998 が組めない場合、NZ 無効化の下で Comfey 216（20-40）+ Shaymin 30 で削りつつロック/デッキアウトに寄せる。Night Stretcher で **discard の Prism を回収**して復帰を狙う。

---

## §2 ルール相互作用の棚卸し

- **エネの進化持ち上がり**: 進化ラインが**存在しない**（全て Basic）。先貼り論点は無効。
- **ダメカン配置 vs ダメージ**: 該当効果なし（NZ/Shaymin/Acerola は「ダメージ無効」＝配置ではなくダメージを止める。相手が配置効果なら無関係）。
- **技コストの型要求 vs 供給型**: §1 の通り。Yveltal=Prism専属、Comfey/Shaymin=任意。
- **1ターン1回制**: サポート枠が過密（Boss/Xerosic/Lillie/Colress/Acerola が全て supporter=1/ターン競合）。序盤 Colress(NZ探索) と中盤 Boss/Xerosic(妨害) が枠を奪い合う。
- **自己ドロー/山掘り地平**: Pokegear/PokePad/Poffin/Night Stretcher/Colress の**5種サーチ＋Comfey215＋Lillie** という巨大エンジン。速い勝ち筋が無いため**自己山切れが最大の敗因源** `[PROBE P5]`。
- **ACE SPEC / スタジアム競合**: NZ が唯一の ACE SPEC（＝他 ACE 不可, 1枚制限 `[PROBE P4]`）。**相手がスタジアムを張ると NZ は trash され、NZ 自身のテキストで discard から回収不能**＝ロックが恒久崩壊する最大の脆弱点。Colress は「山から」しか持って来られない。
- **特殊エネの供給型**: Prism（任意/1エネ）、Telepath（P固定, 付与時ベンチ増強）。基本 P×3。

### §2b 自己害の棚卸し ★（60枚走査）

**自軍除去/自傷/強制入替の能動効果は無し**（唯一の特性 Shaymin Flower Curtain は受動・無害）。ただし**山とカード経済に対する自己害が2系統**あり、**catastrophic**：

| カード | 自己害 | 発火してよい盤面条件（状態変数） |
|---|---|---|
| **Comfey 215** | 自山3ミル＋**相手に3ドロー**（相手のデッキアウトを遠ざける） | `hand<H かつ deckCount>K かつ prize-plan（deckout-plan では禁止）` |
| **Lillie's Determination** | **手札全部を山に戻す**（NZ/Prism/Boss を抱えていると埋める）＋実質山圧迫 | `NZ/必要Prism/Boss を持っていない かつ deckCount>K` |
| 5種サーチ+Colress | 毎使用で山を薄くする（速い勝ち筋が無い） | `所持ベースの必要時のみ かつ deckCount>K` |

→ **catastrophic 仮説 H1（自己山切れ）と H2（型死）**。H1 は `[PROBE P5]` で実証（無行動の相手に自滅）。

---

## §3 サイド算術

- こちらの全ポケモン（Comfey/Yveltal/Shaymin）は**非ルールボックス=サイド1**。安いトレードで、NZ 下では ex/V から KO されない → 実質**取られないサイド**。
- 環境典型打点 ~200。NZ が無い素の状態では Comfey(70)/Yveltal(110)/Shaymin(80) は毎ターン即 KO 圏（NZ の有無が生死を分ける）。
- **必要 KO 数**: 6サイド。相手 ex を KO すると**2枚**取れるため、実際は ex を3体 KO で決着し得る。Yveltal 110 では 210-260HP の ex を**2-3発**で落とす → チェイン装填は不要（Prism は同一 Yveltal に残留、毎ターン 998 再使用可）。むしろ**1体の Yveltal を維持**する方が重要。
- **チェイン要件**: 主砲は NZ 下で取られにくいので「2体目の装填」は薄くてよいが、Prism が3枚しかないため**予備 Yveltal は 997(1 Prism)止まり**。

---

## §4 フェーズプラン（計測可能）

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| 序盤 | `NZ 未設置 or Yveltal 0体 or Prism<1確保` | round3 までに NZ 設置（Colress/Pokegear で探索）＋Comfey engine 起動。サーチは**欠けているピースを埋める時のみ** | NZ 未着なら Pokegear→Colress 連鎖。**全サーチ乱用は禁止**（自滅 H1） |
| 中盤 | `1体の Yveltal に Prism2枚` | 998 online、110/ターン。相手手札≤3(Xerosic)・エネ剥ぎ(Hammer/Fan)。`nonattacking_turn_rate<0.25` | Prism 喪失なら Night Stretcher で回収してから掘る |
| 終盤 | `相手サイド≤2 or 相手が資源ロック` | サイドで詰める（ex KO=2枚）or デッキアウト。**Comfey215 停止**（相手に引かせる）。NZ 死守。Acerola 解禁 | 自山 `deckCount≤K` なら全サーチ/215 を停止し残サイドで詰め |

**使うカード/使わないカード**: 序盤=Colress/Poffin/PokePad/Comfey215/Telepath。中盤=Prism→Yveltal, Hammer/Fan/Xerosic/Boss。終盤=Boss/Acerola、Comfey215 は不使用。

---

## §5 スコアカード（1-5）＋対立軸

| 軸 | 点 | このデッキ固有の定義／検証指標 |
|---|---|---|
| 速度 | 2 | 勝ち筋 online は Yveltal 装填後（~round4-6+）。`first_attack_turn[689]` |
| 火力曲線 | 2 | Yveltal 110 固定＋Comfey 20-40。バースト無し |
| 安定性 | 4 | 巨大ドロー/サーチ（が自己山切れの諸刃）。`hand_size_dist` |
| 継戦力 | 3 | Night Stretcher で Yveltal/Prism 回収。ただし NZ は ACE で回収不能=上限 |
| 対応力 | 4 | 対 ex は NZ/Shaymin/Acerola＋エネ破壊で最強。対**非ex攻撃者**とスタジアム上書きに脆弱 |
| 資源経済 | 2 | エネ10枚(ダーク源3)と薄く、ドローエンジンが自山を食う。`loss_share[deckout]` |
| 妨害耐性 | 3 | 急所は NZ(1枚ACE) と Prism(3枚)。ここを割られると機能停止 |
| サイド構造 | 3 | 全サイド1の安トレード＋相手 ex は2枚源。ただし取り切りは遅い |
| **lock_integrity(発明軸)** | 3 | NZ 稼働率×Prism 可用性。勝敗の中心変数 |

### 対立軸（tensions）★
1. **ドロー安定性 ↔ 自己山切れ**: NZ/Prism/盤面を探す同じエンジンが自分を殺す。バランス点=`サーチは欠損ピース所持ベース かつ deckCount>K(K∈{8..12})`。**掘りを止める側**が必須。
2. **Comfey215 のカード増 ↔ 相手の反デッキアウト**: 215 は相手を3ドローさせる。`hand<H かつ prize-plan の時のみ、deckout-plan では禁止`。
3. **Prism→Yveltal(ダーク) ↔ Prism を無色フィラーに**: `1体の Yveltal が Prism2枚になるまで Prism は 689 専属`。
4. **ロック維持(NZ稼働) ↔ サイド圧**: 必ず攻撃（消極性は有害）しつつ、Boss 対象は実 KO 価値で選ぶ。**NZ をテンポで捨てない**（ACE, 回収不能）。

P3 含意: tension 側の指標を直す時は必ず対の指標（例: 掘り抑制↔盤面完成率）を同時監視。

---

## §6 カード別 使用宣言（全60枚）

各行: `名 ×n | 意図 | 発火条件 | 期待プレイ率/試合 | 腐る条件`。数値ゲートは較正パラメータ（締めすぎリスク付き）。

- **Boss's Orders ×4** | 装填Yveltalの実KO対象を引き摺る/キー狙撃 | `fire_if: 装填Yveltal が引いた対象を KO 可 OR 盤面キー除去`（**表示active打点でなく実打点で判定**, H7）| ~0.4/枚 | 相手ベンチが空/実打点不足で締めすぎるとリーサル逃す
- **Xerosic's Machinations ×4** | 相手手札を3に固定し再構築封じ | `fire_if: opp_hand>3`（A∈{4,5}）| ~0.3 | 既に手札≤3 なら空撃ち
- **Comfey ×4** | 序盤ドロー(215)/NZ下チップ(216)/70HP壁 | `215: hand<H かつ deck>K かつ prize-plan` `216: NZ下で殴れる時` | ~0.9 | 山薄/deckout-plan で 215 は自害
- **Pokégear 3.0 ×4** | 不足サポート（特に Colress→NZ, Boss）を掘る | `fire_if: 必要サポ不所持 かつ deck>K` | ~0.35 | 手札に既にサポ十分/山薄
- **Lillie's Determination ×4** | 死札で溢れた手を引き直し | `fire_if: NZ/必要Prism/Boss 非所持 かつ deck>K`（K∈{8,10,12}）| ~0.2 | キー抱え時に撃つと埋没(H9)
- **Poké Pad ×4** | 非ルールボックス（Comfey/Yveltal/Shaymin）補充 | `fire_if: 盤面欠損 かつ deck>K` | ~0.3 | 盤面完成後/山薄
- **Buddy-Buddy Poffin ×4** | ベンチ増強。**≤70HP基本=実質Comfeyのみ**（Shaymin80/Yveltal110は不可）| `fire_if: 序盤Comfey不足 かつ deck>K` | ~0.35 | Comfey充足後は腐り
- **Colress's Tenacity ×4** | **NZ＋エネを直接サーチ**（NZファインダー, 最優先）| `fire_if: NZ 未所持/未設置`（+Prismを一緒に取る）| ~0.4 | NZ設置後は価値低下（エネ回収に転用）
- **Night Stretcher ×4** | KO Yveltal/Comfey 回収 or **discardのPrism(ダーク源)回収** | `fire_if: 主砲/Prism が discard にあり必要` | ~0.3 | discard に有効対象なし
- **Crushing Hammer ×4** | 相手エネをコインで破壊 | `fire_if: opp に剥がすエネあり` | ~0.4 | 相手エネ0で空撃ち(H8)
- **Acerola's Mischief ×4** | 相手サイド≤2で自軍1体を相手exから保護 | `legal_if: opp_prizes<=2 かつ キー被弾予兆` | ~0.15 | 序盤は使用不可の死札
- **Telepath Psychic Energy ×4** | Comfey(P)へ→支払い＋Basic P(Comfey)ベンチ増強 | `attach: Comfey に。Yveltal には付けない（ダーク不可）` | ~0.8 | Yveltal に付けると無駄
- **Prism Energy ×3** | **Yveltalダーク専属**（998=2枚）。唯一のダーク源 | `attach: 1体のYveltalが2枚になるまで689優先` | ~0.7 | Comfey/Shaymin に流すと型死(H3)
- **Basic {P} Energy ×3** | Comfeyの P / Yveltal998 の無色枠(3枚目) / Shaymin | `attach: P需要 or Yveltal無色スロット` | ~0.6 | 余剰時のみ壁付け
- **Yveltal ×2** | 主砲。1体を Prism2 で 998(110/T)。逃0。2体目=997トラッパー/予備 | `arm: Prism>=2 で attacker 扱い（Psychicのみは非装填）` | ~0.6 | Prism 不足時は 997 止まり/型死
- **Handheld Fan ×2** | 被弾時に攻撃側エネを相手ベンチへ移動（エネ denial）| `attach: 前線で被弾する壁/active` | ~0.5 | ベンチ待機に付けると不発(H12)
- **Shaymin ×1** | ベンチ番: Flower Curtain で自ベンチ非ルールボックスを**全攻撃(非exも)**から保護＝NZの穴埋め。強制active時30チップ | `keep on bench` | ~0.6 | 相手が active 直接狙いのみだと価値低
- **Neutralization Zone ×1 (ACE SPEC)** | ロック本体: 非ルールボックスへの ex/V ダメージを両者無効 `[PROBE P3]`。即設置・死守 | `play ASAP, keep up` | ~0.9 | 相手スタジアム上書きで消滅=回収不能で恒久崩壊

**L0 既定の監査（項目別）**:
1. 特性無条件使用 → 能動特性は無し（Shaymin は受動）。**空振り＝無害**。§2b 特性起因なし。
2. エネは表示最大アタッカーへ → 最大=Yveltal110 で狙いは正しいが、**型を見ず Psychic を Yveltal に貼る＝型死**（H2）／**Prism を Comfey に流す**（H3）。**要修正（catastrophic）**。
3. ドローサポは手札≤5で発火 → Lillie が NZ/Prism を巻き込み埋没＋山圧迫（H9）。**要ゲート**。
4. サーチ/ボールを毎ターン無条件 → **自己山切れで自滅**（H1, `[PROBE P5]` で 27/40 自滅）。**要 need ベース＋薄山停止（catastrophic）**。
5. 進化は表示最大優先 → **進化ライン無し。無害**。
6. 逃げは装填(エネ数)なら逃さない → **Psychic だけの型死 Yveltal を「装填」と誤認**し active 固定（H5）。逃0なのに無駄に居座る。**要型対応装填判定**。
7. gust は自active表示≥60時 → active が Comfey(20)/型死Yveltal だと **Boss を撃たない/価値を逃す**（H7）。**実打点/サイド価値ベースに**。
8. KO後昇格は表示最大 → 未装填(Prismなし)Yveltal を晒す恐れ（H5関連）。**装填済みを昇格**。

---

## §7 敗因仮説と L2 規則候補

**予想敗因分布**:
- `loss_share[deckout]（自分）`が最大 — 巨大ドローエンジン×遅い勝ち筋 `[PROBE P5]`。
- テンポ/型死 — Yveltal に非ダークが乗り主砲が沈黙 `[PROBE P1]`。
- ロック崩壊 — NZ を上書き/prize され回収不能（ACE）。
- 非ex攻撃者相手 — NZ が効かず素で殴られる（Shaymin/Acerola で部分カバー）。

**適用パターン（優先順）**:
1. `自己害効果の無条件発火` + `山切れ負け` → サーチ/215/Lillie に **need ゲート＋薄山停止**（H1,H9,H10）。
2. `型死ポケモンを装填扱い` + `エネが意図しないポケモンへ` → **型対応の装填判定＋Prism給餌計画**（H2,H3,H5）。
3. `キーカードのプレイ率~0` → NZ を **Colress 所持ベースで探索・設置**（H4）。
4. `gust が価値を逃す` / `妨害の浪費` → **実打点・相手盤面条件ベース**（H7,H8）。
5. `攻撃せず抱え込み` → 装填後は攻撃優先（H6, 逃0で居座り理由が薄い）。

### L2 要否の結論: **L2 必要**
- catastrophic 2本（H1 自己山切れ・H2 型死）が**プローブで実証**され、汎用 L0 の既定（無条件サーチ／表示最大への型無視給餌）が**このデッキの勝ち筋そのものを壊す**。
- 勝ち筋がグラインド/資源否定であり「最良表示アタッカーに給餌して殴る」汎用床とは前提が逆。
- よって **L2 作成を推奨**。中核規則: (i) サーチ/ドローの need ゲート＋薄山停止、(ii) Prism→Yveltal 専属の型対応給餌と型対応装填判定、(iii) NZ の所持ベース探索・死守、(iv) 妨害/gust の実打点・相手状態ゲート。
