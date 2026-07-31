# P0 静的解析（盲検） — `marnie_grimmsnarl`

対象: cabt シミュレータ / L2 Discovery Pipeline P0。カードDB + ルール知識 + シミュレータプローブのみから解析。
検証済み(verified)/未検証(assumed) を各主張に明記する。

## 0. 60枚全リスト（攻撃・特性テキスト）

### ポケモン（20枚）
| cid | 名前 | ×n | 種別 | HP | 弱点/抵抗/逃 | 攻撃/特性 |
|---|---|---|---|---|---|---|
| 646 | Marnie's Impidimp | 4 | 種(闇) | 70 | 弱=草/逃1 | Filch [C] 0「1ドロー」／ Corkscrew Punch [D] 10 |
| 647 | Marnie's Morgrem | 2 | 1進化(闇)←Impidimp | 100 | 弱=草/逃1 | Corkscrew Punch [D][D] 60 |
| 648 | **Marnie's Grimmsnarl ex** | 3 | 2進化(闇)←Morgrem | **320** | 弱=草/逃2 | 特性 **Punk Up**：手札からこのポケに進化させた時、山から基本闇エネ最大5枚を自分の「マールの」ポケに好きに付ける／ **Shadow Bullet [D][D] 180**（+ベンチ1体に30、弱抵無視） |
| 112 | Munkidori | 4 | 種(超) | 110 | 弱=闇/抵=闘/逃1 | 特性 **Adrena-Brain**：自番1回、闇エネが付いていれば自分のポケ1体からダメカン最大3個を相手ポケ1体へ移す／ Mind Bend [P][C] 60（混乱） |
| 103 | Snorunt | 2 | 種(水) | 60 | 弱=鋼/逃1 | Astonish [W][C] 20（手札1枚山戻し） |
| 104 | Froslass | 2 | 1進化(水)←Snorunt | 90 | 弱=鋼/逃1 | 特性 **Freezing Shroud**：ポケモンチェックで特性持ちポケ全て（両者、Froslass除く）にダメカン1／ Frost Smash [W][C] 60 |

### エネルギー（10枚）
- 基本闇エネ ×10 （**このデッキ唯一のエネ供給。超も水も無い**）

### トレーナー（30枚）
| cid | 名前 | ×n | 種別 | 効果 |
|---|---|---|---|---|
| 1259 | Spikemuth Gym | 4 | スタジアム | 各プレイヤー自番1回、山から「マールの」ポケ1枚をサーチ→手札 |
| 1152 | Poké Pad | 4 | グッズ | ルールボックス無しポケ1枚サーチ→手札 |
| 1219 | Team Rocket's Petrel | 4 | サポート | トレーナー1枚サーチ→手札 |
| 1227 | Lillie's Determination | 4 | サポート | 手札を山に戻し6枚ドロー（サイド6枚残なら8枚） |
| 1079 | Rare Candy | 3 | グッズ | 種→2進化（1進化飛ばし）。初手番/新規種は不可 |
| 1086 | Buddy-Buddy Poffin | 3 | グッズ | HP70以下の種を最大2体ベンチに（→Impidimp/Snorunt のみ。Munkidori 110 は不可） |
| 1231 | Dawn | 3 | サポート | 種・1進化・2進化を各1枚サーチ→手札（=Impidimp+Morgrem+Grimmsnarl を1枚で揃える） |
| 1097 | Night Stretcher | 3 | グッズ | トラッシュのポケ or 基本エネ1枚→手札 |
| 1182 | Boss's Orders | 2 | サポート | 相手ベンチ1体をバトル場へ |
| 1197 | Xerosic's Machinations | 1 | サポート | 相手手札を3枚まで捨てさせる |
| 1161 | Handheld Fan | 1 | ポケの道具 | 付いたポケがバトル場で攻撃を受けた時、攻撃元のエネ1個を相手ベンチへ移す |
| 1080 | **Unfair Stamp** | 1 | グッズ **ACE SPEC** | 前の相手番に自分のポケがKOされた時のみ。両者手札を山に戻し、自分5枚/相手2枚ドロー |

**型死の即断（verified by DB）**: 供給エネは闇のみ。従って
- Munkidori `Mind Bend [P][C]` は**永遠に撃てない**（超エネ不在）→ Munkidori は**特性専用**（Adrena-Brain）。
- Snorunt `Astonish [W][C]`・Froslass `Frost Smash [W][C]` も**永遠に撃てない**（水エネ不在）→ Froslass ラインは**特性専用**（Freezing Shroud）。

---

## §1 勝ち筋の仕様（win condition spec）

### 主勝ち筋: 「Grimmsnarl ex を毎ターン Shadow Bullet で回し、Froslass+Munkidori のスプレッドで補助KO」
手順:
1. **T1**: Impidimp をバトル場に（Poffin/Poké Pad/Spikemuth Gym で確保）。ベンチに Munkidori・Snorunt・2体目 Impidimp を並べる。
2. **T2–T3**: `Rare Candy` で Impidimp→**Grimmsnarl ex**（または Morgrem 経由）。進化時 **Punk Up が発火**し山から闇エネ最大5枚を「マールの」ポケに配分（**verified**）。→ アクティブ Grimmsnarl に [D][D]、余りを**2体目のマール・ライン**に先付けしてチェイン準備。
3. **T3以降**: 毎ターン `Shadow Bullet` = **アクティブ180 + 相手ベンチ1体に30**（弱抵無視、**verified**）。
4. **並行してスプレッド機関を起動**: **Munkidori に手貼りで闇エネ1個**を付け Adrena-Brain を on（**要注意: Punk Up は「マールの」ポケ限定で Munkidori に付かない**、後述）。Froslass の Freezing Shroud で毎チェック両者の特性持ちに+10、Munkidori で自分側のダメカンを相手へ移送（最大3/体）。
5. Grimmsnarl(320HP) は環境典型打点(~200)で**1発では落ちない**タンク。相手が2ターンかけて倒す間に180+スプレッドで先行KOしサイドレースを取る。

### 副勝ち筋 / プランB
- Grimmsnarl 未完成時: `Morgrem Corkscrew Punch [D][D] 60` を繋ぎに（消極的に殴らないより有害でない）。
- 主砲が枯れた時: `Night Stretcher`+`Rare Candy`+3枚目 Grimmsnarl で再建。Froslass+Munkidori の**純スプレッドだけ**でも遅いKOは可能（ただし単独では火力不足）。

### 実打点式（_expected_dmg 候補）
- **648 Shadow Bullet**: `active = 180`（弱点で×2）; `bench_snipe = 30`（固定・弱抵無視・対象は自分で選ぶ相手ベンチ1体）。**verified**（ctx=15 DAMAGE が相手ベンチを提示）。
- **112 Adrena-Brain（実効到達）**: `reach = 30 × (闇エネ付き Munkidori 数)`。移送元は自分のダメカン、上限は各体3個。**verified**（ctx=16/40/13: 自Munkidoriから抜いて相手へ配置）。
- **104 Freezing Shroud（毎チェック）**: `opp_chip = 10 × (相手の特性持ちポケ数)`, `self_chip = 10 × (自分の特性持ちポケ数; Munkidori＋Grimmsnarl＋余剰Froslassは自身除外)`。**verified**（ctx=34 SKILL_ORDER で毎チェック発火）。
- **647 Corkscrew Punch**: `60`（固定）。

複合の実効: あるベンチ脅威に対し `30(bullet) + 10(froslass) + 30×#Munkidori(adrena)` を**アクティブに触れず**毎ターン与えられる。

---

## §2 ルール相互作用の棚卸し

- **エネは進化で持ち上がる**: Impidimp/Morgrem に闇を先付け → Grimmsnarl へ持ち上がる。Punk Up と併用で、下位ラインへの先貼りは**2体目の主砲の即時装填**に有効。**すべき**。
- **Punk Up は「マールの」ポケ限定（verified）**: ATTACH_FROM の選択肢がベンチのマール・ライン(area5)のみ。**Munkidori/Snorunt/Froslass には Punk Up でエネが行かない**。→ Munkidori の起動は**手貼り（1/ターン）に依存**。ここがエネ経済の急所。
- **ダメージ配置 vs ダメージ**: Shadow Bullet ベンチ30 と Adrena-Brain と Freezing Shroud はいずれも**弱点・抵抗を無視**（配置/固定）。→ 相手の抵抗・軽減を貫通する。
- **型要求 vs 供給**: 上記の通り Munkidori/Snorunt/Froslass の技は**型死**。攻撃要員は事実上 **Grimmsnarl と（繋ぎの）Morgrem のみ**。
- **1ターン1回制の衝突**: サポート14枚（Petrel4/Lillie's4/Dawn3/Boss2/Xerosic1）が**1枚/ターン**を奪い合う。手貼り1回も Munkidori 起動 vs Grimmsnarl 追加装填で競合。逃げも1回。→ シーケンス圧が高い。
- **ドロー/山掘り**: 能動ドローは Lillie's（山に手札を戻して6/8ドロー）が主。Poké Pad/Petrel/Dawn/Poffin/Spikemuth は**サーチ**（ドローでない）。→ 「手札≤5でドロー」的な汎用閾値が想定する**受動ドローは存在しない**。
- **デッキアウト地平**: サーチ多用で山は薄くなるが Lillie's は手札を山に戻すため純粋なミルではない。Night Stretcher で回収。長期戦での軽度リスク。
- **ACE SPEC**: Unfair Stamp（1枚制限、KO被弾後の巻き返しドロー）。**山に1枚のみ**。
- **スタジアム**: Spikemuth Gym は「マールの」ポケサーチを**両者に**与えるが、相手は通常マールを持たない → **実質一方的**。維持推奨。

---

## §3 サイド算術（prize arithmetic）

- **Grimmsnarl ex: HP320 / サイド2枚**。環境典型打点~200では**1発耐える**（弱点＝草の相手のみ ×2 で ~360 必要=実質2パン確定を早める）。→「主砲は毎ターンは取られない」タンク前提。相手に2ターンを強要しつつこちらは180+スプレッドで先行。
- Impidimp/Morgrem/Munkidori/Snorunt/Froslass はサイド1枚。**Munkidori(110)・Froslass(90)・Snorunt(60) は非攻撃の裏方**で、ここを Boss で引っ張られKOされると**エンジンが1枚で崩れる**（特に Munkidori/Froslass はサイド1枚で高価値損失）。
- 必要KO: 相手 ex 主体なら3KO（2+2+2/一部1）。チェイン要件: **2体目 Grimmsnarl まで装填**できれば主砲KO後も180継続。3枚目は Night Stretcher 前提。

---

## §4 フェーズプラン

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| 序盤 | `own_active is Impidimp` 未成立 or `Grimmsnarl not in play` | T1: Impidimp をアクティブ、ベンチに Munkidori・Snorunt・予備Impidimp。Spikemuth 設置。Dawn/Petrel で進化線 or Rare Candy を確保 | 種薄い→Poffin/Poké Pad連打。Grimmsnarl 手札に無→Dawn or Spikemuth×3で掘る |
| 中盤 | `648 in play` かつ `Punk Up 発火済` | 毎ターン Shadow Bullet 180+30。**≥1 Munkidori に闇エネ1**（Adrena-Brain on）。Froslass 稼働。ベンチ30とAdrenaを**同一の落とし切り対象**に集約 | Munkidori に闇未着→手貼り優先を Munkidori へ振る。主砲被弾→2体目ライン先付けで即復帰 |
| 終盤 | `own_prizes_taken ≥ 3` | Boss で裏の詰めどころを引き、Shadow Bullet+スプレッドで残りサイドを刈る。Unfair Stamp は被KO時のみ | 主砲全滅→Night Stretcher で Grimmsnarl 回収→Rare Candy 再建。山薄→Lillie's でリソース更新 |

**使う/使わないカード宣言**:
- 序盤に使う: Poffin, Poké Pad, Spikemuth, Dawn, Petrel, Rare Candy。使わない: Boss（詰め用）, Xerosic, Unfair Stamp, Handheld Fan。
- 中盤: 手貼り（Munkidori→Grimmsnarl の順で分配）, Adrena-Brain, Freezing Shroud, Shadow Bullet, Night Stretcher。
- 終盤: Boss, Xerosic（相手手札干渉）, Unfair Stamp（被KO後）。

---

## §5 多面的評価軸（deck scorecard）

1. **速度** — 評価: **3/5** / 根拠: 2進化 ex 到達が主勝ち筋。Rare Candy3+Dawn3+Spikemuth4 で T2–T3 到達を狙えるが 2進化ゆえ分布は広い。種10枚でマリガン率~25.9%（verified: 計算）/ 検証指標: `first_attack_turn`中央値≤3、`nonattacking_turn_rate`低 / 修正: 遅い時は Morgrem 60 で繋ぐ。
2. **火力曲線** — 評価: **4/5** / 根拠: online後は 180+30+スプレッドで盤面全体に毎ターン到達 / 検証指標: online後の`post_ko_attack_rate`≈1 / 修正: スケール成分（Adrena/Froslass）を共有知覚に昇格。
3. **安定性** — 評価: **3/5** / 根拠: マリガン率~25.9%、種の実質価値は Impidimp のみ（Munkidori/Snorunt 先頭は攻撃死）/ 検証指標: `first_attack_turn`分布、初手 Impidimp 到達率 / 修正: 先頭を Impidimp に矯正、支援種を active に置かない。
4. **継戦力** — 評価: **4/5** / 根拠: Grimmsnarl3+Night Stretcher3+Rare Candy3+Punk Up の即装填でチェイン容易 / 検証指標: `post_ko_attack_rate` / 修正: 2体目ライン先付けの上限つき装填。
5. **対応力** — 評価: **3/5** / 根拠: 壁=Boss(2)で貫通、手札干渉=Xerosic/Unfair、エネ破壊=Handheld Fan(1)、スプレッド耐性は無い / 検証指標: `gust_targets`, `play_rate[1182]` / 修正: Boss を実打点(180)でKO可能対象へ。
6. **資源経済** — 評価: **3/5** / 根拠: 闇エネ10枚+Punk Up(山から最大5)。**手貼り1回が Munkidori 起動と主砲追加装填で競合**。Punk Up が山エネを大量消費すると後半枯れ / 検証指標: `energy_attach_share[112]`が>0（Munkidoriに最低1）かつ<過剰 / 修正: 給餌計画（active648に2、余りは2体目ライン、Munkidoriに手貼りで1）。
7. **妨害耐性** — 評価: **3/5** / 根拠: 裏の Munkidori/Froslass を Boss で狙われるとエンジン崩壊（サイド1枚で高価値）。Unfair Stamp で手札リセット復帰可 / 検証指標: `loss_share[board]` / 修正: 支援ポケを複数並べ冗長化。
8. **サイドレース構造** — 評価: **4/5** / 根拠: 320タンク＋2プライズだが「1発で落ちない」ため被トレード有利。草弱点デッキ相手のみ不利 / 検証指標: `loss_share[prize]` / 修正: 対草はスプレッドで速攻。

**追加軸（このデッキ固有）**
9. **スプレッド機関の稼働率** — 評価: **要監視** / 根拠: Froslass の自傷(+10/チェック)は Munkidori(闇付き)が相手へ移送して初めて益になる。**verified**: греーディ試行で Munkidori に闇未着のまま Froslass 自傷が積もり自軍 Munkidori が70/110ダメージに（機関が死に自傷のみ残る）/ 検証指標: `trigger_fire[adrena_brain]`が online後>0、自軍 Munkidori の被ダメ / 修正: Munkidori 起動を「闇エネ所持状態」で強制。
10. **Punk Up 配分の質** — 評価: **要監視** / 根拠: 汎用「最良アタッカーに集中」で5枚全部を Grimmsnarl に乗せると 3枚が死蔵し2体目が育たない / 検証指標: `energy_attach_share[648]`が~2–3で頭打ち / 修正: 配分ヒューリスティク（active2＋予備ライン）。

---

## §6 カード別「使用宣言」（全60枚）

書式: `cid 名前 ×n | 意図 | 発火条件（状態変数） | 期待プレイ率/試合 | 腐る条件`

- **646 Marnie's Impidimp ×4** | 主砲の起点/先頭要員 | `own_active empty` or 進化元不足 | 高(~1.0) | Grimmsnarl 過剰・エネ無しで先頭放置
- **647 Marnie's Morgrem ×2** | Rare Candy無時の進化中継/繋ぎ攻撃60 | `Rare Candy 不在 & Impidimp in play` | 中(~0.5) | Rare Candy で常に飛ばされる
- **648 Marnie's Grimmsnarl ex ×3** | 主砲/Punk Up エネ加速 | `Impidimp/Morgrem in play & (RareCandy or Morgrem)` | 高(~1.0) | 進化元/Candy 不足で塩漬け
- **112 Munkidori ×4** | **特性専用**スプレッド送り手 | `闇エネ≥1 attached & 自軍にダメカン有` | 中〜高(~0.7) | 闇エネ未着で Adrena 死・自傷のみ / 技は永遠に不可
- **103 Snorunt ×2** | Froslass の種（**攻撃死**） | Froslass 進化の踏み台 | 中(~0.6) | 先頭 active で攻撃できず塩
- **104 Froslass ×2** | **特性専用**両者チップ | `in play`（自動発火） | 中(~0.6) | Munkidori未起動だと自傷過多
- **7 基本闇エネ ×10** | 唯一の供給。手貼り＋Punk Up | active648に2, Munkidoriに1, 予備ライン | 高 | Munkidori放置で機関死
- **1259 Spikemuth Gym ×4** | マール・サーチ（一方的） | 毎ターン | 高(~1.0/設置後) | 相手スタジアムに上書きされる
- **1152 Poké Pad ×4** | 非exポケ確保（Impidimp/Munkidori/Snorunt/Morgrem） | 種/進化元不足 | 中(~0.7) | 手札に既に揃う
- **1219 Team Rocket's Petrel ×4** | 任意トレーナー・チュートル | 特定トレーナー欲時 | 中(~0.6) | サポート枠を Lillie's/Dawn と競合
- **1227 Lillie's Determination ×4** | 主ドロー（山戻し6/8） | `手札にコンボ核が無い & 手札小` | 中(~0.6) | **核を抱えた手札で撃つと山送りで自壊** / サイド6でだけ8
- **1079 Rare Candy ×3** | 1進化飛ばし加速 | `Impidimp(既存) & Grimmsnarl in hand & 非初手番` | 高(~0.9) | 初手番/新規種には不可
- **1086 Buddy-Buddy Poffin ×3** | 種展開(HP70以下→Impidimp/Snorunt) | 序盤ベンチ薄 | 中(~0.7) | Munkidori(110)は取れない・盤面完成後
- **1231 Dawn ×3** | 進化線一括サーチ(種+1+2) | Grimmsnarl線不足 | 中(~0.6) | サポート枠競合
- **1097 Night Stretcher ×3** | ポケ/闇エネ回収・再建 | `Grimmsnarl or エネ in discard & 要再建` | 中(~0.5) | トラッシュに対象無
- **1182 Boss's Orders ×2** | 裏の詰め/エンジン狙撃 | `KO可能 or 相手裏に脆い鍵` | 中(~0.5) | 早期に浪費・盤面詰み前
- **1197 Xerosic's Machinations ×1** | 手札干渉 | `相手手札4枚以上 & 詰め局面` | 低(~0.2) | 序盤浪費
- **1161 Handheld Fan ×1** | エネ移送妨害 | active648に付与、相手が大型アタッカー | 低(~0.2) | 相手エネ加速なし
- **1080 Unfair Stamp ×1 (ACE)** | 被KO後巻き返しドロー | `直前相手番に自ポケKO` | 低(~0.25) | KOされない限り不可

---

## §7 敗因仮説と L2 規則候補

**予想敗因分布**:
- **テンポ負け(最大)**: 2進化到達遅延＋種先頭の質（Munkidori/Snorunt 先頭は攻撃0）で初撃が遅れる。
- **エンジン不起動負け**: Munkidori に闇が乗らず Adrena-Brain 死＝Froslass 自傷だけ残る（**verified**で観測）。汎用エンジンが Grimmsnarl に手貼りを集中すると再現。
- **エネ配分ミス**: Punk Up 5枚を主砲1体に死蔵、2体目が育たず主砲KO後に失速。
- **サイド損**: 裏 Munkidori/Froslass を Boss され機関崩壊。
- **山切れ**: 軽度（Lillie's 山戻しで緩和）。

**汎用パターン適用候補（優先順）**:
1. `キーカードのプレイ率~0`→ **Munkidori 起動を「闇エネ所持」ベース発火に**（手貼り優先を Munkidori に振る）。最優先。
2. `エネが意図しないポケモンへ`→ **給餌計画**: Punk Up は active648に2＋予備マール・ライン、Munkidori/Snorunt/Froslass には**乗せない**（Punk Up 対象外＝verified、手貼りは Munkidori に1のみ）。
3. `スケール技が過小評価される`→ Adrena-Brain(30×#Munkidori) と Froslass(10×#特性持ち) と bullet 30 を**共有知覚に昇格**し、bench30/Adrena を**同一の落とし切り対象**へ集約。
4. `主砲KO後に殴れない`→ 2体目マール・ラインの**上限つき先付け**（Punk Up 余剰＋手持ち闇）。
5. `gust が価値を逃す`→ Boss は Shadow Bullet 180 でKO可能 or 裏の Munkidori/Froslass 級の高価値へ。
6. `攻撃せず抱え込み`→ online後は毎ターン Shadow Bullet（消極性抑止）。
7. `無差別ベンチ展開`→ ベンチ許可: Impidimp×2, Munkidori×1〜2, Snorunt→Froslass×1。Munkidori 過剰展開は自傷面積を増やす。

**新パターン提案（テンプレ）**:
- 症状「自傷特性(Froslass)が味方エンジン(Munkidori)未起動のまま自軍を削る」→ 条件「self_chip源(Froslass)が場 かつ 闇付きMunkidori=0」→ 修正「手貼りを Munkidori に強制 or Froslass の自傷を移送できる体制が整うまで支援ポケの並べ過ぎを抑制」。

---

## 検証ステータスまとめ
- **verified（シミュレータプローブ）**: Punk Up が進化時発火し山から闇エネを「マールの」ポケ(bench)限定で配分 / Adrena-Brain が自Munkidoriのダメカンを相手へ移送（最大3・要闇エネ）/ Freezing Shroud が毎チェック自動発火（両者特性持ち）/ Shadow Bullet が相手ベンチ1体へ30 / Munkidori 闇未着だと Adrena 死＋Froslass自傷が自軍Munkidoriに蓄積(70/110観測)。
- **verified（カードDB）**: 型死（Munkidori[P][C]・Snorunt/Froslass[W][C]は供給闇のみで不可）/ Poffin のHP70フィルタ（Munkidori不可）/ Unfair Stamp=ACE SPEC / Grimmsnarl HP320・逃2・弱草・サイド2。
- **assumed（未検証）**: Rare Candy 経由でも Punk Up が発火する点（テキスト「手札から進化させた時」に合致、通常進化での発火は verified）/ マリガン率25.9%（解析計算）/ 環境典型打点200 の外部前提。
