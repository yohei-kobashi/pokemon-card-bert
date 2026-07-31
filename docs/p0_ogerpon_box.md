# P0 静的解析: `ogerpon_box`

> BLIND zero-shot 解析。カードDB (`agents._engine._CARDS/_ATTACKS`) とルール知識、`tools/arena.py`+`cg-lib` プローブのみを根拠にする。
> エネルギー型コード: 0=C(無), 1=G(草), 2=R(炎), 3=W(水), 4=L(雷), 5=P(超), 6=F(闘), 7=D(悪), 8=M(鋼)。

## 0. 60枚ダンプ（全カード + テキスト）

### ポケモン（全21枚・すべて Basic、進化ラインは無い）
| cid | ×n | 名前 | HP | 型 | 弱点 | 逃 | prize | 技/特性 |
|---|---|---|---|---|---|---|---|---|
| 756 | 4 | **Mega Kangaskhan ex** (megaEx→**3 prize**) | 300 | C | F | 3 | 3 | ATK `Rapid-Fire Combo`[CCC] 200: コインを裏が出るまで投げ、表1枚ごとに+50。 / 特性 `Run Errand`: アクティブ時1回、2ドロー。 |
| 1071 | 3 | **Meowth ex** | 170 | C | F | 1 | 2 | ATK `Tuck Tail`[CCC] 60: **このポケモンと付いたカード全てを手札に戻す**。 / 特性 `Last-Ditch Catch`: ベンチに出した時、サポート1枚をサーチ。 |
| 184 | 2 | **Latias ex** (resist F) | 210 | P | D | 2 | 2 | ATK `Eon Blade`[PPC] 200: 次の自分番、攻撃不可。 / 特性 `Skyliner`: **自分の場の Basic は逃げエネ0**。 |
| 272 | 2 | **Lillie’s Clefairy ex** | 190 | P | M | 1 | 2 | ATK `Full Moon Rondo`[PC] 20: **ベンチ数（両者）ごとに+20**。 / 特性 `Fairy Zone`: 相手の {N} ポケモンの弱点を {P} に。 |
| 112 | 2 | **Munkidori** (resist F) | 110 | P | D | 1 | 1 | ATK `Mind Bend`[PC] 60: 相手アクティブこんらん。 / 特性 `Adrena-Brain`: D エネ付きなら、自分のポケモンのダメカンを最大3個相手ポケモンへ移す。 |
| 108 | 1 | **Wellspring Mask Ogerpon ex** (Tera) | 210 | W | L | 1 | 2 | ATK `Sob`[C] 20: 相手逃げ不可。 / `Torrential Pump`[WCC] 100: このポケの3エネを山に戻せば、相手ベンチ1体に120。 |
| 140 | 1 | **Fezandipiti ex** | 210 | D | F | 1 | 2 | ATK `Cruel Arrow`[CCC] **表示0/実100**: 相手ポケ1体に100（ベンチ弱抵抗無視）。 / 特性 `Flip the Script`: 前の相手番に自分のポケがKOされていれば3ドロー。 |
| 141 | 1 | **Pecharunt ex** | 190 | D | F | 1 | 2 | ATK `Irritated Outburst`[DD] **表示0**: 相手が取ったサイド枚数×60。 / 特性 `Subjugating Chains`: ベンチの D ポケ(Pecharunt除く)を入替、**その新アクティブをどく**。 |
| 689 | 1 | **Yveltal** (resist F) | 110 | D | L | 0 | 1 | ATK `Clutch`[D] 20: 相手逃げ不可。 / `Dark Feather`[DDC] 110。 |
| 791 | 1 | **Moltres** | 120 | R | W | 1 | 1 | ATK `Fighting Wings`[R] 20: **相手アクティブが ex なら+90（→110）**。 |
| 209 | 1 | **Chien-Pao** | 120 | W | M | 1 | 1 | ATK `Icicle Loop`[WWC] 120: 自分のエネ1個を手札へ。 / 特性 `Snow Sink`: ベンチに出した時、**場のスタジアムを1枚トラッシュしてよい**。 |
| 96 | 1 | **Teal Mask Ogerpon ex** (Tera) | 210 | G | R | 1 | 2 | ATK `Myriad Leaf Shower`[GGG] 30: **両アクティブの付与エネ数ごとに+30**。 / 特性 `Teal Dance`: 手札の基本Gをこのポケに付け、1ドロー。 |
| 979 | 1 | **Koraidon ex** (Tera) | 230 | F | P | 2 | 2 | ATK `Tera`[FC] 50: ベンチにいる間、攻撃ダメージを全て無効。 / `Orichalcum Fang`[FFC] 200: 前の相手番に自分のポケがKOされていれば+120（→320）。 |

### トレーナー / エネルギー（39枚）
| cid | ×n | 名前 | 種別 | 効果 |
|---|---|---|---|---|
| 1198 | 4 | **Crispin** | Sup | 山から**別々の型の基本エネ2枚**をサーチ、1枚を手札に、もう1枚を自分のポケ1体に付ける。 |
| 1182 | 4 | **Boss’s Orders** | Sup | 相手ベンチ1体をアクティブに（gust）。 |
| 1121 | 4 | **Ultra Ball** | Item | **手札2枚を捨てて**、山からポケモン1枚をサーチ。 |
| 1250 | 4 | **Area Zero Underdepths** | Stadium | Tera を場に持つ側はベンチ上限8。**Tera が場から無くなるとベンチを5まで捨てる**。退場時も両者5まで。 |
| 1116 | 3 | **Energy Switch** | Item | 自分のポケの基本エネ1個を別の自ポケへ移す。 |
| 1097 | 2 | **Night Stretcher** | Item | トラッシュのポケモン or 基本エネ1枚を手札へ。 |
| 1210 | 1 | **Brock’s Scouting** | Sup | 山から Basic 2枚 or 進化1枚をサーチ。 |
| 1188 | 1 | **Ciphermaniac’s Codebreaking** | Sup | 山から2枚を山上へ（次2ドローを固定）。 |
| 1205 | 1 | **Cyrano** | Sup | 山から**ポケモンex最大3枚**をサーチ。 |
| 1227 | 1 | **Lillie's Determination** | Sup | 手札を山に戻し6ドロー（サイド6なら8）。 |
| 1159 | 1 | **Hero’s Cape** | Tool (**ACE SPEC**) | 付けたポケの HP +100。 |
| 1 | 4 | Basic {G} | Energy | G |
| 7 | 2 | Basic {D} | Energy | D |
| 5 | 2 | Basic {P} | Energy | P |
| 3 | 1 | Basic {W} | Energy | W |
| 6 | 1 | Basic {F} | Energy | F |
| 2 | 1 | Basic {R} | Energy | R |
| 16 | 2 | **Prism Energy** | 特殊 | C を供給。**Basic に付けると全型を供給（同時に1型のみ）**。 |

エネルギー総数=13（基本11 + Prism2）。基本の内訳は G4 / D2 / P2 / W1 / F1 / R1 の**レインボー**。

---

## §1 勝ち筋の仕様（win condition spec）

### 主勝ち筋（Doctrine A: Kangaskhan-tank）
**Mega Kangaskhan ex を 300/(Cape で)400 HP の壁兼主砲**にする。
- 手順: T1-2 で Kangaskhan をアクティブ、`Run Errand`(2ドロー) を毎ターン回し、Meowth ex `Last-Ditch Catch` で Crispin/Boss/Cyrano を確保。CCC は**無色=どのエネでも良い**ので Energy Switch と手貼りで3個を最速装填。
- 以降毎ターン `Rapid-Fire Combo` を撃つ。**実打点式**: `dmg = 200 + 50·H`（H=最初の裏までの表数、幾何分布 p=0.5）。E[dmg]=**250**、ただし最頻値=200（P(H=0)=0.5）。プローブ実測: {200,250,300,400} を確認（1092、verified）。
- 200 は HP≤200 の ex を**確定OHKO**、230(Koraidon)級は 250 必要=50%、Mega(300)は 300 必要=25%。よって「200-HP ex を毎ターン1体割る」がベースライン。
- 300 HP により相手は1KOに300ダメージ必要。Cape で400。**このタンク性が勝ち筋の本体**。

### 副勝ち筋 / プランB（toolbox 供給）
Kangaskhan が組めない/落ちた時、**Cyrano/Crispin で状況特化アタッカーを供給**する:
- vs ex 主体: **Moltres `Fighting Wings`[R] 110** をたった1エネで供給（R1枚 or Prism）。1エネで ex を脅かす最速手。
- スナイプ: **Fezandipiti `Cruel Arrow` 実100**（表示0）を任意対象へ。ベンチの育成中主砲/低HPを直接割る。
- 広ベンチ火力: Area Zero でベンチ8、**Clefairy `Full Moon Rondo` = 20+20·(自ベンチ+相ベンチ)**。実測340まで確認（両者8ベンチ、371 verified）。
- リベンジKO: **Koraidon `Orichalcum Fang` 200/+120=320**（前ターン自ポケKO時）、`Tera` でベンチ無敵化して温存。
- 詰め: **Pecharunt `Irritated Outburst`[DD] = 60·(相手が取ったサイド)**。相手がサイド4なら240。終盤の一撃。

### 複数ドクトリン
- **A: Kangaskhan-tank**（4枚積みが示す本命）— Kangaskhan にエネ集中、Cape 装備、3-2-KOで削る。
- **B: Toolbox-flex** — マッチごとに最良アタッカーへ分散給餌、Kangaskhan は選択肢の一つ。
判別: `energy_attach_share[756]`（Aなら高集中）、tech アタッカーの `play_rate`、`first_attack_turn`。**両立仕様として記述**（決め打ちしない）。

---

## §2 ルール相互作用の棚卸し

- **進化でのエネ持ち上がり**: 対象外。**全21枚が Basic**、進化ラインは無い。下位への先貼り概念は存在しない。
- **ダメージ vs ダメカン配置**: 全攻撃はダメージ（弱抵抗適用）。ただし Munkidori `Adrena-Brain` は**ダメカン移動**（配置扱い、弱抵抗無視で相手へ移す＝有益）。Fezandipiti/Wellspring のベンチ打点は「弱抵抗無視」明記。
- **技コストの型 vs 供給型**（型死検査、下表）: Kangaskhan/Fezandipiti は無色=常に充足。tech は 1-of の型エネ + Prism 依存。
- **1ターン1回制**: サポートは Crispin(給餌) と Boss(gust) と Cyrano(サーチ) が競合。手貼りは1回、Crispin/Teal Dance/Munkidori/Prism で実質増やす。gust を撃つ番は Crispin 給餌不可＝**給餌とgustのサポート枠衝突**が中盤の律速。
- **山掘り→デッキアウト地平**: `Run Errand` 2/T + 通常1 + Ultra Ball(掘り) で ~3-4/T 消費。全Basicで薄い。プローブ実測（ランダム自己対戦30戦）で**大半が deckCount=0 到達**（＝deckout は現実の終端条件）。良プレイ下でも Run Errand 常用は 山を早める。
- **ACE SPEC**: Hero’s Cape が唯一の ACE SPEC（デッキに1枚制約）。+100HP は Kangaskhan を400にし主勝ち筋を支える最重要1枚。
- **スタジアム競合**: 自分の Area Zero は Tera(Wellspring/Teal/Koraidon いずれか) が場にいる限り有効。相手スタジアムを Chien-Pao `Snow Sink` で割れるが**自分の Area Zero も割り得る**（§2b）。
- **特殊エネ**: Prism は Basic に付けば全型1個供給＝tech の型充足を1個だけ補助（Kangaskhan含む全ポケが Basic なので万能1個）。

### 型検査（全アタッカー: コスト型 vs 供給）
| アタッカー | コスト | 必要型 | 供給源 | 判定 |
|---|---|---|---|---|
| Mega Kangaskhan | CCC | 無色×3 | 何でも | ◎ 常時充足 |
| Fezandipiti | CCC | 無色×3 | 何でも | ◎ |
| Meowth (Tuck Tail) | CCC | 無色×3 | 何でも | ◎（ただし自バウンス技） |
| Latias (Eon Blade) | PPC | P×2 | 基本P2 + Prism | ○ 逼迫 |
| Clefairy (Rondo) | PC | P×1 | 基本P2 + Prism | ○ |
| Munkidori (Mind Bend) | PC | P×1 | 基本P2 | ○ |
| Pecharunt (Outburst) | DD | D×2 | 基本D2 + Prism | ○ |
| Yveltal (Dark Feather) | DDC | D×2 | 基本D2 + Prism | ○ 逼迫 |
| Wellspring (Torrential) | WCC | W×1 | **基本W1** + Prism | △ 1-of依存 |
| Chien-Pao (Icicle) | WWC | W×2 | **基本W1** + Prism | ▲ ほぼ型不足（W2は Crispin+Prism必須） |
| Teal Mask (Myriad) | GGG | G×3 | 基本G4 + Teal Dance | ○（Teal Dance で自給） |
| Koraidon (Orichalcum) | FFC | F×2 | **基本F1** + Prism | △ 1-of依存 |
| Moltres (Fighting Wings) | R | R×1 | 基本R1 or Prism | ○ 最軽量 |

**含意**: Chien-Pao(WW) / Koraidon(FF) / Wellspring(W) は**単独では型が揃わず、Crispin の型指定給餌が前提**。Crispin が引けない/枯れると tech ラインは事実上死ぬ。L0 が Crispin の給餌先を Kangaskhan(無色) に浪費すると型死が顕在化。

---

## §2b 自己害の棚卸し ★（60枚走査結果 = 該当あり）

| カード | 自己害の内容 | 発火してよい盤面条件（状態変数） | severity |
|---|---|---|---|
| **Meowth ex `Tuck Tail`** (1071) | このポケ+付与エネを**手札に戻す**。攻撃なので選択制だが、L0が60打点技として選ぶと主砲/最後の壁をバウンス。 | `bench_count>=1` かつ「Meowthを戻したい積極理由（エネ回収/退避）」がある時のみ。**最後の1体では絶対不可**。 | catastrophic |
| **Pecharunt `Subjugating Chains`** (141) | ベンチのDポケを強制アクティブ化し**その新アクティブをどく**。特性→L0無条件使用で意図せぬ露出+自どく。 | 「Dポケを前に出したい」意図が明確で、露出させるポケが壁/囮の時のみ。装填済み主砲を前に出す用途では禁止。 | catastrophic |
| **Chien-Pao `Snow Sink`** (209) | ベンチ登場時に**場のスタジアムをトラッシュ**。自分の Area Zero を割ると Tera 喪失時ベンチ5落ちの連鎖。 | 「場のスタジアムが**相手の**もの」かつ割りたい時のみ。自 Area Zero が出ている時は発火禁止。 | major |
| **Area Zero Underdepths** (1250) | **Tera が場から消えるとベンチを5まで捨てる**（＝ポケ喪失）。退場時も両者5落ち。 | `bench_count>5` を作るのは Tera を場に維持できる時のみ。Tera が Wellspring/Teal/Koraidon の**1枚ずつ**しかないので露出注意。 | major |
| **Ultra Ball** (1121) | 使用に**手札2枚トラッシュ**。1-of の型エネ(W/F/R)・キーex を捨てると型死/勝ち筋喪失。 | `hand` に「捨ててよい余剰(重複基本エネ/腐りサポート)」が2枚ある時のみ。1-of資源しか無い手では禁止。 | major |
| **Lillie's Determination** (1227) | 手札を山に戻し6ドロー。抱えた Cape/主砲/型エネを流す。 | `hand_size<=2` かつ手札が実質死んでいる時のみ。 | minor |
| **Torrential Pump** (108) | Wellspring の3エネを山へ（自エネ喪失）してベンチ120。 | ベンチKOで勝負が決まる時のみ（任意効果）。 | minor |
| **Icicle Loop** (209) | Chien-Pao のエネ1個を手札へ（自脱エネ）。 | 連撃継続を捨ててでも回収したい時のみ。 | minor |
| **Latias `Eon Blade`** (184) | 次番攻撃不可（自テンポロック）。 | リーサル or 相手に次番攻撃されない盤面のみ。 | minor |

→ severity=catastrophic 仮説を最低1本立てる（H1）。汎用エンジンは特性を無条件使用するため、**Subjugating Chains / Snow Sink / Tuck Tail** が最大の事故源。

---

## §3 サイド算術（prize arithmetic）

- **Mega Kangaskhan ex は 3-prize**。落ちると相手は一気に3枚。相手は Kangaskhan を**2回割れば勝ち**（6枚）。よって「主砲は毎ターン取られる前提」では**破綻**。300/(Cape)400 HP で**取られない前提**が必須。
- 他アタッカーは ex=2, Munkidori/Yveltal/Moltres/Chien-Pao=1。tech で1prizeを刻む余地はあるが本命は Kangaskhan の壁。
- 自打点: Rapid-Fire 200 で HP≤200 ex を毎ターンOHKO（相手サイド2/回）。相手が ex 主体なら **3KO(=6prize)で勝ち**、我は3回攻撃通せば良い。
- チェイン要件: Kangaskhan を主軸にするなら**2体目の装填**が要る（1体目が2-3回殴って落ちる前に）。CCC無色なので装填は容易だが、3-prize を2体連続で失うと即負け＝**同時に2体を晒さない**（Area Zero の露出/gust に注意）。

---

## §4 フェーズプラン（計測可能）

| フェーズ | 遷移条件（状態変数） | 「計画通り」の定義 | 逸脱時リカバリ |
|---|---|---|---|
| **序盤** | `turn<=2` かつ `energy_on(756)<3` | T1-2に Kangaskhan or 有効スターターがアクティブ、Meowth `Last-Ditch` でサポート確保、Area Zero 設置、`Run Errand` 起動。`bench_size>=2` | スターター事故→ Brock's Scouting/Ultra Ball で Basic 確保、Cyrano で ex 呼ぶ |
| **中盤** | `energy_on(attacker)>=3` かつ `first_attack_turn` 到達 | 毎ターン Rapid-Fire で 200+ を通しサイドを取る（`nonattacking_turn_rate≈0`）、Cape を Kangaskhan に、型エネは Crispin で tech へ経路指定 | 主砲落ち→2体目装填 or Moltres/Fezandipiti で刻む |
| **終盤** | `my_prizes<=3` | Boss で裏の脅威/低HPを釣って詰め、Cruel Arrow/Irritated Outburst の実打点でリーサル、`deckCount>5` を維持し deckout 回避 | whiff(200止まり)→ gust対象を2HKO圏に、Night Stretcher でエネ/ポケ再利用 |

各フェーズの使用/不使用: 序盤=Ultra Ball/Meowth/Crispin/Area Zero を使う、Boss は温存。中盤=Crispin(給餌)/Energy Switch/手貼り。終盤=Boss/tech アタッカー/Night Stretcher。Lillie's Determination は手札枯れ時のみ。

---

## §5 スコアカード + 対立軸

| 軸 | 点(1-5) | このデッキ固有の定義 / 検証指標 |
|---|---|---|
| 速度 | 3 | Kangaskhan は無色装填で T2-3 online。`first_attack_turn` |
| 火力曲線 | 4 | Rapid-Fire E=250(最頻200)。scale技は表示過小。`expected_dmg` 実装で補正 |
| 安定性 | 4 | Meowth+Cyrano+Crispin+Ultra Ball+Brock で厚い。全Basicでブリック少。`play_rate[1071/1205]` |
| 継戦力 | 4 | 無色主砲＋Night Stretcher 回収＋Run Errand。ただし3-prize喪失は重い |
| 対応力 | 4 | toolbox: 壁=Kangaskhan/Latias自由逃げ、スナイプ=Fezandipiti、対ex=Moltres、広火力=Clefairy。`play_rate[tech]` |
| 資源経済 | 2 | エネ13枚(1-of多数)、Run Errand+Ultra Ball で山を焼く。`loss_share[deckout]`、`hand_size_dist` |
| 妨害耐性 | 3 | 主砲300HPで手貼り剥がしに強いが gust で2体目露出に弱い。`gust_targets` |
| サイドレース構造 | 2 | **3-prize 主砲が最大の弱み**。取られると2KO負け。`loss_share[prize]` |
| **(発明) 型充足率** | 2 | tech の必要型が揃った割合。Crispin依存。`energy_attach_share[cid]` |

### 対立軸（tensions）★
1. **山掘り × デッキアウト**: Run Errand(2/T)+Ultra Ball が engine 兼敗因。balance: `deckCount<=8 のとき Run Errand/Ultra Ball を止める`。
2. **ベンチ幅(Area Zero 8＝Clefairy火力) × 自己ベンチ崩壊/gust露出**: balance: `bench_size>5 は「場に Tera>=1」の間のみ`。
3. **主砲タンク(3-prize温存) × 攻撃即応**: balance: `Cape装備前 or 2体目未装填の間は Kangaskhan を2体同時に晒さない`。
4. **エネ集中(Kangaskhan無色) × 型エネ温存(tech)**: balance: `Crispin/1-of型エネ(W/F/R) は必要 tech が場にいる時のみそこへ、いなければ Kangaskhan へ`。
P3含意: これらは対で監視しないと whack-a-mole。

---

## §6 カード別「使用宣言」（全60枚）

| カード ×n | 意図 | 発火条件（状態変数） | 期待プレイ率/試合 | 腐る条件 |
|---|---|---|---|---|
| Mega Kangaskhan ex ×4 | 主砲兼壁 | 常時アクティブ候補、`energy>=3`で Rapid-Fire | 0.95 | 相手が300+を毎ターン出す/闘弱点相手 |
| Meowth ex ×3 | サポ確保エンジン | ベンチ着地時 `Last-Ditch`; 終盤の詰めで Tuck Tail は**禁止気味** | 0.9(特性) | ベンチ枠無/最後の1体で Tuck Tail 誤爆 |
| Latias ex ×2 | 逃げ0付与ピボット / 予備砲 | `Skyliner`常時; Eon Blade は次番攻撃不可を許容できる時 | 0.6(特性)/0.15(技) | Pのみ2枠必要で装填遅い |
| Lillie’s Clefairy ex ×2 | 広ベンチ火力 | `bench_total>=6`（Area Zero下）で Rondo | 0.2 | ベンチ狭い/P不足 |
| Munkidori ×2 | ダメカン移送でOHKO圏調整 | `D energy on self` かつ相手が10-30残り | 0.3 | D未装填 |
| Wellspring Ogerpon ex ×1 | ベンチ壁(Tera)/スプレッド | Tera でベンチ無敵温存; Torrential は W揃い時 | 0.2 | W枯渇 |
| Fezandipiti ex ×1 | スナイプ100 / KO後ドロー | `Flip the Script`=自ポケ被KO時; Cruel Arrow は割りたいベンチ低HPがある時 | 0.3 | 3エネ必要、表示0でL0が選ばない |
| Pecharunt ex ×1 | 悪ピボット / 詰め砲 | Outburst は `opp_prizes_taken>=3`; Chains は**壁を前に出す時のみ** | 0.15 | 序盤(相手サイド0で0打点)/自どく誤爆 |
| Yveltal ×1 | 逃げ縛り / 軽量D砲 | Clutch で相手足止め、Dark Feather は D揃い | 0.1 | D逼迫 |
| Moltres ×1 | 対ex 1エネ110 | `opp_active is ex`、R1供給時 | 0.25 | 相手アクティブ非ex(20のみ) |
| Chien-Pao ×1 | 対鋼/水砲・相手スタジアム割り | `opp stadium exists` で Snow Sink; Icicle は WW揃い | 0.1 | 自Area Zero を割る誤爆/W不足 |
| Teal Mask Ogerpon ex ×1 | G自己加速 / Tera壁 / スケール砲 | `Teal Dance`で毎ターンG付け1ドロー | 0.25(特性) | 手札にG無 |
| Koraidon ex ×1 | リベンジ320 / ベンチ無敵 | `Tera`温存→被KO次番に Orichalcum で320 | 0.15 | F逼迫、単騎装填困難 |
| Crispin ×4 | 型指定エネ加速（tech給餌の要） | 手貼り前、`tech attacker in play で必要型`; いなければ Kangaskhan へ | 0.8 | 山にエネ枯れ |
| Boss’s Orders ×4 | gust（詰め/裏の主砲落とし） | `my_prizes<=4` かつ釣る価値のある裏 | 0.6 | 序盤の浪費/裏に的なし |
| Ultra Ball ×4 | ポケサーチ | 捨て札2枚に**余剰**がある時のみ | 0.7 | 1-of資源しか無い手で誤爆 |
| Area Zero Underdepths ×4 | Tera時ベンチ8（Clefairy火力/展開） | 序盤設置、`Tera in play` 維持 | 0.8(設置) | Tera 0体で自ベンチ崩壊トリガ |
| Energy Switch ×3 | エネ再配置（主砲へ集中/tech転送） | `energy misplaced` 時 | 0.5 | 基本エネのみ対象(Prism不可) |
| Night Stretcher ×2 | ポケ/基本エネ回収 | トラッシュに主砲/型エネがある時 | 0.4 | 序盤に的無 |
| Brock’s Scouting ×1 | Basic確保(事故回避) | `bench<2 or no attacker` | 0.3 | 展開済み |
| Ciphermaniac’s Codebreaking ×1 | 山上固定(次2ドロー保証) | キー2枚を次に引きたい時 | 0.15 | ドロー手段被り |
| Cyrano ×1 | ex 3枚サーチ（toolbox展開の核） | tech アタッカーを揃えたい時 | 0.35 | ex 既に手札 |
| Lillie's Determination ×1 | 手札リフレッシュ | `hand_size<=2 で死に手` | 0.2 | 良い手札を流す |
| Hero’s Cape ×1 (ACE) | Kangaskhan→400HP | 主砲アクティブ時に即装備 | 0.6 | 非主砲に付ける/序盤で腐る |
| Basic G ×4 | Teal/汎用・無色コスト | Teal Dance/手貼り | — | — |
| Basic D ×2 | Pecharunt/Yveltal/Munkidori特性 | D砲・Adrena-Brain条件 | — | — |
| Basic P ×2 | Latias/Clefairy/Munkidori | P砲 | — | — |
| Basic W ×1 | Wellspring/Chien-Pao | Crispin経由給餌 | — | W tech不使用で腐り |
| Basic F ×1 | Koraidon | Crispin経由給餌 | — | Koraidon不使用で腐り |
| Basic R ×1 | Moltres | 1エネ110 | — | Moltres不使用で腐り |
| Prism ×2 | 全型1個の穴埋め | tech の型充足1個 | — | — |

**L0 既定監査（項目ごと）**:
1. **特性無条件使用** → Subjugating Chains(自どく/強制入替)・Snow Sink(自Area Zero割り) が事故。Run Errand/Teal Dance/Last-Ditch/Adrena-Brain/Skyliner/Fairy Zone/Flip the Script は無害〜有益。→ 盤面ゲート必要（H1）。
2. **手貼りは表示最大へ** → Kangaskhan(200)へ集中は正しいが、型エネ(W/F/R)まで Kangaskhan の無色枠に吸われ tech 型死（H3）。
3. **ドローサポ手札≤5発火** → Lillie's Determination が良い手札で発火し得る（1-of、minorだが監視）。
4. **サーチ/ボール無条件毎ターン** → Ultra Ball が捨て札コストで1-of資源を焼く（H9）+ deckout加速（H7）。
5. **進化は表示最大優先** → 進化ライン無し、非該当。
6. **逃げ=装填済みなら逃がさない** → Latias Skyliner で全Basic逃げ0＝逃げ判定ほぼ無効化（副作用小）。ただし型死壁(Chien-Pao等)を「装填済み」と誤認しない（H8 少）。
7. **gust は自表示≥60時のみ** → Kangaskhan200なら常に条件満たすが、Fezandipiti(0)/Pecharunt(0)アクティブ時に gust しない＝実打点無視（H2/H6）。
8. **KO後昇格=表示最大** → 未装填の2体目 Kangaskhan/tech を晒し得る（3-prize露出、H4）。

---

## §7 敗因仮説と L2 規則候補

予想敗因分布: **prize(3-prize主砲の連続被KO)** 最大 → **board(自己害/型死で殴れない)** → **deckout(掘り過多+stall)** → tempo。

適用パターン:
- scale技過小評価 → 実打点式を共有知覚に昇格（Cruel Arrow/Irritated Outburst/Full Moon Rondo/Myriad/Orichalcum）。**最優先**。
- キーカードプレイ率~0 → tech アタッカー/型エネの発火を所持ベースへ。
- エネ意図外へ → 型指定給餌計画（Crispin は tech 優先、無ければ Kangaskhan）。
- 自己害無条件発火 → 盤面ゲート（§2b宣言を実装）。**catastrophic**。
- 山切れ → 掘り停止側（deckCount<=8で Run Errand/Ultra Ball 抑制）。
- 新パターン提案: 「**3-prize タンクの露出制御**」＝症状: Kangaskhan(mega)が Cape無/2体同時晒しで連続被KO→条件: `energy_on(2体目Kangaskhan)==0 or no Cape` で前進→修正: Cape装備までタンク前進を抑制、2体目は装填後のみ露出。

### L2 が必要か？
**必要**。根拠: (a) §2b に catastrophic な自己害が複数（特性無条件使用の事故源）、(b) 表示0/表示低の scale 技が4種あり L0 の gust/昇格/選択が実打点を知らない、(c) 3-prize 主砲の露出制御という L0 既定に無い専用ロジック、(d) 1-of レインボーの型指定給餌。これらは汎用床では表現できず、**H1/H2/H3 の catastrophic〜major が支持される見込みが高い**ため L2 作成を推奨。
