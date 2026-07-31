# 決定アルゴリズム — L0 ＋ 7分類（2026-07-11）

各メソッドが「**情報取得 → その情報からどうプレイを決めるか**」を書き出したもの。
`agents/engine_v2.py` の実装と対応。表記: `[impl]`=実装済 / `[plan]`=§10 語彙で予定。
用語は `docs/engine_v2_spec.md`（§2,§3,§8,§10）に準拠。

凡例:
- `me`/`opp` = 自分/相手の `SideView`（active: PokemonView|None, bench: [PokemonView]）。
- PokemonView: `hp, energy_count, ready`(今払える技), `best_ready_dmg`, `best_potential_dmg`, `loaded`(=energy≥最安コスト且つ攻撃可), `role`。
- `ctx.my_active_dmg` = 今 active が出せる最大打点（提示 ATTACK 由来、scaling は nominal）。
- 判定メソッドは「option index の list」か `None`（下位へ委譲）を返す。

---

# Part A — L0 `BasePolicy`（汎用 develop-and-attack 床）

## A-1. 知覚（Ctx を構築、副作用なし）

### assess_self / assess_opponent → SideView
```
情報: obs.current.players[mi] / [1-mi]
決定: active・bench を PokemonView 化（各技を _can_pay で ready 判定、potential=最大打点）。
      cardId マスク時（相手 active facedown）は None 扱い。roles を各カードに付与。
```

### assess_self_setup → {MULLIGAN, DEVELOPING, READY, STALLED}
```
情報: me.active, me.bench の loaded/potential
決定:
  active is None                                  -> MULLIGAN
  active.potential==0 かつ 攻撃可能なベンチが居る   -> STALLED   (activeが非アタッカー、控えの方が良い)
  active.loaded かつ bench 非空                    -> READY
  それ以外                                        -> DEVELOPING
```

### assess_opponent_setup → {CAN_KO_ME_NOW, CAN_KO_ME_NEXT, LOW}
```
情報: opp.active.best_ready_dmg / best_potential_dmg, me.active.hp
決定:
  opp.active or me.active が None                 -> LOW
  opp.best_ready_dmg     >= me.active.hp          -> CAN_KO_ME_NOW
  opp.best_potential_dmg >= me.active.hp          -> CAN_KO_ME_NEXT
  それ以外                                        -> LOW
```

### assess_ko_targets → [PokemonView]
```
情報: ctx.my_active_dmg, opp.inplay()
決定: my_active_dmg>0 のとき hp<=my_active_dmg の相手ポケ全部（gust/攻撃対象の土台）。
```

### assess_prize_race → PrizeInfo{mine, opp, can_close, posture}
```
情報: 双方 prizes_left, my_active_dmg, opp.active
決定:
  can_close = (opp.active存在 かつ my_active_dmg>=opp.active.hp) かつ
              (mine <= prize_value(opp.active)  または  opp.bench 空)   # このKOで勝ち切る
  posture   = "race" if mine<=opp else "stall"
```

## A-2. MAIN — 優先ラダー（choose_main）
```
情報: ctx（上記知覚 + option バケット evolves/abilities/plays/attaches/attacks/retreat/end）
決定:
  turnActionCount>=40 (ループ保険): 攻撃可→最大打点 / 終了 / 先頭
  以下を上から評価し、最初に非 None を返す:
    1 step_lethal   : attacks かつ prize.can_close かつ ko_targets  -> 最大打点で即勝ち
    2 step_ability  : decide_ability
    3 step_evolve   : decide_evolve
    4 step_bench    : decide_bench
    5 step_trainer  : decide_trainer
    6 step_attach   : decide_energy_target
    7 step_attack   : decide_attack
    8 step_retreat  : decide_retreat
    9 step_end      : END（無ければ攻撃）
```
> 攻撃はラダー終盤＝**展開/進化/貼りを先に済ませてから殴る**（攻撃はターン終了のため）。
> lethal だけは「勝ち切る/強制KO」なので前倒し。

## A-3. 判定メソッド

### decide_ability
```
情報: ctx.abilities（提示された特性起動 option）
決定: あれば abilities[0]（明白に有益な無償特性を使う。有害特性は稀、ループは40手保険で防止）。
```

### decide_evolve
```
情報: 各 EVOLVE option の進化先カード（hand_card）
決定: 進化後の best_dmg が最大の option を選ぶ（ほぼ常に HP↑・打点↑）。
```

### decide_bench
```
情報: me.bench 枚数, plays の中の基本ポケモン
決定: bench < bench_target(=3) のとき、手札の基本ポケモン1体をベンチへ（KO即負け回避）。
```

### decide_trainer
```
情報: supporterPlayed, my_active_dmg, me.hand_count, plays（効果テキストで種別検出）
決定（上から）:
  サポート未使用なら:
    draw系 かつ hand<=draw_threshold(5)           -> draw
    gust系 かつ my_active_dmg>=60 かつ 相手が居る   -> gust
    draw系 かつ hand<=draw_threshold+1            -> draw
  Rare Candy / サーチ系グッズ                      -> 出す
  攻撃が未成立(attacks 空) なら 加速グッズ          -> 出す
  スタジアム未設置なら スタジアム                   -> 出す
  それ以外 -> None
```

### decide_energy_target
```
情報: attaches を「エネ添付」「ツール添付」に分離, 各 option の対象 PokemonView
score(pk) = attacker_score(pk)                     # 基礎打点+ex（装填数では釣られない）
            + 150 if energy_count < 最安コスト       # 攻撃到達へ寄せる
            (非アタッカーは -10 で最後回し)
決定:
  エネ添付があり energyAttached 未使用           -> score 最大の対象へ手貼り
  ツール添付                                     -> アタッカーへツール（非アタッカーには付けない）
```

### decide_attack
```
情報: attacks, opp.active.hp, my_active_dmg
決定:
  最大打点技 best を選ぶ
  opp.active に対し dmg>=hp                       -> best（KO）
  dmg>=attack_min_dmg(10)                        -> best（有用打点）
  それ以外                                        -> None（＝逃げ/終了へ）
```

### decide_retreat  ★ping-pong 安全（§2.2）
```
情報: retreat option, retreated, me.active, me.bench
決定:
  逃げ不可 / 既に逃げた                            -> None
  今ターン攻撃可能(attacks 非空)                    -> None   （殴れるなら逃げない）
  active.loaded（装填済み or 装填中の主砲）          -> None   （エネを捨てない=ping-pong 回避）
  ベンチに「今殴れる/より強い attacker」が居る       -> 逃げる
  それ以外                                         -> None
```

### decide_active(mode=setup|promote)
```
情報: 各 option の cardId
決定: best_dmg(cardId) が最大の順に並べて返す（＝最良アタッカーをバトル場へ）。
```

### decide_target(kind)
```
情報: opp 側 option の PokemonView
決定:
  kind="gust"        : (KO可 1/0, target_score) 降順   # 今KOできる相手を最優先で引きずり出す
  kind="energy_strip": (activeか, エネ数, target_score) 降順  # 最も装填/脅威の相手から剥ぐ
  それ以外(attack/spread/effect):
     自分側 option は attacker_score（自軍最良へ）
     相手側は target_score、spread は +(10000-hp)   # KOに近い相手へ寄せる
```

### decide_acquire（サーチ/ドロー/回収/ATTACH_FROM）
```
情報: option（cardId=None の場自体参照 or カード）
決定:
  自分の場ポケへのエネ添付先(cardId=None,自分)      -> 10000 + attacker_score（最良アタッカーへ）
  トラッシュ回収スロット(cardId=None,DISCARD)       -> ポケモンは 5000+HP / その他 1000
  通常カード                                        -> card_usefulness（進化・アタッカー>埋め草）
  有用度降順で返す。
```

### decide_discard
```
情報: option のカード
決定: card_usefulness 昇順（有用度の低いものから捨てる）。
```

## A-4. サブ選択ディスパッチ（choose_sub）
```
YES/NO      : MULLIGAN=No(手札維持) / IS_FIRST=Yes(先攻) / それ以外=Yes(発動)
SETUP/TO_ACTIVE          -> decide_active(setup)
SETUP_BENCH/TO_BENCH/FIELD-> setup_score 降順
SWITCH(相手を動かす=gust) -> decide_target(gust)
DAMAGE/COUNTER系          -> decide_target(spread)
EFFECT_TARGET            -> decide_target(effect)
DISCARD_ENERGY(相手)      -> decide_target(energy_strip)
TO_HAND/LOOK/EVOLVES/ATTACH-> decide_acquire
DISCARD/TO_DECK系         -> decide_discard
COUNT                    -> 最大の数
```

---

# Part B — L1 7分類（override するメソッドとアルゴリズム）

各分類は BasePolicy を継承し、**下記メソッドだけ差し替え**、他は L0 に委譲。
`[impl]`=実装済 / `[plan]`=§10 の役割語彙で強化予定。

## B-1. AGGRO（速攻・テンポ）
```
decide_trainer [impl]:
  情報: opp.bench, my_active_dmg
  決定: サポート未使用 かつ 相手ベンチに hp<=my_active_dmg が居る（gustでKO成立）
        -> gust を出す（サイドを1ターン早く取る）。それ以外は L0.decide_trainer。
draw_threshold=4（手札を薄く保ちテンポ維持）
[plan] decide_energy_target: fast_attacker（最安コスト）へ集中。
```
> 初版で行った「ベンチ切詰＋最速アタッカー誤配」は退行→撤去済。L0 が既に強い床のため薄い上乗せに留める。

## B-2. BEATDOWN（主砲2キャップ＋backup）
```
decide_energy_target [impl]:
  情報: 各エネ添付先 PokemonView, self.primary_ids, energy_cap(=need or 最安コスト)
  score(pk):
    非アタッカー                         -> -10
    pk が primary:
        energy_count >= cap             -> score*0.01   # 既に2キャップ→backupへ溢れさせる
        else                            -> 1000+score   # まず primary を cap まで
    pk が backup(他アタッカー):
        energy_count < 最安コスト         -> 500+score    # primary の後に backup 装填
        else                            -> score*0.1
  決定: score 最大へ手貼り（過剰装填を避け gust 全損を防ぐ）
[plan] primary/backup 検出を精密化（純engine=Dunsparce等を primary から除外＝誤り修正）。
```

## B-3. RAMP（加速→単一 payoff を overload）
```
decide_trainer [impl]:
  情報: plays の加速グッズ
  決定: 攻撃可能でも**加速を先にプレイ**（大技ターンを前倒しで組む）。無ければ L0。
decide_energy_target: L0 のまま（L0 は元々「最良アタッカー1体に集中」＝overload と同義）
[plan] payoff(overload対象)/accel(proactive)/fuel を役割で明示し、payoff に cap 無しで集中。
```

## B-4. SPREAD（ダメカン移動で複数KO）
```
decide_target(kind) [impl]:
  情報: kind in {spread, effect} のとき remainDamageCounter, 各相手 hp
  sc(pk) = (KO可: hp<=remainDamageCounter*10 なら1, 10000-hp)   降順
  決定: **今KOにできる相手**を最優先、無ければ残HP最小へ寄せる（同時KOを組む）
  他 kind は L0.decide_target。
[plan] mover（Munkidori/Dusknoir/Froslass）を毎ターン起動（decide_ability で優先）、
       spreader 攻撃を優先。finisher+gust で複数プライズを取り切る。
```

## B-5. TOOLBOX（対面でアタッカー選択）
```
_matchup_score(pk) [impl]:
  = attacker_score(pk) + 250 if pk.energyType == opp.active.weakness   # 弱点を突く
decide_energy_target [impl]: matchup_score 最大の attacker へ（+150 if 未到達）
decide_active [impl]: matchup_score（場ポケ）/ best_dmg（カード）で降順選択
[plan] attacker_pool[etype,cost,prize] を保持し、selector/flex_energy で対面最適を組む。
```

## B-6. CONTROL（妨害・停滞・グラインド）
```
main_ladder [impl]: [lethal, ability, evolve, DISRUPT, bench, trainer, attach, attack, retreat, end]
  ※ step_disrupt を攻撃/展開より前に挿入。
decide_disrupt [impl]:
  情報: plays の妨害カード(_DENIAL: Crushing/Enhanced Hammer, Xerosic, Eri, Petrel, Judge, Iono)
  決定: サポートは未使用時のみ、該当を1枚プレイ（エネ/手札破壊を継続）。無ければ None。
attack_min_dmg=30（レースでなく chip/ユーティリティのため閾値↑）
[plan] wall/lock/denial/recovery/chip を役割で分離。decide_retreat は相手アタッカーに応じ
       適切な wall へ pivot（弱点対応、正当かつ頻繁）。deckout 管理。
```

## B-7. COMBO（組み立て→ペイオフ）
```
decide_trainer [impl]:
  情報: me.hand_count
  決定: 手札薄い(<=draw_threshold)ときサーチ優先で掘る。無ければ L0。
combo_online(ctx) [impl・暫定]:
  primary が loaded  または  my_active_dmg>=80  -> True
[plan/L2] ASSEMBLE→EXECUTE ゲート:
  情報: piece 在場, payoff 装填（profile の online 述語）
  決定: online 前は掘り優先で**劣化攻撃を撃たない**（lethal を除く）。online 後にペイオフ実行。
  ※ generic な gate は誤推論 primary で starvation（A/B 実証）→ **per-deck の正しい roles + online 述語で L2 実装**。
```

---

# Part C — 情報→決定の全体像（1手の流れ）
```
act(obs):
  ctx = perceive(obs)                       # A-1 の6知覚で Ctx 構築
  if MAIN:  return choose_main(ctx)          # A-2 ラダー（各 step が B の override or A の判定を呼ぶ）
  else:     return choose_sub(ctx)           # A-4（同じ判定を再利用）
  例外時は合法な最小選択にフォールバック（決してクラッシュしない）
```
各 archetype は「知覚は共通、**判定の一部だけ自分の語彙で上書き**」という構造。
役割語彙（§10）が入ると、各 decide_* は generic ではなく archetype 固有ロールを読んで
より正確に primary/payoff/mover/wall 等を狙う。
