# 汎用エンジン v2 仕様（ドラフト）

対象: `agents/_engine.py` の刷新。3層の役割分担で「最低限の普遍エンジン → 4アーキタイプ →
（将来）個別デッキ」を積み上げる。本書は **Stage 1（最低限）** と **Stage 2（4分類）** までを固める。

---

## 0. 設計思想 — 3層アーキテクチャ

| 層 | 名前 | 役割 | メタ情報 |
|---|---|---|---|
| L0 | 最低限・普遍エンジン | どんな60枚でも合法かつ「概ね正しい」手を返す。上位層のフォールバックの床。 | 不要（カード事実のみ） |
| L1 | アーキタイプ方策 ×4 | aggro / midrange / control / combo。デッキの**カード役割メタ情報**を解釈し、その計画に沿ってL0の判断を上書き。 | デッキプロファイル（役割） |
| L2 | 個別デッキ方策（将来 Stage 3） | 特殊なデッキだけ bespoke に上書き。 | 個別 |

**制御フロー**（全 SelectContext 共通）:
```
decision = perdeck(obs)              # L2（任意・将来）
        or archetype(obs, profile)   # L1
        or minimal(obs)              # L0（必ず合法手を返す＝床）
```
各層は「具体的な選択肢インデックス」か `None`（下位へ委譲）を返す。**L0 は決して None を返さない**。

この設計の狙い: 現行エンジンに溜まった個別 special-case を、**役割ベースの一般原則**に置き換え、
「なぜその手か」を役割とアーキタイプから説明可能にする。今回発見した ping-pong 問題のような
「役割を知らないがゆえの誤り」を構造的に排除する。

---

## 1. データモデル

### 1.1 カード事実（既存 `_CARDS` / `_ATTACKS`）
hp / cardType / stage(基本・1進化・2進化) / ex・megaEx / energyType / weakness / resistance /
retreatCost / attacks[{damage, cost, text}] / abilities[{name, text}]。

### 1.2 デッキプロファイル（新規メタ情報・デッキ単位）
```python
profile = {
  "archetype": "aggro" | "midrange" | "control" | "combo",
  "energy_types": ["FIGHTING"],          # デッキが使うエネ型
  "main_attackers": [1056],              # 勝ち筋の主砲
  "combo_pieces": [...],                 # combo デッキのみ: 組み立てセット
  "roles": {                             # cardId -> 役割記述
     1056: {"role": "attacker", "tier": "primary", "energy_need": 3},
     140:  {"role": "attacker", "tier": "backup"},
     305:  {"role": "engine",   "subrole": "draw"},
     1086: {"role": "search",   "targets": "basic<=70hp"},
     7:    {"role": "energy",   "etype": "DARKNESS"},
  },
}
```
役割は **自動推論のデフォルト + デッキ単位の上書き**（§4）。

### 1.3 役割タクソノミ（語彙）
- **attacker**（tier: primary/backup, energy_need, 大型フィニッシャーか）
- **engine**（特性でデッキを回すポケモン: draw / accel / search / ダメージ配置）
- **accelerator**（エネ加速）
- **draw**（手札補充: サポート/グッズ/特性）
- **search**（特定カードサーチ）
- **disruption**（相手妨害: 手札・エネ破壊 / ベンチ呼び出し / 特殊状態）
- **wall**（ダメージ軽減・高HP・盤面停滞）
- **pivot**（入れ替え/逃げ補助）
- **recovery**（トラッシュ回収）
- **evolution_piece**（進化の種でしかない下位: Snorunt→Froslass, Abra→Kadabra 等）
- **tech**（状況対応の1枚差し）
- **energy**（基本/特殊 + 型）

---

## 2. L0 — 最低限・普遍エンジン（汎用 develop-and-attack 床）

**原則**: どんなデッキにも成り立つルールのみ。役割メタ情報を使わず、カード**事実**（hp/damage/stage/cost）
だけで判断。必ず合法手を返す。

**設計指針（確定・重要）**: L0 は「シンプルな**アグロ**」ではなく、**deck-agnostic な汎用床**である。
攻撃＝サイドを取る＝あらゆるデッキ共通の勝ち筋なので、形は「盤面を作って殴る」＝アグロ的に見える。
だが要件は **アグロ以外（control/combo）のデッキが L0 単体に落ちても"それなりに"機能する**こと。
したがって **アグロ特化の最適化は L0 に入れない**——それらは L1 AGGRO の仕事（§3.2）。

L0 が実装するのは「健全な盤面を作り、進化・有益な無償特性・ドローを使い、**意味のある**攻撃をし、
危険な逃げをしない」だけ。現行 `_choose_main` の main-line 計算・hints・special-case は引き継がず clean に。

**L0 の汎用アグロ vs L1 AGGRO の特化**（同じ「殴る」でも別物）:
| 観点 | L0 汎用床（全デッキの下支え） | L1 AGGRO（アグロ専用特化） |
|---|---|---|
| 展開 | **健全な**ベンチ（~2–3）を作る。必要ならサーチで種を揃える | 最小限に切り詰め（attacker ライン+1–2）。過剰サーチしない |
| 攻撃 | **意味があれば**攻撃（ラダーは展開・進化を先に） | 最速アタッカーで T1–2 から攻撃強行。テンポ最優先 |
| 逃げ | §2.2 ping-pong 安全規則のみ | ほぼ逃げない（テンポ損） |
| gust | **明確な KO 価値**がある時のみ | サイドレース加速のため積極的に |
| 妨害/回収 | 無償で明白に有益なものだけ | 無償でなければ無視 |

→ 帰結: **L1 AGGRO は薄い veneer ではなく、L0 の汎用判断を「速度優先」に振り直す実質的な特化層**。
一方 **CONTROL / COMBO は L0 の「意味があれば殴る」を大きく上書き**（壁・妨害・組み立て優先）。
L0 は中庸な床、4アーキタイプはいずれもそこからの逸脱として定義される（AGGRO も例外でない）。

### 2.1 MAIN 判断順（普遍の優先ラダー、上から最初に該当したものを実行）
1. **強制セットアップ**（開幕）: 合法な基本をバトル場へ / 基本をベンチ展開。
2. **リーサル攻撃**: 相手バトル場を**きぜつさせる**攻撃が可能で自滅しないなら実行（サイドが勝利条件）。
3. **有益な無償特性**: 手札が少ない/盤面未発達のとき、ドロー・加速の無償特性を使う（「カード/エネ増」は一般に善。L0 は明白に有益なものだけ＝手札≤K のドロー等）。
4. **進化**: より上の段階へ進化（ほぼ常に HP↑・攻撃↑）。
5. **ベンチ展開**: 安全枚数（例 ≥2–3）までベンチに基本を並べる（KO で即負けを防ぐ）。
6. **ドロー/サーチサポート**: 手札が少ないとき、ドロー/サーチのサポートを1枚（サポートは1ターン1枚）。効果テキストで汎用検出（役割不要）。
7. **エネ手貼り**: そのターンのエネを、使える攻撃に近づくなら**バトル場**へ。無ければ盤面で最も攻撃力の高いポケモンの最安攻撃コストへ。
8. **攻撃**: 最も「有用」な攻撃（一定閾値以上のダメージ、なければ手持ちの最大）。ターン終了。
9. **ターン終了**。

### 2.2 逃げルール（普遍・ping-pong 安全）★今回の発見を一般化
逃げは既定ラダーに**入れない**。L0 が逃げるのは以下を**すべて**満たすときのみ:
- activeが今ターン攻撃不能、**かつ**エネを与えても次ターンも攻撃不能（＝タイプ違い等の非アタッカー）、**かつ**
- ベンチに「厳密により良い active」（今攻撃できる/本物のアタッカー）がいる、**かつ**
- その逃げが**装填済みアタッカーのエネを捨てない**。

→ **「まだ攻撃できないだけの主砲」を逃がさない**。mega_zygarde/marnie で見た ping-pong を構造的に排除。

### 2.3 サブ選択（普遍デフォルト）
- ダメージ対象 → 相手バトル場、または（スプレッドは）KO に最も近い相手（残HP最小）。
- 呼び出し(gust) → 今ターン KO できるベンチ、無ければ最高価値。
- サーチ/手札加入 → 有用度順（進化・アタッカー > 埋め草）。盤面エネ添付先は自軍最良アタッカーへ。
- トラッシュ/山戻し → 有用度の低い順。
- 相手エネ破壊 → 最も装填/脅威の高い相手から。
- KO 後の昇格 → 最高HP/最良アタッカー（L0 は wall 概念を持たない）。

### 2.4 普遍ガードレール
- 回避可能な自山切れ（deckout）を招く手を打たない。
- リーサル/追加KOがあるのに攻撃未使用でターンを終えない。
- 1ターン1回制限（サポート・エネ・逃げ・特性）を尊重。

---

## 3. L1 — アーキタイプ方策 ×4

各アーキタイプは、プロファイルの役割を引数に **L0 ラダーへの上書き集合**を提供。意見がなければ L0 に委譲。

### 3.1 共有機構
プロファイルから解決: primary/backup attacker, engine, draw/search, disruption, wall, combo pieces。
役割対応ヘルパ: `best_attack_target` / `attacker_to_load` / `should_attack_now` /
`development_priority` / `retreat_plan`。

### 3.2 AGGRO（速攻: サイドを取り切る、テンポ > カード）
L0 の**汎用床を「速度優先」に振り直す実質的な特化**（薄い veneer ではない。§2 の対比表参照）。
L0 が中庸に「健全な盤面を作って意味があれば殴る」のに対し、AGGRO は展開を削り攻撃を前倒しする。
- 展開: L0 の健全ベンチを**切り詰め**、アタッカーライン + ベンチ1–2 のみ。過剰サーチしない。
- エネ: **最速アタッカー**（最安コストで殴り出せる）へ。T1–2 から攻撃開始（L0 より前倒し）。
- 攻撃: 毎ターン。KO/最大サイド圧を優先。gust を**サイドレース加速**に積極使用（L0 は KO 価値がある時のみ）。
- 逃げ: ほぼしない（テンポ損）。KO 後に装填済みアタッカーを昇格するときだけ。
- 妨害/回収: 無償でなければ無視。
- 見る役割: attacker/accelerator/draw/search/energy の5つ（§8.8）。

### 3.3 MIDRANGE（中庸: 強い盤面を作ってから効率よく攻撃）
最多クラス（34/52）。一律では扱えないため**操縦挙動で4サブ型に細分**する（2026-07-11 確定）。
`archetype="midrange"` + `subtype`（tuning.json）。※現行 `_kanga_focus` は **beatdown** の種。

| subtype | 勝ち筋の機序 | **操縦ノブ（他サブ型と異なる点）** | 主なデッキ |
|---|---|---|---|
| **beatdown** | 低コスト主砲＋backup で効率的にサイド交換 | primary を **2キャップ**し backup を予備装填。**過剰装填しない**（gust 全損回避＝Grimmsnarl 2キャップと同型） | cynthia_garchomp, garchomp_lucario, mega_feraligatr, crustle, ceruledge, ns_zoroark, hydreigon, mega_gengar |
| **ramp** | 加速して**単一の大型/オーバーコスト技**を撃つ | 加速最優先で**1体に集中 overload**（beatdown と真逆）。1発大技で KO | mega_venusaur, mega_zygarde, ethan_hooh, mega_gardevoir, manectric, iono_bellibolt, deck, volcanion_box, archaludon, okidogi_box |
| **spread** | 攻撃＋ダメカン移動特性で盤面全体に散らし複数KO | Munkidori/Dusknoir/Froslass を**毎ターン起動**、残HP最小を狙い**同時KO**を組む。gust で露出 | dragapult, dragapult_dusknoir, dragapult_blaziken, mega_starmie, mega_froslass, mega_diancie, marnie_grimmsnarl, black_kyurem, omatsuri, mega_absol |
| **toolbox** | 多数の入替可能な ex を相手に合わせ選択 | **対面でアタッカーを選ぶ**（固定 primary なし）、柔軟なエネ配分 | ogerpon_box, lillies_clefairy, mega_dragonite, flareon, mega_latias, raging_bolt |

共通（全サブ型）: 有用な攻撃が立ったら攻撃、Boss/gust で KO を組む、逃げは pivot 目的のみ（ping-pong しない）。
**ramp を4分類の外へ出すのではなく midrange のサブ型に据える**——「overload(ramp) vs 2キャップ(beatdown)」が
真逆の操縦になるため、この2つを分けるのが細分の最大の目的。境界例（ns_zoroark: beatdown↔ramp /
raging_bolt: toolbox↔ramp / mega_gengar: beatdown↔spread）は tuning で上書き可、A/B で精緻化。
実装: `MidrangePolicy` を基底に、subtype ごとに `decide_energy_target`（2キャップ vs overload）・
`decide_target`（spread は残HP最小・多点）・attacker 選択（toolbox は対面依存）を差し替え。

### 3.4 CONTROL（制圧: 妨害・停滞・グラインドで消耗戦に勝つ、サイドは二次的）
- 展開: **wall と disruption** を優先。ロックを確立。
- エネ: wall/ユーティリティ/妨害を先に。アタッカーへは最小限。
- トレーナー: **妨害**（手札・エネ破壊）、gust で孤立、recovery で延命を優先。自山切れ回避。
- 攻撃: サイドレースでなく chip/妨害/ユーティリティのため。wall を前に維持。
- 逃げ/pivot: 相手の現アタッカーに応じた**適切な wall へ pivot**（弱点対応）。aggro と違い**正当かつ頻繁**な pivot。
- deckout 管理: 自分の山を管理し、相手を資源枯渇させる。

### 3.5 COMBO（コンボ: 特定エンジンを組み立て、ペイオフを実行）
- 展開: **combo pieces** を最優先（search/draw で完成させる）。組み立て中に「productive な手が無くても正しい」。
- エネ: コンボが（ほぼ）完成してからペイオフアタッカーへ。
- トレーナー: 掘り切る（search/draw）。妨害/tech はここぞのターンまで温存。
- 攻撃: 早い段階で劣化攻撃を撃たない。完成後にペイオフ（多くは特性→攻撃の連鎖、時に反復）を実行。
- **「コンボ online?」ゲート**: プロファイル定義の述語（必要ピースが場/手札に揃ったか）で
  ASSEMBLE モード → EXECUTE モードに切替。

### 3.6 アーキタイプ別サブ選択
- Aggro/Midrange: サイド最大化（KO / スプレッドは残HP最小）。
- Control: 妨害最大化（要のエネを剥ぐ / ロックを崩す相手を gust / アタッカーを機能停止させる位置にダメージ）。
- Combo: サーチ結果を欠けているコンボピースへ / ダメージをペイオフ成立方向へ。

---

## 4. メタ情報の付与方法（役割の決め方）
**推奨: ハイブリッド**
- **自動推論**でデフォルト付与: 最高ダメージの進化ポケ→primary attacker / ドロー・加速テキストの特性ポケ→engine /
  ドローテキストのグッズ・サポート→draw / サーチテキスト→search / エネ破壊・gust・手札干渉→disruption /
  高HP+ダメージ軽減→wall / トラッシュ回収→recovery / 基本・特殊エネ→energy(+型)。
- **デッキ単位の上書き**（tuning.json/プロファイル）: 推論が外す場合、および
  combo_pieces / archetype タグ（本質的に手動）を指定。

---

## 5. 移行計画（現行コードから）
**推奨: 段階的・非破壊**
1. 現 `_choose_main`/`_choose_sub` から **L0** を抽出（special-case を削ぎ、clean な最低限ルール + ping-pong安全な逃げ）。
2. 4アーキタイプ方策を実装。`_kanga_focus` を MIDRANGE/AGGRO の種にする。
3. プロファイル/役割層を追加（自動推論 + tuning 上書き）。各デッキの archetype を tuning.json でタグ付け。
4. **アーキタイプ単位で**デッキを移行。既存 bespoke 方策（marnie/alakazam/crustle_stall 等）は
   アーキタイプ層が同等以上になるまで **L2 上書き**として残す。
5. 各移行を機構的指標（初撃ターン・逃げ規律・エネ浪費・WR A/B）で検証。
   ローカル WR は弱い信号（今回 45戦はノイズ支配と判明）＝床としてのみ使い、**live スコアを真実**とする。

---

## 6. 決定事項（2026-07-11 確定）
- **役割メタの付与 = ハイブリッド**（自動推論のデフォルト + デッキ単位の手動上書き）。
- **archetype タグ = デッキ1つに単一の主分類**（ハイブリッドデッキも主たる勝ち筋で分類）。
- **移行 = 段階的・非破壊**（bespoke を L2 上書きとして併存させ、アーキ単位で移行）。
- **L0 = 汎用 develop-and-attack 床**（アグロ特化ではない。deck-agnostic で、control/combo デッキが
  単体で落ちてもそれなりに機能する中庸な床。現 `_choose_main` の複雑さは引き継がない。§2 参照）。
- **L1 AGGRO = 実質的な特化層**（L0 の薄い veneer ではなく、汎用床を「速度優先」に振り直す。§3.2）。
  4アーキタイプはいずれも L0 からの逸脱として定義（AGGRO も例外でない）。

## 7. 次アクション（実装フェーズ）
1. ~~**Stage 1**: L0 を新規実装（`agents/engine_v2.py`、現 `_engine.py` は非破壊で温存）。~~ **✅ 完了（2026-07-11）**。
   `BasePolicy` に知覚6 + 判定9 + 展開2 + `infer_roles` + `choose_main` ラダー + `choose_sub` + `make_policy`/`act` shim。
   スケール技（表示0ダメージ）はテキスト検出で nominal 化（Alakazam 等の床を救済）。
   **検証結果（全52デッキ・各12戦）**: クラッシュ0。床 WR vs random 平均 **78%**、vs 現行チューニング済みエンジン平均 **38.8%**
   （L0 は床なので現行に負けるのは想定内＝hints/bespoke の付加価値を L1/L2 で足す）。
   既知の弱点（＝L1/L2 の担当）: 加速依存の重量級コンボ（ns_zoroark 16.7%）は hint 無しの床では回り切らない。
2. ~~**Stage 2**: 4アーキ方策を `BasePolicy` サブクラスで実装~~ **✅ 実装＋A/B済（2026-07-11）**。
   **midrange を4サブ型に昇格し top-level 7分類**（aggro/beatdown/ramp/spread/toolbox/control/combo）。
   `agents/engine_v2.py` に7サブクラス、`_ARCHETYPES` 登録、`tuning.json` の `archetype` で自動ディスパッチ。
   **A/B（L1 vs 素のL0・同デッキ対戦）**: aggro **57.9%**✅ / control 52.1 / toolbox 51.4 / ramp 51.3 /
   beatdown 47.9 / spread 47.5 / combo 47.1（退行ゼロ、aggro大勝、ramp/toolbox/control勝ち越し）。
   学び: 初版で aggro（ベンチ切詰＋最速アタッカー誤配）と combo（assemble-gate starvation）が退行→修正。
   **combo の assemble-gate は generic では有害（誤推論 primary）→ L2 に委譲**（`combo_online`＋正しい roles を per-deck で）。
3. **未着手**: (a) デッキ agent wrapper を engine_v2 に配線（アーキ単位移行）。(b) bespoke（marnie/alakazam/
   crustle_stall 等）を L2 サブクラス化。(c) infer_roles の primary 誤り6件を tuning で上書き（alakazam/ns_zoroark/
   metagross/mega_froslass/mamoswine/mega_diancie）。(d) beatdown/spread の精緻化、live スコア検証。

---

## 8. OOP 具体設計（継承モデル）

3層を**クラス継承**で実装する。L0 = 基底クラス `BasePolicy`。L1 = 4アーキタイプ（`BasePolicy` を継承）。
L2 = 個別デッキ（対応するアーキタイプを継承）。**判断ロジックは「知覚（盤面把握）」と「判定（意思決定）」の
2群のメソッドに分解**し、上位クラスは必要なメソッドだけを選択的に override する（テンプレートメソッド + フックの形）。

### 8.1 クラス階層
```
BasePolicy                     # L0: 汎用の知覚 + 判定。単体で合法かつ概ね正しい手を返す（床）
 ├─ AggroPolicy                # L1: L0 の薄い上乗せ（最速アタッカー・毎ターン攻撃）
 ├─ MidrangePolicy             # L1: primary 2キャップ・backup 予備装填・pivot 計画
 │    └─ MarnieGrimmsnarl      # L2: bench→active Dark 配分・Grimmsnarl入替・Punk Up 分配
 ├─ ControlPolicy              # L1: ラダー再構成（妨害優先）・wall pivot・deckout 管理
 │    └─ CrustleStall          # L2
 └─ ComboPolicy                # L1: ASSEMBLE/EXECUTE ゲート・掘り優先
      └─ AlakazamPolicy        # L2: Powerful Hand スケール・Repelling Veil tech
```
生成は `make_policy(deck, profile)` が `profile["archetype"]`（+個別デッキ表）から適切なクラスを選ぶ。

### 8.2 知覚オブジェクト（読み取り専用・obs をデコード）
```python
class PokemonView:            # 場の1体を扱いやすい形に
    pk; card                  # 生 obs / _CARDS エントリ
    hp_now; hp_max
    energy_count; energy_by_type      # {EnergyType: n}
    ready_attacks             # 今払える [(attackId, dmg, cost)]
    best_ready_dmg            # 今出せる最大打点（Powerful Hand 等のスケールも解決）
    best_potential_dmg        # 満タン時の打点
    role                      # プロファイル由来（L0 は None）
    def can_ko(self, target_hp): ...

class SideView:               # 片側プレイヤーの盤面
    active: PokemonView|None
    bench:  list[PokemonView]
    hand_count; prizes_left; deck_count; discard; energy_in_play
    supporter_played; energy_attached; retreated     # 1ターン1回制フラグ（自分側のみ有効）

class Ctx:                    # 1回の act() 中だけ生きる判断コンテキスト
    obs; sel; mi
    me: SideView; opp: SideView
    my_setup:  SetupState     # MULLIGAN / DEVELOPING / READY / STALLED   (知覚3)
    opp_threat: ThreatState   # CAN_KO_ME_NOW / CAN_KO_ME_NEXT / LOW      (知覚4)
    ko_targets: list          # 今ターン倒せる相手 [(pk, attackId, cost, prize)]  (知覚5)
    prize:      PrizeInfo      # 残りサイド・勝ち切れるか(can_close)・姿勢(race/stall) (知覚6)
    opts                      # バケット: evolves/abilities/plays/attaches/attacks/retreat_idx/end_idx
    needs_active; is_gust; is_opp_energy   # sub-select 分岐用フラグ
    def hand_card(o); def field_pk(o)      # option → カード/場ポケ 解決
```

### 8.3 判定メソッド一覧（L0 `BasePolicy` に定義）
知覚6 + 判定9 + 展開ヘルパ2。上位クラスは必要なものだけ override する。

**知覚（盤面把握）6種** — 純関数、`Ctx` を組み立てる。副作用なし。
| # | メソッド | 返り値 | L0 の既定 |
|---|---|---|---|
| 1 | `assess_self(obs) -> SideView` | 自分の HP・盤面 | active/bench を PokemonView 化、各アタッカーの ready/potential 打点、1ターン制フラグ |
| 2 | `assess_opponent(obs) -> SideView` | 相手の HP・盤面 | 同上（cardId マスク時も HP・エネ数は可視、role は不明） |
| 3 | `assess_self_setup(me,opp) -> SetupState` | 自分のセットアップ判定 | active 無=MULLIGAN／ready アタッカー無 or ベンチ薄=DEVELOPING／装填済み+ベンチ充足=READY／active が当分攻撃不能=STALLED |
| 4 | `assess_opponent_setup(opp,me) -> ThreatState` | 相手のセットアップ判定 | 相手 active の potential 打点 vs 自分 active HP → 次ターン KO 可否・装填済み本数 |
| 5 | `assess_ko_targets(ctx) -> [KoOpt]` | **倒せる相手ポケモンの判定** | 今ターン KO 可能な相手 active/bench の集合＝ [(pokemon, attackId, cost, prize_value)]。攻撃・gust・スプレッドの共通土台 |
| 6 | `assess_prize_race(me,opp) -> PrizeInfo` | **倒すべき残りポケモン数** | 自分の残りサイド、相手ポケの prize 期待値(ex=2/mega=3)、今ターン/次ターンで勝ち切れるか、**先行/劣勢の姿勢**（race か stall か） |

**判定（意思決定）9種** — option index 群 or `None`（下位ステップ/L0へ委譲）。
| # | メソッド | 役割 | L0 の既定 |
|---|---|---|---|
| 7 | `decide_trainer(ctx) -> idxs?` | サポート・グッズ使用判定 | ドロー（手札≤K）／KO を成立させる gust／サーチ・Rare Candy・加速。効果テキストで汎用検出（役割不要） |
| 8 | `decide_energy_target(ctx) -> idxs?` | エネルギー付与先判定 | 最良アタッカーを最安攻撃コストへ近づける先。装填済み主砲へは過剰装填しない床ルール |
| 9 | `decide_active(ctx, mode) -> idxs?` | バトル場に出すポケモン判定 | mode=`setup`（開幕）/`promote`（KO後昇格）。最高打点アタッカー。※逃げは #11 に分離 |
| 10 | `decide_attack(ctx) -> idxs?` | 使用する技の判定 | `assess_ko_targets`/`assess_prize_race` を使いリーサル優先、無ければ閾値以上の最大打点。撃てないなら None |
| 11 | `decide_retreat(ctx) -> idxs?` | **逃げ判定** | 既定は §2.2 ping-pong 安全規則（装填中の主砲を逃がさない）。AGGRO=ほぼ None／CONTROL=wall へ正当な pivot |
| 12 | `decide_ability(ctx) -> idxs?` | 特性使用判定 | 明白に有益な無償特性（ドロー・加速・ダメージ配置）を起動。marnie の Adrena-Brain/Punk Up 等の能動特性の起動点。CONTROL は温存判断で override |
| 13 | `decide_target(ctx, kind) -> idxs?` | 対象選択の統合 | kind=`attack`/`spread`/`gust`/`effect`/`energy_strip`。`assess_ko_targets`＋prize 価値を核に、KO 可能→高価値の順。スプレッドは残HP最小へ寄せる |
| 14 | `decide_acquire(ctx) -> idxs?` | 取得先の判定 | サーチ/ドロー/回収（Night Stretcher 等）で進化・アタッカー優先の順 |
| 15 | `decide_discard(ctx) -> idxs?` | トラッシュ/山戻しの判定 | 有用度が低い順 |

**展開ヘルパ**（ラダーが呼ぶ）: `decide_evolve`（高打点フォームへ）／`decide_bench`（安全枚数まで基本展開）。
**L0 に置かない**: `combo_online(ctx)` 述語は L1 `ComboPolicy` 専用（ASSEMBLE→EXECUTE ゲート）。

### 8.4 テンプレートメソッド（`act` の骨格）
```python
class BasePolicy:
    def __init__(self, deck, profile=None):
        self.deck, self.profile = deck, profile or {}

    def act(self, obs, sel):
        ctx = self._perceive(obs, sel)
        return self.choose_main(ctx) if sel.context == MAIN else self.choose_sub(ctx)

    def _perceive(self, obs, sel):
        me, opp = self.assess_self(obs), self.assess_opponent(obs)
        return Ctx(obs, sel, me, opp,
                   self.assess_self_setup(me, opp),
                   self.assess_opponent_setup(opp, me),
                   self._bucket(sel))

    # MAIN = 優先ラダー（§2.1）。各 step は上の判定メソッドを薄く包む
    def choose_main(self, ctx):
        for step in self.main_ladder():
            r = step(ctx)
            if r is not None:
                return r
        return self._fallback(ctx)

    def main_ladder(self):
        return [self.step_forced_setup, self.step_lethal, self.step_ability,
                self.step_evolve, self.step_bench, self.step_trainer,
                self.step_attach, self.step_attack, self.step_retreat, self.step_end]

    # step_* は各判定メソッドへのブリッジ。例:
    def step_attach(self, ctx):  return self.decide_energy_target(ctx)   # 判定8
    def step_attack(self, ctx):  return self.decide_attack(ctx)          # 判定10
    def step_trainer(self, ctx): return self.decide_trainer(ctx)         # 判定7
    def step_ability(self, ctx): return self.decide_ability(ctx)         # 判定12
    def step_retreat(self, ctx): return self.decide_retreat(ctx)         # 判定11（ping-pong安全）
    def step_lethal(self, ctx):                                          # 知覚5+6 → 判定10
        return self.decide_attack(ctx) if ctx.ko_targets and ctx.prize.can_close else None
    def step_forced_setup(self, ctx):
        return self.decide_active(ctx, mode="setup") if ctx.needs_active else None
    def step_bench(self, ctx):   return self.decide_bench(ctx)           # 展開
    def step_evolve(self, ctx):  return self.decide_evolve(ctx)          # 展開
```
**サブ選択も同じ判定を再利用**して MAIN/sub の二重実装を解消:
```python
    def choose_sub(self, ctx):
        c = ctx.sel.context
        if c in (SETUP_ACTIVE, TO_ACTIVE): return self.decide_active(ctx, mode="setup")    # 判定9
        if c == SWITCH and ctx.is_gust:    return self.decide_target(ctx, "gust")           # 判定13（assess_ko_targets）
        if c in (ATTACH_FROM, ATTACH_TO):  return self.decide_energy_target(ctx)            # 判定8（Punk Up 等）
        if c in DAMAGE_CTXS:               return self.decide_target(ctx, "spread")         # 判定13（残HP最小へ）
        if c == EFFECT_TARGET:             return self.decide_target(ctx, "effect")         # 判定13
        if c == DISCARD_ENERGY and ctx.is_opp_energy: return self.decide_target(ctx, "energy_strip")
        if c in ACQUIRE_CTXS:              return self.decide_acquire(ctx)                  # 判定14（search/回収）
        if c in DISCARD_CTXS:              return self.decide_discard(ctx)                  # 判定15
        return self._sub_default(ctx)
```

### 8.5 上位クラスの override 指針（何を差し替えるか）
- **AggroPolicy**: `decide_energy_target`（最速＝最安アタッカー優先）だけ薄く override。ラダーは L0 のまま。
- **MidrangePolicy**: `decide_energy_target`（primary を cost で 2キャップ→backup へ）＋`decide_retreat`（役割 pivot、装填済みを ping-pong しない）。※現 `_kanga_focus` の一般化。
- **ControlPolicy**: `main_ladder` を再構成（step_trainer の妨害を step_attack より前に）＋`decide_retreat`（相手アタッカーに応じた wall pivot、逃げは正当・頻繁）＋`decide_ability`（温存判断）＋deckout ガード強化。
- **ComboPolicy**: `main_ladder` に `step_assemble_gate` を挿入し、`combo_online(ctx)` 述語が False の間は掘り（search/draw）を優先し劣化攻撃を撃たない。True で EXECUTE（判定8）へ。
- **L2 個別**: 対応アーキを継承し、必要な判定メソッド1–2個と（必要なら）特定 sub-select だけ override。

### 8.6 既存関数エンジンとの橋渡し（非破壊移行）
- 新 OOP は別モジュール `agents/engine_v2.py` に実装。既存 `agents/_engine.py:act(...)` はそのまま残す。
- デッキラッパは opt-in で切替: `EngineV2 = make_policy(DECK, PROFILE); return EngineV2.act(obs, sel)`。
- 互換 shim: `engine_v2.act_compat(obs_dict, deck, profile)` を用意し、`to_observation_class` 変換・例外フォールバック（`_mk` 相当）・deck-selection フェーズ処理を L0 に内包。挙動が現行以上を確認したデッキから順に差し替え。

### 8.7 カード役割の自動判定（L0 `infer_roles`）
役割推論は**純粋なカードDB関数**なので基底クラス `BasePolicy` に置く（L0 が提供、L1 が消費）。
`profile["roles"]` 未指定の cardId をこれで埋め、指定済みは手動値を優先（§4 ハイブリッド）。
```python
def infer_roles(self, deck) -> dict:      # {cardId: {"role":..., ...}}
    roles = {}
    lines = self._evolution_lines(deck)   # name チェーンから進化ラインを構築
    best_potential = self._best_potential_by_line(deck)  # ライン毎の満タン打点
    for cid in set(deck):
        c = _CARDS[cid]
        roles[cid] = self._infer_one(cid, c, deck, lines, best_potential)
    roles.update(self.profile.get("roles", {}))   # 手動上書きが勝つ
    return roles
```
`_infer_one` の判定規則（既存の keyword 集合 `_DRAW_SUPPORTERS`/`_SEARCH_ITEMS`/`_GUST`/`_ENERGY_ACCEL`/
`_SWITCH_CARDS` とカード事実を利用。上から最初に一致したもの）:

| 判定 | 条件（カード事実 / テキスト / 名前） | 付与 role |
|---|---|---|
| energy | cardType ∈ {BASIC_ENERGY, SPECIAL_ENERGY} | `energy` (+etype=energyType) |
| draw | supporter/item 名 ∈ `_DRAW_SUPPORTERS`、または特性/効果テキスト ~ "draw" | `draw` |
| search | item 名 ∈ `_SEARCH_ITEMS`、またはテキスト ~ "search your deck" | `search` |
| accelerator | 名 ∈ `_ENERGY_ACCEL`、または特性/技テキスト ~ "attach … Energy from"（山/トラッシュ→場） | `accelerator` |
| evolution accel | 名 ~ "Rare Candy" | `accelerator` (subrole=`evolution`) |
| disruption | 名 ∈ `_GUST`→(subrole=gust) / テキスト ~ 相手の手札破壊→(hand) / 相手エネ破壊→(energy) / 特殊状態ロック | `disruption` (+subrole) |
| pivot | 名 ∈ `_SWITCH_CARDS`、または自分の active を入替える効果 | `pivot` |
| recovery | 名 ~ Night Stretcher/Energy Recycler/Super Rod、またはテキスト ~ "from your discard pile … to your hand/deck" | `recovery` |
| wall | POKEMON かつ hp≥高閾値 かつ 特性/技テキスト ~ ダメージ軽減・無効・防御 | `wall` |
| engine | POKEMON かつ 有益特性（draw/accel/search/ダメージ配置）を持ち、自身が primary attacker でない | `engine` (+subrole) |
| attacker(primary) | POKEMON かつ 自ラインの満タン打点が最大 or ex/megaEx（`main_attackers` hint 優先） | `attacker` (tier=primary, energy_need=最安打点コスト) |
| attacker(backup) | POKEMON かつ ダメージ技を持つ（primary 以外） | `attacker` (tier=backup) |
| evolution_piece | POKEMON かつ deck 内の attacker ラインの下位進化 かつ 自身の最大打点が低い | `evolution_piece` |
| tech | 上記に当てはまらない 1〜2枚差しトレーナー | `tech` |

**L0 自身が使うのは最小限**（§2 の床＝カード事実主体）: primary/backup attacker と energy と evolution_piece
だけ参照（エネ集中先・進化の種）。draw/search/gust 等は L0 では**テキスト検出**で足り、
役割を本格的に使うのは L1（アーキタイプが計画に沿って役割別に動く）。

### 8.8 アグロデッキにおける役割リスト
アグロ（速攻・テンポ優先）で意味を持つ役割と、L0/AGGRO での扱い。**太字＝アグロの中核**。

| 役割 | アグロでの意味 | 自動判定の主な手掛かり | L0/AGGRO の使い方 |
|---|---|---|---|
| **attacker(primary)** | 勝ち筋の主砲。T1–2 から殴り出す最速のフィニッシャー | 自ライン満タン打点最大 / ex・megaEx | エネを集中、最優先で active・毎ターン攻撃 |
| **attacker(backup)** | primary が倒された後の後続。切れ目なく殴り続ける | ダメージ技を持つ他ポケ | primary が cost 到達後に予備装填、KO 後に昇格 |
| **accelerator** | エネ加速。攻撃開始を1ターン早める（アグロの生命線） | `_ENERGY_ACCEL` / "attach … Energy from" | 攻撃が未成立の間、能動的にプレイ |
| **draw** | 手札を切らさずプレッシャー継続 | `_DRAW_SUPPORTERS` / "draw" | 手札薄いとき（≤K）に1枚。テンポを崩さない範囲 |
| **search** | 主砲・エネ・ボールを最速で引き込む | `_SEARCH_ITEMS` / "search your deck" | 序盤に attacker ライン/加速を探す |
| **energy** | 攻撃コストそのもの（型一致） | BASIC/SPECIAL ENERGY | 毎ターン手貼りを attacker へ |
| **disruption(gust)** | Boss で相手ベンチのサイド要員を引き摺り出し KO＝サイド加速 | `_GUST` | リーサル/価値の高い KO を組めるときだけ |
| pivot | 逃げエネの節約／攻撃できる active を前に出す入替補助 | `_SWITCH_CARDS` | 稀。攻撃続行のための入替のみ（逃げは基本しない） |
| evolution_piece | 進化アタッカーを使うアグロでの種（純アグロは基本無し。Rare Candy で飛ばす） | attacker ラインの下位進化 | 進化の踏み台。エネは載せない |
| tech | 状況対応の1枚差し（対面メタ等） | 上記外の少数トレーナー | 無償でなければ後回し |

**アグロで比重が下がる/使わない役割**: `wall`・`recovery`・`engine`（重い特性セットアップ）・
`disruption(hand/energy ロック)` は速度を犠牲にするためアグロの計画からは外れる（これらが中核化するのは
CONTROL/COMBO）。→ ゆえに **AGGRO 特化は attacker/accelerator/draw/search/energy の5役割だけを見て**、
L0 の中庸な汎用床を「速度優先」に振り直す（展開切り詰め・攻撃前倒し・積極 gust）。§3.2。

### 8.9 例: marnie を L2 として圧縮（現 bespoke → 継承 override）
現 `policies.py` の marnie 特殊ロジックは、`MidrangePolicy` を継承した `MarnieGrimmsnarl` の
**override 3点**に収まる（プロファイルで Munkidori/Froslass/Snorunt=wall, Grimmsnarl=primary(need2), Impidimp/Morgrem=evolution_piece と宣言）:
1. `decide_energy_target`: Grimmsnarl 未完なら Dark を **bench Munkidori→active Munkidori** の順、完成後は Grimmsnarl 2キャップ→Marnie's ライン分配。
2. `decide_retreat`: 装填済み Grimmsnarl がベンチにいる時だけ入替（それ以外は None＝逃げない）。§2.2 と同義なので基底で概ね賄え、marnie は「Grimmsnarl 入替」条件のみ足す。
3. `decide_ability`（Adrena-Brain の起動）＋`decide_target`（付け替え先＝残HP最小の相手）＋`decide_energy_target` の `ATTACH_FROM`（Punk Up 分配）を override。
現在の `full_control` / `suppress_engine_retreat` フラグは、OOP では「メソッドを override したか否か」に自然に吸収される（フラグ不要）。

---

## 9. デッキ4分類の作業方法論（2026-07-11 確定）

52デッキを archetype（aggro / midrange / control / combo）にタグ付けするための手順。最終タグは
`agents/tuning.json` の per-deck `archetype`（ハイブリッド＝自動＋手動）へ、作業記録は
`docs/deck_archetypes.md` に置く。

**重要な前提（試走で判明）**: 内部の役割カウントだけの自動分類は破綻する。Boss/Iono/gust・Night Stretcher 等の
**妨害/回収トレーナーは全アーキタイプが積む**ため、シェアを数えても control を分離できない（試走で 38/52 が
control に誤集約）。⇒ 分類の主軸は「**各デッキの勝ち筋の読解（primaryが何をするか・ロック/組み立ての有無）＋
戦法テキスト**」。内部シグナル（primaryの`energy_need`・スケール技か・進化段・加速依存度・最速攻撃コスト）は
**判断の入力**であって判断そのものではない。

### 9.1 手順
1. **戦法テキストの記録（全52デッキ・Web検索）**: 各デッキの勝ち筋・キーエンジン/コンボ/ロック・速度を
   テキスト化。**我々の実際の60枚に紐付ける**（代用構築・post-rotation Standard のため一般記事と差異あり）。
   カード要約は `deck_summary.json`（primary/energy_need/scaling/エネ型/mega_ex/全カード名）を土台に使う。
2. **4分類の決定（デッキ内容＋テキスト）**: §9.2 ルーブリックで aggro/midrange/control/combo を判定。
   **決めがたいものは OTHERS バケットへ**（COMBO には落とさない＝assemble-gate の誤爆を避ける）。
3. **OTHERS の精査**: OTHERS（＋COMBO）を再検討し、必要なら**追加カテゴリを提案**。

### 9.2 分類ルーブリック（勝ち筋で判断・内部特徴は証拠）
- **AGGRO**: 安いアタッカー（多くは基本/1進化、攻撃コスト≤2、セットアップ最小）でサイドを取り切る速攻。
  多カードのコンボ/ロックに依存せず T1–2 から殴る。
- **MIDRANGE**: 進化/Mega-ex を立て 2アタッカー体制で効率よく殴る。中程度のセットアップ、Rare Candy/加速で
  primary＋backup を装填。速攻でもロックでも組み立て依存でもない「標準的ビートダウン」。
- **CONTROL**: 妨害/停滞/ロック/デッキアウトで勝つ（サイドは二次的）。壁・手札/エネ破壊が**計画の中核**
  （単なる差し込みではない）、回収でグラインド。
- **COMBO**: 特定の多パーツエンジンを**組み立ててからペイオフ**（コスト超過アタッカーへの加速連鎖・特性ループ・
  組み上がった盤面に依存するスケールアタッカー）。online まで非生産的ターンが正しい。
- **OTHERS**: どれにも綺麗に収まらない/2分類で曖昧。step3 レビュー用の保留枠。

### 9.3 デッキ調査の出力スキーマ（1デッキ1レコード）
```
deck / primary_attacker / win_condition(1–2文) / key_pieces(engine/combo/lock) /
speed(fast|mid|slow) / archetype(aggro|midrange|control|combo|others) /
confidence(high|med|low) / evidence(web|card|internal) / notes
```

---

## 10. archetype 別 役割語彙（2026-07-11 確定・実装前ロック）

**原則**: 役割は archetype 相対である。同じカードでも archetype によって果たす役割が違う
（例: Munkidori は spread では中核 `mover`、beatdown では補助 `finisher-enabler`）。単一 generic taxonomy
（§1.3/§8.7）を全 archetype で共有する現状が弱点。infer_roles の primary 誤り6件もこれが原因。
→ 各 archetype 方策が**自分の語彙で役割を再解釈**する。

### 10.1 機構（AUGMENT 方式・L0 非破壊）
- `BasePolicy.infer_roles`（§8.7）は **generic ベース**のまま（`role` = attacker/engine/… を全カードに付与）。
- 各サブクラスは `infer_roles` を **override** し、`super().infer_roles(deck)` の結果に
  **archetype 固有タグを AUGMENT**（generic `role` は残す＝L0 フォールバック判定が壊れない）。
  ```python
  class SpreadPolicy(BasePolicy):
      def infer_roles(self, deck):
          roles = super().infer_roles(deck)          # generic base（role= は保持）
          for cid, r in roles.items():
              if self._is_counter_mover(cid):         # archetype 固有検出
                  r["spread_role"] = "mover"           # AUGMENT（消費は decide_ability/decide_target）
          return roles                                 # profile 手動上書きは super 内で既に適用済（ハイブリッド）
  ```
- 各 archetype の decide_* は**自分の固有タグ**を読む。generic しか無ければ L0 挙動にフォールバック。

### 10.2 7 archetype の役割語彙（固有タグ / 検出 / 消費する判定）
凡例: 検出は「generic role ＋ カード事実/テキスト」から。消費は L1 の decide_* メソッド。

**aggro** — 速度に必要な最小語彙
| 固有タグ | 検出 | 消費 |
|---|---|---|
| `fast_attacker` | role=attacker かつ cheapest_cost≤2（同点は dmg/コスト最大） | decide_energy_target（最速へ集中） |
| `accel`/`draw` | generic 継承 | decide_trainer |

**beatdown** — primary を 2キャップ＋backup
| `primary` (+`energy_cap`) | 攻撃を持つ非engineの中で potential_dmg 最大の**進化/ex ライン**（Dunsparce 等の純engineは除外＝primary誤り修正） | decide_energy_target（cap まで） |
| `backup` | 他のダメージ技持ち attacker | decide_energy_target（cap後に装填） |
| `pivot`/`gust` | generic 継承 | retreat / trainer |

**ramp** — 単一 payoff を overload
| `payoff` (+`overload=True`) | attacker で cheapest_cost≥3 か scaling、potential_dmg 最大の単一体 | decide_energy_target（cap無しで集中） |
| `accel` (+`proactive=True`) | role=accelerator、または ability に "attach … Energy from"（山/トラッシュ→場） | decide_trainer（攻撃可能でも先に） |
| `fuel` | payoff/accel を支える energy・recovery | trainer/acquire |

**spread** — ダメカン移動で複数KO
| `spreader` | 攻撃テキストに "to … Benched"/ばらまき | decide_attack（優先） |
| `mover` | ability に "damage counter" ＋ ("move"/"put")（Munkidori/Dusknoir/Froslass） | decide_ability（毎ターン起動）/ decide_target（KO成立へ寄せ） |
| `finisher`/`gust` | 高打点 attacker / generic gust | trainer（KO を組む） |

**toolbox** — 対面でアタッカー選択（固定 primary なし）
| `attacker_pool[etype,cost,prize]` | 全 attacker を pool 化（各に type/コスト/サイド枚数） | decide_active/energy（弱点対面で選択） |
| `selector` | attacker をサーチ/選ぶ engine（Noctowl 等）・search | decide_trainer/acquire |
| `flex_energy` | Prism/Rainbow/多色加速(Crispin) | decide_energy_target（型を跨ぐ配分） |

**control** — 妨害/停滞/グラインド
| `wall` | 高HP＋ダメージ軽減/無効（generic wall） | decide_retreat（対面に応じ pivot） |
| `lock` | テキスト "can't"/"prevent"/"lock"（Neutralization Zone 等） | decide_disrupt（確立を優先） |
| `denial` | エネ/手札破壊（Crushing/Enhanced Hammer, Xerosic, Judge, Iono） | decide_disrupt（攻撃より先） |
| `recovery`/`chip` | generic recovery / 低打点の遅い win-con attacker | acquire / decide_attack(閾値↑) |

**combo** — 組み立て→ペイオフ（主に L2）
| `piece` | 組み立てる engine（Slowking+Codebreaking, Genesect+Hilda, Metagross engine） | (L2) combo_online |
| `enabler` | 完成させる search/draw | decide_trainer（掘り） |
| `payoff` | 実行アタッカー | (L2) decide_attack gate |
| `online述語` | piece 在場 ＋ payoff 装填（**per-deck・profile 定義**） | (L2) combo_online |
※ combo の piece/payoff は generic 検出が困難 → **profile 手動指定を主**とする（combo はどのみち L2）。

### 10.3 profile 拡張（ハイブリッド上書き）
`tuning.json` の per-deck エントリで固有タグを手動上書きできる（自動検出が外す場合）:
```json
"alakazam":   {"archetype":"combo",  "roles":{"743":{"role":"attacker","payoff":true}}},
"ns_zoroark": {"archetype":"beatdown","roles":{"906":{"role":"attacker","tier":"primary"}}}
```
→ infer_roles 誤り6件（alakazam/ns_zoroark/metagross/mega_froslass/mamoswine/mega_diancie）はここで是正。

### 10.4 実装順（スコープ: 高価値の4つから）
1. **beatdown**（primary/backup 精密化＝primary誤り修正の主効果）→ **ramp**（payoff/accel）→
   **spread**（mover/spreader）→ **toolbox**（attacker_pool/matchup）。各実装後に L1-vs-L0 A/B。
2. aggro は現状で勝ち越し済み＝`fast_attacker` は軽微改良。control/combo は L2 と併せて後回し。

### 10.5 実装結果（2026-07-11 ✅）
- **機構**: 各サブクラス `infer_roles` override＝`super()`＋固有タグ AUGMENT。`_apply_profile_roles`
  （JSON文字列キー→int正規化・マージ）、`_manual_primary`＋`_enforce_primary`（宣言 primary を単一化、
  0ダメージ copier も含め他アタッカーを backup へ降格）を BasePolicy に追加。
- **beatdown**: primary を「純engine除外の最大打点アタッカー」で再選出（Dunsparce型 primary 誤りを構造的に排除）。
- **ramp**: payoff/overload/accel(proactive) タグ＋加速の能動プレイ。※payoff集中の energy override は
  「ベンチの payoff にも集中して今殴れる場面を逃す」で退行（45.0%）→撤去し energy は L0 に委譲（overloadと同義）。
- **spread**: mover（Munkidori/Dusknoir/Froslass＝ダメカン移動特性）を毎ターン優先起動＋KO成立へ寄せる。
- **toolbox**: attacker_pool[etype,cost]／flex_energy タグ＋弱点対面でアタッカー/エネ選択。
- **tuning 上書き6件**: alakazam(743)/ns_zoroark(293)/metagross(641)/mega_froslass(40)/mamoswine(283)/
  mega_diancie(766) の primary/payoff を是正（`roles` 上書き）。
- **A/B（L1-vs-L0）**: spread **47.5→53.0**✅（mover優先＋KO寄せ）、beatdown 47.9→49.2、toolbox 51.7、
  ramp 49.1（≒中立、能動加速が価値）。**全52デッキ クラッシュ0**。役割は L2 でも正しく参照可能に。
- **残り3分類も実装（7分類完了）**: **aggro**=`fast_attacker` タグ（メタのみ・挙動不変。速度限定の energy
  誤配は既に退行実証済のため energy は L0 据置）。**control**=`denial`/`lock`/`chip` タグ＋decide_disrupt に
  lock札の能動設置を追加（stadium lock は DB にテキスト無く未検出＝現行挙動は実質不変、profile 手動 lock 可）。
  **combo**=`payoff`/`enabler`(search+draw)/`piece`(主に profile) タグ＝**L2 の assemble-gate 用メタ**（挙動は
  L0＋掘りのまま）。→ aggro/combo は挙動不変・L2準備、control は lock 検出時のみ発火。全52クラッシュ0。
  ※ A/B は 30戦/デッキで ±9% のノイズ支配、単一runの数値差（例 aggro 57.9↔47.7）は挙動不変のノイズ。

---

## 11. L1-vs-L0 厳密検証（2026-07-11）

**受け入れ基準**: 同一デッキで L1 を両側 L0 と対戦させたとき、L1 のアーキタイプ認識により
**プレイングが安定し L1 が有意に勝ち越す**こと。ハーネス: `tools`外の `l1_verify.py`（並列・60戦/デッキ×52・n=3086）。

**結論: 基準未達 — 全体は互角（50.55%, z=+0.6, 有意でない）**。
- **control のみ一貫して有意に L1 勝ち**（57–59%, 2run とも z>2）。他は互角圏。
- **beatdown の実退行を発見・修正**: 2キャップの cap を `_cheapest_cost`（安いチップ技コスト）にしていたため
  主砲が本命技コストに届かず 42.9%（z=-3.1）。cap を `_main_cost`（最大打点技コスト）に修正 → 51.4%。
- **ノイズの重大知見**: 60戦/デッキでも run 間で **archetype 平均 ±6%・個別デッキ ±12%** 振れる
  （ogerpon 60→35 等。cgエンジン RNG が run 毎に変わり試合分散が大きい）。
  → **信頼できるのは全体集計(n≳3000)と control のみ**。個別デッキ/単一run/archetype 平均は判断材料に不可。

**なぜ互角か**: 現状 override が薄く、L0 が既に有能な床。beatdown の例のように「効くはずの override が害」も起こる。
**有意な勝ち越しに必要なこと**: (a) 各 archetype override の強化（特に aggro/ramp/spread/toolbox は現状ほぼ L0 と差が出ていない）、
(b) L2（combo/bespoke）、(c) 低分散な評価（多ゲーム or 固定シード）で小さなエッジを検出できるようにする。

### 11.1 divergence 診断 — なぜ非control L1 が勝ち越せないか（2026-07-11）
「L1 が同一局面で L0 と**違う手を選ぶ割合**」を全選択で測定（`divergence.py`）。結果:
| archetype | L0との相違率 | 解釈 |
|---|---|---|
| **control** | **15.5%** | 実際に大きく違う手→**有意に勝つ** |
| beatdown | 2.3% | attach(2キャップ)のみ |
| aggro | 2.2% | L0≒aggro なので当然 |
| toolbox | 1.0% | 対面適応はミラーでは発火しにくい |
| ramp | 0.4% | ほぼ L0 と同一 |
| spread | 0.7% | ほぼ L0 と同一 |
| **combo** | **0.0%** | L0 と**完全同一** |

**構造的結論**: L0 の「展開→殴る＋既に賢いサブ選択（残HP最小へ配置・最良アタッカーへエネ・KO可能をgust）」が、
aggro/beatdown/ramp/spread/toolbox の**操縦を既にほぼ完全実装している**。これらは**デッキ構築が違うだけで、
1手ごとの操縦は L0 と同じで正しい**。実際に別操縦が要るのは:
- **control**（レースせず妨害/停滞）＝実装済・勝つ。
- **combo**（online まで攻撃我慢）＝per-deck の online 述語が要る＝**L2**（generic gate は starvation で有害と実証）。

**帰結・方針**:
1. aggro/beatdown/ramp/spread/toolbox は **L0 が正しい pilot**。無理に divergence を作る override（ramp の
   payoff集中エネ／patience、beatdown の cheapest cap 等）は neutral〜有害。archetype タグは**メタ情報/構築**
   としては有効だが、L1 の pilot 差分は小さくてよい。
2. **toolbox の価値（対面適応）は同デッキ・ミラー A/B では原理的に測れない** → **フィールド横断（round-robin
   vs 多様な相手）で評価すべき**。ミラー互角＝toolbox が無意味の証明にはならない。
3. 勝ち越しを増やす投資先は **combo の L2** と **control 系の深化**、および**評価法をミラー→フィールド横断へ**。

### 11.2 掘り下げ — ramp/spread/toolbox のフィールド評価（2026-07-11）
「対面適応はミラー（同デッキ）では原理的に測れない」ため、多様な10デッキのパネル相手に
**L1 pilot vs L0 pilot**で勝率比較（`field_eval.py`）。
| archetype | ミラー相違 | フィールド Δ(L1−L0) | 結論 |
|---|---|---|---|
| **toolbox** | 1.0% | **+2.1%** (z+1.0) | ✅小だが実在。type-matchup は L0 が無視する唯一の軸。ミラーでは不可視だった |
| **spread** | 0.7% | −0.2% (z−0.15) | フィールドでも互角。L0 の残HP最小配置＝spread。例外: dusknoir系 mover +14% |
| **ramp** | 0.4% | （測定不要） | L0 の「最良アタッカーにエネ集中＋撃てる時に撃つ」＝ramp と数学的に同一 |

**確定した理解**: ramp/spread の共通パターンは正しいが **L0 のコア（エネ集中・残HP最小狙い・特性発火）に帰着**するため
独自 pilot 価値なし。**toolbox のみ**が L0 に無い matchup 軸を足せる（強化: 弱点突き+300/弱点回避−150、
フィールドで小幅プラス）。spread の mover 優先は damage-mover デッキ（dragapult_dusknoir 等）だけ効く＝archetype 内
不均一なので L2/個別で伸ばす領域。**pilot 上の archetype 価値は control（大）> toolbox（小・field）> 他≈L0**。

### 11.3 評価法の転換 — matchup ベース評価（2026-07-11・重要）
**ミラー（同デッキ L1 vs L0）は archetype 価値を系統的に隠す**——両側が同じ archetype 特性を持つため
相殺される（例: beatdown の「主砲ロード後 backup 装填」耐性は、相手が主砲を KO してくる時だけ効くが、
ミラーでは双方同じなので 0 になる）。**正しい評価は「その特徴が刺さる相手」との対戦**。

**ノイズ対策（必須）**: cg エンジンは battle_start にシード引数が無く、1プロセスで連続対戦すると RNG
ストリームを共有し試合が相関→実効 n が激減（同一ポリシーでも n=1280 で ±3%/z≈2 振れた）。
`Pool(maxtasksperchild=1)` ＋細かいバッチで**プロセス分離**すると ~binomial に収束（`eval_paired.py`）。

**matchup 行列**（`matchup_matrix.py`、各 test-archetype を L1/L0 で全相手 archetype と対戦・L1−L0 差）2run 結果:
| archetype | Δ run1 | Δ run2 | 判定 |
|---|---|---|---|
| **beatdown** | +7.9 | +7.7 | ✅堅牢に正（ミラーでは ~tied だった＝隠れていた価値） |
| **control** | +4.1 | +6.1 | ✅堅牢に正 |
| ramp | −2.8 | −3.2 | ⚠️堅牢に負（proactive-accel が有害・要修正） |
| toolbox | +5.1 | +1.9 | 小さく正（ノイズ有） |
| spread | +6.7 | −2.5 | ❓不安定（per-cell n 過小）・要追試 |
| aggro | −0.6 | +2.6 | ≒L0（設計通り） |
| combo | −0.4 | −1.1 | ≒L0（L2 案件） |

**帰結**: (1) beatdown の 2キャップ（main_cost 版）は**保持**（field で +7.8%、ミラーの ~tied は誤誘導だった）。
(2) 今後 L1 評価は**必ず matchup/field で**（ミラーは archetype 差の測定に不適）。理想は**実フィールド分布
（leaderboard メタ）で重み付け**。(3) ramp の proactive-accel は有害→見直し。spread/toolbox は n を増やして再測。
評価ツール: `matchup_matrix.py` / `eval_paired.py` / `field_eval.py`。

**3run 確定（ramp 修正後）**: beatdown +7.9/+7.7/+5.1（平均+6.9・堅牢✅）／control +4.1/+6.1/+7.4（+5.9✅）／
**ramp −2.8/−3.2/−0.5**（proactive-accel を**グッズ限定**にしてサポート浪費を除去→中立に回復）／toolbox +5.1/+1.9/+0.7
（~+2.5・小正）／spread +6.7/−2.5/0.0（不安定・保留）／aggro/combo ≒0。
**確定所見**: L1 archetype 層は **beatdown・control で明確に L0 を上回る**（matchup 評価でのみ可視）。ramp 修正は
「有害 override を L0 に戻すのでなく、原因（サポート浪費）を特定し独自性を残して有害部分だけ除去」した好例。
