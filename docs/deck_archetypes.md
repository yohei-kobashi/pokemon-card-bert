# デッキ4分類（作業結果 2026-07-11）

全52デッキを Web 調査＋カード内容で分類（手順・ルーブリックは `engine_v2_spec.md` §9）。

**分布**: aggro 7 / midrange 31 / control 4 / combo 10 / others 0

`★`=境界（別アーキタイプの解釈あり）。最終タグは `agents/tuning.json` の per-deck `archetype` へ（ハイブリッド）。

| arch | ★ | deck | conf | primary(実体) | 勝ち筋 | 境界メモ |
|---|---|---|---|---|---|---|
| aggro |  | hop_zacian | high | Hop's Zacian ex | Hop's Choice Band(コスト-1&+30)で0-1エネ即起動Insta-Strikeを1-2T目から、有利プライズトレードでレ | low-cost fast metal aggro; hand-disrupt is tempo not wincon |
| aggro | ★ | mega_heracross | med | Mega Heracross ex | Mega Heracross ex x4単体を最速で立て2エネJuggernaut Hornを毎ターン、テンポで押す | no evolution/combo, single-attacker race; could be midrange (counter) |
| aggro |  | mega_lopunny | high | Mega Lopunny ex | 1エネRapid Smashers(最大180)を連打しDudunsparce/Fezandipitiでドロー継続する速攻 | pure race |
| aggro |  | mega_lucario | med | Mega Lucario ex | Mega Lucario exを4-4で連打、加点/加速で序盤から圧 | boost package -> midrange-ish but single-attacker race core |
| aggro |  | staryu | high | Mega Starmie ex | 1エネJetting Blow(120+ベンチ50)で毎ターン展開しサイド走る、Poffin/Salvatoreで即起動 | 1-energy minimal-setup race; Cinderace backup not combo |
| aggro | ★ | trevenant_control | med | Hop's Trevenant | 安いStage1 Hop's Trevenantで殴り、Hop'sポケ気絶で+100、Choice Band/Postwick/Snorla | FILENAME MISLEADING: Hop's Trevenant swarm/aggro, no lock. Could be midrange (2-attacker) |
| aggro |  | zangoose | high | Zangoose ex | 1エネBasic Zangoose exを即攻撃、Dunsparce展開と厚い加速/ドローで継戦し序盤からサイドレース | single-energy all-basic, textbook fast aggro |
| midrange |  | archaludon | high | Archaludon ex | Archaludon exに進化しAssemble Alloyで鋼加速、3エネMetal Defender 220で殴る、Cinderace | built-in accel, standard evolve-beatdown |
| midrange |  | black_kyurem | med | Black Kyurem ex | 水手張りしBlizzard Burst(130+サイド数拡散)で削りサイド取り切る | single Basic ex; slight aggro lean |
| midrange |  | crustle | med | Mega Kangaskhan ex | Jumbo Ice CreamでMega Kangaskhan exを立てDouble-Punchingでサイド進める、Crustle壁をe | Kangaskhan beatdown + Crustle wall tech (control angle) |
| midrange |  | cynthia_garchomp | high | Cynthia's Garchomp ex | Gabiteサーチで一気にGarchompを立て+30補正で高打点ビート | stage2 beatdown |
| midrange | ★ | deck | med | Mega Abomasnow ex | Snover→Mega Abomasnow exで2T目から大型ビート、Kyogreでエネ再供給 | borders combo (35-energy consistency engine) |
| midrange |  | dragapult | high | Dragapult ex | Dragapult exを立てPhantom Diveでスプレッド、Boss/Munkidoriで取り切る | classic spread midrange |
| midrange |  | dragapult_blaziken | high | Dragapult ex + Blaziken ex | Blaziken exで炎エネ加速しDragapult exを高速起動、二枚看板で押す | accel a normal-cost attacker, not over-cost combo |
| midrange |  | dragapult_dusknoir | high | Dragapult ex | Phantom Diveで撒きDusknoirのCursed Blastでダメカン追加移動し複数KO | spread tempo; disruption tech but core=beatdown |
| midrange |  | ethan_hooh | med | Ethan's Magcargo/Ho-Oh ex | Ho-Oh exのGolden Flameで炎加速しMagcargo Lava Burst(エネ5トラッシュ350) | accel->big attack, combo-ish but Ho-Oh self-sufficient |
| midrange |  | flareon | high | Flareon ex + eeveelution toolbox | NoctowlエンジンでセットアップしFlareon加速、状況でイーブイ進化を使い分けるツールボックス | flexible toolbox midrange |
| midrange |  | garchomp_lucario | high | Cynthia's Garchomp ex/Mega Lucario | Garchomp Draconic Buster260とLucario Aura Jab加速をAir Balloonで回す闘2軸ビート | 2-attacker fighting beatdown w/ energy recycle |
| midrange | ★ | hydreigon | med | Mega Sharpedo ex/Hydreigon ex | Toxtricity自傷加速でSharpedo Hungry Jaws+150 OHKO、Munkidoriダメカン管理 | strong combo synergy (self-damage->Sharpedo); midrange/combo border |
| midrange |  | lillies_clefairy | med | Mega Kangaskhan ex + Lillie's Clef | 多数の2枚サイドexを並べ殴るツールボックス、Area Zeroでベンチ8にしClefairy Full Moon Rondoスケール | toolbox beatdown w/ board-scale element |
| midrange | ★ | manectric | med | Mega Manectric ex | Eelektrik Dynamotorでトラッシュ雷を繰返し加速しMega Manectric exを撃ち直す2メガビート | Eelektrik reload loop leans combo; could be combo |
| midrange | ★ | marnie_grimmsnarl | med | Marnie's Grimmsnarl ex | Froslass+Munkidoriでダメカン撒き、タフな2進化Grimmsnarlで殴りAdrena-Brainで回復/カン移動 | spread+resilient stage2 prize wincon; BUT our build very disruption-heavy -> could be cont |
| midrange |  | mega_diancie | med | Mega Diancie ex | Wondrous Patch/Powerglassでエネ積みGarland Ray(捨てエネ×120=240)連打、Diamond Coat | tanky mega beatdown+spread; reload loop leans combo |
| midrange |  | mega_dragonite | med | Mega Dragonite ex | 複数exを並べ加速で殴り分ける盤面ビート | unfocused ex toolbox, no Rare Candy |
| midrange |  | mega_feraligatr | med | Mega Feraligatr ex | Stage2立てMortal Crunch(ダメージ済み2倍=400)、Feraligatr/Munkidoriで事前ダメージ | damage-setup synergy, core stage2 beatdown |
| midrange |  | mega_gardevoir | med | Mega Gardevoir ex | Gardevoir Fantasia Forceで加速し超タイプexツールボックスを回す | accel toolbox; combo-adjacent |
| midrange |  | mega_gengar | med | Mega Gengar ex | Gengar Void Gale230自己加速+Marnie's Grimmsnarl悪加速の2枚看板、Boss x4で押す悪箱 | disruption-heavy but wincon=prizes; aggro-ish |
| midrange |  | mega_latias | med | Mega Latias ex | Blaziken ex+Crispin x4で加速しMega Latias ex(全エネ破棄300)等の大型Megaを連続起動 | accel over-cost megas; combo-adjacent |
| midrange |  | mega_starmie | med | Mega Starmie ex/Dragapult ex | 1エネJetting Blowとスプレッド+Munkidoriで複数同時KO | 2-attacker spread board-build |
| midrange |  | mega_venusaur | med | Mega Venusaur ex | Teal Dance/Solar Transferで草加速し4エネJungle Dump240連打の耐久ビート | accel-dependent 2-evo, combo-ish but gradual accel |
| midrange |  | mega_zygarde | med | Mega Zygarde ex | Barbaracle加速+Fighting GongでGaia Wave200を連打制圧 | accel-dependent big attacker; core=beatdown |
| midrange |  | ns_zoroark | med | N's Zoroark ex | Zoroark Night JokerでN's Zekrom(250)等の技をコピー、Trade高速ドロー+N's PP Up加速+N's  | summary primary(Zekrom) is backup; real=Zoroark copier; aggro-ish |
| midrange |  | ogerpon_box | high | Mega Kangaskhan ex + Ogerpon toolb | 多色トゥールボックス箱、Crispin/Prismで加速し相手に最適アタッカーを選んで殴る | slop box toolbox beatdown |
| midrange |  | okidogi_box | med | Okidogi + Bloodmoon Ursaluna | 闘単ビート、Barbaracle加速でOkidogi早期起動しUrsaluna Mad Bite(310)で締め | has disruption tech but core=fighting beatdown; mild combo flavor |
| midrange | ★ | omatsuri | med | Dipplin (Do the Wave) | ベンチ最大展開しDipplin Do the Wave(ベンチ×20)をFestival Leadで相手前に2回攻撃 | board-build beatdown; bench-scale+double-attack borders combo |
| midrange |  | raging_bolt | med | Raging Bolt ex + Mega Kangaskhan e | Teal Dance+Crispin+Energy Switchで溜めRaging Bolt Bellowing Thunder(捨てエネ× | energy-discard scaling leans combo; Kangaskhan box makes it midrange |
| midrange | ★ | rockets_mewtwo | med | Team Rocket's Mewtwo ex | Spidops/Tarountulaで TRポケ4体並べPower Saver解除、ベンチ充填しErasure Ball(280-330)で | tribal beatdown; 4-TR requirement+loading = assemble feel, could be combo |
| midrange | ★ | volcanion_box | med | Volcanion ex | 全Basic炎箱。Crispin/Cyrano+18炎エネで加速しSteam Up(重ね掛け)で火力底上げ、Reshiram ex等で叩く | all-basic box beatdown; could read combo (Steam Up+accel) but soft boost -> midrange |
| control | ★ | chandelure | med | Chandelure (Stage2) | Hammer/Xerosic/Eriでエネ断ち、Gravity Gemstoneで相手逃げエネを上げChandelureの逃げエネ参照打点を | non-ex chip + heavy denial; Gravity Gemstone combo finish -> could be combo |
| control |  | comfey_yveltal | high | Yveltal | Crushing Hammer x4/Xerosic x4でエネ枯らし、Acerola/Colressで回復し続け1エネYveltalで削り | disruption/recovery grind IS win con |
| control |  | crustle_stall | high | Crustle+Cornerstone Ogerpon ex | 壁で攻撃を無効化し続け回復とダメージ移動で相手をリソース/デッキ切れに追い込む二重ロック | textbook stall/lock |
| control |  | cubchoo_control | high | Dudunsparce(minor) | エネ破壊+特性ロック+ミルで相手を機能停止させ枯渇/山切れで勝つ | disruption IS win condition |
| combo |  | alakazam | med | Alakazam (Stage2) | 手札最大化しPowerful Handで手札×2ダメカン、Dunsparceドローで手札を膨らませEnhanced Hammerで妨害 | real attacker=Alakazam not Dudunsparce; defensible as midrange |
| combo | ★ | ceruledge | med | Ceruledge ex | Lunatone+Solrockでエネをトラッシュに溜め1エネAbyssal Flames(トラッシュ枚数加算)を大打点で撃つ | 1-energy but engine-dependent discard loop; could be aggro |
| combo | ★ | doublade | med | Aegislash/Doublade (hand-scaling) | 手札に鋼ラインを多数抱えSword Stashを枚数×60で撃つ、Genesect/Hildaで手札構築 | could be midrange evolution beatdown |
| combo |  | hydrapple | med | Hydrapple ex (scaling) | 草エネ蓄積+Meganium倍化エンジンを組みHydrapple exのスケール技でワンショット | ramp+doubling assemble engine; Teal Mask early beat -> midrange possible |
| combo | ★ | iono_bellibolt | med | Raging Bolt ex + Iono's Bellibolt  | Bellibolt特性で雷加速、Levincia回収しRaging Bolt Bellowing Thunder(捨てエネ数スケール) | accel-chain powers over-cost scaling attacker; could be midrange |
| combo |  | mamoswine | med | Mamoswine ex | Rare Candyで複数の2進化を並べBlaziken加速、Rumbling March(ベンチ2進化×40)を盤面依存でスケール | assemble 4 stage2 lines; payoff scales with benched stage2 count; setup non-productive unt |
| combo | ★ | mega_absol | med | Mega Absol ex | 毒/ダメカン操作で相手アクティブにちょうど6カウンター乗せTerminal Period即気絶、揃わねばClaw 200 | 6-counter poison combo instant-KO; midrange fallback + TR disruption -> midrange/control a |
| combo |  | mega_froslass | med | Greninja ex | Greninja Deadly Shuriken+Dusknoirカウンター移動+Froslass/Munkidoriアビリティでダメカン撒 | summary primary(Latias) wrong; ability-assemble spread; some control lean |
| combo |  | metagross | med | Steven's Metagross ex (engine)+too | Metagross加速エンジンを組みGenesect/Latias/Empoleon等のexツールボックスを起動 | summary primary wrong; engine-assemble toolbox; midrange toolbox possible |
| combo |  | slowking | high | Slowking (Seek Ascension copy) | トップ操作(Codebreaking/Academy)でSeek Ascensionが無ルール箱のKyurem/Metagressの技をコピ | top-deck-manip + copy engine assemble-then-payoff; Metagross has no line=copy target |

---

## step 3: OTHERS（★16件）の個別解決（2026-07-11）

方針: **4分類を維持**し、境界16件を一旦 OTHERS 保留 → **挙動ルール**で解決。
> 判定基準: 「エンジンが online になる前に攻撃すると価値を失うか？」
> No（削りながら組める）→ combo でない（midrange等）／ Yes（揃うまで攻撃無価値）→ combo。

**最終分布**: aggro 7 / midrange 34 / control 4 / combo 7。OTHERS は解決済みで空。

| deck | 元(agent) | → 最終 | 理由（挙動ルール） |
|---|---|---|---|
| mega_heracross | aggro | **aggro** | 単体アタッカーを最速で立て毎ターン殴る純レース。組み立て/進化なし |
| trevenant_control | aggro | **aggro** | Hop's Trevenant swarm。安いStage1で削りサイド先取（ファイル名は誤解）。ロックなし |
| deck | midrange | **midrange** | Mega Abomasnowは部分エネで T2 から殴れる＝gate無し。35エネは安定エンジン |
| hydreigon | midrange | **midrange** | Hydreigon exで殴りつつ組める。Sharpedo自傷はシナジーであってgateでない |
| manectric | midrange | **midrange** | Manectricは随時80+で殴れる。Eelektrik reloadは加速ループ（gate無し） |
| marnie_grimmsnarl | midrange | **midrange** | 勝ち筋=Grimmsnarl殴り＋spread。disruptionは支援。bespoke L2で個別対応 |
| omatsuri | midrange | **midrange** | Do the Waveは盤面を作りつつ削れる（漸増）。gate無しの盤面ビート |
| volcanion_box | midrange | **midrange** | Steam Upは漸増ソフトブースト、随時殴れる。全Basic箱ビート |
| ceruledge | combo | **midrange** | 1エネで随時殴れ、トラッシュ加算で伸びるランプ型ビート（gate無し） |
| iono_bellibolt | combo | **midrange** | Bellibolt 230で随時殴れる。Raging Boltは伸びしろ（gate無し） |
| mega_absol | combo | **midrange** | Claw 200の生産的攻撃あり。6カウンターTerminal Periodは締め（gate無し） |
| mega_froslass | combo | **midrange** | Greninja/Dusknoirで削りつつ組めるspread。勝ち筋=サイド取り |
| chandelure | control | **control** | 非exチップ＋重いエネ破壊/ロックが勝ち筋。denialグラインド |
| metagross | combo | **combo** | Metagross加速エンジンを組むまでtoolboxが機能しない＝assemble-gate。slow |
| rockets_mewtwo | midrange | **combo** | TRポケ4体でPower Saver解除＋ベンチ充填まで大技が撃てない＝gate |
| doublade | combo | **combo** | 手札に鋼ライン多数を揃えるまで打点が出ない（単体~60）。Genesectは組立用 |

**確定コンボ（assemble-gate 妥当）**: slowking, hydrapple, mamoswine, alakazam, metagross, rockets_mewtwo, doublade。
**タグ反映先**: `agents/tuning.json` の per-deck `archetype`（L0は無視／L1が profile 経由で読む・非破壊）。
**要 tuning 上書き（infer_roles の primary 誤り訂正）**: alakazam→Alakazam / ns_zoroark→N's Zoroark ex / metagross→Steven's Metagross ex / mega_froslass→Greninja ex / mamoswine→Mamoswine ex / mega_diancie→Mega Diancie ex。

---

## midrange 細分類（4サブ型・2026-07-11 確定）

midrange 34件を**操縦挙動**で細分（`archetype=midrange`＋`subtype`、tuning.json）。
`ramp` は4分類の外ではなく midrange サブ型に据える（overload vs 2キャップが真逆の操縦）。

| subtype | 操縦ノブ | デッキ |
|---|---|---|
| **beatdown** | primaryを2キャップ+backup予備装填（過剰装填しない） | ceruledge, crustle, cynthia_garchomp, garchomp_lucario, hydreigon, mega_feraligatr, mega_gengar, ns_zoroark |
| **ramp** | 加速最優先で1体にoverload、1発大技 | archaludon, deck, ethan_hooh, iono_bellibolt, manectric, mega_gardevoir, mega_venusaur, mega_zygarde, okidogi_box, volcanion_box |
| **spread** | ダメカン移動特性を毎ターン、残HP最小で同時KO | black_kyurem, dragapult, dragapult_blaziken, dragapult_dusknoir, marnie_grimmsnarl, mega_absol, mega_diancie, mega_froslass, mega_starmie, omatsuri |
| **toolbox** | 対面でアタッカー選択（固定primaryなし） | flareon, lillies_clefairy, mega_dragonite, mega_latias, ogerpon_box, raging_bolt |

境界: ns_zoroark(beatdown↔ramp), raging_bolt(toolbox↔ramp), mega_gengar(beatdown↔spread), ceruledge(beatdown↔ramp), dragapult_blaziken(spread↔beatdown)。tuning上書き＋A/Bで精緻化。
