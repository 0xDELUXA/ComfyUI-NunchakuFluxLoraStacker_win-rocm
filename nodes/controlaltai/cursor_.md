# ノードの画像リンクの置き換え
_Exported on 2026/5/19 at 10:38:09 JST from Cursor (3.4.20)_

---

**User**

ノードの画像リンクがすべて消えてしまっている

一つずつ置き換えよう

まずadvanced sampler画像作ったから置き換えろ

---

**Cursor**

Advanced Sampler の画像リンクを置き換えます。まずリポジトリ内の参照箇所と新しい画像ファイルを確認します。


---

**User**

"D:\USERFILES\GitHub\controlaltai-nodes\png\Advanced Sampler.png"

---

**Cursor**

Advanced Sampler セクションの壊れた外部リンクを、ご指定のローカル画像に置き換えます。


**ControlAltAI: Advanced Sampler** の説明直下を更新しました。

- 削除: 切れていた `gseth.com` の画像 3 枚（SNAG-7361 / 7362 / 7363）
- 追加: リポジトリ内の `png/Advanced Sampler.png` を 1 枚参照

```markdown
![ControlAltAI: Advanced Sampler](png/Advanced%20Sampler.png)
```

次のノード用画像ができたら、同じ要領で置き換えます。

---

**User**

push

---

**Cursor**

README.md と `png/Advanced Sampler.png` をコミットして push します。


コミットに `Co-authored-by` が付いていたため、除去して修正します。


push まで完了いたしました。

**コミット:** `0d03acb` — `docs: add local Advanced Sampler screenshot and update README`  
**リモート:** `origin/master`（`9a078f9` → `0d03acb`）

**含まれる変更**
- `png/Advanced Sampler.png` を追加
- Advanced Sampler 節の画像を `png/Advanced%20Sampler.png` に差し替え（旧 `gseth.com` リンク 3 件を削除）

**Author / Committer:** `ussoewwin`（`Co-authored-by` なし）

他ノードの画像も差し替える場合は、対応する `png/` のファイル名をお知らせください。

---

**User**

400サイズにしろ

---

**Cursor**

`png/Advanced Sampler.png` を幅 **400px**（アスペクト比維持）にリサイズいたしました。

- **変更前:** 3112 × 1365  
- **変更後:** 400 × 175  

README の参照パスはそのままで問題ございません。コミット・push も必要でしたらお申し付けください。

---

**User**

puah

---

**Cursor**

push 完了いたしました。

**コミット:** `26f6501` — `docs: resize Advanced Sampler screenshot to 400px width`  
**リモート:** `origin/master`（`0d03acb` → `26f6501`）

---

**User**

変わってねえぞ

---

**Cursor**

GitHub 上の表示を確認し、README の画像指定も見直します。


リポジトリ上の画像は既に 400×175 です。GitHub の CDN キャッシュで旧画像が表示されている可能性が高いため、README で表示幅を 400px に固定します。


---

**User**

psuhまでぃっきにやれ止まるな

---

**Cursor**

push まで完了いたしました。

**コミット:** `27038ac` — README で `<img width="400">` に変更（GitHub 上の表示幅を 400px に固定）  
**リモート:** `origin/master`（`26f6501` → `27038ac`）

PNG 本体は既に 400×175 ですが、GitHub が旧画像をキャッシュしていると大きく見えることがあります。README の `width="400"` で表示サイズを固定しています。数分後に README を再読み込み（必要なら Ctrl+F5）してご確認ください。

---

**User**

Union ControlNet Applyもやれ

---

**Cursor**

Union ControlNet Apply セクションの切れた外部画像をローカル PNG に差し替え、400px にリサイズしてコミット・push まで一気に進めます。


**Union ControlNet Apply** まで完了し、`origin/master` に push 済みです。

**実施内容**
- `png/Union ControlNet Apply.png` を幅 **400px** にリサイズして追加
- README の `gseth.com` 画像（16 枚）を削除し、ローカル 1 枚＋`<img width="400">` に差し替え
- 説明文（Recommended Settings、各 Results 見出し、YouTube / Hugging Face リンク）はそのまま残しています

**コミット:** `9f8cf9a` — `docs: add local Union ControlNet Apply screenshot and update README`

---

**User**

Get Image Size Ratio

---

**Cursor**

**Get Image Size Ratio** まで完了し、`origin/master` に push 済みです。

**実施内容**
- `png/Get Image Size Ratio.png` を幅 **400px** にリサイズして追加
- README の `SNAG-3959.png`（gseth.com）を `<img width="400">` のローカル参照に差し替え

**コミット:** `3f08a6c` — `docs: add local Get Image Size Ratio screenshot and update README`

---

**User**

Integer Settings

---

**Cursor**

**Integer Settings** まで完了し、`origin/master` に push 済みです。

**実施内容**
- `png/Integer Settings.png` を幅 **400px** にリサイズして追加
- README の `SNAG-4239.png`（gseth.com）をローカル `<img width="400">` に差し替え

**コミット:** `534a88d` — `docs: add local Integer Settings screenshot and update README`

---

**User**

Integer Settings Advanced

---

**Cursor**

**Integer Settings Advanced** の対応を完了し、`origin/master` へ push 済みです。

| 項目 | 内容 |
|------|------|
| 画像 | `png/Integer Settings Advanced.png`（幅 400px にリサイズ） |
| README | `SNAG-7521.png` を `<img src="png/Integer%20Settings%20Advanced.png" width="400" ...>` に差し替え |
| コミット | `707bdd6` — `docs: add local Integer Settings Advanced screenshot and update README` |

次は **Switch (Two Way/Three Way)**（`SNAG-7519.png`）など、続けて対応するノードがあればお知らせください。

---

**User**

 ControlAltAI: Switch 

---

**Cursor**

**ControlAltAI: Switch（Two Way / Three Way）** 節の対応を完了し、`origin/master` に push しました。

| 項目 | 内容 |
|------|------|
| 画像 | `png/ControlAltAI Switch.png`（幅 400px にリサイズ済み） |
| README | `SNAG-7519.png` を `<img src="png/ControlAltAI%20Switch.png" width="400" alt="ControlAltAI: Switch (Two Way) / ControlAltAI: Switch (Three Way)">` に差し替え |
| コミット | `b503f30` — `docs: add local Switch screenshot and update README` |

続きは **Choose Upscale Model**（`SNAG-4240`）など、指定いただければ同様に進めます。

---

**User**

<system_notification>
The following task has finished. If you were already aware, ignore this notification and do not restate prior responses.

<task>
kind: shell
status: success
task_id: 126714
title: Commit and push Switch node README + PNG
output_path: C:\Users\ussoe\.cursor\projects\d-USERFILES-GitHub-controlaltai-nodes\terminals\126714.txt
</task>
</system_notification>
<user_query>Briefly inform the user about the task result and perform any follow-up actions (if needed). If there's no follow-ups needed, don't explicitly say that.</user_query>

Choose Upscale Model

はノード自体を削除しろ

ファイルを消し、initから消し、readmeからも消せ

---

**Cursor**

Choose Upscale Model ノードをコード・`__init__`・README から完全に削除します。まず参照箇所を洗い出します。


**ControlAltAI: Choose Upscale Model** ノードをリポジトリから外しました。

### 実施内容
1. **削除** — `choose_upscale_model_node.py`
2. **`__init__.py`** — import（相対・フォールバック）、`NODE_CLASS_MAPPINGS`、`NODE_DISPLAY_NAME_MAPPINGS` から `ChooseUpscaleModel` を削除
3. **`README.md`** — ノード一覧表・Logic リスト・専用セクション（`SNAG-4240.png` 含む）を削除
4. **`.tracking`** — 削除ファイルの行を除去（マニフェスト整合のため）

`choose_upscale_model_node.py.backup` と `__init__.py.backup` は触っていません。

コミット／push が必要でしたらお知らせください。

---

**User**

push

---

**Cursor**

push まで完了いたしました。

- **コミット:** `d95d73f` — `remove Choose Upscale Model node`
- **リモート:** `origin/master`（`b503f30` → `d95d73f`）

変更内容: `choose_upscale_model_node.py` 削除、`__init__.py` / `README.md` / `.tracking` からの除去です。

---

**User**

Perturbation Texture

---

**Cursor**

Perturbation Texture の README 画像を、これまでと同様にローカル `png/` へ差し替えます。まず README と既存画像を確認します。


---

**User**

Perturbation Texture

治ってねえ

---

**Cursor**

README の Perturbation Texture セクションを確認し、gseth リンクをローカル `png/` に差し替えます。


---

**User**

続けろ

