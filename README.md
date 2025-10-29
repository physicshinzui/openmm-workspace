# 01_hello_world

このディレクトリは OpenMM を使ったシンプルな分子動力学 (MD) ワークフローのサンプルです。  
YAML でパラメータを管理し、エネルギー最小化 → NVT → NPT → Production という流れでシミュレーションを実行し、結果を解析するための基本的なスクリプトを含みます。

## 内容物
- `01_md.py` — メインの MD 実行スクリプト。PyYAML で設定を読み込み、MDAnalysis の selection で位置拘束原子を指定できます。
- `config.yaml` — 入力パラメータと入出力ファイルの指定。シミュレーションの段階やレポート間隔もここで管理します。
- `requirements.txt` — 必須パッケージ一覧 (`openmm` を含む適切な環境に加えて `pyyaml`, `MDAnalysis` が必要)。
- `utils/monitor_basic_quantity.py` — `md_log.txt` を読み込み、時間軸 (ns) に変換した各種物理量（ポテンシャル/運動/全エネルギー、温度、密度、ns/day など）をまとめて可視化するスクリプト。
- `utils/trjconv.py` — トラジェクトリ後処理のための補助スクリプト。
- `1AKI.pdb` — 初期構造。
- `top.pdb` / `minimized.pdb` / `traj.dcd` / `md_log.txt` — シミュレーション結果（それぞれトポロジー、エネルギー最小化後構造、軌跡、ログ）。
- `traj_fixed.dcd`, `pbc.py` — 参考や補助用の追加ファイル。

## セットアップ
1. 適切な Python 環境を用意し、OpenMM が動作することを確認します。
2. 必要なパッケージをインストールします。
   ```bash
   pip install -r requirements.txt
   ```

## シミュレーションの実行
```bash
python 01_md.py --config config.yaml
```

処理の流れ:
1. 入力 PDB を読み取り、溶媒・イオンを追加。
2. エネルギー最小化を行い、`paths.minimized` で指定したファイルへ構造を書き出し。
3. `restraints.selection` で選んだ原子に位置拘束を掛けたまま NVT と NPT を実行。
4. 拘束を外し、Production ランを続行。
5. 指定した間隔で `traj.dcd` と `md_log.txt` に出力。

## 設定ファイル (`config.yaml`)
主なキーは次の通りです。

- `paths`  
  - `pdb` / `topology` / `minimized` / `trajectory` / `log` / `checkpoint`: 入出力ファイルパス。
- `force_fields`: ForceField XML ファイル群。
- `thermodynamics`: 温度、圧力、摩擦係数、タイムステップ。デフォルトでは 0.004 ps (=4 fs) のタイムステップを採用しており、HMR を前提とした値です。
- `system`: 非結合カットオフ、ソルベントパディング、イオン濃度、`hydrogen_mass` による HMR 設定（例: 4.0 amu）。HMR を無効にする場合はこのキーを削除し、ステップサイズも適宜 0.002 ps に戻してください。
- `simulation`: 各フェーズのステップ数 (`nvt_steps`, `npt_steps`, `production_steps`)。既定値では NVT/NPT が 100 ps、Production が 1 µs になります。
- `reporting`: DCD 出力間隔、標準出力/ログへのレポート間隔。
- `restraints`: 
  - `force_constant` — 位置拘束の力定数 (kJ/mol/nm²)。
  - `selection` — MDAnalysis の選択式。例: `"protein and not name H*"`、`"name CA or name CB"` など。選択が空集合になるとエラーになります。

## 結果ファイル
- `top.pdb` — 初期構造を基に水・イオンを追加したトポロジー。
- `minimized.pdb` — エネルギー最小化後の構造。
- `traj.dcd` — Production まで含むトラジェクトリ。
- `md_log.txt` — ステップ数・時間(ps)・各種エネルギー・温度・体積・密度・ns/day などを CSV 形式で記録。
- `checkpoint.chk` — `CheckpointReporter` が書き出す最新のシミュレーション状態。ファイル名は `paths.checkpoint` で変更できます。

## 解析
`utils/monitor_basic_quantity.py` を使うと `md_log.txt` の内容を簡単に可視化できます。Matplotlib が必要です。
```bash
python utils/monitor_basic_quantity.py
```
表示されるグラフで収束状況を確認してください。必要に応じてスクリプトを改造し、任意の物理量をプロットできます。

## 備考
- `pbc.py` は周期境界条件関連の補助スクリプトです。
- 他のシステムで流用する際は、力場ファイルや溶媒条件を適宜差し替えてください。
- 生成された `traj.dcd` や `md_log.txt` は `.gitignore` 済みで、そのままワークスペースに残せます。
- `checkpoint.chk` などチェックポイントを保存しておけば `--restart` オプションで Production を再開できます。

## リスタート (Checkpoint)

シミュレーションを途中で止めた後に再開する場合は、`CheckpointReporter` が書き出すファイルを利用します。

- `config.yaml` の `paths.checkpoint` で保存先を指定できます（デフォルトは `checkpoint.chk`）。
- 実行時に `python 01_md.py --config config.yaml --restart` とすると、チェックポイントを読み込んで Production ステージだけを再開します。
- 別のファイル名を使いたい場合は `--checkpoint custom.chk` のように CLI 引数で上書きできます。
- チェックポイントは `config.reporting.log_interval` の頻度で更新されます。
