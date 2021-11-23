# fMRI_GAN
- このリポジトリにデータはないため、自分で用意してください  

# その他  
- バージョン情報  
```
print(torch.__version__) # 1.9.1+cu111
```
- pytorchのバージョンによっては以下の処理を追加する必要がある。
```
label_real = label_real.type_as(d_out_real.view(-1))
label_fake = label_fake.type_as(d_out_fake.view(-1))
```
- CUDAのバージョンによるエラー
  - 下記コマンドでCUDAのバージョンを確認
```
nvidia-smi
```
  - 環境に合わせてインストールし直す。以下はその例
```
pip3 install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
```
