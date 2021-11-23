# fMRI_GAN
- このリポジトリにデータはないため、自分で用意してください  

# その他  
- pytorchのバージョンによっては以下の処理を追加する必要がある。
```
label_real = label_real.type_as(d_out_real.view(-1))
label_fake = label_fake.type_as(d_out_fake.view(-1))
```
