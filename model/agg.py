class Proj_img(nn.Sequential):
    def __init__(self, embedding_dim=768, proj_dim=768, drop_proj=0.3):
        super().__init__(
            nn.Linear(embedding_dim, proj_dim),
            ResidualAdd(nn.Sequential(
                nn.GELU(),
                nn.Linear(proj_dim, proj_dim),
                nn.Dropout(drop_proj),
            )),
            nn.LayerNorm(proj_dim),
        )
    def forward(self, x):
        return x

class AGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.prj = nn.Linear(feature_inp_dim, 512 + 256)  

        self.mha = nn.MultiheadAttention(embed_dim=768, num_heads=1, batch_first=True)
        self.norm = nn.LayerNorm([768])

        self.mlp = Mlp(768)
        self.mlp_head = MlpHead(768, num_classes=len(CLASSES), head_dropout=.5)  
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.prj(x)  

        x_att, w_att = self.mha(x, x, x)
        x = x_att + x
        x = self.norm(x) 

        x_mlp = self.mlp(x)
        x = x_mlp.mean([0])  
        x = self.mlp_head(x)
        x = self.sigmoid(x)
        return x

class combine(nn.Module):

    def __init__(self, proj_img, agg):

        super().__init__()

        self.proj_img = proj_img

        self.agg = agg
 
    def forward(self, x):

        img_feat = self.proj_img(x)  

        seq = self.agg(img_feat)          
        return seq
 