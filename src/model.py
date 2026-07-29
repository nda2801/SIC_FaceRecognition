import torch
from torch import nn
from torch.nn import functional as F


# ===========================
# Patch Embedding
# ===========================

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()

        self.number_of_patches = (image_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, images):
        patches = self.projection(images)
        patches = patches.flatten(2)
        return patches.transpose(1, 2)


# ===========================
# Feed Forward
# ===========================

class FeedForward(nn.Module):
    def __init__(self, embed_dim, hidden_dim, dropout):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.layers(x)


# ===========================
# LayerScale
# ===========================

class LayerScale(nn.Module):
    def __init__(self, dim, init_value=1e-5):
        super().__init__()
        self.gamma = nn.Parameter(init_value * torch.ones(dim))

    def forward(self, x):
        return self.gamma * x


# ===========================
# DropPath
# ===========================

class DropPath(nn.Module):

    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):

        if self.drop_prob == 0.0 or not self.training:
            return x

        keep_prob = 1 - self.drop_prob

        shape = (x.shape[0],) + (1,) * (x.ndim - 1)

        random_tensor = keep_prob + torch.rand(
            shape,
            dtype=x.dtype,
            device=x.device,
        )

        random_tensor.floor_()

        return x.div(keep_prob) * random_tensor


# ===========================
# Transformer Block
# ===========================

class TransformerEncoderBlock(nn.Module):

    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_ratio,
        dropout,
        drop_path=0.1,
    ):
        super().__init__()

        hidden_dim = int(embed_dim * mlp_ratio)

        self.norm_attention = nn.LayerNorm(embed_dim)

        self.attention = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.layer_scale1 = LayerScale(embed_dim)

        self.drop_path1 = DropPath(drop_path)

        self.norm_mlp = nn.LayerNorm(embed_dim)

        self.mlp = FeedForward(
            embed_dim,
            hidden_dim,
            dropout,
        )

        self.layer_scale2 = LayerScale(embed_dim)

        self.drop_path2 = DropPath(drop_path)

    def forward(self, tokens):

        normalized = self.norm_attention(tokens)

        attention_output, _ = self.attention(
            normalized,
            normalized,
            normalized,
            need_weights=False,
        )

        tokens = tokens + self.drop_path1(
            self.layer_scale1(attention_output)
        )

        mlp_output = self.mlp(
            self.norm_mlp(tokens)
        )

        tokens = tokens + self.drop_path2(
            self.layer_scale2(mlp_output)
        )

        return tokens


# ===========================
# Face Vision Transformer
# ===========================

class FaceVisionTransformer(nn.Module):

    def __init__(
        self,
        image_size,
        patch_size,
        in_channels,
        embed_dim,
        depth,
        num_heads,
        mlp_ratio,
        dropout,
        face_embedding_dim,
    ):
        super().__init__()

        self.patch_embedding = PatchEmbedding(
            image_size,
            patch_size,
            in_channels,
            embed_dim,
        )

        number_of_patches = self.patch_embedding.number_of_patches

        self.class_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.position_embedding = nn.Parameter(
            torch.zeros(
                1,
                number_of_patches + 1,
                embed_dim,
            )
        )

        self.embedding_dropout = nn.Dropout(dropout)

        drop_rates = torch.linspace(
            0,
            0.1,
            depth,
        ).tolist()

        self.encoder_blocks = nn.Sequential(

            *[
                TransformerEncoderBlock(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    dropout,
                    drop_path=drop_rates[i],
                )

                for i in range(depth)
            ]
        )

        self.final_norm = nn.LayerNorm(embed_dim)

        self.embedding_head = nn.Linear(
            embed_dim,
            face_embedding_dim,
        )

        nn.init.trunc_normal_(
            self.class_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.position_embedding,
            std=0.02,
        )

    def forward_features(self, images):

        patch_tokens = self.patch_embedding(images)

        batch_size = patch_tokens.shape[0]

        class_tokens = self.class_token.expand(
            batch_size,
            -1,
            -1,
        )

        tokens = torch.cat(
            (class_tokens, patch_tokens),
            dim=1,
        )

        tokens = self.embedding_dropout(
            tokens + self.position_embedding
        )

        tokens = self.encoder_blocks(tokens)

        tokens = self.final_norm(tokens)

        return tokens[:, 0]

    def forward(self, images):

        features = self.forward_features(images)

        embeddings = self.embedding_head(features)

        return F.normalize(
            embeddings,
            p=2,
            dim=1,
        )


# ===========================
# Build Model
# ===========================

def build_model(cfg):

    return FaceVisionTransformer(
        image_size=cfg.image_size,
        patch_size=cfg.patch_size,
        in_channels=3,
        embed_dim=cfg.embed_dim,
        depth=cfg.depth,
        num_heads=cfg.num_heads,
        mlp_ratio=cfg.mlp_ratio,
        dropout=cfg.dropout,
        face_embedding_dim=cfg.face_embedding_dim,
    )


if __name__ == "__main__":

    from config import get_parser

    cfg = get_parser()

    model = build_model(cfg)

    images = torch.randn(
        2,
        3,
        cfg.image_size,
        cfg.image_size,
    )

    embeddings = model(images)

    print(images.shape)

    print(embeddings.shape)

    print(torch.linalg.vector_norm(embeddings, dim=1))

    print(
        sum(
            p.numel()
            for p in model.parameters()
        )
    )