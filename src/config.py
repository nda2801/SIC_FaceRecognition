import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="SIC FaceViT for Face Recognition"
    )

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="sic_facevit_vggface2_semi_hard",
    )

    parser.add_argument(
        "--loss_type",
        type=str,
        default="triplet",
        choices=["triplet", "supcon"],
        help="Training loss",
    )

    parser.add_argument(
        "--dataset_root",
        type=str,
        default="dataset/vggface2",
    )

    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument(
        "--validation_identity_ratio",
        type=float,
        default=0.5,
    )

    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--identities_per_batch", type=int, default=8)
    parser.add_argument("--images_per_identity", type=int, default=4)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument("--epochs", type=int, default=100)

    parser.add_argument("--early_stop", type=int, default=10)
    parser.add_argument("--early_stop_min_delta", type=float, default=1e-4)

    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--patch_size", type=int, default=16)
    parser.add_argument("--embed_dim", type=int, default=192)
    parser.add_argument("--depth", type=int, default=12)
    parser.add_argument("--num_heads", type=int, default=3)
    parser.add_argument("--mlp_ratio", type=float, default=4.0)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--face_embedding_dim", type=int, default=128)

    # Triplet Loss
    parser.add_argument("--triplet_margin", type=float, default=0.2)

    # Supervised Contrastive Loss (InfoNCE)
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
    )

    parser.add_argument("--verification_pairs", type=int, default=10000)

    parser.add_argument(
        "--gallery_images_per_identity",
        type=int,
        default=5,
    )

    cfg = parser.parse_args()

    if not 0.0 < cfg.validation_identity_ratio < 1.0:
        parser.error("validation_identity_ratio phai nam trong khoang (0, 1)")

    if cfg.image_size % cfg.patch_size != 0:
        parser.error("image_size phai chia het cho patch_size")

    if cfg.identities_per_batch < 2:
        parser.error("identities_per_batch phai lon hon hoac bang 2")

    if cfg.images_per_identity < 2:
        parser.error("images_per_identity phai lon hon hoac bang 2")

    if cfg.verification_pairs < 2:
        parser.error("verification_pairs phai lon hon hoac bang 2")

    if cfg.gallery_images_per_identity < 1:
        parser.error("gallery_images_per_identity phai lon hon hoac bang 1")

    if cfg.loss_type == "supcon":
        if cfg.temperature <= 0:
            parser.error("temperature phai lon hon 0")

    return cfg


if __name__ == "__main__":
    print(get_parser())