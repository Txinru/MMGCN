import argparse


def parameter_parser():

    parser = argparse.ArgumentParser(description="Run MMGCN.")

    parser.add_argument("--dataset-path",
                        nargs="?",
                        default="../datasets",
                        help="Training datasets.")

    parser.add_argument("--epoch",
                        type=int,
                        default=651,
                        help="Number of training epochs. Default is 651.")

    parser.add_argument("--gcn-layers",
                        type=int,
                        default=2,
                        help="Number of Graph Convolutional Layers. Default is 2.")

    parser.add_argument("--out-channels",
                        type=int,
                        default=128,
                        help="out-channels of cnn. Default is 128.")

    parser.add_argument("--miRNA-number",
                        type=int,
                        default=853,
                        help="miRNA number. Default is 853.")

    parser.add_argument("--fm",
                        type=int,
                        default=256,
                        help="miRNA feature dimensions. Default is 256.")

    parser.add_argument("--disease-number",
                        type=int,
                        default=591,
                        help="disease number. Default is 591.")

    parser.add_argument("--fd",
                        type=int,
                        default=256,
                        help="disease number. Default is 256.")

    parser.add_argument("--view",
                        type=int,
                        default=2,
                        help="views number. Default is 2(2 datasets for miRNA and disease sim)")


    parser.add_argument("--validation",
                        type=int,
                        default=5,
                        help="5 cross-validation.")


    return parser.parse_args()