import argparse




if __name__ == '__main__':
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument('K', action='store', type=int)
    my_parser.add_argument('N', action='store', type=int)
    my_parser.add_argument('d', action='store', type=int)
    my_parser.add_argument('MAX_ITER', action='store', type=int)

    args = my_parser.parse_args()

    k_mean(args.K, args.N, args.d, args.MAX_ITER)