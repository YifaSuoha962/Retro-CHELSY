import os
import logging

from utils.parsing import get_parser, post_setting_args
from utils.retro_datasets import select_reader
from utils.preprocess_smi import reaction_sequence
from utils.rxn_feats import Data_Preprocess
from utils.eval_split import Data_Split


proj_dir = os.path.dirname(os.path.realpath(__file__))
output_dir = os.path.join(proj_dir, 'preprocessed')

def main(args):
    # init dataset
    raw_reader = select_reader(proj_dir, output_dir, args.dataset_name)  # , proj_dir
    logging.basicConfig(filename=os.path.join(raw_reader.out_dir, 'preprocess.log'),
                        format='%(asctime)s %(message)s', level=logging.INFO)
    for k, v in args.__dict__.items():
        logging.info('args -> {0}: {1}'.format(k, v))

    raw_reader.csv_process()
    logging.info('csv/txt preprocess finish.')

    # extract products, reactants, rxn_types and save them separately
    csv_preprocess = reaction_sequence(
        root_dir=output_dir,
        output_dir=output_dir,
        data_name=args.dataset_name,
        split_type=args.split_type
    )
    csv_preprocess.get_vocab_dict()
    csv_preprocess.smi_encode()
    logging.info('maximum reaction length: {0}'.format(csv_preprocess.max_len))
    logging.info('average reaction length: {0}'.format(csv_preprocess.avg_len))
    logging.info('total reaction data count: {0}'.format(csv_preprocess.reaction_count))
    logging.info('\n')


    if args.graph_preprocess:
        graph_preprocess = Data_Preprocess(
            output_dir=output_dir,
            data_name=args.dataset_name,
            split_type=args.split_type,
            need_atom=args.need_atom,
            need_graph=args.need_graph,
            self_loop=args.self_loop,
            vnode=args.vnode
        )
    if args.split_preprocess:
        eval_split = Data_Split(
            output_dir=output_dir,
            data_name=args.dataset_name,
            split_type=args.split_type,
            seed=args.seed
        )
        eval_split.split(
            split_len=args.split_len,
            split_name=args.split_name
        )


if __name__ == '__main__':
    parser = get_parser(mode='preprocess')
    args = post_setting_args(parser)

    main(args)

