import argparse
import os
import pandas as pd
import numpy as np


def main(args):
    df = pd.read_csv(args.filename,
                     sep='&',
                     header=None,
                     skiprows=4,
                     engine='python')
    df = df.dropna()
    df[0] = df[0].apply(lambda x: x.strip())
    df = df.set_index(0)
    df[10] = df[10].apply(lambda x: x.replace('\\', '').strip())
    df = df.astype(float)
    df = df*100
    if 'Avg Halting' in df.index:
        df.loc[['Avg Halting', 'Std Halting']] = df.loc[['Avg Halting', 'Std Halting']] / 100
    df = df.applymap(lambda x: "{:.1f}".format(x))
    df.loc['Character Accuracy ± Std'] = df.loc['Character Accuracy'] + '±' + df.loc['Character Accuracy Std']
    df.loc['Sequence Accuracy ± Std'] = df.loc['Sequence Accuracy'] + '±' + df.loc['Sequence Accuracy Std']
    df = df.drop(['Character Accuracy', 'Character Accuracy Std', 'Sequence Accuracy', 'Sequence Accuracy Std'])
    if 'Avg Halting' in df.index:
        df.loc['Avg Halting ± Std'] = df.loc['Avg Halting'] + '±' + df.loc['Std Halting']
        df = df.drop(['Avg Halting', 'Std Halting'])        

    df.index.name = 'Nesting'
    original_filename = args.filename.split('/')[-1].split('.')[0]
    df.to_latex(os.path.join(args.output_dir, f'{original_filename}_paper.tex'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('filename')
    parser.add_argument('output_dir')
    args = parser.parse_args()
    main(args)
