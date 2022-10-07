import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')


def split_title_line(title_text, max_words=5):
    """
    A function that splits any string based on specific character
    (returning it with the string), with maximum number of words on it
    """
    seq = title_text.split()
    return '\n'.join([' '.join(seq[i:i + max_words]) for i in range(0, len(seq), max_words)])


def plot_alignment(alignment, path, info=None, split_title=False):
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111)

    im = ax.imshow(alignment, aspect='auto',
                   origin='lower', interpolation='none')
    fig.colorbar(im, ax=ax)
    xlabel = 'Decoder timestep'
    title = split_title_line(info) if split_title else info
    plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel('Encoder timestep')
    plt.tight_layout()
    plt.savefig(path, format='png')
    plt.close()
