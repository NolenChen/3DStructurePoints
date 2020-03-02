class LogUtils():
    def __init__(self, fname, filemode):
        self.logf = open(fname, filemode)

    def write(self, text, need_display=True):
        if need_display is True:
            print(text)

        self.logf.write(text + '\n')
        self.logf.flush()

    def close(self):
        self.logf.close()

    def write_args(self, cmd_args):
        self.logf.write('cmd arguments:\n')
        for k in cmd_args.__dict__:
            val = cmd_args.__dict__[k]
            self.logf.write('{0}:  {1}\n'.format(k, val))

    def write_correspondence_accuracy(fname, dis_ratio, dis_threshold):
        with open(fname, 'w') as f:
            f.write('Correspondence Accuracy:\n')
            f.write('distance ratio: ')
            for i in range(dis_ratio.shape[0]):
                f.write(' {0}'.format(dis_ratio[i]))
            f.write('\n')

            f.write('distance threshold: ')
            for i in range(dis_threshold.shape[0]):
                f.write(' {0}'.format(dis_threshold[i]))
            f.write('\n')
            f.close()

















