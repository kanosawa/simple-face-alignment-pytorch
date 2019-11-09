if __name__ == '__main__':
    for n in range(1, 26):
        filename_w = '{0:06d}.pts'.format(n)
        filename_r = '{0:06d}_.pts'.format(n)
        with open('/root/datasets/face_landmark/mayu/' + filename_w, 'w') as file_w:
            try:
                with open('/root/datasets/face_landmark/mayu/' + filename_r, 'r') as file_r:
                    # header
                    file_w.write('version: 1\n')
                    for i in range(27):
                        file_w.write(file_r.readline().replace('\t', ' '))
            except IOError:
                print('file is nothing')

            