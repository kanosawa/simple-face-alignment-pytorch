if __name__ == '__main__':
    for n in range(1, 11):
        filename_w = '{0:06d}.pts'.format(n)
        filename_r = '{0:06d}_.pts'.format(n)
        with open('/root/datasets/madomagi/madomagi_face_tmp/homura/' + filename_w, 'w') as file_w:
            try:
                with open('/root/datasets/madomagi/madomagi_face_tmp/homura/' + filename_r, 'r') as file_r:
                    # header
                    file_w.write('version: 1\n')
                    file_w.write('n_points: 68\n')
                    file_w.write('{\n')
                    for i in range(2):
                        file_r.readline()
                    # contour & brow
                    for i in range(27):
                        file_w.write(file_r.readline().replace('\t', ' '))
                    # nose
                    nose = file_r.readline().replace('\t', ' ')
                    for i in range(9):
                        file_w.write(nose)
                    # eye
                    for i in range(12):
                        file_w.write(file_r.readline().replace('\t', ' '))
                    # lip
                    lips = []
                    for i in range(12):
                        lip = file_r.readline().replace('\t', ' ')
                        file_w.write(lip)
                        lips.append(lip)
                    lip_no_list = [0, 2, 3, 4, 6, 8, 9, 10]
                    for i in lip_no_list:
                        file_w.write(lips[i])
                    file_w.write('}\n')
            except IOError:
                print('file is nothing')

            