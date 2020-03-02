import numpy as np
def read_obj_file(fname):
    vertices = []
    faces = []
    try:
        f = open(fname)

        for line in f:
            if line[:2] == "v ":
                strs = line.split()
                v0 = float(strs[1])
                v1 = float(strs[2])
                v2 = float(strs[3])
                vertex = [v0, v1, v2]
                vertices.append(vertex)

            elif line[0] == "f":
                strs = line.split()
                f0 = int(strs[1].split('/')[0])-1
                f1 = int(strs[2].split('/')[0])-1
                f2 = int(strs[3].split('/')[0])-1
                face = [f0, f1, f2]

                faces.append(face)

        f.close()
    except IOError:
        print(".obj file not found.")

    vertices = np.array(vertices)
    faces = np.array(faces)

    return vertices, faces


def rotate_obj_file(fname, rot_mat):

    lines = []
    with open(fname) as fin:

        for line in fin:
            lines.append(line)

        for i in range(len(lines)):
            line = lines[i]
            if line[:2] == "v ":
                strs = line.split(' ')
                vertex = np.array([float(strs[1]), float(strs[2]), float(strs[3])])
                vertex = np.matmul(rot_mat, vertex)
                line = "v {0} {1} {2}\n".format(vertex[0], vertex[1], vertex[2])
                lines[i] = line
            elif line[:3] == 'vn ':
                strs = line.split(' ')
                vn = np.array([float(strs[1]), float(strs[2]), float(strs[3])])
                vn = np.matmul(rot_mat, vn)
                line = "vn {0} {1} {2}\n".format(vn[0], vn[1], vn[2])
                lines[i] = line

    with open(fname, 'w') as f:
        for line in lines:
            f.write(line)


def write_off_file(fname, vertices, faces):
    with open(fname, 'w') as f:
        vnum = len(vertices)
        fnum = len(faces)
        f.write('COFF\n')
        f.write('{0} {1} {2}\n'.format(vnum, fnum, 0))
        for i in range(0, vnum):
            f.write('{0} {1} {2}\n'.format(vertices[i][0], vertices[i][1], vertices[i][2]))

        fnum = len(faces)
        for i in range(0, fnum):
            f.write('3 {0} {1} {2}\n'.format(faces[i][0], faces[i][1], faces[i][2]))

def read_off_file(fname):
    vertices = []
    faces = []
    try:
        f = open(fname)
        head = f.readline()
        strline = f.readline()
        strs = strline.split(' ')
        vnum = int(strs[0])
        fnum = int(strs[1])
        for i in range(0, vnum):
            strline = f.readline()
            strs = strline.split(' ')
            v0 = float(strs[0])
            v1 = float(strs[1])
            v2 = float(strs[2])
            vertex = [v0, v1, v2]
            vertices.append(vertex)

        for i in range(0, fnum):
            strline = f.readline()
            strs = strline.split(' ')
            f0 = int(strs[1])
            f1 = int(strs[2])
            f2 = int(strs[3])
            face = [f0, f1, f2]
            faces.append(face)

        f.close()
    except IOError:
        print(".off file not found.")

    vertices = np.array(vertices)
    faces = np.array(faces)
    return vertices, faces


def write_obj_file(fname, vertices, faces):
    with open(fname, 'w') as f:
        vnum = len(vertices)
        for i in range(0, vnum):
            f.write('v {0} {1} {2}\n'.format(vertices[i][0], vertices[i][1], vertices[i][2]))

        fnum = len(faces)
        for i in range(0, fnum):
            f.write('f {0} {1} {2}\n'.format(faces[i][0]+1, faces[i][1]+1, faces[i][2]+1))





