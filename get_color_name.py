from numpy import linalg


def csv_reader(datafile, has_header):
    data = []
    if has_header:
        with open(datafile, "r") as f:
            header = f.readline().split(",")
            counter = 0
            for line in f:
                data.append(line)
                fields = line.split(",")
                counter += 1
    else:
        with open(datafile, "r") as f:
            data = f.read().splitlines()
        data = [line.split(',#') for line in data]
    return data


def get_color_name(R, G, B, color_csv_path):
    data = csv_reader(color_csv_path, True)
    minimum = 1000

    for i in range(len(data)):
        start_r = data[i].find("R/RGB:,") + len("R/RGB:,")
        end_r = data[i].find(",G/RGB:")
        substring_r = data[i][start_r:end_r]

        start_g = data[i].find("G/RGB:,") + len("G/RGB:,")
        end_g = data[i].find(",B/RGB:")
        substring_g = data[i][start_g:end_g]

        start_b = data[i].find("B/RGB:,") + len("B/RGB:,")
        end_b = data[i].find("\n")
        substring_b = data[i][start_b:end_b]

        d = abs(R - int(substring_r)) + abs(G - int(substring_g)) + abs(B - int(substring_b))

        if d <= minimum:
            minimum = d
            start = data[i].find("Name:,") + len("Name:,")
            end = data[i].find(",Hex")
            cname = data[i][start:end]
    return cname


def get_simple_color_names(R, G, B, color_csv_path):
    data = csv_reader(color_csv_path, has_header=False)
    min_diff = 1000
    min_idx = 0
    for i in range(len(data)):
        ref_color = data[i]
        #print(i, ref_color[0])
        R_ref = int(ref_color[1][0:2], 16)
        G_ref = int(ref_color[1][2:4], 16)
        B_ref = int(ref_color[1][4:6], 16)
        diff = (R - R_ref) ** 2 + (G - G_ref) ** 2 + (B - B_ref) ** 2
        if (diff < min_diff):
            min_idx = i
            min_diff = diff
    return data[min_idx][0]

print(get_simple_color_names(255, 255, 255, 'colors_x11.csv'))
