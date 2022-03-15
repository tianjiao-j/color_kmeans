def csv_reader(datafile):
    data = []
    with open(datafile, "r") as f:
        header = f.readline().split(",")
        counter = 0
        for line in f:
            data.append(line)
            fields = line.split(",")
            counter += 1
    return data


def get_color_name(R, G, B, color_csv_path):
    data = csv_reader(color_csv_path)
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
