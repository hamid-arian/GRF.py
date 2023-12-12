def split_sequence(start, end, num_parts):
    result = []
    if num_parts == 1:
        return [start, end + 1]

    length = end - start + 1
    if num_parts > length:
        return list(range(start, end + 2))

    part_length_short = length // num_parts
    part_length_long = -(-length // num_parts)  # Equivalent to ceil
    cut_pos = length % num_parts

    for i in range(start, start + cut_pos * part_length_long, part_length_long):
        result.append(i)
    
    for i in range(start + cut_pos * part_length_long, end + 2, part_length_short):
        result.append(i)

    return result
